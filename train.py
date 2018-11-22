import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import argparse
import numpy as np
import pandas as pd
import yaml
import scipy.io
import tqdm

from addict import Dict
from itertools import zip_longest
from PIL import Image, ImageFilter
from tensorboardX import SummaryWriter

from models.deeplabv2 import DeepLabV2
from models.msc import MSC
from models.discriminator import Discriminator
from dataset import PartAffordanceDataset, PartAffordanceDatasetWithoutLabel
from dataset import CenterCrop, ToTensor, Normalize



''' one-hot representation '''

def one_hot(label, n_classes, device, requires_grad=True):
    one_hot_label = torch.eye(n_classes, dtype=torch.long, requires_grad=requires_grad, device=device)[label].transpose(1, 3).transpose(2, 3)
    return one_hot_label
    

''' scheduler for learning rate '''

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    optimizer.param_groups[0]["lr"] = new_lr
    optimizer.param_groups[1]["lr"] = 10 * new_lr
    optimizer.param_groups[2]["lr"] = 20 * new_lr


def poly_lr_scheduler_d(optimizer, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    optimizer.param_groups[0]["lr"] = new_lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = 10 * new_lr


''' model, weight initialization, get params '''

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.weight, 1)


def DeepLabV2_ResNet101_MSC(n_classes):
    return MSC(
        scale=DeepLabV2(
            n_classes=n_classes, n_blocks=[3, 4, 23, 3], pyramids=[6, 12, 18, 24]
        ),
        pyramids=[0.5, 0.75],
    )


def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias



''' training '''

def full_train(model, sample, criterion_ce_full, optimizer, device):

    ''' full supervised learning for segmentation network'''

    model.train()
    model.scale.freeze_bn()

    x, y = sample['image'], sample['class']

    x = x.to(device)
    y = y.to(device)

    h = model(x)     # shape => (N, 8, H/8, W/8)
    h = F.interpolate(h, size=(256, 320), mode='bilinear', align_corners=True)

    loss_ce = criterion_ce_full(h, y)

    optimizer.zero_grad()
    loss_ce.backward()
    optimizer.step()

    return loss_ce.item()



def adv_train(
        model, model_d, sample, criterion_ce_ful, criterion_bce, 
        optimizer, optimizer_d, ones, zeros, device):

    ''' full supervised and adversarial learning '''
    
    model.train()
    model.scale.freeze_bn()
    model_d.train()

    # train segmentation network
    x, y = sample['image'], sample['class']

    batch_len = len(x)

    x = x.to(device)
    y = y.to(device)

    h = model(x)     # shape => (N, 8, H/8, W/8)
    h = F.interpolate(h, size=(256, 320), mode='bilinear', align_corners=True)

    h_ = h.detach()    # h_ is for calculating loss for discriminator
    y_ = y.detach()    # y_is for the same purpose.  shape => (N, H, W)

    d_out = model_d(h)    # shape => (N, 1, H, W)
    d_out = d_out.squeeze()
    
    loss_ce = criterion_ce_full(h, y)
    loss_adv = criterion_bce(d_out, ones[:batch_len])
    loss_full = loss_ce + 0.01 * loss_adv

    optimizer.zero_grad()
    optimizer_d.zero_grad()
    loss_full.backward()
    optimizer.step()


    # train discriminator
    seg_out = model_d(h_)    # shape => (N, 1, H, W)
    seg_out = seg_out.squeeze()
    
    y_ = one_hot(y_, CONFIG.n_classes, device)    # shape => (N, 8, H, W)
    true_out = model_d(y_)    # shape => (N, 1, H, W)
    true_out = true_out.squeeze()

    loss_d_fake = criterion_bce(seg_out, zeros[:batch_len])
    loss_d_real = criterion_bce(true_out, ones[:batch_len])
    loss_d = loss_d_fake + loss_d_real

    optimizer.zero_grad()
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()

    return loss_full.item(), loss_d.item()



def semi_train(
        model, model_d, sample, criterion_ce_semi, criterion_bce, 
        optimizer, optimizer_d, ones, zeros, device):

    ''' semi supervised learning '''
    
    model.train()
    model.scale.freeze_bn()
    model_d.eval()

    # train segmentation network
    x = sample['image']
    batch_len = len(x)

    x = x.to(device)

    h = model(x)     # shape => (N, 8, H/8, W/8)
    h = F.interpolate(h, size=(256, 320), mode='bilinear', align_corners=True)

    _, h_ = torch.max(h, dim=1)    # to calculate the crossentropy loss. shape => (N, H, W)

    with torch.no_grad():
        d_out = model_d(h)    # shape => (N, 1, H, W)
        d_out = d_out.squeeze()

    loss_adv = criterion_bce(d_out, ones[:batch_len])


    # if the pixel value of the output from discriminator is more than a threshold,
    # its value is viewd as one from true label. Else, its value is ignored(value=255).
    h_[d_out < 0.2] = 255

    loss_ce = criterion_ce_semi(h, h_)
    loss_semi = 0.001 * loss_adv + 0.1 * loss_ce

    optimizer.zero_grad()
    optimizer_d.zero_grad()
    loss_semi.backward()
    optimizer.step()

    return loss_semi.item()



''' validation '''

def eval_model(model, test_loader, device='cpu'):
    model.eval()
    
    intersections = torch.zeros(8)   # the dataset has 8 classes including background
    unions = torch.zeros(8)
    
    for sample in test_loader:
        x, y = sample['image'], sample['class']
        
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            ypred = model(x)    # ypred.shape => (N, 8, H/8, W/8)
            ypred = F.interpolate(ypred, size=(256, 320), mode='bilinear', align_corners=True)
            _, ypred = ypred.max(1)    # y_pred.shape => (N, 256, 320)

            p = one_hot(ypred, 8, device)
            t = one_hot(y, 8, device)
            
            intersection = torch.sum(p & t, (0,2,3))
            union = torch.sum(p | t, (0, 2, 3))
            
            intersections += intersection.float()
            unions += union.float()
    
    """ iou[i] is the IoU of class i """
    iou = intersections / unions
    
    return iou



def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''
    
    parser = argparse.ArgumentParser(description='adversarial learning for affordance detection')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument('--device', type=str, default='cpu', help='choose a device you want to use')

    return parser.parse_args()

args = get_arguments()



def main(config, device):

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))

    if CONFIG.writer_flag:
        writer = SummaryWriter(CONFIG.result_path)


    ''' DataLoader '''
    train_data_with_label = PartAffordanceDataset('train_with_label_4to1.csv',
                                            transform=transforms.Compose([
                                                CenterCrop(),
                                                ToTensor(),
                                                Normalize()
                                            ]))

    train_data_without_label = PartAffordanceDatasetWithoutLabel('train_without_label_4to1.csv',
                                            transform=transforms.Compose([
                                                CenterCrop(),
                                                ToTensor(),
                                                Normalize()
                                            ]))

    test_data = PartAffordanceDataset('test.csv',
                                transform=transforms.Compose([
                                    CenterCrop(),
                                    ToTensor(),
                                    Normalize()
                                ]))

    train_loader_with_label = DataLoader(train_data_with_label, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers)
    train_loader_without_label = DataLoader(train_data_without_label, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers)
    test_loader = DataLoader(test_data, batch_size=CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers)

    
    ''' define model, optimizer, loss '''
    model = DeepLabV2_ResNet101_MSC(CONFIG.n_classes)
    model_d = Discriminator(CONFIG.n_classes)

    model.apply(init_weights)
    model_d.apply(init_weights)

    state_dict = torch.load(CONFIG.pretrained_model)
    model.load_state_dict(state_dict, strict=False)
    
    model.to(args.device)
    model_d.to(args.device)

    optimizer = optim.SGD(
                        params=[{
                            "params": get_params(model, key="1x"),
                            "lr": CONFIG.learning_rate,
                            "weight_decay": 5.0e-4,
                                },
                                {
                            "params": get_params(model, key="10x"),
                            "lr": 10 * CONFIG.learning_rate,
                            "weight_decay": 5.0e-4,
                                },
                                {
                                "params": get_params(model, key="20x"),
                                "lr": 20 * CONFIG.learning_rate,
                                "weight_decay": 0.0,
                                }],
                        momentum=0.9)

    

    optimizer_d = optim.Adam(model_d.parameters(), lr=CONFIG.learning_rate_d, betas=(0.9,0.99))

    if CONFIG.class_weight_flag:
        class_weight = torch.tensor([0.0057, 0.4689, 1.0000, 1.2993, 
                                    0.4240, 2.3702, 1.7317, 0.8149])    # refer to dataset.py
        criterion_ce_full = nn.CrossEntropyLoss(weight=class_weight.to(args.device))
    else:
        criterion_ce_full = nn.CrossEntropyLoss()

    criterion_ce_semi = nn.CrossEntropyLoss(ignore_index=255)
    criterion_bce = nn.BCEWithLogitsLoss()

    # supplementary constant for discriminator
    ones = torch.ones(CONFIG.batch_size, 256, 320).to(args.device)
    zeros = torch.zeros(CONFIG.batch_size, 256, 320).to(args.device)


    ''' training '''

    losses_full = []
    losses_semi = []
    losses_d = []
    val_iou = []
    mean_iou = []
    best_iou = 0.0

    for epoch in tqdm.tqdm(range(CONFIG.max_epoch)):
        
        epoch_loss_full = 0.0
        epoch_loss_d = 0.0
        epoch_loss_semi = 0.0
        
        poly_lr_scheduler(
            optimizer=optimizer,
            init_lr=CONFIG.learning_rate,
            iter=epoch - 1,
            lr_decay_iter=10,
            max_iter=CONFIG.max_epoch,
            power=0.9,
        )

        poly_lr_scheduler_d(
            optimizer=optimizer_d,
            init_lr=CONFIG.learning_rate_d,
            iter=epoch - 1,
            lr_decay_iter=10,
            max_iter=CONFIG.max_epoch,
            power=0.9,
        )
        


        # only supervised learning
        if epoch < 50:
            for i, sample in enumerate(train_loader_with_label):

                loss_full = full_train(model, sample, criterion_ce_full, optimizer, args.device)
                
                epoch_loss_full += loss_full

            losses_full.appned(epoch_loss_full / i)
            losses_d.append(0.0)
            losses_semi.append(0.0)


        # adversarial and supervised learning
        if 50 <= epoch < 100:
            for i, sample in enumerate(train_loader_with_label):
                
                loss_full, loss_d = adv_train(
                                        model, model_d, sample, criterion_ce_full, criterion_bce,
                                        optimizer, optimizer_d, ones, zeros, args.device)
                
                epoch_loss_full += loss_full
                epoch_loss_d += loss_d
                
            losses_full.append(epoch_loss_full / i)   # mean loss over all samples
            losses_d.append(epoch_loss_d / i)
            losses_semi.append(0.0)
            
        
        # semi-supervised learning
        if epoch >= 100:
            for i, (sample1, sample2) in enumerate(zip(train_loader_with_label, train_loader_without_label)):
                
                loss_full, loss_d = adv_train(
                                        model, model_d, sample1, criterion_ce_full, criterion_bce,
                                        optimizer, optimizer_d, ones, zeros, args.device)
                
                epoch_loss_full += loss_full
                epoch_loss_d += loss_d

                loss_semi = semi_train(
                                        model, model_d, sample2, criterion_ce_semi, criterion_bce,
                                        optimizer, optimizer_d, ones, zeros, args.device)

                epoch_loss_semi += loss_semi

            losses_full.append(epoch_loss_full / i)   # mean loss over all samples
            losses_d.append(epoch_loss_d / i)
            losses_semi.append(epoch_loss_semi / i)

        
        # validation
        val_iou.append(eval_model(model, test_loader, args.device))
        mean_iou.append(val_iou[-1].mean().item())

        if best_iou < mean_iou[-1]:
            best_iou = mean_iou[-1]
            torch.save(model.state_dict(), CONFIG.result_path + '/best_iou_model.prm')
            torch.save(model_d.state_dict(), CONFIG.result_path + '/best_iou_model_d.prm')

        if epoch%50 == 0 and epoch != 0:
            torch.save(model.state_dict(), CONFIG.result_path + '/epoch_{}_model.prm'.format(epoch))
            torch.save(model_d.state_dict(), CONFIG.result_path + '/epoch_{}_model_d.prm'.format(epoch))

        if writer is not None:
            writer.add_scalar("loss_full", losses_full[-1], epoch)
            writer.add_scalar("loss_d", losses_d[-1], epoch)
            writer.add_scalar("loss_semi", losses_semi[-1], epoch)
            writer.add_scalar("mean_iou", mean_iou[-1], epoch)
            writer.add_scalars("class_IoU", {'iou of class 0': val_iou[-1][0],
                                            'iou of class 1': val_iou[-1][1],
                                            'iou of class 2': val_iou[-1][2],
                                            'iou of class 3': val_iou[-1][3],
                                            'iou of class 4': val_iou[-1][4],
                                            'iou of class 5': val_iou[-1][5],
                                            'iou of class 6': val_iou[-1][6],
                                            'iou of class 7': val_iou[-1][7]}, epoch)

        print('epoch: {}\tloss_full: {:.5f}\tloss_d: {:.5f}\tloss_semi: {:.5f}\tmean IOU: {:.3f}'
            .format(epoch, losses_full[-1], losses_d[-1], losses_semi[-1], mean_iou[-1]))

    torch.save(model.state_dict(), CONFIG.result_path + '/final_model.prm')
    torch.save(model_d.state_dict(), CONFIG.result_path + '/final_model_d.prm')

if __name__ == '__main__':
    main(args.config, args.device)
