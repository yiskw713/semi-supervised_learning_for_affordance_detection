import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import click
import numpy as np
import pandas as pd
import scipy.io
import tqdm

from PIL import Image, ImageFilter
from tensorboardX import SummaryWriter

from models.deeplabv2 import DeepLabV2
from models.msc import MSC
from models.discriminator import Discriminator
from dataset import PartAffordanceDataset, PartAffordanceDatasetWithoutLabel
from dataset import CenterCrop, ToTensor, Normalize



''' one-hot representation '''

def one_hot(label, n_classes):
    one_hot_label = torch.eye(n_classes, requires_grad=True)[label].transpose(1, 3).transpose(2, 3)
    return one_hot_label
    

''' scheduler for learning rate '''

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    optimizer.param_groups[0]["lr"] = new_lr
    optimizer.param_groups[1]["lr"] = 10 * new_lr
    optimizer.param_groups[2]["lr"] = 20 * new_lr


def poly_lr_scheduler_D(optimizer, init_lr, iter, lr_decay_iter, max_iter, power):
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

def full_train(
        model, model_D, sample, criterion_ce_full, criterion_bce, 
        optimizer, optimizer_D, ones, zeros, device):

    ''' full supervised learning '''
    
    model.train()
    model.freeze_bn()
    model_D.train()

    # train segmentation network
    batch_len = len(sample)
    x, y = sample['image'], sample['class']

    x = x.to(device)
    y = y.to(device)

    h = model(x)     # shape => (N, 8, H/8, W/8)
    h = F.interpolate(h, size=(256, 320), mode='bilinear')

    h_ = h.detach()    # h_ is for calculating loss for discriminator
    y_ = h.detach()    # y_is for the same purpose

    D_out = model_D(h)    # shape => (N, 1, H/32, W/32)
    D_out = F.interpolate(D_out, size=(256, 320), mode='bilinear')    # shape => (N, 1, H, W)
    D_out = D_out.squeeze()
    
    loss_ce = criterion_ce_full(h, y)
    loss_adv = criterion_bce(D_out, ones[:batch_len])
    loss_full = loss_ce + 0.01 * loss_adv

    optimizer.zero_grad()
    optimizer_D.zero_grad()
    loss_full.backward()
    optimizer.step()


    # train discriminator
    seg_out = model_D(h_)    # shape => (N, 1, H/32, W/32)
    seg_out = F.interpolate(seg_out, size=(256, 320), mode='bilinear')    # shape => (N, 1, H, W)
    seg_out_ = seg_out.squeeze()
    
    y_ = one_hot(y_, 8)    # shape => (N, 8, H, W)
    true_out = model_D(y_)    # shape => (N, 1, H/32, W/32)
    true_out = F.interpolate(true_out, size=(256, 320), mode='bilinear')    # shape => (N, 1, H, W)
    true_out = true_out.squeeze()

    loss_D_fake = criterion_bce(seg_out, zeros[:batch_len])
    loss_D_real = criterion_bce(true_out, ones[:batch_len])
    loss_D = loss_D_fake + loss_D_real

    optimizer.zero_grad()
    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

    return loss_full.item(), loss_D.item()



def semi_train(
        model, model_D, sample, criterion_ce_semi, criterion_bce, 
        optimizer, optimizer_D, ones, zeros, device):

    ''' semi supervised learning '''
    
    model.train()
    model.freeze_bn()
    model_D.eval()

    # train segmentation network
    batch_len = len(sample)
    x = sample['image']

    x = x.to(device)

    h = model(x)     # shape => (N, 8, H/8, W/8)
    h = F.interpolate(h, size=(256, 320), mode='bilinear')

    h_, _ = torch.max(h, dim=1)    # to calculate the crossentropy loss. shape => (N, H, W)

    with torch.no_grad():
        D_out = model_D(h)    # shape => (N, 1, H/32, W/32)
        D_out = F.interpolate(D_out, size=(256, 320), mode='bilinear')    # shape => (N, 1, H, W)
        D_out = D_out.squeeze()

    loss_adv = criterion_bce(D_out, ones[:batch_len])


    # if the pixel value of the output from discriminator is more than a threshold,
    # its value is viewd as one from true label. Else, its value is ignored(value=255).
    h_[D_out < 0.2] = 255

    loss_ce = criterion_ce_semi(h, h_)
    loss_semi = 0.001 * loss_adv + 0.1 * loss_ce

    optimizer.zero_grad()
    optimizer_D.zero_grad()
    loss_semi.backward()
    optimizer.step()

    return loss_semi.item()



''' validation '''

def eval_model(model, test_loader, device='cpu'):
    model.eval()
    
    intersection = torch.zeros(8)   # the dataset has 8 classes including background
    union = torch.zeros(8)
    
    for sample in test_loader:
        x, y = sample['image'], sample['class']
        
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            _, ypred = model(x).max(1)    # y_pred.shape => (N, 256, 320)
        
        for i in range(8):
            y_i = (y == i)           
            ypred_i = (ypred == i)   
            
            inter = (y_i.byte() & ypred_i.byte()).float().sum().to('cpu')
            intersection[i] += inter
            union[i] += (y_i.float().sum() + ypred_i.float().sum()).to('cpu') - inter
    
    """ iou[i] is the IoU of class i """
    iou = intersection / union
    
    return iou



@click.command()
@click.option("--pretrained_model", type=str, default=None,
                        help="if you use a pretrained model. If so, write the path of params")
@click.option("--class_weight_flag", type=bool, default=True,
                        help="if you want to use class weight, input True. Else, input False")
@click.option("--batch_size", type=int, default=10,
                        help="number of batch size: number of samples sent to the network at a time")
@click.option("--num_workers", type=int, default=4,
                        help="number of workers for multithread data loading")
@click.option("--max_epoch", type=int, default=1000,
                        help="the number of epochs for training")
@click.option("--learning_rate", type=float, default=0.00025,
                        help="base learning rate for training segmentation network")
@click.option("--learning_rate_D", type=float, default=0.0001,
                        help="base learning rate for training discriminator")
@click.option("--n_classes", type=int, default=8,
                        help="number of classes in the dataset including background")
@click.option("--device", type=str, default='cpu',
                        help="the device you'll use (cpu or cuda:0 or so on)")
@click.option("--writer_flag", type=bool, default=True,
                        help="if you want to use SummaryWriter in tesorboardx, input True. Else, input False")
@click.option("--result_path", type=str, default='./result',
                        help="select your directory to save the result")
def main(
    pretrained_model, class_weight_flag, batch_size, num_worker, max_epoch, learning_rate, 
    learning_rate_D, n_classes, device, writer_flag, result_path):
    

    if writer_flag:
        writer = SummaryWriter(result_path)


    ''' DataLoader '''
    train_data_with_label = PartAffordanceDataset('train_with_label.csv',
                                            transform=transforms.Compose([
                                                CenterCrop(),
                                                ToTensor(),
                                                Normalize()
                                            ]))

    train_data_without_label = PartAffordanceDatasetWithoutLabel('train_with_label.csv',
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

    train_loader_with_label = DataLoader(train_data_with_label, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_loader_without_label = DataLoader(train_data_without_label, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=False, num_workers=num_workers)

    
    ''' define model, optimizer, loss '''
    model = DeepLabV2_ResNet101_MSC(n_classes)
    model_D = Discriminator(n_classes)

    model.apply(init_weights)
    model_D.apply(init_weights)

    state_dict = torch.load(pretrained_model)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.to(device)

    optimizer = optim.SGD(
                        params=[{
                            "params": get_params(model.module, key="1x"),
                            "lr": learning_rate,
                            "weight_decay": 5.0e-4,
                                },
                                {
                            "params": get_params(model.module, key="10x"),
                            "lr": 10 * learning_rate,
                            "weight_decay": 5.0e-4,
                                },
                                {
                                "params": get_params(model.module, key="20x"),
                                "lr": 20 * learning_rate,
                                "weight_decay": 0.0,
                                }],
                        momentum=0.9)

    optimizer_D = optim.Adam(model_D.parameters(), lr=learning_rate_D, betas=(0.9,0.99))

    if class_weight_flag:
        class_weight = torch.tensor([0.0057, 0.4689, 1.0000, 1.2993, 
                                    0.4240, 2.3702, 1.7317, 0.8149])    # refer to dataset.py
        criterion_ce_full = nn.CrossEntropyLoss(weight=class_weight.to(device))
    else:
        criterion_ce_full = nn.CrossEntropyLoss()

    criterion_ce_semi = nn.CrossEntropyLoss(ignore_index=255)
    criterion_bce = nn.BCEWithLogitsLoss()

    # supplementary constant for discriminator
    ones = torch.ones(batch_size, 256, 320).to(device)
    zeros = torch.zeros(batch_size, 256, 320).to(device)


    ''' training '''

    losses_full = []
    losses_semi = []
    losses_D = []
    val_iou = []
    mean_iou = []
    best_iou = 0.0

    for epoch in range(max_epoch):
        
        epoch_loss_full = 0.0
        epoch_loss_D = 0.0
        epoch_loss_semi = 0.0
        
        poly_lr_scheduler(
            optimizer=optimizer,
            init_lr=learning_rate,
            iter=epoch - 1,
            lr_decay_iter=10,
            max_iter=max_epoch,
            power=0.9,
        )

        poly_lr_scheduler_D(
            optimizer=optimizer_D,
            init_lr=learning_rate_D,
            iter=epoch - 1,
            lr_decay_iter=10,
            max_iter=max_epoch,
            power=0.9,
        )
        
        
        # only supervised learning
        if epoch < 10:
            for i, sample in enumerate(train_loader_with_label):
                
                loss_full, loss_D = full_train(
                                        model, model_D, sample, criterion_ce_full, criterion_bce,
                                        optimizer, optimizer_D, ones, zeros, device)
                
                epoch_loss_full += loss_full
                epoch_loss_D += loss_D
                
            losses_full.append(epoch_loss_full / i)   # mean loss over all samples
            losses_D.append(epoch_loss_D / i)
            losses_semi.append(0.0)
            
        
        # semi-supervised learning
        if epoch >= 10:
            for i, (sample1, sample2) in enumerate(zip(train_loader_with_label, train_loader_without_label)):
                
                loss_full, loss_D = full_train(
                                        model, model_D, sample1, criterion_ce_full, criterion_bce,
                                        optimizer, optimizer_D, ones, zeros, device)
                
                epoch_loss_full += loss_full
                epoch_loss_D += loss_D

                loss_semi += semi_train(
                                        model, model_D, sample2, criterion_ce_semi, criterion_bce,
                                        optimizer, optimizer_D, ones, zeros, device)

                epoch_loss_semi += loss_semi

            losses_full.append(epoch_loss_full / i)   # mean loss over all samples
            losses_D.append(epoch_loss_D / i)
            losses_semi.append(epoch_loss_semi / i)

        
        # validation
        val_iou.append(eval_model(model, test_loader, device))
        mean_iou.append(val_iou[-1].mean().item())

        if best_iou < mean_iou[-1]:
            best_iou = mean_iou[-1]
            torch.save(model.state_dict(), result_path + '/best_iou_model.prm')
            torch.save(model_D.state_dict(), result_path + '/best_iou_model_D.prm')

        if epoch%50 == 0 and epoch != 0:
            torch.save(model.state_dict(), result_path + '/epoch_{}_model.prm'.format(epoch))
            torch.save(model_D.state_dict(), result_path + '/epoch_{}_model_D.prm'.format(epoch))

        if writer is not None:
            writer.add_scalar("loss_full", losses_full[-1], epoch)
            writer.add_scalar("loss_D", losses_D[-1], epoch)
            writer.add_scalar("loss_semi", losses_semi[-1], epoch)
            writer.add_scalar("loss", mean_iou[-1], epoch)
            writer.add_scalars("class_IoU", {'iou of class 0': val_iou[-1][0],
                                           'iou of class 1': val_iou[-1][1],
                                           'iou of class 2': val_iou[-1][2],
                                           'iou of class 3': val_iou[-1][3],
                                           'iou of class 4': val_iou[-1][4],
                                           'iou of class 5': val_iou[-1][5],
                                           'iou of class 6': val_iou[-1][6],
                                           'iou of class 7': val_iou[-1][7]}, epoch)

        print('epoch: {}\tloss_full: {.5f}\tloss_D: {.5f}\tloss_semi: {.5f}\tmean IOU: {.3f}'
            .format(epoch, losses_full[-1], losses_D[-1], losses_semi[-1], mean_iou[-1]))



if __name__ == 'main':
    main()