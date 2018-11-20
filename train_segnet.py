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
import random
import scipy.io
import tqdm
import yaml

from addict import Dict
from itertools import zip_longest
from PIL import Image, ImageFilter
from tensorboardX import SummaryWriter

from models.SegNet import SegNetBasic
from models.discriminator import Discriminator
from dataset import PartAffordanceDataset, PartAffordanceDatasetWithoutLabel
from dataset import CenterCrop, ToTensor, Normalize



''' one-hot representation '''

def one_hot(label, n_classes, dtype, device):
    one_hot_label = torch.eye(n_classes, dtype=dtype, device=device)[label].transpose(1, 3).transpose(2, 3)
    return one_hot_label
    


''' model, weight initialization '''

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
            nn.init.constant_(m.bias, 0)



''' training '''

def full_train(model, sample, criterion_ce_full, optimizer, device):

    ''' full supervised learning for segmentation network'''

    model.train()

    x, y = sample['image'], sample['class']

    x = x.to(device)
    y = y.to(device)

    h = model(x)     # shape => (N, 8, H, W)

    loss_ce = criterion_ce_full(h, y)

    optimizer.zero_grad()
    loss_ce.backward()
    optimizer.step()

    return loss_ce.item()



def adv_train(
        model, model_d, sample, criterion_ce_full, criterion_bce, 
        optimizer, optimizer_d, real, fake, config, device):

    ''' full supervised and adversarial learning '''
    
    model.train()
    model_d.train()

    # train segmentation network
    x, y = sample['image'], sample['class']

    batch_len = len(x)

    x = x.to(device)
    y = y.to(device)

    h = model(x)     # shape => (N, 8, H, W)

    h_ = h.detach()    # h_ is for calculating loss for discriminator
    y_ = y.detach()    # y_is for the same purpose.  shape => (N, H, W)

    if config.gaussian:
        h += torch.rand((batch_len, 8, 256, 320)).to(device)

    d_out = model_d(h)    # shape => (N, 1, H, W)
    d_out = d_out.squeeze()
    
    loss_ce = criterion_ce_full(h, y)
    loss_adv = criterion_bce(d_out, real[:batch_len])
    loss_full = loss_ce + 0.01 * loss_adv

    optimizer.zero_grad()
    optimizer_d.zero_grad()
    loss_full.backward()
    optimizer.step()


    # train discriminator
    if config.gaussian:
        h_ += torch.rand((batch_len, 8, 256, 320)).to(device)

    seg_out = model_d(h_)    # shape => (N, 1, H, W)
    seg_out = seg_out.squeeze()
    
    y_ = one_hot(y_, 8, dtype=torch.float, device)    # shape => (N, 8, H, W)
    true_out = model_d(y_)    # shape => (N, 1, H, W)
    true_out = true_out.squeeze()

    if config.flip_label:
        if random.random() > 0.2:
            loss_d_fake = criterion_bce(seg_out, fake[:batch_len])
            loss_d_real = criterion_bce(true_out, real[:batch_len])
        else:
            loss_d_fake = criterion_bce(seg_out, real[:batch_len])
            loss_d_real = criterion_bce(true_out, fake[:batch_len])
    
    
    loss_d = loss_d_fake + loss_d_real

    optimizer.zero_grad()
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()

    return loss_full.item(), loss_d.item()



def semi_train(
        model, model_d, sample, criterion_ce_semi, criterion_bce, 
        optimizer, optimizer_d, real, fake, config, device):

    ''' semi supervised learning '''
    
    model.train()
    model_d.eval()

    # train segmentation network
    x = sample['image']
    batch_len = len(x)

    x = x.to(device)

    h = model(x)     # shape => (N, 8, H, W)

    _, h_ = torch.max(h, dim=1)    # to calculate the crossentropy loss. shape => (N, H, W)

    with torch.no_grad():
        if config.gaussian:
            h += torch.rand((batch_len, 8, 256, 320)).to(device)
        
        d_out = model_d(h)    # shape => (N, 1, H, W)
        d_out = d_out.squeeze()

    loss_adv = criterion_bce(d_out, real[:batch_len])


    # if the pixel value of the output from discriminator is more than a threshold,
    # its value is viewd as one from true label. Else, its value is ignored(value=255).
    if config.noisy_label_flag:
        h_[d_out < 0.35] = 255
    else:
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
    
    intersections = torch.zeros(8).to(device)   # the dataset has 8 classes including background
    unions = torch.zeros(8).to(device)
    
    for sample in test_loader:
        x, y = sample['image'], sample['class']
        
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            ypred = model(x)    # ypred.shape => (N, 8, H, W)
            _, ypred = ypred.max(1)    # y_pred.shape => (N, 256, 320)

            p = one_hot(ypred, 8, dtype=torch.long, device)
            t = one_hot(y, 8, dtype=torch.long, device)
            
            intersection = torch.sum(p & t, (0, 2, 3))
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
    train_data_with_label = PartAffordanceDataset(CONFIG.labeled_data,
                                            transform=transforms.Compose([
                                                CenterCrop(),
                                                ToTensor(),
                                                Normalize()
                                            ]))

    train_data_without_label = PartAffordanceDatasetWithoutLabel(CONFIG.unlabeled_data,
                                            transform=transforms.Compose([
                                                CenterCrop(),
                                                ToTensor(),
                                                Normalize()
                                            ]))

    test_data = PartAffordanceDataset(CONFIG.test_data,
                                transform=transforms.Compose([
                                    CenterCrop(),
                                    ToTensor(),
                                    Normalize()
                                ]))

    train_loader_with_label = DataLoader(train_data_with_label, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers)
    train_loader_without_label = DataLoader(train_data_without_label, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers)
    test_loader = DataLoader(test_data, batch_size=CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers)

    
    ''' define model, optimizer, loss '''
    model = SegNetBasic(3, CONFIG.n_classes)
    model_d = Discriminator(CONFIG.n_classes)

    model.apply(init_weights)
    model_d.apply(init_weights)
    
    model.to(args.device)
    model_d.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate)

    optimizer_d = optim.Adam(model_d.parameters(), lr=CONFIG.learning_rate_d)

    if CONFIG.class_weight_flag:
        class_weight = torch.tensor([0.0057, 0.4689, 1.0000, 1.2993, 
                                    0.4240, 2.3702, 1.7317, 0.8149])    # refer to dataset.py
        criterion_ce_full = nn.CrossEntropyLoss(weight=class_weight.to(args.device))
    else:
        criterion_ce_full = nn.CrossEntropyLoss()

    criterion_ce_semi = nn.CrossEntropyLoss(ignore_index=255)
    criterion_bce = nn.BCELoss()

    # supplementary constant for discriminator
    if CONFIG.noisy_label_flag:
        real = torch.full((CONFIG.batch_size, 256, 320), 0.8).to(args.device)
        fake = torch.full((CONFIG.batch_size, 256, 320), 0.2).to(args.device)        
    else:
        real = torch.ones(CONFIG.batch_size, 256, 320).to(args.device)
        fake = torch.zeros(CONFIG.batch_size, 256, 320).to(args.device)


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
        
        # only supervised learning
        if epoch < 50:
            for i, sample in enumerate(train_loader_with_label):

                loss_full = full_train(model, sample, criterion_ce_full, optimizer, args.device)
                
                epoch_loss_full += loss_full

            losses_full.append(epoch_loss_full / i)
            losses_d.append(0.0)
            losses_semi.append(0.0)


        # adversarial and supervised learning
        if 50 <= epoch < 100:
            for i, sample in enumerate(train_loader_with_label):
                
                loss_full, loss_d = adv_train(
                                        model, model_d, sample, criterion_ce_full, criterion_bce,
                                        optimizer, optimizer_d, real, fake, CONFIG, args.device)
                
                epoch_loss_full += loss_full
                epoch_loss_d += loss_d
                
            losses_full.append(epoch_loss_full / i)   # mean loss over all samples
            losses_d.append(epoch_loss_d / i)
            losses_semi.append(0.0)
            
        
        # semi-supervised learning
        if epoch >= 100:
            cnt_semi = 0
            
            for i, (sample1, sample2) in enumerate(zip_longest(train_loader_with_label, train_loader_without_label)):
                
                loss_full, loss_d = adv_train(
                                        model, model_d, sample1, criterion_ce_full, criterion_bce,
                                        optimizer, optimizer_d, real, fake, args.device)
                
                epoch_loss_full += loss_full
                epoch_loss_d += loss_d

                if sample2 is not None:
                    loss_semi = semi_train(
                                            model, model_d, sample2, criterion_ce_semi, criterion_bce,
                                            optimizer, optimizer_d, real, fake, CONFIG, args.device)
                    epoch_loss_semi += loss_semi
                    cnt_semi += 1

            losses_full.append(epoch_loss_full / i)   # mean loss over all samples
            losses_d.append(epoch_loss_d / i)
            losses_semi.append(epoch_loss_semi / cnt_semi)

        
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
