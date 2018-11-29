import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import argparse
import numpy as np
import pandas as pd
import random
import scipy.io
import skimage.io
import sys
import tqdm
import yaml

from addict import Dict
from itertools import zip_longest
from PIL import Image, ImageFilter
from tensorboardX import SummaryWriter

from models.FCN8s import FCN8s
from models.SegNet import SegNetBasic
from models.UNet import UNet
from models.discriminator import Discriminator
from dataset import PartAffordanceDataset, PartAffordanceDatasetWithoutLabel, CenterCrop, ToTensor, Normalize
from dataset import crop_center_numpy, crop_center_pil_image



def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''
    
    parser = argparse.ArgumentParser(description='adversarial learning for affordance detection')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument('--device', type=str, default='cpu', help='choose a device you want to use')

    return parser.parse_args()



""" validation """

def one_hot(label, n_classes, dtype, device, requires_grad=True):
    one_hot_label = torch.eye(n_classes, dtype=dtype, requires_grad=requires_grad, device=device)[label].transpose(1, 3).transpose(2, 3)
    return one_hot_label



''' training '''

def full_train(model, sample, criterion, optimizer, config, device):

    ''' full supervised learning for segmentation network'''

    x, y = sample['image'], sample['label']
            
    x = x.to(device)
    y = y.to(device)

    h = model(x)

    loss = criterion(h, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item().to('cpu')



def adv_train(model, model_d, sample, criterion, criterion_bce, optimizer, optimizer_d, real, fake, config, device):

    ''' full supervised and adversarial learning '''
    
    model.train()
    model_d.train()

    # train segmentation network
    x, y = sample['image'], sample['label']

    batch_len = len(x)
            
    x = x.to(device)
    x_ = x.detach()
    y = y.to(device)

    h = model(x)     # shape => (N, n_classes, H, W)

    h_ = h.detach()    # h_ is for calculating loss for discriminator
    y_ = y.detach()    # y_is for the same purpose.  shape => (N, H, W)

    if config.gaussian:
        h += torch.rand((batch_len, config.n_classes, config.height, config.width)).to(device)

    xh = torch.cat((x, h), dim=1)
    d_out = model_d(xh)    # shape => (N, 1, H, W)
    d_out = d_out.squeeze()
    
    loss_full = criterion(h, y)
    loss_adv = criterion_bce(d_out, real[:batch_len])
    loss = loss_full + config.adv_weight * loss_adv

    optimizer.zero_grad()
    optimizer_d.zero_grad()
    loss.backward()
    optimizer.step()


    # train discriminator
    if config.gaussian:
        h_ += torch.rand((batch_len, config.n_classes, config.height, config.width)).to(device)

    xh_ = torch.cat((x_, h_), dim=1)
    seg_out = model_d(xh_)    # shape => (N, 1, H, W)
    seg_out = seg_out.squeeze()
    
    y_ = one_hot(y_, config.n_classes, torch.float, device)    # shape => (N, n_classes, H, W)
    xy_ = torch.cat((x_, y_), dim=1)
    true_out = model_d(xy_)    # shape => (N, 1, H, W)
    true_out = true_out.squeeze()

    if config.flip_label:
        if random.random() > config.flip_label_th:
            loss_d_fake = criterion_bce(seg_out, fake[:batch_len])
            loss_d_real = criterion_bce(true_out, real[:batch_len])
        else:
            loss_d_fake = criterion_bce(seg_out, real[:batch_len])
            loss_d_real = criterion_bce(true_out, fake[:batch_len])
    else:
        loss_d_fake = criterion_bce(seg_out, fake[:batch_len])
        loss_d_real = criterion_bce(true_out, real[:batch_len])
    
    
    loss_d = loss_d_fake + loss_d_real

    optimizer.zero_grad()
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()

    return loss.item().to('cpu'), loss_d.item().to('cpu')



def semi_train(model, model_d, sample, criterion, criterion_bce, optimizer, optimizer_d, real, fake, config, device):

    ''' semi supervised learning '''
    
    model.train()
    model_d.eval()

    # train segmentation network
    x = sample['image']

    batch_len = len(x)

    x = x.to(device)
    
    h = model(x)     # shape => (N, n_classes, H, W)

    _, h_ = torch.max(h, dim=1)    # to calculate the crossentropy loss. shape => (N, H, W)

    with torch.no_grad():
        if config.gaussian:
            h += torch.rand((batch_len, config.n_classes, config.height, config.width)).to(device)
        
        xh = torch.cat((x, h), dim=1)
        d_out = model_d(xh)    # shape => (N, 1, H, W)
        d_out = d_out.squeeze()

    loss_adv = criterion_bce(d_out, real[:batch_len])


    # if the pixel value of the output from discriminator is more than a threshold,
    # its value is viewd as one from true label. Else, its value is ignored(value=255).
    h_[d_out < config.d_th] = 255

    loss_semi = criterion(h, h_)
    
    loss = config.adv_weight * loss_adv + config.semi_weight * loss_semi

    optimizer.zero_grad()
    optimizer_d.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item().to('cpu')


def eval_model(model, test_loader, config, device):
    model.eval()
    
    intersections = torch.zeros(config.n_classes).to(device)   # including background
    unions = torch.zeros(config.n_classes).to(device)
    
    for sample in test_loader:
        x, y = sample['image'], sample['label']
        
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            h = model(x) 
            _, ypred = h.max(1)    # y_pred.shape => (N, H, W)
            
            p = one_hot(ypred, 8, torch.long, device, requires_grad=False)
            t = one_hot(y, 8, torch.long, device, requires_grad=False)
            
            intersection = torch.sum(p & t, (0, 2, 3))
            union = torch.sum(p | t, (0, 2, 3))
            
            intersections += intersection.float()
            unions += union.float()
    
    """ iou[i] is the IoU of class i """
    iou = intersections / unions
    
    return iou.to('cpu')



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




def main():

    args = get_arguments()

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))

    # writer
    if CONFIG.writer_flag:
        writer = SummaryWriter(CONFIG.result_path)
    else:
        writer = None


    """ DataLoader """
    labeled_train_data = PartAffordanceDataset(CONFIG.labeled_data,
                                                config=CONFIG,
                                                transform=transforms.Compose([
                                                        CenterCrop(CONFIG),
                                                        ToTensor(),
                                                        Normalize()
                                                ]))

    if CONFIG.train_mode == 'semi':
        unlabeled_train_data = PartAffordanceDatasetWithoutLabel(CONFIG.unlabeled_data,
                                                                config=CONFIG,
                                                                transform=transforms.Compose([
                                                                    CenterCrop(CONFIG),
                                                                    ToTensor(),
                                                                    Normalize()
                                                                ]))
    else:
        unlabeled_train_data = None

    test_data = PartAffordanceDataset(CONFIG.test_data,
                                    config=CONFIG,
                                    transform=transforms.Compose([
                                        CenterCrop(CONFIG),
                                        ToTensor(),
                                        Normalize()
                                    ]))

    train_loader_with_label = DataLoader(labeled_train_data, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers)
    
    if unlabeled_train_data is not None:
        train_loader_without_label = DataLoader(unlabeled_train_data, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers)
    
    test_loader = DataLoader(test_data, batch_size=CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers)


    """ model """
    if CONFIG.model == 'FCN8s':
        model = FCN8s(CONFIG.in_channel, CONFIG.n_classes)
    elif CONFIG.model == 'SegNetBasic':
        model = SegNetBasic(CONFIG.in_channel, CONFIG.n_classes)
    elif CONFIG.model == 'UNet':
        model = UNet(CONFIG.in_channel, CONFIG.n_classes)
    else:
        print('This model doesn\'t exist in the model directory')
        sys.exit(1)


    if CONFIG.train_mode == 'full':
        model.apply(init_weights)
        model.to(args.device)
    elif CONFIG.train_mode == 'semi':
        model.load_state_dict(torch.load(CONFIG.pretrain_model))
        model.to(args.device)
        model_d = Discriminator(CONFIG)
        model_d.apply(init_weights)
        model_d.to(args.device)
    else:
        print('This training mode doesn\'t exist.')
        sys.exit(1)



    """ class weight after center crop. See dataset.py """
    if CONFIG.class_weight_flag:
        class_num = torch.tensor([2078085712, 34078992, 15921090, 12433420, 
                                    38473752, 6773528, 9273826, 20102080])

        total = class_num.sum().item()

        frequency = class_num.float() / total
        median = torch.median(frequency)

        class_weight = median / frequency
        class_weight = class_weight.to(args.device)
    else:
        class_weight = None



    """ supplementary constant for discriminator """
    if CONFIG.noisy_label_flag:
        if one_label_smooth:
            real = torch.full((CONFIG.batch_size, CONFIG.height, CONFIG.width), CONFIG.real_label).to(args.device)
            fake = torch.zeros(CONFIG.batch_size, CONFIG.height, CONFIG.width).to(args.device)
        else:
            real = torch.full((CONFIG.batch_size, CONFIG.height, CONFIG.width), CONFIG.real_label).to(args.device)
            fake = torch.full((CONFIG.batch_size, CONFIG.height, CONFIG.width), CONFIG.fake_label).to(args.device)        
    else:
        real = torch.ones(CONFIG.batch_size, CONFIG.height, CONFIG.width).to(args.device)
        fake = torch.zeros(CONFIG.batch_size, CONFIG.height, CONFIG.width).to(args.device)


    """ optimizer, criterion """
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
    optimizer_d = optim.Adam(model_d.parameters(), lr=CONFIG.learning_rate)

    criterion = nn.CrossEntropyLoss(class_weight)
    criterion_bce = nn.BCELoss()    # discriminator includes sigmoid layer


    losses_full = []
    losses_semi = []
    losses_d = []
    val_iou = []
    mean_iou = []
    mean_iou_without_bg = []
    best_mean_iou = 0.0

    for epoch in tqdm.tqdm(range(CONFIG.max_epoch)):

        epoch_loss_full = 0.0
        epoch_loss_d = 0.0
        epoch_loss_semi = 0.0

        # only supervised learning
        if CONFIG.train_mode == 'full':    

            for i, sample in enumerate(train_loader_with_label):

                loss_full = full_train(model, sample, criterion, optimizer, CONFIG, args.device)
                
                epoch_loss_full += loss_full

            losses_full.append(epoch_loss_full / i)
            losses_d.append(0.0)
            losses_semi.append(0.0)

        
        # semi-supervised learning
        elif CONFIG.train_mode == 'semi':
            
            # first, adveresarial learning
            if epoch < CONFIG.adv_epoch:
                
                for i, sample in enumerate(train_loader_with_label):
                
                    loss_full, loss_d = adv_train(
                                            model, model_d, sample, criterion, criterion_bce,
                                            optimizer, optimizer_d, real, fake, CONFIG, args.device)
                
                    epoch_loss_full += loss_full
                    epoch_loss_d += loss_d
                    
                losses_full.append(epoch_loss_full / i)   # mean loss over all samples
                losses_d.append(epoch_loss_d / i)
                losses_semi.append(0.0)
                    
            # semi-supervised learning
            else:
                cnt_full = 0
                cnt_semi = 0
                
                for (sample1, sample2) in zip_longest(train_loader_with_label, train_loader_without_label):
                    
                    if sample1 is not None:
                        loss_full, loss_d = adv_train(
                                                model, model_d, sample1, criterion, criterion_bce,
                                                optimizer, optimizer_d, real, fake, CONFIG, args.device)
                        
                        epoch_loss_full += loss_full
                        epoch_loss_d += loss_d
                        cnt_full += 1

                    if sample2 is not None:
                        loss_semi = semi_train(
                                                model, model_d, sample2, criterion, criterion_bce,
                                                optimizer, optimizer_d, real, fake, CONFIG, args.device)
                        epoch_loss_semi += loss_semi
                        cnt_semi += 1

                losses_full.append(epoch_loss_full / cnt_full)   # mean loss over all samples
                losses_d.append(epoch_loss_d / cnt_full)
                losses_semi.append(epoch_loss_semi / cnt_semi)


        else:
            print('This train mode can\'t be used. Choose full or semi')
            sys.exit(1)


        # validation
        val_iou.append(eval_model(model, test_loader, CONFIG, args.device))
        mean_iou.append(val_iou[-1].mean().item())
        mean_iou_without_bg.append(val_iou[-1][1:].mean().item())

        if best_mean_iou < mean_iou[-1]:
            best_mean_iou = mean_iou[-1]
            torch.save(model.state_dict(), CONFIG.result_path + '/best_mean_iou_model.prm')
            if CONFIG.train_mode == 'semi':
                torch.save(model_d.state_dict(), CONFIG.result_path + '/best_mean_iou_model_d.prm')

        if epoch%50 == 0 and epoch != 0:
            torch.save(model.state_dict(), CONFIG.result_path + '/epoch_{}_model.prm'.format(epoch))
            if CONFIG.train_mode == 'semi':
                torch.save(model_d.state_dict(), CONFIG.result_path + '/epoch_{}_model_d.prm'.format(epoch))

        if writer is not None:
            writer.add_scalar("loss_full", losses_full[-1], epoch)
            writer.add_scalar("loss_d", losses_d[-1], epoch)
            writer.add_scalar("loss_semi", losses_semi[-1], epoch)
            writer.add_scalar("mean_iou", mean_iou[-1], epoch)
            writer.add_scalar("mean_iou_without_background", mean_iou_without_bg[-1], epoch)
            writer.add_scalars("class_IoU", {'iou of class 0': val_iou[-1][0],
                                            'iou of class 1': val_iou[-1][1],
                                            'iou of class 2': val_iou[-1][2],
                                            'iou of class 3': val_iou[-1][3],
                                            'iou of class 4': val_iou[-1][4],
                                            'iou of class 5': val_iou[-1][5],
                                            'iou of class 6': val_iou[-1][6],
                                            'iou of class 7': val_iou[-1][7]}, epoch)

        print('epoch: {}\tloss_full: {:.5f}\tloss_d: {:.5f}\tloss_semi: {:.5f}\tmean IOU: {:.3f}\tmean IOU w/ bg: {:.3f}'
            .format(epoch, losses_full[-1], losses_d[-1], losses_semi[-1], mean_iou[-1], mean_iou_without_bg[-1]))


    torch.save(model.state_dict(), CONFIG.result_path + '/final_model.prm')
    if CONFIG.train_mode == 'semi':
        torch.save(model_d.state_dict(), CONFIG.result_path + '/final_model_d.prm')


if __name__ == '__main__':
    main()