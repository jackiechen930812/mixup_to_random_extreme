#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from utils import progress_bar

import random
from PIL import Image, ImageFilter
import Gau_noise

import mixup as mp
import mixup_v2 as mp_v2

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')  # 0.02
# change base learning rate
parser.add_argument('--lr', default=0.004, type=float, help='learning rate')  # 0.02
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')  # 24
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
# modify weight decay
# parser.add_argument('--decay', default=0.05, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    # Gau_noise.AddGaussianNoise(0.0, 8.0, 1.0),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=False,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=8)

testset = datasets.CIFAR10(root='./data', train=False, download=False,
                           transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=8)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                            + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    net = models.__dict__[args.model]()

if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log' + '_' + args.model + '_epoch50_i3_gua8.0' + '_'
           # logname = ('results/log' +  '_' + args.model + '_epoch50_4_2_gua_matrix_2.0_'
           + str(args.seed) + '.csv')

net = net.to(device)
net = torch.nn.DataParallel(net)
print(torch.cuda.device_count())
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
#                       weight_decay=args.decay)
# change optimizer method
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.decay)

# import warm_up module
import pytorch_warmup as warmup

# add warmup_scheduler
warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

# adjust lr_scheduler
# import Consine Annealing lr scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        batch_size = inputs.size()[0]
        one_third = int(batch_size / 3)

        inputs_v2, targets_a_v2, targets_b_v2, lam_v2 = mp_v2.mixup_data(inputs, targets, args.alpha)
        inputs_v1, targets_a_v1, targets_b_v1, lam_v1 = mp.mixup_data(inputs, targets, args.alpha)

        inputs_mix = torch.cat((inputs[:one_third], inputs_v1[one_third:2 * one_third], inputs_v2[2 * one_third:]), 0)
        inputs_mix = inputs_mix.float()
        inputs_v2 = inputs_v2.float()
        inputs_v1 = inputs_v1.float()
        inputs_or = inputs.float()

        outputs_mix = net(inputs_mix)
        outputs_or = outputs_mix[:one_third]

        outputs_v1 = outputs_mix[one_third:2 * one_third]
        outputs_v2 = outputs_mix[2 * one_third:]

        loss_or = criterion(outputs_or, targets[:one_third])
        loss_v1 = mp.mixup_criterion(criterion, outputs_v1, targets_a_v1[one_third:2 * one_third],
                                     targets_b_v1[one_third:2 * one_third], lam_v1)
        loss_v2 = mp.mixup_criterion(criterion, outputs_v2, targets_a_v2[2 * one_third:], targets_b_v2[2 * one_third:],
                                     lam_v2)

        # print(loss_v2)
        # print(loss_v1)
        # print(loss_or)

        loss = (loss_v2 + loss_or + loss_v1) / 3

        train_loss += loss.item()
        _, predicted = torch.max(outputs_v2.data, 1)

        total += targets.size(0)
        correct += (lam_v2 * predicted.eq(targets_a_v2[2 * one_third:].data).cpu().sum().float()
                    + (1 - lam_v2) * predicted.eq(targets_b_v2[2 * one_third:].data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # add warmup_scheduler
        warmup_scheduler.dampen()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), reg_loss / (batch_idx + 1),
                        100. * correct / total, correct, total))
    scheduler.step()
    return (train_loss / batch_idx, reg_loss / batch_idx, 100. * correct / total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(testloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total,
                            correct, total))
        acc = 100. * correct / total
        if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
            checkpoint(acc, epoch)
        if acc > best_acc:
            best_acc = acc
    return (test_loss / batch_idx, 100. * correct / total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint/ResNet18/'):
        os.mkdir('checkpoint/ResNet18/')
    torch.save(state, './checkpoint/ResNet18/ckpt.t7_' + args.model + '_epoch50_i3_gua8.0' + '_'
               # torch.save(state, './checkpoint/ResNet18/ckpt.t7_' +  args.model + '_epoch50_4_2_gua_matrix_2.0'  + '_'
               + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    adjust_learning_rate(optimizer, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                            test_acc])
