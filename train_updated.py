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
from PIL import Image,ImageFilter
import Gau_noise 

import mixup as mp
import mixup_v2 as mp_v2

# 导入warm up模块
import pytorch_warmup as warmup

# 导入mixup， cutmix可以由mixup直接实现
from mixup_new import mixup_data, mixup_criterion, rand_bbox

# 导入余弦退火学习率衰减方式
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0.0 < smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        """
        写法1
        """
        # logprobs = F.log_softmax(x, dim=-1)
        # nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        # nll_loss = nll_loss.squeeze(1)  # 得到交叉熵损失
        # # 注意这里要结合公式来理解，同时留意预测正确的那个类，也有a/K，其中a为平滑因子，K为类别数
        # smooth_loss = -logprobs.mean(dim=1)
        # loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        """
        写法2
        """
        y_hat = torch.softmax(x, dim=1)
        # 这里cross_loss和nll_loss等价
        cross_loss = self.cross_entropy(y_hat, target)
        smooth_loss = -torch.log(y_hat).mean(dim=1)
        # smooth_loss也可以用下面的方法计算,注意loga + logb = log(ab)
        # smooth_loss = -torch.log(torch.prod(y_hat, dim=1)) / y_hat.shape[1]
        loss = self.confidence * cross_loss + self.smoothing * smooth_loss
        return loss.mean()

    def cross_entropy(self, y_hat, y):
        return - torch.log(y_hat[range(len(y_hat)), y])


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')  #0.02
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=64, type=int, help='batch size') #24
parser.add_argument('--epoch', default=500, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data

# rand data augmentation
print('==> Preparing data..')
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.CenterCrop(32),
        # transforms.Grayscale(),
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
    Gau_noise.AddGaussianNoise(0.0, 2.0, 1.0),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=False,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=0)

testset = datasets.CIFAR10(root='./data', train=False, download=False,
                           transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=0)


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
logname = ('results/log' +  '_' + args.model + '_epoch50_test'
# logname = ('results/log' +  '_' + args.model + '_epoch50_4_2_gua_matrix_2.0_'
           + str(args.seed) + '.csv')

# if use_cuda:
#     net.cuda()
#     net = torch.nn.DataParallel(net)
#     print(torch.cuda.device_count())
#     cudnn.benchmark = True
#     print('Using CUDA..')
net = net.to(device)
net = torch.nn.DataParallel(net)
print(torch.cuda.device_count())
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
#                       weight_decay=args.decay)
# optimizer = optim.AdamW(net.parameters(), lr=0.0005, betas=(0.9, 0.999))

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

# 学习率调度
scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)  # Tmax

# 加入warmup_scheduler
# warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

# def mixup_data(x, y, alpha=1.0, use_cuda=True):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1

#     batch_size = x.size()[0]
#     if use_cuda:
#         index = torch.randperm(batch_size).cuda()
#     else:
#         index = torch.randperm(batch_size)

#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam


# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # if use_cuda:
        #     inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = inputs.to(device), targets.to(device)

        if epoch <=20:
            inputs, targets_a, targets_b, lam = mp_v2.mixup_data(inputs, targets,
                                                           args.alpha)
        else:
            inputs, targets_a, targets_b, lam = mp.mixup_data(inputs, targets,
                                                           args.alpha)
        # inputs, targets_a, targets_b = map(Variable, (inputs,
        #                                               targets_a, targets_b))
        inputs = inputs.float() 
        outputs = net(inputs)
        # print('outputs:',outputs)
    #    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss = mp.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        # train_loss += loss.data[0]
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        # print('predicted',predicted)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 是否加入梯度裁剪
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)

        # 更新参数
        optimizer.step()

        # 加入warmup_scheduler
        # warmup_scheduler.dampen()



        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))

    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # test_loss += loss.data[0]
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(testloader),
                        'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total,
                            correct, total))
        acc = 100.*correct/total
        if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
            checkpoint(acc, epoch)
        if acc > best_acc:
            best_acc = acc
    return (test_loss/batch_idx, 100.*correct/total)


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
    torch.save(state, './checkpoint/ResNet18/ckpt.t7_' +  args.model + '_epoch50_test'  + '_'
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
