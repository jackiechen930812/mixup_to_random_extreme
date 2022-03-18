'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import models
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import csv
import argparse

from models import *
from utils import progress_bar

import random
from PIL import Image,ImageFilter

import Gau_noise
import mixup as mp
import cutmix as cx
import mixup_v2 as mp_v2
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--mixup', type=str, default='ori', help='mixup method')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    Gau_noise.AddGaussianNoise(0.0, 8.0, 1.0),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
# net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth' + args.name + '_'
                            + str(args.seed))
    # net.load_state_dict(checkpoint['net'])
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    net = models.__dict__[args.model]()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log' +  '_' + args.model + '_epoch200_gau_cutmix_'
           + str(args.seed) + '.csv')


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.mixup == 'ori':
            inputs, targets_a, targets_b, lam = mp.mixup_data(inputs, targets,
                                                              args.alpha)
            inputs = inputs.float()
            outputs = net(inputs)
            loss = mp.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            train_loss += loss.item()
            _, predicted = outputs.max(1)

        elif args.mixup == 'cutmix':
            inputs, lam = cx.mixup_data(inputs, targets, args.alpha)
            inputs = inputs.float()
            outputs = net(inputs)
            loss = cx.mixup_criterion(criterion, outputs, targets, lam)
            train_loss += loss.item()
            _, predicted = outputs.max(1)

        elif args.mixup == 'matrix':
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
            loss_v1 = mp.mixup_criterion(criterion, outputs_v1, targets_a_v1[one_third:2 * one_third], targets_b_v1[one_third:2 * one_third], lam_v1)
            loss_v2 = mp.mixup_criterion(criterion, outputs_v2, targets_a_v2[2 * one_third:], targets_b_v2[2 * one_third:], lam_v2)

            loss = (loss_v2 + loss_or + loss_v1) / 3
            train_loss += loss.item()
            _, predicted = torch.max(outputs_v2.data, 1)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            _, predicted = outputs.max(1)

        # train_loss += loss.item()
        # _, predicted = outputs.max(1)
        total += targets.size(0)
        correct = 0.0
        # correct += predicted.eq(targets).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), reg_loss / (batch_idx + 1),
                        100. * correct / total, correct, total))

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
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            progress_bar(batch_idx, len(testloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total,
                            correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            # 'net': net.state_dict(),
            'net': net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/ResNet18/'):
            os.mkdir('checkpoint/ResNet18/')

        torch.save(state, './checkpoint/ResNet18/ckpt.pth_' + args.model + '_epoch200_gau_cutmix' + '_'
                   + str(args.seed))
        best_acc = acc
    return (test_loss / batch_idx, 100. * correct / total)

if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                            test_acc])
        scheduler.step()


