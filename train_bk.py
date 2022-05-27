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

from torch.utils.data.distributed import DistributedSampler

from utils import progress_bar

import random
from PIL import Image,ImageFilter

import Gau_noise
import mixup as mp
import cutmix as cx
import mixup_v2 as mp_v2
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--model', default="resnest101", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--mixup', type=str, default='ori', help='mixup method')
parser.add_argument('--epoch', default=100, type=int,
                    help='total epochs to run')
parser.add_argument('--input_size', default=224, type=int,
                    help='the size of input image')
parser.add_argument('--batch_size', default=32, type=int,
                    help='total epochs to run')                   
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
# print('==> Preparing data..')

# # Model
# print('==> Building model..')
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

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.pth' + args.name + '_'
#                             + str(args.seed))
#     # net.load_state_dict(checkpoint['net'])
#     net = checkpoint['net']
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch'] + 1
#     rng_state = checkpoint['rng_state']
#     torch.set_rng_state(rng_state)
# else:
#     print('==> Building model..')
#     net = models.__dict__[args.model]()
#     # net = get_model(args.model)
#     net = net.to(device)


# 使用DDP进行多进程分布式训练，local_rank表示系统自动分配的进程号，
torch.cuda.set_device(args.local_rank)  # 这里设定每一个进程使用的GPU是一定的，即一张gpu一个进程
device = torch.device('cuda', args.local_rank)
torch.distributed.init_process_group(backend='nccl') #初始化

# 固定随机种子，使用相同的参数进行不同进程上模型的初始化
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

net = models.__dict__[args.model]()
net = net.to(device)


transform_train = transforms.Compose([
    transforms.Resize([args.input_size,args.input_size]), 
    transforms.RandomCrop([args.input_size,args.input_size], padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    # Gau_noise.AddGaussianNoise(0.0, 8.0, 1.0),
    transforms.Resize([args.input_size,args.input_size]), 
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batch_size = args.batch_size
trainset = torchvision.datasets.ImageFolder('./imagenet-1k/train/',transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, num_workers=2,sampler=DistributedSampler(trainset)) # 这个sampler会自动分配数据到各个gpu上

testset = torchvision.datasets.ImageFolder('./imagenet-1k/val/',transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size,  num_workers=2,sampler=DistributedSampler(testset,shuffle=False))

net = torch.nn.parallel.DistributedDataParallel(net,
                                        device_ids=[args.local_rank], 
                                        output_device=args.local_rank,
                                        find_unused_parameters=True) #将模型分布到多张gpu上

cudnn.benchmark = True

if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log' +  '_' + args.model + '_epoch200_' + str(args.mixup)
           + str(args.seed) + '.csv')


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


# Training
def train(epoch):
    if args.local_rank == 0:  #仅显示第一个进程的训练过程，都显示的话会非常混乱
        print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        #print(inputs.shape)
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
        if args.local_rank == 0:  #仅显示第一个进程的训练过程，都显示的话会非常混乱
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
            if args.local_rank == 0:
                progress_bar(batch_idx, len(testloader),
                            'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss / (batch_idx + 1), 100. * correct / total,
                                correct, total))

    # Save checkpoint.
    if args.local_rank == 0:
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

            torch.save(state, './checkpoint/ResNet18/ckpt.pth_' + args.model + '_epoch200_none' + '_'
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
        if args.local_rank == 0: #仅记录第一个进程的测试结果，由于参数共享，不同进程测试结果相同，无需重复保存
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                                test_acc])
        scheduler.step()


