
from __future__ import print_function
import numpy as np
import torch
import models
import torchvision.transforms as transforms
import argparse
import json

from torch.utils.data.distributed import DistributedSampler
import random
import matplotlib.pyplot as plt
from dataset_img_9 import get_imagenet_dataloader


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--model', default="resnest101", type=str,
                    help='model type (default: resnest50)')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--mixup', type=str, default='cutmix', help='mixup method')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')

parser.add_argument('--input_size', default=224, type=int,
                    help='the size of input image')
parser.add_argument('--batch_size', default=16, type=int,
                    help='total epochs to run')
parser.add_argument('--local_rank', type=int, default=0)

parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
args = parser.parse_args()


transform_train = transforms.Compose([
    transforms.Resize([args.input_size,args.input_size]),
    transforms.RandomCrop([args.input_size,args.input_size], padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize([args.input_size,args.input_size]),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batch_size = args.batch_size
train_dir='./imagenet-1k/train'
test_dir = './imagenet-1k/val'


trainset =  get_imagenet_dataloader(train_dir, batch_size=batch_size, transform = transform_train,train=True, val_data='ImageNet-A',)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, num_workers=2,sampler=DistributedSampler(trainset)) # 这个sampler会自动分配数据到各个gpu上

testset = get_imagenet_dataloader(test_dir, batch_size=1,transform = transform_test, train=False, val_data='ImageNet-A',)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1,  num_workers=2,sampler=DistributedSampler(testset,shuffle=False))