import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data.distributed import DistributedSampler
import models
import argparse

from new_dataset import New_Dataset


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--local_rank', type=int, default=0)

args = parser.parse_args()

torch.cuda.set_device(args.local_rank)  # 这里设定每一个进程使用的GPU是一定的，即一张gpu一个进程
device = torch.device('cuda', args.local_rank)
torch.distributed.init_process_group(backend='nccl') #初始化


transform_test = transforms.Compose([
    transforms.ToTensor(), #255 1
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# saved_model_path = 'checkpoint/resnest50/ckpt.pth_resnest50_ori_last_model.pth'
# saved_model_path = 'checkpoint/resnest50/ckpt.pth_resnest50_one_fourth_last_model.pth'
# saved_model_path = 'checkpoint/resnest50/ckpt.pth_resnest50_cutmix_last_model.pth'
saved_model_path = 'checkpoint/resnest50/ckpt.pth_resnest50_none_last_model.pth'


radius = [160]
# radius = [32, 64, 96, 128, 160, 192, 224]

for r in radius:
    test_low = './data/test_data_low_' + str(r) + '.npy'

    testset_low = New_Dataset(test_low,'./data/test_label.npy',transform_test)

    testloader = torch.utils.data.DataLoader(testset_low, batch_size=10,
                                            shuffle=False, num_workers=8)

    # test_high = './data/test_data_high_' + str(r) + '.npy'
    # testset_high = New_Dataset(test_high,'./data/test_label.npy',transform_test)
    #
    # testloader = torch.utils.data.DataLoader(testset_high, batch_size=10,
    #                                           shuffle=False, num_workers=8)
    checkpoint = torch.load(saved_model_path)
    net = checkpoint['net']
    net.eval()
    # net.eval().cuda()

    # 模型在原测试集上的准确率
    correct = 0
    total = 0

    for images, labels in testloader:
        
        images = images.cuda()
        outputs = net(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
        
    print('frequency =',r,', test accuracy: %.2f %%' % (100 * float(correct) / total))
    


