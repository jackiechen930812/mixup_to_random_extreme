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

import models

from new_dataset import New_Dataset

transform_test = transforms.Compose([
    transforms.ToTensor(), #255 1
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

saved_model_path = './checkpoint/ckpt.pthResNet18_200_one_fourth_20220520'

radius = [4, 8, 12, 16, 20, 24, 28]
for r in radius:
    # test_low = './data/CIFAR10/test_data_low_' + str(r) + '.npy'
    test_high = './data/CIFAR10/test_data_high_' + str(r) + '.npy'


    # testset = New_Dataset(test_low,'./data/CIFAR10/test_label.npy',transform_test)
    testset_high_4 = New_Dataset(test_high,'./data/CIFAR10/test_label.npy',transform_test)
    # testset = testset.__add__(testset_high_4)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=10,
    #                                          shuffle=False, num_workers=8)

    testloader = torch.utils.data.DataLoader(testset_high_4, batch_size=10,
                                             shuffle=False, num_workers=8)
    checkpoint = torch.load(saved_model_path)
    net = checkpoint['net']
    net.eval().cuda()

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

