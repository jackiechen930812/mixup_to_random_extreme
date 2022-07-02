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

import torchattacks
from torchattacks import PGD
import models


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')   #防止生成adv样本时报错 https://github.com/pytorch/pytorch/issues/11201

transform_test = transforms.Compose([
    transforms.ToTensor(), #255 1
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = datasets.CIFAR10(root='./data', train=False, download=False,
                           transform= transform_test,)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=8)

# saved_model_path = './checkpoint/ResNet18/ckpt.t7_ResNet18_epoch50_2_1_baseline_20220103'  
saved_model_path = './checkpoint/ResNet18/ckpt.pth_ResNet18_epoch200_none_20220424'
pgd_saved_path = "./data/cifar10_test_pgd_2-0.pt"
if os.path.exists(pgd_saved_path) == False :
    checkpoint = torch.load(saved_model_path)
    net = checkpoint['net']
    net = net.eval()
    # atk = PGD(net, eps=8/255, alpha=2/255, steps=4)
    atk = PGD(net, eps=6.0 / 255, alpha=1.0 / 255, steps=40)
    atk.set_return_type('int') # 0-255 Save as integer. float 0-1
    atk.save(data_loader=testloader, save_path=pgd_saved_path, verbose=True)



##测试所存模型在的准确率
adv_images, adv_labels = torch.load(pgd_saved_path)
adv_data = TensorDataset(adv_images.float()/255, adv_labels)
adv_loader = DataLoader(adv_data, batch_size=128, shuffle=False)

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
    
print('Before PGD attack, accuracy: %.2f %%' % (100 * float(correct) / total))


# 模型在被PGD攻击后的测试集样本上的准确率
correct_adv = 0
total_adv = 0

for images, labels in adv_loader:
    
    images = images.cuda()
    outputs = net(images)
    
    _, predicted = torch.max(outputs.data, 1)
    
    total_adv += labels.size(0)
    correct_adv += (predicted == labels.cuda()).sum()
    
print('After PGD attack, accuracy: %.2f %%' % (100 * float(correct_adv) / total_adv))

