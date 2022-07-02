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

from attack.PixelAttack import attack_all
from attack.differential_evolution import differential_evolution
from time import strftime

import torch.multiprocessing
from attack.PixelAttack import attack_all
from attack.differential_evolution import differential_evolution
from time import strftime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.multiprocessing.set_sharing_strategy(
    'file_system')  # 防止生成adv样本时报错 https://github.com/pytorch/pytorch/issues/11201

transform_test = transforms.Compose([
    # Gau_noise.AddGaussianNoise(0.0, 8.0, 1.0),
    transforms.ToTensor(),  # 255 1
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = datasets.CIFAR10(root='./data', train=False, download=False,
                           transform=transform_test, )
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=0)

saved_model_path = 'checkpoint/ResNet18/cifar10/ckpt.pth_ResNet18_epoch200_matrix_20220328'
pgd_saved_path = "./data/cifar10_test_pgd_1-0.pt"

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

print('Before one-pixel attack, accuracy: %.2f %%' % (100 * float(correct) / total))

from attack.pixel_attack import attack
success = 0
length = 0
for image, label in testloader:
    img = image.numpy()
    label = label.numpy()
    result = attack(img, label, net, target=None, pixel_count=1)
    if result[3] == True:
        success += 1
        print('Success')
    else:
        print('Fail')

print('Success rate of one pixel attack:', success/length)

