import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import torchvision
from tqdm import tqdm, trange

from utils import imgnormalize, gkern, get_gaussian_kernel

seed_num = 1
random.seed(seed_num)
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.backends.cudnn.deterministic = True
variance = np.random.uniform(0.5, 1.5, size=(3,))
neg_perturbations = - variance
variance = np.hstack((variance, neg_perturbations))
variance = np.append(variance, 0)
liner_interval = variance


def parse_arguments():
    parser = argparse.ArgumentParser(description='transfer_attack')
    parser.add_argument('--source_model', nargs="+", default=['resnet50'])  # 替代模型
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_iterations', type=int, default=50)
    parser.add_argument('--lr', type=eval, default=1.0 / 255.)
    parser.add_argument('--linf_epsilon', type=float, default=32)
    parser.add_argument('--di', type=eval, default="True")
    parser.add_argument('--result_path', type=str, default='transfer_res')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    adv_img_folder = os.path.join(args.result_path, 'images')  # 对抗样本保存文件夹
    if not os.path.exists(adv_img_folder):
        os.makedirs(adv_img_folder)
    norm = imgnormalize()  # 标准化处理类
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU ID
    source_model_names = args.source_model  # 替代模型
    num_source_models = len(source_model_names)  # 替代模型的数量
    source_models = []  # 根据替代模型的名称分别加载对应的网络模型
    for model_name in source_model_names:
        print("Loading: {}".format(model_name))
        source_model = models.__dict__[model_name](pretrained=True).eval()
        for param in source_model.parameters():
            param.requires_grad = False  # 不可导
        source_model.to(device)  # 计算环境
        source_models.append(source_model)  # ensemble

    # TI 参数设置
    channels = 3  # 3通道
    kernel_size = 5  # kernel大小
    kernel = gkern(kernel_size, 1).astype(np.float32)  # 3表述kernel内元素值得上下限
    gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=7)
    gaussian_filter.weight.data = gaussian_kernel  # 高斯滤波，高斯核的赋值
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=2)
    gaussian_smoothing = get_gaussian_kernel(kernel_size=5, sigma=1, channels=3, use_cuda=True)  # 高斯核（过滤部分高频信息） 5，1
    print('start atttacking....')
    idx = 0
    for X_ori, labels_gt in tqdm(trainloader):
        X_ori = X_ori.to(device)
        labels_gt = labels_gt.to(device)
        delta = torch.zeros_like(X_ori, requires_grad=True).to(device)  # 噪声大小的初始化
        X_ori = gaussian_smoothing(X_ori)  # 对输入图片进行高斯滤波
        for t in trange(args.max_iterations):
            g_temp = []
            for tt in range(len(liner_interval)):
                c = liner_interval[tt]
                X_adv = X_ori + c * delta  # 如果使用了DI，则不用顶点浮动
                X_adv = nn.functional.interpolate(X_adv, (224, 224), mode='bilinear', align_corners=False)  # 插值到224
                logits = 0
                for source_model_n, source_model in zip(source_model_names, source_models):
                    logits += source_model(norm(X_adv))  # ensemble操作
                logits /= num_source_models
                loss = -nn.CrossEntropyLoss()(logits, labels_gt)  # 交叉熵
                loss.backward()  # 梯度回传
                # MI + TI 操作
                grad_c = delta.grad.clone()  # 同时使用MI和TI
                grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)
                grad_a = grad_c
                g_temp.append(grad_a)
            g0 = 0.0
            for j in range(7):
                g0 += g_temp[j]  # 求均值，抵消噪声【多次DI随机，消除噪声，保留有效信息】
            g0 = g0 / 7.0
            delta.grad.zero_()  # 梯度清零
            # 无穷范数攻击
            delta.data = delta.data - args.lr * torch.sign(g0)
            delta.data = delta.data.clamp(-args.linf_epsilon / 255., args.linf_epsilon / 255.)
            delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori  # 噪声截取操作

        for i in range(X_ori.shape[0]):
            adv_final = (X_ori + delta)[i].cpu().detach().numpy()
            adv_final = (adv_final * 255).astype(np.uint8)
            file_path = os.path.join(adv_img_folder, str(labels_gt[i].item()))
            if not os.path.exists(file_path):
                os.mkdir(file_path)
            file_path = os.path.join(file_path, f"{idx}.jpg")
            adv_x_255 = np.transpose(adv_final, (1, 2, 0))
            im = Image.fromarray(adv_x_255)
            im.save(file_path, quality=99)
            idx += 1

if __name__ == '__main__':
    main()
