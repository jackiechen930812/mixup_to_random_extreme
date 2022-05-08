from torch.utils.data import DataLoader, TensorDataset
import torchattacks
from torchattacks import PGD
import models
import numpy as np
import random
import torch
import torchvision
from PIL import Image,ImageFilter
import os
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing
from torch.utils.data.distributed import DistributedSampler

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_size', default=224, type=int,
                    help='the size of input image')
parser.add_argument('--batch_size', default=16, type=int,
                    help='total epochs to run')
parser.add_argument('--local_rank', type=int, default=0)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.multiprocessing.set_sharing_strategy(
    'file_system')  # 防止生成adv样本时报错 https://github.com/pytorch/pytorch/issues/11201



# 使用DDP进行多进程分布式训练，local_rank表示系统自动分配的进程号，
torch.cuda.set_device(args.local_rank)  # 这里设定每一个进程使用的GPU是一定的，即一张gpu一个进程
device = torch.device('cuda', args.local_rank)
torch.distributed.init_process_group(backend='nccl') #初始化



import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')   #防止生成adv样本时报错 https://github.com/pytorch/pytorch/issues/11201

transform_test = transforms.Compose([
    transforms.Resize([args.input_size,args.input_size]),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_dir = './imagenet-100/val'
testset = torchvision.datasets.ImageFolder(test_dir,transform=transform_test)
#testloader = torch.utils.data.DataLoader(
#    testset, batch_size=16,  num_workers=2,sampler=DistributedSampler(testset,shuffle=False))

testloader = torch.utils.data.DataLoader(testset, batch_size=16, num_workers=2)

saved_model_path = './checkpoint/ResNet18/ckpt.pth_resnest50_epoch200_matrix_20220502'
pgd_saved_path = "./data/imagenet100_test_pgd_matrix_1-1.pt"

if os.path.exists(pgd_saved_path) == False :
    checkpoint = torch.load(saved_model_path)
    net = checkpoint['net']
    # net = net.eval()
    net = net.to(device)
    # net = net.eval().cuda()
    # atk = PGD(net, eps=8/255, alpha=2/255, steps=4)
    atk = PGD(net, eps=6.0 / 255, alpha=1.0 / 255, steps=40)
    atk.set_return_type('int') # 0-255 Save as integer. float 0-1
    atk.save(data_loader=testloader, save_path=pgd_saved_path, verbose=True)


##测试所存模型在的准确率
adv_images, adv_labels = torch.load(pgd_saved_path)
adv_data = TensorDataset(adv_images.float()/255, adv_labels)
#adv_loader = DataLoader(adv_data, batch_size=16, shuffle=False, sampler=DistributedSampler(adv_data))
adv_loader = DataLoader(adv_data, batch_size=16, shuffle=False)
checkpoint = torch.load(saved_model_path)
net = checkpoint['net']
net.eval()
#net.eval().cuda()

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



