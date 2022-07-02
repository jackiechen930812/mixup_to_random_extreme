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
from dataset_img_9 import get_imagenet_dataloader
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

batch_size = args.batch_size

# 使用DDP进行多进程分布式训练，local_rank表示系统自动分配的进程号，
torch.cuda.set_device(args.local_rank)  # 这里设定每一个进程使用的GPU是一定的，即一张gpu一个进程
device = torch.device('cuda', args.local_rank)
torch.distributed.init_process_group(backend='nccl') #初始化


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean            #均值
        self.variance = variance    #方差
        self.amplitude = amplitude  #倍数，原始图像所加的高斯噪声的倍数

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

transform_test = transforms.Compose([
    transforms.Resize([args.input_size,args.input_size]),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_dir = './imagenet-1k/val'
testset = get_imagenet_dataloader(test_dir, batch_size=batch_size,transform = transform_test, train=False, val_data='ImageNet-A',)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size,  num_workers=2,sampler=DistributedSampler(testset,shuffle=False))

# saved_model_path = 'checkpoint/resnest50/ckpt.pth_resnest50_cutmix_last_model.pth'
# saved_model_path = 'checkpoint/resnest50/ckpt.pth_resnest50_ori_last_model.pth'
saved_model_path = 'checkpoint/resnest50/ckpt.pth_resnest50_one_fourth_last_model.pth'
# saved_model_path = 'checkpoint/resnest50/ckpt.pth_resnest50_none_last_model.pth'

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

print('Before Gaussian Noise attack, accuracy: %.2f %%' % (100 * float(correct) / total))


# 模型在被GAU Noise攻击后的测试集样本上的准确率
correct_adv = 0
total_adv = 0
transform_test_adv = transforms.Compose([
    AddGaussianNoise(0.0, 15.0, 1.0),
    transforms.Resize([args.input_size,args.input_size]),
    transforms.ToTensor(), #255 1
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset_adv = get_imagenet_dataloader(test_dir, batch_size=batch_size,transform = transform_test_adv, train=False, val_data='ImageNet-A',)
testloader_adv = torch.utils.data.DataLoader(
    testset_adv, batch_size=16,  num_workers=2,sampler=DistributedSampler(testset_adv,shuffle=False))


for images, labels in testloader_adv:
    images = images.cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)

    total_adv += labels.size(0)
    correct_adv += (predicted == labels.cuda()).sum()

print('After Gaussian Noise attack, accuracy: %.2f %%' % (100 * float(correct_adv) / total_adv))