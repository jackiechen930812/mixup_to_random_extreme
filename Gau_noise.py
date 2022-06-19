import numpy as np
import random
import torch
from PIL import Image,ImageFilter
import os
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing
device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.multiprocessing.set_sharing_strategy(
    'file_system')  # 防止生成adv样本时报错 https://github.com/pytorch/pytorch/issues/11201

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
    # Gau_noise.AddGaussianNoise(0.0, 8.0, 1.0),
    transforms.ToTensor(), #255 1
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = datasets.CIFAR10(root='./data', train=False, download=False,
                           transform= transform_test,)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=8)
saved_model_path = 'checkpoint/ckpt.pthResNet18_200_ori_20220520'
# gau_saved_path = "./data/cifar10_test_pgd_1-0.pt"


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
    AddGaussianNoise(0.0, 8.0, 1.0),
    transforms.ToTensor(), #255 1
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset_adv = datasets.CIFAR10(root='./data', train=False, download=False,
                           transform= transform_test_adv,)
testloader_adv = torch.utils.data.DataLoader(testset_adv, batch_size=10,
                                         shuffle=False, num_workers=8)
for images, labels in testloader_adv:
    images = images.cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)

    total_adv += labels.size(0)
    correct_adv += (predicted == labels.cuda()).sum()

print('After Gaussian Noise attack, accuracy: %.2f %%' % (100 * float(correct_adv) / total_adv))