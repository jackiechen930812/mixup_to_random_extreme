import numpy as np
import random
import torch
from PIL import Image,ImageFilter
import os
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing
import math
device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.multiprocessing.set_sharing_strategy(
    'file_system')  # 防止生成adv样本时报错 https://github.com/pytorch/pytorch/issues/11201


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

# class AddGaussianNoise(object):
#
#     def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
#
#         self.mean = mean            #均值
#         self.variance = variance    #方差
#         self.amplitude = amplitude  #倍数，原始图像所加的高斯噪声的倍数
#
#     def __call__(self, img):
#         img = np.array(img)
#         h, w, c = img.shape
#         N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
#         N = np.repeat(N, c, axis=2)
#         img = N + img
#         img[img > 255] = 255
#         img = Image.fromarray(img.astype('uint8')).convert('RGB')
#         return img

transform_test = transforms.Compose([
    transforms.ToTensor(), #255 1
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = datasets.CIFAR10(root='./data', train=False, download=False,
                           transform= transform_test,)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=8)
# saved_model_path = 'checkpoint/ResNet18/cifar10/ckpt.pth_ResNet18_ori_last_model.pth'
# saved_model_path = 'checkpoint/ResNet18/cifar10/ckpt.pth_ResNet18_none_last_model.pth'
# saved_model_path = 'checkpoint/ResNet18/cifar10/ckpt.pth_ResNet18_cutmix_last_model.pth'
# saved_model_path = 'checkpoint/ResNet18/cifar10/ckpt.pth_ResNet18_one_fourth_last_model.pth'
# saved_model_path = 'checkpoint/ResNet18/cifar10/ckpt.pth_ResNet18_puzzlemix_last_model.pth'
saved_model_path = 'checkpoint/ResNet18/cifar10/ckpt.pth_ResNet18_comix_last_model.pth'

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

print('Before Random Erasing attack, accuracy: %.2f %%' % (100 * float(correct) / total))


# 模型在被Random Erase Noise攻击后的测试集样本上的准确率
correct_adv = 0
total_adv = 0
transform_test_adv = transforms.Compose([
    transforms.ToTensor(), #255 1
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    RandomErasing(probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]),
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

print('After Random Erasing attack, accuracy: %.2f %%' % (100 * float(correct_adv) / total_adv))