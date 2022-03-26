import torch
import numpy as np
import skimage

from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
 
class New_Dataset(Dataset):

    def __init__(self, data, labels, transforms):   #初始化函数
        self.data = np.load(data)               #data为图像数据存放地址，
        self.labels = np.load(labels)           #labels为标签存放地址，
        self.transforms = transforms            #对图像进行数据增强

    def __getitem__(self, index):                   #得到dataset中的每一项对应的图像和标签
        image= self.data[index, :, :, :]        #读取每一张图像的数据
        image=Image.fromarray(np.uint8(image))  #转成PIL Image的形式，适合网络输入。(去掉会报错）
        image= self.transforms(image)           #对图像进行数据增强
        label = self.labels[index]              #读取图像对应标签
        return image,label                      #返回图像还有标签

    def __len__(self):
        return self.data.shape[0]               #返回数据的总个数
    
    def __add__(self, other):                   #用于多个dataset拼接成一个dataset，将参数中的other与现有数据集进行拼接
        return ConcatDataset([self, other])
 
