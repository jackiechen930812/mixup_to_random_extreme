__author__ = 'Haohan Wang'

import numpy as np
from scipy import signal
import os
import cv2
import random
def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return np.fft.fftshift(fft(img))


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(np.fft.ifftshift(img))


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask


def generateSmoothKernel(data, r):
    result = np.zeros_like(data)
    [k1, k2, m, n] = data.shape
    mask = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                mask[i,j] = 1
            else:
                mask[i,j] = r
    mask = mask
    for i in range(m):
        for j in range(n):
            result[:,:, i,j] = signal.convolve2d(data[:,:, i,j], mask, boundary='symm', mode='same')
    return result


def generateDataWithDifferentFrequencies_GrayScale(Images, r):
    Images_freq_low = []
    mask = mask_radial(np.zeros([28, 28]), r)
    for i in range(Images.shape[0]):
        fd = fftshift(Images[i, :].reshape([28, 28]))
        fd = fd * mask
        img_low = ifftshift(fd)
        Images_freq_low.append(np.real(img_low).reshape([28 * 28]))

    return np.array(Images_freq_low)

def generateDataWithDifferentFrequencies_3Channel(Images, r):
    Images_freq_low = []
    Images_freq_high = []
    mask = mask_radial(np.zeros([Images.shape[1], Images.shape[2]]), r)
    for i in range(Images.shape[0]):
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * mask
            img_low = ifftshift(fd)
            tmp[:,:,j] = np.real(img_low)
        Images_freq_low.append(tmp)
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * (1 - mask)
            img_high = ifftshift(fd)
            tmp[:,:,j] = np.real(img_high)
        Images_freq_high.append(tmp)

    return np.array(Images_freq_low), np.array(Images_freq_high)




def image_label(imageLabel, label2idx, i):
    """返回图片的label
    """
    if imageLabel not in label2idx:
        label2idx[imageLabel]=i
        i = i + 1
    return label2idx, i

def image2npy(dir_path='./imagenet-100/val'):

    i = 0
    label2idx = {}
    data = []
    for (root, dirs, files) in os.walk(dir_path):
        for Ufile in files:
            # Ufile是文件名
            img_path = os.path.join(root, Ufile)        # 文件的所在路径
            File = root.split('/')[-1]                  # label名称
            img_data = cv2.imread(img_path)             # 读取图像
            img_data = cv2.resize(img_data,(224,224))   # resize成input_size的尺寸，此处需统一尺寸，方便后续生成npy
            label2idx, i = image_label(File, label2idx, i) #得到所有的类别
            label = label2idx[File]
            data.append([np.array(img_data), label])    # 存储image和label数据
    random.shuffle(data) 

    img = np.array([i[0] for i in data])      # 图像
    label = np.array([i[1] for i in data])    # 标签

    print(len(img), len(label))
    np.save('./data/test_label', label)       #保存标签的npy文件
    return img, label, len(img)


if __name__ == '__main__':
    import sys
    version = sys.version_info

    eval_images, eval_labels, len_images = image2npy(dir_path='./imagenet-100/val')
    # eval_images, eval_labels = load_datafile('./data/cifar-10-batches-py/test_batch')

    images = np.zeros((len_images, 224, 224, 3), dtype='uint8')

    for i, cur_images in enumerate(eval_images):
        images[i,...] = cur_images

    eval_image_low_32, eval_image_high_32 = generateDataWithDifferentFrequencies_3Channel(images, 32)
    np.save('./data/test_data_low_32', eval_image_low_32)
    np.save('./data/test_data_high_32', eval_image_high_32)

    eval_image_low_64, eval_image_high_64 = generateDataWithDifferentFrequencies_3Channel(images, 64)
    np.save('./data/test_data_low_64', eval_image_low_64)
    np.save('./data/test_data_high_64', eval_image_high_64)

    eval_image_low_96, eval_image_high_96 = generateDataWithDifferentFrequencies_3Channel(images, 96)
    np.save('./data/test_data_low_96', eval_image_low_96)
    np.save('./data/test_data_high_96', eval_image_high_96)

    eval_image_low_128, eval_image_high_128 = generateDataWithDifferentFrequencies_3Channel(images, 128)
    np.save('./data/test_data_low_128', eval_image_low_128)
    np.save('./data/test_data_high_128', eval_image_high_128)

    eval_image_low_160, eval_image_high_160 = generateDataWithDifferentFrequencies_3Channel(images, 160)
    np.save('./data/test_data_low_160', eval_image_low_160)
    np.save('./data/test_data_high_160', eval_image_high_160)


    eval_image_low_192, eval_image_high_192 = generateDataWithDifferentFrequencies_3Channel(images, 192)
    np.save('./data/test_data_low_192', eval_image_low_192)
    np.save('./data/test_data_high_192', eval_image_high_192)

    eval_image_low_224, eval_image_high_224 = generateDataWithDifferentFrequencies_3Channel(images, 224)
    np.save('./data/test_data_low_224', eval_image_low_224)
    np.save('./data/test_data_high_224', eval_image_high_224)



    # np.save('./data/CIFAR10/train_images', train_images)
    # np.save('./data/CIFAR10/train_label', train_labels)

    # train_image_low_4, train_image_high_4 = generateDataWithDifferentFrequencies_3Channel(train_images, 4)
    # np.save('./data/CIFAR10/train_data_low_4', train_image_low_4)
    # np.save('./data/CIFAR10/train_data_high_4', train_image_high_4)

    # train_image_low_8, train_image_high_8 = generateDataWithDifferentFrequencies_3Channel(train_images, 8)
    # np.save('./data/CIFAR10/train_data_low_8', train_image_low_8)
    # np.save('./data/CIFAR10/train_data_high_8', train_image_high_8)

    # train_image_low_12, train_image_high_12 = generateDataWithDifferentFrequencies_3Channel(train_images, 12)
    # np.save('./data/CIFAR10/train_data_low_12', train_image_low_12)
    # np.save('./data/CIFAR10/train_data_high_12', train_image_high_12)

    # train_image_low_16, train_image_high_16 = generateDataWithDifferentFrequencies_3Channel(train_images, 16)
    # np.save('./data/CIFAR10/train_data_low_16', train_image_low_16)
    # np.save('./data/CIFAR10/train_data_high_16', train_image_high_16)


    #eval_image_low_4, eval_image_high_4 = generateDataWithDifferentFrequencies_3Channel(images, 4)
    #np.save('./data/test_data_low_4', eval_image_low_4)
    #np.save('./data/test_data_high_4', eval_image_high_4)


    #eval_image_low_8, eval_image_high_8 = generateDataWithDifferentFrequencies_3Channel(images, 8)
    #np.save('./data/test_data_low_8', eval_image_low_8)
    #np.save('./data/test_data_high_8', eval_image_high_8)

    #eval_image_low_12, eval_image_high_12 = generateDataWithDifferentFrequencies_3Channel(images, 12)
    #np.save('./data/test_data_low_12', eval_image_low_12)
    #np.save('./data/test_data_high_12', eval_image_high_12)

    #eval_image_low_16, eval_image_high_16 = generateDataWithDifferentFrequencies_3Channel(images, 16)
    #np.save('./data/test_data_low_16', eval_image_low_16)
    #np.save('./data/test_data_high_16', eval_image_high_16)

    #eval_image_low_20, eval_image_high_20 = generateDataWithDifferentFrequencies_3Channel(images, 20)
    #np.save('./data/test_data_low_20', eval_image_low_20)
    #np.save('./data/test_data_high_20', eval_image_high_20)

    #eval_image_low_24, eval_image_high_24 = generateDataWithDifferentFrequencies_3Channel(images, 24)
    #np.save('./data/test_data_low_24', eval_image_low_24)
    #np.save('./data/test_data_high_24', eval_image_high_24)

    #eval_image_low_28, eval_image_high_28 = generateDataWithDifferentFrequencies_3Channel(images, 28)
    #np.save('./data/test_data_low_28', eval_image_low_28)
    #np.save('./data/test_data_high_28', eval_image_high_28)
