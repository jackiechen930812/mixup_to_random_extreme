import numpy as np
import random
from PIL import Image,ImageFilter
<<<<<<< HEAD
from torchvision.utils import save_image
=======

>>>>>>> 6ba0baf0edc972d13e42a54073817dbfe9121015
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
<<<<<<< HEAD
        img[img > 255] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img
=======
        img[img > 255] = 255                      
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img
>>>>>>> 6ba0baf0edc972d13e42a54073817dbfe9121015
