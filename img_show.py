import numpy as np
import torch
from torchvision import transforms
import cv2


def save_mix_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 加载到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    input_tensor = cv2.resize(input_tensor,(256,256)) 
    cv2.imwrite(filename, input_tensor)