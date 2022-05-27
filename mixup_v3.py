import numpy as np
import torch

def rand_bbox(img_shape, n):
    batch, channel, img_h, img_w = img_shape[:]  # N, C, H, W
    interval_h = int(img_h / n)
    interval_w = int(img_w / n)
    A = np.zeros((batch, channel, img_h, img_w)) # A in the formula
    B = np.ones((batch, channel, img_h, img_w)) # A+B == I
    num = 0
    for i in range(n):
        for j in range(n):
            if np.random.uniform() > 0.5:
                A[:, :, interval_h*i:interval_h*(i+1), interval_h*j:interval_h*(j+1)] = np.ones((batch, channel, interval_h, interval_w))
                B[:, :, interval_h*i:interval_h*(i+1), interval_h*j:interval_h*(j+1)] = np.zeros((batch, channel, interval_h, interval_w))
                num += 1

    return A, B, num/(n**2)



def mixup_data(x, y, n, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    A, B, lam = rand_bbox(x.shape, n)

    mixed_x = torch.multiply(torch.from_numpy(A).cuda(), x) + torch.multiply(torch.from_numpy(B).cuda(), x[index, :])
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

#
# if __name__ == '__main__':
#     a = np.ones((3, 28, 28))
#     a_torch = torch.from_numpy(a)
#     b = torch.multiply(torch.from_numpy(a), a_torch)
#     print(rand_bbox((3, 9, 9), 3))