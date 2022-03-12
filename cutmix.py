import numpy as np
import torch


def rand_bbox(img_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.
    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh

def cutmix_bbox_and_lam(img_shape, lam,  count=None):
    """ Generate bbox and apply lambda correction.
    """
    yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    bbox_area = (yu - yl) * (xu - xl)
    lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


def mixup_data(x, y, alpha, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    lam = np.random.beta(alpha, alpha)
    lam = float(lam)
    
    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam( x.shape, lam, )
    x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
    

    return x, lam


def mixup_criterion(criterion, pred, target, lam):
    target_flip = target.flip(0)
    y = target * lam + target_flip * (1. - lam)
    return criterion(pred, y.long())


# def rand_bbox(size, lam):
#     W = size[2]
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = np.int(W * cut_rat)
#     # uniform
#     cx = np.random.randint(W)
#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     return bbx1, bbx2
