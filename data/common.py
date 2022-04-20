import random

import numpy as np
import skimage.color as sc

import torch
from torchvision import transforms

def get_patch(*args, patch_size=96, scale=1):
    ih, iw = args[0].shape[:2]
    tp =  (int)(int(scale)* int(patch_size))
    ip = int(patch_size)
    ix = random.randrange(0, iw-ip)
    iy = random.randrange(0, ih-ip)
    tx, ty = int(int(scale) * ix), int(int(scale)  * iy)
    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        args[1][ty:ty + tp, tx:tx + tp, :]
    ]
    
    return ret

def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(convert_rgb_to_y(img),2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)
        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(1.0 / rgb_range)
        return tensor
    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(a) for a in args]

