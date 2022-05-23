import os
import glob

from data import common
# import common

import pickle
import numpy as np
import imageio

import torch
import torch.utils.data as data
from data import multiscalesrdata

class DIV2KLOW(multiscalesrdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        super(DIV2KLOW, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(
                os.path.join(self.dir_lr, 'X{}'.format('2')) , 
                '*' + self.ext[0]))
        )
        names_lr = sorted(
            glob.glob(os.path.join(
                os.path.join(self.dir_lr, 'X{}'.format('4')) , 
                '*' + self.ext[0]))
        )
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2KLOW, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')