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

class DIV2K(multiscalesrdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(DIV2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = names_lr[self.begin - 1:self.end]
        
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        print(self.apath)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SR')
    # Data specifications
    parser.add_argument('--dir_data', type=str, default='/home/yejiannan/SIGGRAPH2022/data',
                        help='dataset directory')
    parser.add_argument('--dir_demo', type=str, default='../test',
                        help='demo image directory')
    parser.add_argument('--data_train', type=str, default='DIV2K',
                        help='train dataset name')
    parser.add_argument('--data_test', type=str, default='Set5',
                        help='test dataset name')
    parser.add_argument('--data_range', type=str, default='1-780/780-800',
                        help='train/test data range')
    parser.add_argument('--ext', type=str, default='img',
                        help='dataset file extension')
    parser.add_argument('--scale', type=str, default='2',
                        help='super resolution scale')
    parser.add_argument('--patch_size', type=int, default=50,
                        help='output patch size')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--chop', action='store_true',
                        help='enable memory-efficient forward')
    parser.add_argument('--no_augment', action='store_true',
                        help='do not use data augmentation')
    
    # Training specifications
    parser.add_argument('--reset', action='store_true',
                        help='reset the training')
    parser.add_argument('--test_every', type=int, default=1000,
                        help='do test per every N batches')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training')
    parser.add_argument('--split_batch', type=int, default=1,
                        help='split the batch into smaller chunks')
    parser.add_argument('--self_ensemble', action='store_true',
                        help='use self-ensemble method for test')
    parser.add_argument('--test_only', action='store_true',
                        help='set this option to test the model')


    args = parser.parse_args()

    #args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    ###here we redefine the scale
    
    if args.scale=='':
        import numpy as np
        #args.scale = np.linspace(1.1,4,30)
        args.scale = [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]
        #print(args.scale)
    else:
        args.scale = list(map(lambda x: float(x), args.scale.split('+')))
    print(args.scale)


    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
    print(args)
    dataset = DIV2K(args)
    # len(dataloader)

    print(len(dataset))
    hr0, lr0, name = dataset[0]
    print(hr0)
    print(lr0)
    print(name)

    from torch.utils.data.dataloader import DataLoader
    train_dataloader = DataLoader(dataset=dataset,
                                  batch_size=2,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)
    print("dataset len:",len(train_dataloader))
    idx = 0
    for data in train_dataloader:
        lr, hr, name = data
        print(lr.shape)
        print(hr.shape)
        print(name)

        lr = lr[0,...].numpy() * 255.
        lr = lr.transpose([1, 2, 0]).astype(np.uint8)
        imageio.imwrite('GT1.jpg', lr)

        break
