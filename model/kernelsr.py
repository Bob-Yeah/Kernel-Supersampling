
# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common
from model import ImportanceMap
from model import KernelConstruction
from model import RepVGG
from model import Supersampling
import time
import torch
import torch.nn as nn
import math

def make_model(args, parent=False):
    return KernelSR(args)

class KernelSR(nn.Module):
    def __init__(self, args):
        super(KernelSR, self).__init__()
        self.feat_extraction = RepVGG.RepVGGFE()
        # 单通道
        # self.importance_map = ImportanceMap.ImportanceMap(map_layers = 6, feat_layers = 6)
        # self.kernel_construction = KernelConstruction.KernelConstruction(outC = 1)
        # self.supersampling = Supersampling.Supersampling(outC = 1, featC = 6, sep_kernel = True)

        # 3通道统一的特征+3通道分开的kernel
        # self.importance_map = ImportanceMap.ImportanceMap(map_layers = 18)
        # self.kernel_construction = KernelConstruction.KernelConstruction()
        # self.supersampling = Supersampling.Supersampling()

        # 3通道分开的特征+3通道统一的kernel
        # self.importance_map = ImportanceMap.ImportanceMap(map_layers = 6, feat_layers = 18)
        # # importance_map channel为6层，重构为单一54层kernel
        # # feat_map为18层，每通道6层
        # self.kernel_construction = KernelConstruction.KernelConstruction(outC = 1)
        # self.supersampling = Supersampling.Supersampling(outC = 1, featC = 6)

        # 3通道分开的特征+3通道分开的kernel
        self.importance_map = ImportanceMap.ImportanceMap(map_layers = 18, feat_layers = 18)
        self.kernel_construction = KernelConstruction.KernelConstruction(outC = 3)
        self.supersampling = Supersampling.Supersampling(outC = 3, featC = 6, sep_kernel = True)
        
        #逐像素预测kernel
        # self.importance_map = ImportanceMap.ImportanceMap(map_layers = 1, feat_layers = 6)
        # self.supersampling = Supersampling.Supersampling(outC = 1, featC = 6, sep_kernel = True)


    def forward(self, x):
        feat = self.feat_extraction(x)
        # print(feat.shape)
        feat, immap = self.importance_map(feat)
        # print(immap.shape)
        # print(feat.shape)
        kernels = self.kernel_construction(immap)
        # print(kernels.shape)
        out = self.supersampling(feat,kernels)
        # print(out.shape)
        return out
