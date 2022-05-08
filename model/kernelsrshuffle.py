
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
    return KernelSRShuffle(args)

class KernelSRShuffle(nn.Module):
    def __init__(self, args):
        super(KernelSRShuffle, self).__init__()
        self.nonlinearity = nn.Sigmoid()
        self.feat_extraction = RepVGG.RepVGGFE()
        # 单通道

        # self.importance_map = ImportanceMap.ImportanceMap(map_layers = 6, feat_layers = 6)
        # self.kernel_construction = KernelConstruction.KernelConstruction(outC = 1)
        # self.supersampling = Supersampling.Supersampling(outC = 1, featC = 6, sep_kernel = True)
        # self.nonlinearity = nn.ReLU()

        # # 3通道统一的特征+3通道分开的kernel
        # # self.importance_map = ImportanceMap.ImportanceMap(map_layers = 18)
        # # self.kernel_construction = KernelConstruction.KernelConstruction()
        # # self.supersampling = Supersampling.Supersampling()

        # # 3通道分开的特征+3通道统一的kernel
        # # self.importance_map = ImportanceMap.ImportanceMap(map_layers = 6, feat_layers = 18)
        # # # importance_map channel为6层，重构为单一54层kernel
        # # # feat_map为18层，每通道6层
        # # self.kernel_construction = KernelConstruction.KernelConstruction(outC = 1)
        # # self.supersampling = Supersampling.Supersampling(outC = 1, featC = 6)

        # 3通道分开的特征+3通道分开的kernel
        self.importance_map = ImportanceMap.ImportanceMap(map_layers = 18, feat_layers = 18)
        self.kernel_construction = KernelConstruction.KernelConstruction(outC = 3)
        self.supersampling = Supersampling.Supersampling(outC = 3, featC = 6, sep_kernel = True)

        #逐像素预测kernel
        # self.importance_map = ImportanceMap.ImportanceMap(map_layers = 1, feat_layers = 6)
        # self.supersampling = Supersampling.Supersampling(outC = 1, featC = 6, sep_kernel = True)

        # FSRCNN
        # Feature extraction layer.
        # self.feature_extraction = nn.Sequential(
        #     nn.Conv2d(1, 56, (5, 5), (1, 1), (2, 2)),
        #     nn.PReLU(56)
        # )

        # # Shrinking layer.
        # self.shrink = nn.Sequential(
        #     nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)),
        #     nn.PReLU(12)
        # )

        # # Mapping layer.
        # self.map = nn.Sequential(
        #     nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
        #     nn.PReLU(12),
        #     nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
        #     nn.PReLU(12),
        #     nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
        #     nn.PReLU(12),
        #     nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
        #     nn.PReLU(12)
        # )

        # # Expanding layer.
        # self.expand = nn.Sequential(
        #     nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)),
        #     nn.PReLU(56)
        # )

        # # Deconvolution layer.
        # self.deconv = nn.ConvTranspose2d(56, 1, (9, 9), (int(args.scale), int(args.scale)), (4, 4), (int(args.scale) - 1, int(args.scale) - 1))
        # Initialize model weights.
        # self._initialize_weights()

        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.PReLU(6),
            nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)
        )

        # define tail module
        # m_tail = [
        #     common.Upsampler(common.default_conv, 2, 12, act=False),
        #     nn.Conv2d(
        #         12, 3, 3,
        #         padding=(3//2)
        #     )
        # ]
        # self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        feat = self.feat_extraction(x)
        # print(feat.shape)
        # out = self.tail(feat)
        # out = self.nonlinearity(out)
        # return out
        #  kernel SR
        feat, immap = self.importance_map(feat)
        # print(immap.shape)
        # print(feat.shape)
        kernels = self.kernel_construction(immap)
        # print(kernels.shape)
        out = self.supersampling(feat,kernels)
        out = self.nonlinearity(self.tail(out))
        # print(out.shape)
        return out

        # out = self.feature_extraction(x)
        # out = self.shrink(out)
        # out = self.map(out)
        # out = self.expand(out)
        # out = self.deconv(out)
        # return out

    def _initialize_weights(self) -> None:
        from math import sqrt
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias.data)
