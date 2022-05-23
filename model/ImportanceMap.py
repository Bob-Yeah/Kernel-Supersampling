import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torch.nn import init
import numpy as np
import math

class ImportanceMap(nn.Module): 

    def __init__(self,  map_layers = 1, feat_layers = 6, in_c=12):
        super(ImportanceMap, self).__init__()
        self.nonlinearity = nn.PReLU()

        # scale = 2
        # self.upscale = nn.ConvTranspose2d(in_channels=in_c, out_channels=feat_layers, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.conv1 = nn.Conv2d(in_c, 4 * feat_layers, 3,stride=1, padding=1)
        self.shuffle1 = nn.PixelShuffle(2)

        # scale = 4
        self.conv2 = nn.Conv2d(feat_layers, 4 * feat_layers, 3,stride=1, padding=1)
        self.shuffle2 = nn.PixelShuffle(2)

        self.genmap = nn.Conv2d(in_channels=feat_layers, out_channels=map_layers, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # feat = self.nonlinearity(self.upscale(x))
        feat = self.nonlinearity(
            self.shuffle2(
                self.conv2(self.shuffle1(self.conv1(x)))
                )
            )
        out = self.nonlinearity(self.genmap(feat))
        return feat, out

if __name__ == "__main__":
    input_feature = 12
    N = 8
    H = 224
    W = 224

    model = ImportanceMap(18)
    data = torch.ones((N,input_feature,H,W))

    model.cuda()
    data = data.cuda()
    print("data", data.shape)
    feat,out = model(data)
    print("feat", feat.shape)
    print("out", out.shape)