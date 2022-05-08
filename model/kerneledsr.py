from model import common

import torch.nn as nn

from model import ImportanceMap
from model import KernelConstruction
from model import Supersampling

def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return KernelEDSR(args, dilated.dilated_conv)
    else:
        return KernelEDSR(args)

class KernelEDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(KernelEDSR, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = int(args.scale)
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.importance_map = ImportanceMap.ImportanceMap(map_layers = 18, feat_layers = 18, in_c=64)
        self.kernel_construction = KernelConstruction.KernelConstruction(outC = 3)
        self.supersampling = Supersampling.Supersampling(outC = 3, featC = 6, sep_kernel = True)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.nonlinearity = nn.Sigmoid()

    def forward(self, x):
        #特征提取的模块
        x = self.head(x)
        res = self.body(x)
        res += x
        #end += skip connection

        # print(res.shape) # torch.Size([16, 64, 64, 64])
        # x = self.tail(res)
        #预测feat，immap
        feat, immap = self.importance_map(res)
        # print(immap.shape)
        # print(feat.shape)
        #构建kernel
        kernels = self.kernel_construction(immap)
        # print(kernels.shape)
        #应用kernel
        out = self.supersampling(feat,kernels)
        #最后加一层sigmoid激活层
        out = self.nonlinearity(out)
        # print(out.shape) # torch.Size([16, 3, 128, 128])
        return out

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

