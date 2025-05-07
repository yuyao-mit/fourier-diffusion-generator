# blocks.py 
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
import numpy as np
from tools import *
import os

class Rezero(nn.Module):
    "https://arxiv.org/abs/2003.04887"
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1)) 

    def forward(self, x):
        return x + self.g * self.fn(x)  


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Block(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, groups=8):
        super().__init__()
        if groups != 0 and dim_out % groups != 0:
            raise ValueError(f'dim_out ({dim_out}) must be divisible by groups ({groups})')
        padding = (kernel_size - 1) // 2
        if groups == 0:
            self.block = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(dim, dim_out, kernel_size),
                nn.Mish()
            )
        else:
            self.block = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(dim, dim_out, kernel_size),
                nn.GroupNorm(groups, dim_out),
                nn.Mish()
            )

    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, groups=8):
        super().__init__()
        self.block1 = Block(dim, dim_out, kernel_size=kernel_size, groups=groups)
        self.block2 = Block(dim_out, dim_out, kernel_size=kernel_size, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, cond=None):
        h = self.block1(x)
        if cond is not None:
            if cond.shape != h.shape:
                raise ValueError(f'Condition shape {cond.shape} does not match hidden shape {h.shape}')
            h = h + cond
        h = self.block2(h)
        return h + self.res_conv(x)


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim_in
        self.conv = nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim_in
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(dim_in, dim_out, 3, stride=2)

    def forward(self, x):
        return self.conv(self.pad(x))


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block with 5 Convolutions.
    Inspired by: "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks" (Wang et al., 2018)
    URL: https://arxiv.org/abs/1809.00219
    """
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf, gc=32, num_blocks=3):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ResidualDenseBlock_5C(nf, gc) for _ in range(num_blocks)
        ])

    def forward(self, x):
        return self.blocks(x) * 0.2 + x


###### Inspired by "Towards Real-Time 4K Image Super-Resolution" ########
################# https://github.com/eduardzamfir/RT4KSR ################

class ResBlock(nn.Module):
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.
    Diagram:
        ---Conv1x1--Conv3x3-+-Conv1x1--+--
                   |________|
         |_____________________________|
    Args:
        n_feats (int): The number of feature maps.
        ratio (int): Expand ratio.
    """

    def __init__(self, n_feats, ratio=2):
        super(ResBlock, self).__init__()
        self.expand_conv = nn.Conv2d(n_feats, int(ratio*n_feats), 1, 1, 0)
        self.fea_conv = nn.Conv2d(int(ratio*n_feats), int(ratio*n_feats), 3, 1, 0)
        self.reduce_conv = nn.Conv2d(int(ratio*n_feats), n_feats, 1, 1, 0)

    def forward(self, x):
        out = self.expand_conv(x)
        out_identity = out
        
        # explicitly padding with bias for reparameterizing in the test phase
        b0 = self.expand_conv.bias
        out = pad_tensor(out, b0)

        out = self.fea_conv(out) + out_identity
        out = self.reduce_conv(out)
        out += x

        return  out

class RepResBlock(nn.Module):
    def __init__(self, n_feats):
        super(RepResBlock, self).__init__()
        self.rep_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

    def forward(self, x):
        out = self.rep_conv(x)
        return out + x

class RBlock(nn.Module):
    def __init__(self,in_c,is_train=True,act='gelu') -> None:
        super(RBlock,self).__init__()
        
        if is_train:
            self.conv1 = ResBlock(in_c)
        else:
            self.conv1 = RepResBlock(in_c)
        
        # activation
        if act=='gelu':
            self.act = nn.GELU()
        elif act=='relu':
            self.act = nn.ReLU(inplace=True)
        elif act=='softmax':
            self.act = nn.Softmax(dim=1)
        elif act=='softmax2d':
            self.act = nn.Softmax2d()
        elif act=='sigmoid':
            self.act = nn.Sigmoid()
        elif act=='tanh':
            self.act = nn.Tanh()
        elif act=='lrelu':
            self.act=nn.LeakyReLU(0.2)
        else:
            print(act+'is NOT implemented in RBlock')
            assert(0)
        
    def forward(self, x):
        return x+self.act(self.conv1(x))
    

class CALayer(nn.Module):
    """
    Channel Attention Layer
    From "Squeeze-and-Excitation Networks" 
    https://arxiv.org/abs/1709.01507    
    """
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)        
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB)
    """
    def __init__(self, n_feat, kernel_size, reduction,
                 bias=True, bn=False, act_cls=nn.ReLU, res_scale=1.0):
        super(RCAB, self).__init__()

        padding = kernel_size // 2
        layers = []

        for i in range(2):
            layers.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=padding, bias=bias))
            if bn:
                layers.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                layers.append(act_cls(inplace=True) if act_cls in [nn.ReLU, nn.LeakyReLU] else act_cls())

        layers.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*layers)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        return x + res * self.res_scale


class LWGatedConv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0):
        super(LWGatedConv2D, self).__init__()
        self.conv_feature = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv_mask = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.conv_feature(x)
        mask = self.conv_mask(x)
        return features * mask


class ResidualGroup(nn.Module):
    """
    Residual Group: A stack of RCAB blocks + one Conv2d,
    followed by a group-level residual connection.
    """
    def __init__(self, in_c, out_c, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        blocks = [RCAB(in_c, kernel_size, reduction) for _ in range(n_resblocks)]
        blocks.append(nn.Conv2d(in_c, out_c, kernel_size, padding=kernel_size // 2))
        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        return x + self.body(x)

class doubleResidualConv(nn.Module):
    def __init__(self,outc,kernel_size=3,padding=1):
        super(doubleResidualConv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(outc,outc,kernel_size=kernel_size,padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc,outc,kernel_size=kernel_size,padding=padding),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)+x


class FeatureExtraction(nn.Module):
    """
    Flexible feature extraction module.
    
    Args:
        inc (int): Input channels.
        outc (int): Output channels.
        is_train (bool): If True, use ResidualGroups.
        act (str): Activation function type ('relu', 'gelu', etc.).
        midc (List[int]): Mid-layer channel dimensions.
        num_blocks (int): Number of intermediate blocks.
        need_RG (bool): Use ResidualGroups or not.
        need_lwg (bool): Use LWGatedConv2D or standard convs.
    """
    def __init__(self, inc, outc, is_train, act='relu',
                 midc=[32, 32, 32], num_blocks=3,
                 need_RG=True, need_lwg=True,
                 kernel_size=3, padding=1):
        super().__init__()

        netlist = []

        # Activation selector
        activation_dict = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(inplace=True),
            'softmax': nn.Softmax(dim=1),
            'softmax2d': nn.Softmax2d(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'lrelu': nn.LeakyReLU(0.2)
        }
        if act not in activation_dict:
            raise ValueError(f"Activation '{act}' is NOT implemented in FeatureExtraction.")

        if need_RG:
            netlist.append(nn.Conv2d(inc, midc[0], kernel_size=1, padding=0))
            for i in range(num_blocks):
                netlist.append(ResidualGroup(midc[i], midc[i], kernel_size, 16, n_resblocks=2))
                if i < num_blocks - 1:
                    if need_lwg:
                        netlist.append(LWGatedConv2D(midc[i], midc[i + 1], kernel_size, stride=1, padding=padding))
                    else:
                        netlist.append(nn.Conv2d(midc[i], midc[i + 1], kernel_size=1, padding=0))
            netlist.append(nn.Conv2d(midc[-1], outc, kernel_size=1, padding=0))

        else:
            c_in = inc
            for i in range(num_blocks):
                c_out = outc if i == num_blocks - 1 else midc[i]
                if need_lwg:
                    netlist.append(LWGatedConv2D(c_in, c_out, kernel_size, stride=1, padding=padding))
                else:
                    netlist.append(nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding))
                netlist.append(activation_dict[act])
                c_in = c_out

        self.net = nn.Sequential(*netlist)

    def forward(self, x):
        return self.net(x)



