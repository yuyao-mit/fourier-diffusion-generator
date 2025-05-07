# tools.py

import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
import numpy as np
import cv2 as cv
import os
from einops import rearrange,repeat

def pad_tensor(t, pattern):
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0)
    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern
    return t

def inProcess(img):
    return (img / 255.0 - 0.5) * 2

def outProcess(img, out_range='float'):
    """
    Denormalize image from [-1, 1] to [0, 1] or [0, 255]
    """
    img = (img + 1) / 2  # map to [0, 1]
    img = img.clamp(0, 1)

    if out_range == 'float':
        return img
    elif out_range == 'uint8':
        return (img * 255).round().byte()
    else:
        raise ValueError("out_range must be 'float' or 'uint8'")

########################################### https://arxiv.org/abs/2012.09161 ############################################
##############  Modules from "Learning Continuous Image Representation with Local Implicit Image Function" ##############

def rescale(x, r):
    """Return the fractional part of x/r, used to compute sub-pixel offsets."""
    return x / r - (torch.div(x, r, rounding_mode='trunc')).clip(0)

def make_coord(shape, ranges=None, flatten=True, device=None):
    """Make coordinates at grid centers"""
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = (-1, 1) if ranges is None else ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)

    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])

    # --- minimal patch: move to desired device ---
    if device is not None:
        ret = ret.to(device)
    return ret


def to_pixel_samples(img):
    """
    Convert batch image tensor to coordinate-pixel pairs.
    
    Args:
        img: Tensor of shape (B, C, H, W)
    
    Returns:
        coord: (B, H*W, 2) tensor of normalized coordinates in [-1, 1]
        value: (B, H*W, C) tensor of pixel values (e.g., gray or RGB)
    """
    assert img.dim() == 4, "Expected shape (B, C, H, W)"
    B, C, H, W = img.shape
    coord = make_coord((H, W), device=img.device)
    coord = repeat(coord, 'n d -> b n d', b=B)
    value = rearrange(img, "b c h w -> b (h w) c")

    return coord, value



def debug_save(x, name, iter, norm=True, save_dir='tmp_img', iter_range=(190, 200)):
    """
    Save a debug image from tensor x to disk if within iter_range.
    
    Args:
        x: torch.Tensor, shape (B, C, H, W)
        name: str, filename identifier
        iter: int, current training step
        norm: bool, if True assume input in [0,1] and scale to [0,255]
        save_dir: str, directory to save image
        iter_range: tuple, (start, end) range of iterations to save images
    """
    if iter < iter_range[0] or iter > iter_range[1]:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    y = x[0].detach().cpu().numpy().transpose(1, 2, 0)
    if norm:
        y = y.clip(0, 1) * 255
    path = os.path.join(save_dir, f"{name}.png")
    cv.imwrite(path, y.astype(np.uint8))

def jitterUpsample(feature,th,tw,jitter,r,mode='nearest'):
    """samples from LR feature with HR coord and jitter"""
    n=feature.shape[0]
    ih,iw=feature.shape[-2:]
    hc=torch.arange(th,device=feature.device,dtype=torch.float32).repeat_interleave(tw).unsqueeze(0).repeat(n,1)
    wc=torch.arange(tw,device=feature.device,dtype=torch.float32).repeat(th).unsqueeze(0).repeat(n,1)
    jitter_w=jitter[:,0:1].repeat(1,th*tw)
    jitter_h=jitter[:,1:2].repeat(1,th*tw)
    coord=torch.div(torch.cat([wc-jitter_w,hc-jitter_h],dim=-1),r,rounding_mode='trunc').view(n,2,th,tw)
    coord[:,1]=(coord[:,1].clip(0,ih)/(ih-1))*2-1
    coord[:,0]=(coord[:,0].clip(0,iw)/(iw-1))*2-1
    ret=F.grid_sample(feature,coord.permute(0,2,3,1),mode=mode,align_corners=False)
    return ret
