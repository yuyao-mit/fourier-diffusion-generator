# models.py
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config 
from tools import *
from blocks import * 
import lightning as L
from loss import CombinatorialLoss


class DFSR(nn.Module):
    """Deep Fourier Based Super Resolution"""
    def __init__(self, is_train, lr_hidc=32, hr_hidc=32, mlpc=64, jitter_std=0.001):
        super(DFSR, self).__init__()
        self.is_train = is_train
        self.jitter_std = jitter_std
        self.curExtraction = FeatureExtraction(3, lr_hidc * 2, is_train, midc=[32, 48, 48], num_blocks=4, need_RG=False)

        r = 4 if config.Pixelshuffle else 1
        # self.r = torch.tensor(r, dtype=torch.float32)

        self.gbufferConv = nn.Sequential(
            nn.Conv2d((10 - 3 + 1) * r, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            doubleResidualConv(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            doubleResidualConv(64),
            nn.Conv2d(64, hr_hidc * 2, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            doubleResidualConv(hr_hidc * 2)
        )

        self.imgConv = FeatureExtraction(
            lr_hidc + hr_hidc, 3 * r, is_train,
            midc=[64, 56, 48], num_blocks=3, need_RG=True
        )

        self.lastConv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.HFConv = nn.Conv2d(3 * r, 3 * r, kernel_size=3, padding=1)

        self.coef = nn.Conv2d(lr_hidc, mlpc, 3, padding=1)
        self.freq = nn.Conv2d(hr_hidc, mlpc, 3, padding=1)
        self.phase = nn.Conv2d(1, mlpc // 2, kernel_size=1, bias=False)

        self.mlp = nn.Sequential(
            nn.Conv2d(mlpc, mlpc, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlpc, mlpc, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlpc, mlpc, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlpc, 3 * r, kernel_size=1),
        )

    def forward(self, cur_lr, cur_hr, r):
        n, _, ih, iw = cur_lr.shape
        th, tw = cur_hr.shape[-2:]
        if config.Pixelshuffle:
            th //= 2
            tw //= 2
        lr_features = self.curExtraction(cur_lr[:, 0:3])
        amp_feat, recon_lr_feat = torch.split(lr_features, lr_features.shape[1] // 2, dim=1)
        up_lr_features = F.interpolate(recon_lr_feat, (th, tw), mode='bilinear', align_corners=False)
        
        # Generate coords and rel_coord
        if self.is_train:
            jitter = torch.randn(n, 2) * self.jitter_std
        else:
            jitter = torch.zeros(n, 2)
        jitter_w=(jitter[:,0:1]/tw).repeat(1,th*tw).unsqueeze(-1).to(cur_lr.device)
        jitter_h=(jitter[:,1:2]/th).repeat(1,th*tw).unsqueeze(-1).to(cur_lr.device)

        coord = make_coord((th, tw),device=cur_lr.device).unsqueeze(0).repeat(n, 1, 1)
        feat_coord = make_coord((ih, iw),device=cur_lr.device, flatten=False).permute(2, 0, 1).unsqueeze(0).expand(n, 2, ih, iw)
        q_coord = F.grid_sample(feat_coord, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)        
        rel_coord = coord - q_coord + torch.cat([jitter_h,jitter_w],dim=-1)
        rel_coord[:, :, 0] *= ih
        rel_coord[:, :, 1] *= iw
        my_rel_coord = rel_coord.permute(0, 2, 1).view(n, 2, th, tw)

        # G-buffer conv
        hr_inp = torch.cat([cur_hr[:, 0:7], cur_hr[:, -1:]], dim=1)
        if config.Pixelshuffle:
            hr_inp = F.pixel_unshuffle(hr_inp, 2)
        gbuffer_features = self.gbufferConv(hr_inp)
        freq_feat, recon_hr_feat = torch.split(gbuffer_features, gbuffer_features.shape[1] // 2, dim=1)

        # Fourier frequency modulation
        coef = self.coef(amp_feat)
        freq = self.freq(freq_feat)

        q_coef = F.grid_sample(coef, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].view(n, -1, th, tw)
        q_freq = torch.stack(torch.split(freq, 2, dim=1), dim=1)
        q_freq = (q_freq * my_rel_coord.unsqueeze(1)).sum(2)
       
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=torch.float32,device=cur_lr.device)
        if r.ndim == 0:                 # 标量 ➜ (1,1,1,1) 再 repeat 到 batch
            mp_r = r.view(1, 1, 1, 1).repeat(n, 1, th, tw)
        else:                           # [B] ➜ (B,1,1,1) 再 repeat
            mp_r = r.view(-1, 1, 1, 1).repeat(1, 1, th, tw)
        # mp_r = r.view(1, 1, 1, 1).repeat(n, 1, th, tw).to(torch.float32).to(cur_lr.device)
        inv_r = torch.ones(n, 1, th, tw, dtype=torch.float32,device=cur_lr.device) / mp_r
        q_freq += self.phase(inv_r)
        q_freq = torch.cat([torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)], dim=1)

        inp_origin = q_coef * q_freq
        ret = self.HFConv(self.mlp(inp_origin))

        # Fusion
        fused = self.imgConv(torch.cat([up_lr_features, recon_hr_feat], dim=1))

        if config.Pixelshuffle:
            return self.lastConv(F.pixel_shuffle(ret, 2) + F.pixel_shuffle(fused, 2))
        else:
            return self.lastConv(ret + fused)

class DFSRNet(L.LightningModule):
    def __init__(self,vgg_path=config.vgg_path_1,
                 lr_hidc=32, hr_hidc=32, mlpc=64, jitter_std=0.001, lr=2e-4, weight_decay=1e-6):
        super().__init__(); self.save_hyperparameters()
        self.net = DFSR(is_train=True, lr_hidc=lr_hidc, hr_hidc=hr_hidc, mlpc=mlpc, jitter_std=jitter_std)
        self.loss_fn = CombinatorialLoss(loss_weight=config.loss_weight, vgg_path=vgg_path)
        
    def forward(self, lr_img, hr_feat, r):
        self.net.is_train = self.training
        return self.net(lr_img, hr_feat, r)
        
    def _step(self, batch, stage):
        lr_img, hr_feat, mask, r = batch
        pred  = self(lr_img, hr_feat, r)
        loss  = self.loss_fn(pred=pred, label=hr_feat[:, :1], mask=mask)
        self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"))
        return loss

    def training_step  (self, b, _): return self._step(b, "train")
    def validation_step(self, b, _):       self._step(b, "val")
    def test_step      (self, b, _):       self._step(b, "test")

    def configure_optimizers(self):
        # 将 bias / norm 层剔除 weight‑decay
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            (no_decay if n.endswith(("bias", "weight_g", "weight_v")) or "norm" in n.lower()
             else decay).append(p)

        param_groups = [
            {"params": decay,    "weight_decay": self.hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        opt = torch.optim.Adam(param_groups, lr=1e-4, betas=(0.9, 0.999))

        # 每 500 个 epoch 将 lr * 0.5
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5)

        return {"optimizer": opt, "lr_scheduler": sched}
