# loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ssim import ssim
from torchvision.models import vgg16
from torchvision import models

class Charbonnier_loss(nn.Module):
    """
    Charbonnier Loss(Smooth L1 loss variant).
    """
    def __init__(self, epsilon=1e-6):
        super(Charbonnier_loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, label):
        diff = pred - label
        loss = torch.mean(torch.sqrt(diff * diff + self.epsilon))
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self, vgg_path, use_gray_to_rgb=True, normalize=True):
        super().__init__()
        # self.device = device
        # self.model = vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval().to(self.device)
        self.model = vgg16()
        self.model.load_state_dict(torch.load(vgg_path))
        self.model = self.model.features.eval() #.to(self.device)

        self.use_gray_to_rgb = use_gray_to_rgb
        self.normalize = normalize

        for param in self.model.parameters():
            param.requires_grad = False

        self.layer_mapping = {
            '3': "relu1",
            '8': "relu2",
            '15': "relu3",
            '22': "relu4",
            '29': "relu5",
        }
        self.target_layers = list(self.layer_mapping.values())

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)#.to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)#.to(self.device)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, pred, label):
        pred = pred#.to(self.device)
        label = label#.to(self.device)

        if pred.shape[1] == 1 and self.use_gray_to_rgb:
            pred = pred.repeat(1, 3, 1, 1)
            label = label.repeat(1, 3, 1, 1)

        pred = pred * 2 - 1
        label = label * 2 - 1

        if self.normalize:
            pred = (pred - self.mean) / self.std
            label = (label - self.mean) / self.std

        feats_pred, feats_label = self.extract_features(pred, label)

        loss = 0.
        for key in feats_pred:
            loss += F.mse_loss(feats_pred[key], feats_label[key])
        return loss

    def extract_features(self, pred, label):
        feats_pred = {}
        feats_label = {}
        x_pred, x_label = pred, label

        for name, module in self.model._modules.items():
            x_pred = module(x_pred)
            x_label = module(x_label)

            if name in self.layer_mapping:
                key = self.layer_mapping[name]
                feats_pred[key] = x_pred
                feats_label[key] = x_label

        return feats_pred, feats_label

class SpatialMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, pred, label, mask):
        """
        pred: (N, C, H, W) - predicted image
        label: (N, C, H, W) - ground truth image
        mask: (N, 1, H, W) or (N, C, H, W) - spatial weight mask
        """
        """
        pred = pred.to(self.device)
        label = label.to(self.device)
        mask = mask.to(self.device)
        """

        # Ensure mask shape matches pred/label
        mask = mask.float()
        if mask.shape[1] == 1 and pred.shape[1] > 1:
            mask = mask.expand_as(pred)

        diff = torch.abs(pred - label) * mask
        loss = diff.sum() / (mask.sum() + 1e-6)  # Avoid divide-by-zero

        return loss

# loss.py
class CombinatorialLoss(nn.Module):
    def __init__(self, loss_weight, vgg_path, use_gray_to_rgb=True, window_size=11):
        super().__init__()
        
        self.loss_weight = loss_weight  # list or tuple of 4 weights

        self.use_gray_to_rgb = use_gray_to_rgb
        self.l1_loss_fn = nn.L1Loss()
        self.perceptual_loss_fn = PerceptualLoss(vgg_path=vgg_path,use_gray_to_rgb=use_gray_to_rgb)
        self.spatial_mask_loss_fn = SpatialMaskLoss()
        self.window_size = window_size

    def forward(self, pred, label, mask):
        pred = pred#.to(self.device)
        label = label#.to(self.device)
        mask = mask#.to(self.device)

        # 1. L1 loss
        l1 = self.l1_loss_fn(pred, label)

        # 2. SSIM loss
        ssim_loss = 1 - ssim(pred, label, window_size=self.window_size)

        # 3. Perceptual loss
        perceptual = self.perceptual_loss_fn(pred, label)

        # 4. Spatial Mask loss
        spatial_mask = self.spatial_mask_loss_fn(pred, label, mask)

        # Combine
        total = (
            self.loss_weight[0] * l1 +
            self.loss_weight[1] * ssim_loss +
            self.loss_weight[2] * perceptual +
            self.loss_weight[3] * spatial_mask
        )

        return total