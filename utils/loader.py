# loader.py

"""
A PyTorch‑Lightning DataModule for nanoscale grayscale images (1700 × 1600).
Outputs for every mini‑batch:
  • LR tensor  [B, 3, H//r, W//r]
  • HR tensor  [B, 8, H, W]          (channels: 3×gray, albedo, gaussian, sobel‑depth, normal‑variation, fourier)
  • Fourier mask tensor [B, 1, H, W]
  • Downsample scale r
"""

import os
import random
from typing import List, Optional
import cv2 as cv
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split,Sampler

###################### HELPERS #############################
def _pil_to_np_gray(pil: Image.Image) -> np.ndarray:
    """PIL → float32 numpy array in [0,1], shape (H, W)."""
    arr = np.asarray(pil.convert("L"), dtype=np.float32)
    return cv.normalize(arr, None, 0.0, 1.0, cv.NORM_MINMAX)

def _albedo(gray: np.ndarray, thresh: float = 0.05) -> np.ndarray:
    smooth = cv.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    return (np.abs(gray - smooth) > thresh).astype(np.float32)

def _gaussian(gray: np.ndarray, thresh: float = 0.05) -> np.ndarray:
    smooth = cv.GaussianBlur(gray, (9, 9), 1.5)
    return (np.abs(gray - smooth) > thresh).astype(np.float32)

def _sobel_magnitude(gray: np.ndarray, thresh: float = 0.05) -> np.ndarray:
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return (mag > thresh).astype(np.float32)

def _normal_variation(gray: np.ndarray, thresh: float = 0.3) -> np.ndarray:
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    direction = np.arctan2(gy, gx)
    dx = cv.Sobel(direction, cv.CV_32F, 1, 0, ksize=3)
    dy = cv.Sobel(direction, cv.CV_32F, 0, 1, ksize=3)
    var = np.sqrt(dx**2 + dy**2)
    return (var > thresh).astype(np.float32)

def _fourier(gray: np.ndarray, perc: int = 95) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    thr = np.percentile(mag, perc)
    high_freq_mask = (mag >= thr).astype(np.float32)
    f_filtered = f * high_freq_mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered)))
    img_back = cv.normalize(img_back, None, 0, 1, cv.NORM_MINMAX)
    return (img_back > 0.1).astype(np.float32)




###################### DATASET #############################
class _NanoDataset(Dataset):
    """
    单张灰度 TIFF/PNG/JPG/EXR → LR / HR‑8c / Fourier mask
    本版本 **固定单一 r**，因此同一 DataLoader 内所有样本尺寸一致，
    可直接用默认 collate_fn；在重建 dataloader 时再切换 r。
    """
    def __init__(self,
                 image_paths: List[str],
                 r: int,
                 high_res: int):
        super().__init__()
        self.image_paths = image_paths
        self.r = int(r)
        self.hr_size = high_res
        self.crop_origin_max = 1600 - high_res

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        pil = Image.open(self.image_paths[idx]).convert("L").crop((0, 0, 1600, 1600))
        x0, y0 = random.randint(0, self.crop_origin_max), random.randint(0, self.crop_origin_max)
        gray_np = _pil_to_np_gray(pil.crop((x0, y0, x0 + self.hr_size, y0 + self.hr_size)))

        gray3 = np.stack([gray_np] * 3)
        hr = np.concatenate([gray3,
                             _albedo(gray_np)[None],
                             _gaussian(gray_np)[None],
                             _sobel_magnitude(gray_np)[None],
                             _normal_variation(gray_np)[None],
                             _fourier(gray_np)[None]], 0)

        lr_sz = self.hr_size // self.r
        lr_np = cv.resize(gray_np, (lr_sz, lr_sz), interpolation=cv.INTER_AREA)
        lr_3  = np.stack([lr_np] * 3)

        lr_t   = torch.from_numpy(lr_3).float()
        hr_t   = torch.from_numpy(hr ).float()
        mask_t = torch.from_numpy(_fourier(gray_np)[None]).float()
        return lr_t, hr_t, mask_t, self.r #torch.tensor(self.r, dtype=torch.float32)

###################### LIGHTNING DATAMODULE #####################
class NanoDataLoader(L.LightningDataModule):
    """
    使用 `Trainer(..., reload_dataloaders_every_n_epochs=100)` 时，
    每隔 100 个 epoch 会重新调用 `setup()`，从而在 r_list 中
    轮换缩放倍率，保证整个 epoch 内 batch 尺寸一致。
    """
    SUPPORTED_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".exr")

    def __init__(self,
                 root_dir: str,
                 r_list: List[int] = (2, 4, 6, 8, 12),
                 high_res: int = 240,
                 batch_size: int = 8,
                 num_workers: int = 8,
                 seed: int = 42):
        super().__init__()
        self.save_hyperparameters()
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(root_dir)
        self.image_paths = [os.path.join(root_dir, f)
                            for f in os.listdir(root_dir)
                            if f.lower().endswith(self.SUPPORTED_EXT)]
        if not self.image_paths:
            raise FileNotFoundError("NO IMAGE FOUND")
        self._r_cycle_idx = 0                  # 记录当前使用 r 的索引

    # ---------- split ----------
    def setup(self, stage: Optional[str] = None):
        r_current = self.hparams.r_list[self._r_cycle_idx % len(self.hparams.r_list)]
        self._r_cycle_idx += 1                 # 下次 dataloader 重建时切换
        ds_full = _NanoDataset(self.image_paths, r_current, self.hparams.high_res)

        n_total = len(ds_full)
        n_tr, n_val = int(0.8 * n_total), int(0.1 * n_total)
        n_test = n_total - n_tr - n_val
        g = torch.Generator().manual_seed(self.hparams.seed)
        self.train_ds, self.val_ds, self.test_ds = random_split(
            ds_full, [n_tr, n_val, n_test], generator=g)

    # ---------- loaders ----------
    def _dataloader(self, ds, shuffle):
        return DataLoader(ds,
                          batch_size=self.hparams.batch_size,
                          shuffle=shuffle,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True)

    def train_dataloader(self): return self._dataloader(self.train_ds, True)
    def val_dataloader  (self): return self._dataloader(self.val_ds,   False)
    def test_dataloader (self): return self._dataloader(self.test_ds,  False)
