# loader.py
import os, random
from typing import List, Optional, Tuple

import cv2 as cv
import lightning as L
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# --------------------------- DATASET --------------------------- #
class _GetFourier(Dataset):
    """
    从给定目录读取灰度图 → 随机裁剪 patch → NumPy FFT（幅度+可选相位）
    返回:
        freq  : [n_patches, 1, res, res]  (torch.float64)
        label : [1, 1, 100, 1600]         (torch.float64)
        phase : 同 freq，如果 return_phase=True，否则 None
    """
    SUPPORTED_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

    def __init__(
        self,
        root_dir: str,
        res: int = 256,
        n_patches: int = 4,
        return_phase: bool = False,
    ) -> None:
        super().__init__()
        self.root_dir, self.res, self.n, self.return_phase = root_dir, res, n_patches, return_phase

        self.img_paths: List[str] = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(self.SUPPORTED_EXT)
        ]
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {root_dir}")

        # 下方 100 行用于 label，假设原图尺寸 1600×1600
        self.y_max, self.x_max = 1600 - res - 100, 1600 - res

    # ---------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.img_paths)

    # ---------------------------------------------------------- #
    def _read_gray(self, path: str) -> np.ndarray:
        """OpenCV 读取 → 灰度 → float64 [0,1]"""
        im = cv.imread(path, cv.IMREAD_UNCHANGED)
        if im is None:
            raise RuntimeError(f"Cannot open image: {path}")

        # 若为彩色，转灰度
        if im.ndim == 3:
            im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        return im.astype(np.float64) / 255.0  # [H, W] float64

    # ---------------------------------------------------------- #
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        img_np = self._read_gray(self.img_paths[idx])                # [H, W] float64

        # label: 取图像底部 100×1600
        label_np = img_np[-100:, :][None, None, ...]                 # [1,1,100,1600]

        mags, phases = [], []
        for _ in range(self.n):
            y0 = random.randint(0, self.y_max)
            x0 = random.randint(0, self.x_max)
            patch = img_np[y0 : y0 + self.res, x0 : x0 + self.res]   # [res, res]

            # NumPy FFT
            fft = np.fft.fft2(patch)
            fft_shifted = np.fft.fftshift(fft)
            mag = np.log1p(np.abs(fft_shifted))                      # [res, res] float64
            mags.append(mag[None, ...])                              # [1, res, res]

            if self.return_phase:
                pha = np.angle(fft_shifted)                          # [-π, π]
                phases.append(pha[None, ...])                        # [1, res, res]

        # 组装 batch 并转 Torch Tensor (float64 保精度)
        freq = torch.from_numpy(np.stack(mags, axis=0))              # [n,1,res,res]
        label = torch.from_numpy(label_np)
        phase = (
            torch.from_numpy(np.stack(phases, axis=0))
            if self.return_phase
            else None
        )

        return freq, label, phase


# ----------------------- LIGHTNING DATAMODULE ------------------ #
class FourierNanoDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        res: int = 256,
        n_patches: int = 16,
        batch_size: int = 1,
        num_workers: int = 8,
        val_split: float = 0.1,
        seed: int = 42,
        return_phase: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_ds = self.val_ds = self.test_ds = None

    # ---------------------------------------------------------- #
    def setup(self, stage: Optional[str] = None):
        full_ds = _GetFourier(
            root_dir=self.hparams.root_dir,
            res=self.hparams.res,
            n_patches=self.hparams.n_patches,
            return_phase=self.hparams.return_phase,
        )

        n_total = len(full_ds)
        n_val = int(self.hparams.val_split * n_total)
        n_test = n_val
        n_train = n_total - n_val - n_test

        g = torch.Generator().manual_seed(self.hparams.seed)
        self.train_ds, self.val_ds, self.test_ds = random_split(
            full_ds, [n_train, n_val, n_test], generator=g
        )

    # ---------------------------------------------------------- #
    def _loader(self, ds, shuffle: bool):
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, True)

    def val_dataloader(self):
        return self._loader(self.val_ds, False)

    def test_dataloader(self):
        return self._loader(self.test_ds, False)
