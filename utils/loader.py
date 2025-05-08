import os, random
from typing import List, Optional, Tuple

# import lightning as L
from lightning.pytorch import LightningDataModule

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torch.fft import fft2, fftshift, angle  

def _pil_to_float_tensor(pil: Image.Image) -> torch.Tensor:
    """PIL → float64 torch tensor in [0,1], shape [H,W]."""
    arr = np.asarray(pil, dtype=np.float64) / 255.0
    return torch.from_numpy(arr)


# --------------------------- DATASET --------------------------- #
class FourierNanoDataset(Dataset):
    SUPPORTED_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

    def __init__(self, 
                 root_dir: str, 
                 res: int = 256, 
                 n_patches: int = 4, 
                 return_phase: bool = False) -> None:
        super().__init__()
        self.root_dir, self.res, self.n, self.return_phase = root_dir, res, n_patches, return_phase
        self.img_paths: List[str] = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(self.SUPPORTED_EXT)
        ]
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {root_dir}")

        self.y_max, self.x_max = 1600 - res - 100, 1600 - res

    def __len__(self) -> int: 
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        img  = Image.open(self.img_paths[idx]).convert("L")
        im_t = _pil_to_float_tensor(img)  # [H, W]

        label = im_t[-100:, :].unsqueeze(0).unsqueeze(0)  # [1,1,100,1600]

        mags, phases = [], []
        for _ in range(self.n):
            y0, x0 = random.randint(0, self.y_max), random.randint(0, self.x_max)
            patch  = im_t[y0:y0+self.res, x0:x0+self.res]  # [res, res]

            f    = fftshift(fft2(patch))
            mag  = torch.log1p(torch.abs(f))
            mag /= mag.max() + 1e-8
            mags.append(mag.unsqueeze(0))  # [1, res, res]

            if self.return_phase:
                pha = angle(f)
                phases.append(pha.unsqueeze(0))  # [1, res, res]

        freq_patches = torch.stack(mags)  # [n, 1, res, res]

        if self.return_phase:
            phase_patches = torch.stack(phases)  # [n, 1, res, res]
            return freq_patches, label, phase_patches
        else:
            return freq_patches, label, None

class FourierNanoDataModule(LightningDataModule):
    def __init__(self,
                 root_dir: str,
                 res: int = 256,
                 n_patches: int = 4,
                 batch_size: int = 8,
                 num_workers: int = 8,
                 val_split: float = 0.1,
                 seed: int = 42,
                 return_phase: bool = False):  
        super().__init__()
        self.save_hyperparameters()
        self.train_ds = self.val_ds = self.test_ds = None

    # ---------------------------------------------------------- #
    def setup(self, stage: Optional[str] = None):
        full_ds = FourierNanoDataset(
            root_dir=self.hparams.root_dir,
            res=self.hparams.res,
            n_patches=self.hparams.n_patches,
            return_phase=self.hparams.return_phase  
        )

        n_total = len(full_ds)
        n_val   = int(self.hparams.val_split * n_total)
        n_test  = n_val
        n_train = n_total - n_val - n_test

        g = torch.Generator().manual_seed(self.hparams.seed)
        self.train_ds, self.val_ds, self.test_ds = random_split(
            full_ds, [n_train, n_val, n_test], generator=g)

    # ---------------------------------------------------------- #
    def _loader(self, ds, shuffle: bool):
        return DataLoader(ds,
                          batch_size=self.hparams.batch_size,
                          shuffle=shuffle,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True)

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader  (self): return self._loader(self.val_ds,   False)
    def test_dataloader (self): return self._loader(self.test_ds,  False)
