# train_dfsr.py
import argparse
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers   import CSVLogger
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent / ".."))
from utils.models import DFSRNet
from utils.loader import NanoDataLoader
from utils.config import root_dir_2, vgg_path_2

root_dir = root_dir_2
vgg_path = vgg_path_2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1, help="number of compute nodes")
    parser.add_argument("--gpus",  type=int, default=1, help="GPUs per node")
    parser.add_argument("--epochs",type=int, default=5000)
    args = parser.parse_args()

    seed_everything(42, workers=True) 

    # ---------- datamodule ----------
    dm = NanoDataLoader(
        root_dir=root_dir,
        r_list=[2, 4, 6, 8, 12],
        high_res=240,
        batch_size=16,
        num_workers=4
    )

    # ---------- model ----------
    model = DFSRNet(vgg_path=vgg_path,lr_hidc=32, hr_hidc=32, mlpc=64,
                    jitter_std=0.001, lr=1e-4, weight_decay=1e-6)

    # ---------- callbacks ----------
    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="dfsr_{epoch:07d}",
        save_top_k=-1,                   # 保存所有
        every_n_epochs=500      
    )

    # ---------- logger ----------
    logger = CSVLogger("logs", name="dfsr")

    # ---------- trainer ----------
    trainer = Trainer(
        max_epochs=args.epochs,
        reload_dataloaders_every_n_epochs=100,
        accelerator="gpu",
        devices=args.gpus,
        num_nodes=args.nodes,
        strategy="ddp",                  # 多机多卡
        callbacks=[ckpt_cb],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=1
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
