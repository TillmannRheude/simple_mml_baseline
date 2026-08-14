"""Train and evaluate SimBaMM on the configured UK Biobank cohort."""

from __future__ import annotations

import os

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from codefiles.helpers import (
    build_datamodule,
    build_lightning_module,
    build_model,
    set_all_seeds,
)

os.environ.setdefault("WANDB_SILENT", "true")
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    set_all_seeds(cfg.seed)

    logger = WandbLogger(
        project=cfg.wandb.project,
        group=None if cfg.wandb.group is None else cfg.wandb.group,
        mode=cfg.wandb.mode,
        save_dir=cfg.wandb.save_dir,
    )
    logger.experiment.config.update(
        OmegaConf.to_container(cfg, resolve=True), allow_val_change=True
    )

    datamodule = build_datamodule(cfg)
    lightning_module = build_lightning_module(cfg, build_model(cfg))
    checkpoint = ModelCheckpoint(
        monitor="val/auroc",
        mode="max",
        save_top_k=1,
        filename="simbamm-ukb-{epoch:02d}",
    )
    trainer = pl.Trainer(
        logger=logger,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=[
            EarlyStopping(
                monitor="val/auroc",
                mode="max",
                patience=cfg.trainer.early_stopping_patience,
            ),
            checkpoint,
        ],
    )
    trainer.fit(lightning_module, datamodule=datamodule)
    if cfg.trainer.test_after_fit:
        trainer.test(lightning_module, datamodule=datamodule, ckpt_path="best")
    wandb.finish()


if __name__ == "__main__":
    main()
