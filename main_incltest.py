import signal
import wandb
import torch
import os 
import hydra 
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from codefiles.helpers import set_all_seeds, build_model, build_lightningmodule, build_datamodule

os.environ["WANDB_SILENT"] = "true"
torch.set_float32_matmul_precision("high")

@hydra.main(config_path="config", config_name="config")
def main(cfg) -> None:
    wandb.finish()
    set_all_seeds(seed=cfg.seed)
    wandb.init(
        project=cfg.wandb.project,
        group=None if cfg.wandb.group == "None" else cfg.wandb.group,
        config={key: value for key, value in cfg.items()},
    )

    model = build_model(cfg)
    lightningmodule = build_lightningmodule(cfg, model)
    datamodule = build_datamodule(cfg)

    checkpointaddon = ""
    if "corrupted_data_protocol" in cfg.modelname:
        if cfg.modelname.corrupted_data_protocol:
            checkpointaddon = "_corrupted"
        else:
            checkpointaddon = "_clean"
    if "unimodal" in cfg.unimodal:
        checkpointaddon = f"{cfg.unimodal}"

    checkpoint_name = f"{cfg.modelname.modelname}_{cfg.dataset}_{cfg.split_nr}{checkpointaddon}"
    if os.path.exists(f'/path/to/checkpoints/{checkpoint_name}.ckpt'):
        os.remove(f'/path/to/checkpoints/{checkpoint_name}.ckpt')

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.encoders.monitor.metric, mode=cfg.encoders.monitor.mode,
        dirpath='/path/to/checkpoints/',
        filename=checkpoint_name,
        save_top_k=1,
    )

    if "early_stopping" in cfg.modelname:
        early_stopping = EarlyStopping(monitor=cfg.encoders.monitor.metric, mode=cfg.encoders.monitor.mode, patience=cfg.max_epochs)
        print("Early stopping is disabled")
    else:
        early_stopping = EarlyStopping(monitor=cfg.encoders.monitor.metric, mode=cfg.encoders.monitor.mode, patience=20)

    trainer = pl.Trainer(
        logger=WandbLogger(project=cfg.wandb.project, dir="wandb/"),
        log_every_n_steps=1,
        accelerator='gpu',
        devices=1,
        max_epochs=cfg.max_epochs,  # 20
        precision=cfg.precision,
        enable_checkpointing=True,
        callbacks=[
            early_stopping,
            checkpoint_callback,
        ],
    )
    trainer.fit(lightningmodule, datamodule)

    model_test = build_model(cfg)
    lightningmodule_test = build_lightningmodule(cfg, model_test)
    trainer.test(lightningmodule_test, datamodule=datamodule, ckpt_path="best")  # ckpt_path=f'/path/to/checkpoints/{checkpoint_name}.ckpt'
    wandb.finish()


if __name__ == "__main__":
    main()