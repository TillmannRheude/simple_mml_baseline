import signal
import wandb
import torch
import os 
import hydra 
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from hydra import compose, initialize
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from codefiles.helpers import is_running_in_notebook  # for reloading modules instead of restarting kernel
if is_running_in_notebook():
    from codefiles import helpers
    import importlib
    importlib.reload(helpers)
from codefiles.helpers import set_all_seeds, signal_handler, build_model, build_lightningmodule, build_datamodule

os.environ["WANDB_SILENT"] = "true"
torch.set_float32_matmul_precision("high")

@hydra.main(config_path="config", config_name="config")
def main(cfg) -> None:
    wandb.finish()

    # cfg = {key: value for key, value in cfg.items()}

    # print missing rates
    print(f"Missing rates: {cfg.missing.missing_train}, {cfg.missing.missing_valid}, {cfg.missing.missing_test}")
    
    set_all_seeds(seed=cfg.seed)
    wandb.init(
        project=cfg.wandb.project,
        group=None if cfg.wandb.group == "None" else cfg.wandb.group,
        config={key: value for key, value in cfg.items()},
    )

    model = build_model(cfg)
    lightningmodule = build_lightningmodule(cfg, model)
    datamodule = build_datamodule(cfg)

    # check if cfg.modelname.early_stopping exists in cfg.modelname 
    if "early_stopping" in cfg.modelname:
        early_stopping = EarlyStopping(monitor=cfg.encoders.monitor.metric, mode=cfg.encoders.monitor.mode, patience=cfg.max_epochs)
        print("Early stopping is disabled")
    else:
        early_stopping = EarlyStopping(monitor=cfg.encoders.monitor.metric, mode=cfg.encoders.monitor.mode, patience=5)


    trainer = pl.Trainer(
        logger=WandbLogger(project=cfg.wandb.project, dir="wandb/"),
        log_every_n_steps=1,
        accelerator='gpu',
        devices=1,
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        enable_checkpointing=False,
        callbacks=[
            early_stopping,
        ],
    )
    trainer.fit(lightningmodule, datamodule)
    #trainer.test(ckpt_path="best", datamodule=datamodule)
    wandb.finish()


if __name__ == "__main__":
    main()