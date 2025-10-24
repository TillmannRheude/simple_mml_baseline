import signal
import wandb
import torch
import os 
import hydra 
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from hydra import compose, initialize
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

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

    checkpoint_name = f'{cfg["modelname"]["modelname"]}_{cfg["dataset"]}_{str(cfg["split_nr"])}'
    # if checkpoint already exists, delete it
    if os.path.exists(f'/sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/checkpoints/{checkpoint_name}.ckpt'):
        os.remove(f'/sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/checkpoints/{checkpoint_name}.ckpt')

    print(f"Checkpoint name: {checkpoint_name}")
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.encoders.monitor.metric, mode=cfg.encoders.monitor.mode,
        dirpath='/sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/checkpoints/',
        filename=checkpoint_name,
        save_top_k=1,
    )

    # check if cfg.modelname.early_stopping exists in cfg.modelname 
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
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        # enable_checkpointing=True,
        callbacks=[
            early_stopping,
            checkpoint_callback,
        ],
    )
    trainer.fit(lightningmodule, datamodule)
    trainer.test(ckpt_path=f'/sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/checkpoints/{checkpoint_name}.ckpt', datamodule=datamodule)
    wandb.finish()


if __name__ == "__main__":
    main()