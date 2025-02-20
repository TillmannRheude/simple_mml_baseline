import torch 
import random 
import signal 
import sys

import numpy as np 
import torch.nn as nn 
import pytorch_lightning as pl

from IPython import get_ipython

def is_running_in_notebook():
    try:
        if 'IPKernelApp' in get_ipython().config:
            return True
    except:
        pass
    return False

if is_running_in_notebook():
    from codefiles import architecture
    from codefiles import encoders
    from codefiles import transformer
    from codefiles.lightningmodules import mimic
    from codefiles.lightningmodules import synthetic
    from codefiles.lightningmodules import mosi_mosei
    import importlib
    importlib.reload(architecture)
    importlib.reload(encoders)
    importlib.reload(transformer)
    importlib.reload(mimic)
    importlib.reload(synthetic)
    importlib.reload(mosi_mosei)
from codefiles.architecture import Multimodal_Architecture
from codefiles.encoders import (
    Encoder_Tabular, 
    Encoder_Image, 
    Encoder_Sequence, 
    Encoders, 
    Encoder_MNIST,
    Encoder_MOSI_Language,
    Encoder_MOSI_Vision,
    Encoder_MOSI_Audio
)
from codefiles.transformer import Multimodal_Transformer
from codefiles.lightningmodules.mimic import MIMIC_Lightning_Module
from codefiles.lightningmodules.synthetic import Synthetic_Lightning_Module
from codefiles.lightningmodules.mosi_mosei import MOSI_Lightning_Module
from codefiles.lightningdatamodules.mimic_symile import MIMIC_Symile_Datamodule
from codefiles.lightningdatamodules.mimic_haim import MIMIC_Haim_Datamodule
from codefiles.lightningdatamodules.synthetic import Halved_Fashion_or_Vanilla_MNIST_Datamodule
from codefiles.lightningdatamodules.mosi_mosei import MOSI_MOSEI_Datamodule

def build_model(
        cfg: dict = {},
) -> Multimodal_Architecture:
    
    if cfg.dataset == "mimic_symile":
        output_dim = 10

        encoders = Encoders(
            encoders = nn.ModuleList(
                [
                    Encoder_Image(
                        latent_dim=cfg.modelname.head_transformer.d_model
                    ),
                    Encoder_Tabular(
                        input_dim=50,
                        latent_dim=cfg.modelname.head_transformer.d_model
                    ),
                    Encoder_Sequence(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        vit_nhead=2,
                        vit_dropout=0.0,
                        vit_dim_feedforward=256,
                        vit_num_layers=4,
                        use_pre_convs=True,
                        use_pre_linear=False
                    )
                ]
            )
        )
    elif cfg.dataset == "mimic_haim":
        output_dim = 10

        encoders = Encoders(
            encoders = nn.ModuleList(
                [
                    Encoder_Image(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        grayscale=True
                    ),
                    Encoder_Sequence(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        vit_nhead=2,
                        vit_dropout=0.0,
                        vit_dim_feedforward=256,
                        vit_num_layers=4,
                        use_pre_convs=False,
                        use_pre_linear=True
                    )
                ]
            )
        )
    elif cfg.dataset == "mnist" or cfg.dataset == "fmnist":
        output_dim = 10

        encoders = Encoders(
            encoders = nn.ModuleList(
                [
                    Encoder_MNIST(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                    ),
                    Encoder_MNIST(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                    ),
                ]
            )
        )
    elif cfg.dataset == "mosi" or cfg.dataset == "mosei":
        output_dim = 7

        encoders = Encoders(
            encoders = nn.ModuleList(
                [
                    Encoder_MOSI_Language(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        dataset=cfg.dataset
                    ),
                    Encoder_MOSI_Vision(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        dataset=cfg.dataset
                    ),
                    Encoder_MOSI_Audio(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        dataset=cfg.dataset
                    )
                ]
            )
        )
    else: 
        raise NotImplementedError("Dataset not implemented")
    
    multimodal_transformer = Multimodal_Transformer(
        d_model=cfg.modelname.head_transformer.d_model,
        nhead=cfg.modelname.head_transformer.nhead,
        dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
        dropout=cfg.modelname.head_transformer.dropout,
        num_layers=cfg.modelname.head_transformer.num_layers,
        dim_output=output_dim
    )
    model = Multimodal_Architecture(
        encoders=encoders,
        transformer=multimodal_transformer,
    )

    return model 

def build_lightningmodule(
        cfg: dict = {},
        model: Multimodal_Architecture = Multimodal_Architecture(),
) -> pl.LightningModule:
    
    if cfg.dataset == "mimic_symile" or cfg.dataset == "mimic_haim":
        lightningmodule = MIMIC_Lightning_Module(
            model=model,
            dataset=cfg.dataset,
            params_optimizer={key: value for key, value in cfg.modelname.optimizer.items()},
        )
    elif cfg.dataset == "mnist" or cfg.dataset == "fmnist":
        lightningmodule = Synthetic_Lightning_Module(
            model=model,
            dataset=cfg.dataset,
            params_optimizer={key: value for key, value in cfg.modelname.optimizer.items()},
        )
    elif cfg.dataset == "mosi" or cfg.dataset == "mosei":
        lightningmodule = MOSI_Lightning_Module(
            model=model,
            dataset=cfg.dataset,
            params_optimizer={key: value for key, value in cfg.modelname.optimizer.items()},
        )
    else: 
        raise NotImplementedError("Dataset not implemented")
    
    return lightningmodule

def build_datamodule(
        cfg: dict = {},
) -> pl.LightningDataModule:
    
    datamodule_params = {
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "missing": {key: value for key, value in cfg.missing.items()},
        "variant": cfg.modelname.modelname
    }
    
    if cfg.dataset == "mimic_symile":
        datamodule = MIMIC_Symile_Datamodule(**datamodule_params)
    elif cfg.dataset == "mimic_haim":
        datamodule = MIMIC_Haim_Datamodule(**datamodule_params)
    elif cfg.dataset == "mosi" or cfg.dataset == "mosei":
        datamodule_params["dataset"] = cfg.dataset
        datamodule = MOSI_MOSEI_Datamodule(**datamodule_params)
    elif cfg.dataset == "mnist" or cfg.dataset == "fmnist":
        datamodule_params["dataset"] = cfg.dataset
        datamodule = Halved_Fashion_or_Vanilla_MNIST_Datamodule(**datamodule_params)
    else:
        raise NotImplementedError("Dataset not implemented")
    
    return datamodule

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def signal_handler(sig, frame):
    # for jupyter notebooks with wandb
    signal.signal(sig, signal.SIG_IGN)
    sys.exit(0)