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
    from codefiles.lightningmodules import vgg_sound
    from codefiles.lightningmodules import ch_sims
    from codefiles.methods.imder import imder 
    from codefiles.methods.albef import albef
    from codefiles.methods.mult import multim_transf
    from codefiles.methods.ogm import gradient_modulation
    import importlib
    importlib.reload(architecture)
    importlib.reload(encoders)
    importlib.reload(transformer)
    importlib.reload(mimic)
    importlib.reload(synthetic)
    importlib.reload(mosi_mosei)
    importlib.reload(vgg_sound)
    importlib.reload(ch_sims)
    importlib.reload(imder)
    importlib.reload(albef)
    importlib.reload(multim_transf)
    importlib.reload(gradient_modulation)
from codefiles.architecture import Multimodal_Architecture, Multimodal_CEN_Architecture
from codefiles.encoders import (
    Encoders, 
    Encoder_MNIST,
    Encoder_MOSI_Language,
    Encoder_MOSI_Vision,
    Encoder_MOSI_Audio,
    Encoder_VGG_Video,
    Encoder_VGG_Audio,
    Encoder_CHS_Language,
    Encoder_CHS_Vision,
    Encoder_CHS_Audio,
    Encoder_VisionTouch_Vision,
    Encoder_VisionTouch_Proprio,
    Encoder_VisionTouch_Force,
    Encoder_CREMA_D_Video,
    Encoder_CREMA_D_Audio,
    Encoder_HAIM_Vision,
    Encoder_HAIM_Sequential,
    Encoder_Symile_Sequential,
    Encoder_Symile_Tabular,
    Encoder_Symile_Vision,
    Encoder_Kinetics_Video,
    Encoder_Kinetics_Audio,
)
from codefiles.transformer import Multimodal_Transformer
from codefiles.methods.mbt.mbt import Multimodal_Bottleneck_Transformer
from codefiles.methods.arl.asym_rep_learning import Asymmetric_Representation_Learning_Transformer
from codefiles.methods.dgl.disent_grad_learning import Disentangled_Gradient_Learning_Transformer
from codefiles.methods.mcr.mcr import MCR_Transformer
from codefiles.methods.mixup.mixup import Modality_Mixup_Transformer
from codefiles.methods.lowrank.lowrank import Low_Rank_Matrix_Fusion_Transformer
from codefiles.methods.mmpareto.mmpareto import MMPareto_Transformer
from codefiles.methods.bmml.bmml import Balanced_Multimodal_Transformer
from codefiles.methods.gblend.gblend import GBlend_Transformer
from codefiles.methods.mmp.mmp import Masked_Modality_Projection_Transformer
from codefiles.methods.pdf.pdf import Predictive_Dynamic_Fusion_Transformer
from codefiles.methods.pmr.pmr import Prototypical_Modal_Rebalance_Transformer
from codefiles.encoders import Encoder_INSPECT_Vision, Encoder_INSPECT_EHR
from codefiles.lightningmodules.mimic import MIMIC_Lightning_Module
from codefiles.lightningmodules.inspect import INSPECT_Lightning_Module
from codefiles.lightningmodules.synthetic import Synthetic_Lightning_Module
from codefiles.lightningmodules.mosi_mosei import MOSI_Lightning_Module
from codefiles.lightningmodules.vgg_sound import VGGSound_Lightning_Module
from codefiles.lightningmodules.ch_sims import CH_Sims_Lightning_Module
from codefiles.lightningmodules.vision_touch import VisionTouch_Lightning_Module
from codefiles.lightningmodules.crema_d import CREMAD_Lightning_Module
from codefiles.lightningmodules.kinetics import Kinetics_Lightning_Module
from codefiles.lightningdatamodules.inspect import INSPECT_Datamodule
from codefiles.lightningdatamodules.mimic_symile import MIMIC_Symile_Datamodule
from codefiles.lightningdatamodules.mimic_haim import MIMIC_Haim_Datamodule
from codefiles.lightningdatamodules.synthetic import Halved_Fashion_or_Vanilla_MNIST_Datamodule
from codefiles.lightningdatamodules.mosi_mosei import MOSI_MOSEI_Datamodule
from codefiles.lightningdatamodules.vgg_sound import VGGSound_Datamodule
from codefiles.lightningdatamodules.ch_sims import CH_Sims_Datamodule
from codefiles.lightningdatamodules.vision_touch import VisionTouch_Datamodule
from codefiles.lightningdatamodules.crema_d import CREMAD_Datamodule
from codefiles.lightningdatamodules.kinetics import Kinetics_Datamodule
from codefiles.methods.imder.imder import IMDer
from codefiles.methods.albef.albef import ALBEF
from codefiles.methods.mult.multim_transf import MulT
from codefiles.methods.ogm.gradient_modulation import OGM
from codefiles.encoders import Encoders_RegBN

def get_output_dim(cfg: dict = {}) -> int:
    if cfg.dataset == "mimic_symile":
        return 10
    elif cfg.dataset == "mimic_haim":
        return 10
    elif cfg.dataset == "mosi" or cfg.dataset == "mosei":
        return 7
    elif cfg.dataset == "vgg_sound":
        return 309
    elif cfg.dataset == "crema_d":
        return 6
    elif cfg.dataset == "vision_touch":
        return 1
    elif cfg.dataset == "ch_sims" or cfg.dataset == "ch_sims_v2":
        return 5
    elif cfg.dataset == "kinetics_400" or cfg.dataset == "kinetics_600" or cfg.dataset == "kinetics_700":
        return int(cfg.dataset.split("_")[1])
    else:
        raise NotImplementedError("Dataset not implemented")
    
def get_num_modalities(cfg: dict = {}) -> int:
    if cfg.dataset == "mimic_symile":
        return 3
    elif cfg.dataset == "mimic_haim":
        return 2
    elif cfg.dataset == "mosi" or cfg.dataset == "mosei":
        return 3
    elif cfg.dataset == "vgg_sound":
        return 2
    elif cfg.dataset == "crema_d":
        return 2
    elif cfg.dataset == "vision_touch":
        return 3
    elif cfg.dataset == "ch_sims" or cfg.dataset == "ch_sims_v2":
        return 3
    elif cfg.dataset == "kinetics_400" or cfg.dataset == "kinetics_600" or cfg.dataset == "kinetics_700":
        return 2
    else:
        raise NotImplementedError("Dataset not implemented")

def get_task_type(cfg: dict = {}) -> str:
    if cfg.dataset == "mimic_symile":
        return "bce"
    elif cfg.dataset == "mimic_haim":
        return "bce"
    elif cfg.dataset == "mosi" or cfg.dataset == "mosei":
        return "ce"
    elif cfg.dataset == "vgg_sound":
        return "ce"
    elif cfg.dataset == "crema_d":
        return "ce"
    elif cfg.dataset == "vision_touch":
        return "bce"
    elif cfg.dataset == "ch_sims" or cfg.dataset == "ch_sims_v2":
        return "ce"
    elif cfg.dataset == "kinetics_400" or cfg.dataset == "kinetics_600" or cfg.dataset == "kinetics_700":
        return "ce"
    else:
        raise NotImplementedError("Dataset not implemented")
    
def get_multilabel(cfg: dict = {}) -> bool:
    if cfg.dataset == "mimic_symile":
        return True
    elif cfg.dataset == "mimic_haim":
        return True
    elif cfg.dataset == "mosi" or cfg.dataset == "mosei":
        return False
    elif cfg.dataset == "vgg_sound":
        return False
    elif cfg.dataset == "crema_d":
        return False
    elif cfg.dataset == "vision_touch":
        return False
    elif cfg.dataset == "ch_sims" or cfg.dataset == "ch_sims_v2":
        return False
    elif cfg.dataset == "kinetics_400" or cfg.dataset == "kinetics_600" or cfg.dataset == "kinetics_700":
        return False
    else:
        raise NotImplementedError("Dataset not implemented")

def manual_optimizer(cfg: dict = {}) -> dict:
    if cfg.modelname.modelname == "bmml":
        return True
    elif cfg.modelname.modelname == "mcr":
        return True
    elif cfg.modelname.modelname == "arl":
        return True
    elif cfg.modelname.modelname == "dgl":
        return True
    elif cfg.modelname.modelname == "ogm":
        return True
    elif cfg.modelname.modelname == "gblend":
        return True
    elif cfg.modelname.modelname == "smil":
        return True
    else:
        return False

def build_model(
        cfg: dict = {},
) -> Multimodal_Architecture:
    
    if cfg.dataset == "mimic_symile":
        output_dim = 10
        n_modalities = 3

        encoders = Encoders(
            encoders = nn.ModuleList(
                [
                    Encoder_Symile_Vision(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        params={
                            "vit": cfg.encoders.vision.vit.vit,
                            "vit_dropout": cfg.encoders.vision.vit.dropout,
                        }
                    ),
                    Encoder_Symile_Tabular(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        params={
                            "input_dim": 50,
                            "hidden_dims": cfg.encoders.sequential2.mlp.hidden_dims,
                            "hidden_dropouts": cfg.encoders.sequential2.mlp.hidden_dropouts,
                        }
                    ),
                    Encoder_Symile_Sequential(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        params={
                            "transformer_num_layers": cfg.encoders.sequential1.transformer.num_hidden_layers,
                            "transformer_num_attention_heads": cfg.encoders.sequential1.transformer.num_attention_heads,
                            "transformer_dim_feedforward": cfg.encoders.sequential1.transformer.intermediate_size,
                            "transformer_dropout": cfg.encoders.sequential1.transformer.dropout,
                        }
                    )
                ]
            )
        )
    elif cfg.dataset == "mimic_haim":
        output_dim = 10
        n_modalities = 2

        encoders = Encoders(
            encoders = nn.ModuleList(
                [
                    Encoder_HAIM_Vision(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        params={
                            "vit": cfg.encoders.vision.vit.vit,
                            "vit_dropout": cfg.encoders.vision.vit.dropout,
                        }
                    ),
                    Encoder_HAIM_Sequential(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        params={
                            "transformer_num_layers": cfg.encoders.sequential.transformer.num_hidden_layers,
                            "transformer_num_attention_heads": cfg.encoders.sequential.transformer.num_attention_heads,
                            "transformer_dim_feedforward": cfg.encoders.sequential.transformer.intermediate_size,
                            "transformer_dropout": cfg.encoders.sequential.transformer.dropout,
                        }
                    )
                ]
            )
        )
    elif cfg.dataset == "mnist" or cfg.dataset == "fmnist":
        output_dim = 10
        n_modalities = 2

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
    elif cfg.dataset == "mnist_quarter" or cfg.dataset == "fmnist_quarter":
        output_dim = 10
        n_modalities = 4

        encoders = Encoders(
            encoders = nn.ModuleList(
                [
                    Encoder_MNIST(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        in_features=196,
                    ),
                    Encoder_MNIST(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        in_features=196,
                    ),
                    Encoder_MNIST(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        in_features=196,
                    ),
                    Encoder_MNIST(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        in_features=196,
                    ),
                ]
            )
        )
    elif cfg.dataset == "mosi" or cfg.dataset == "mosei":
        output_dim = 7
        n_modalities = 3

        encoders = Encoders(
            encoders = nn.ModuleList(
                [
                    Encoder_MOSI_Language(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        dataset=cfg.dataset
                    ),
                    Encoder_MOSI_Vision(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        dataset=cfg.dataset,
                        params={
                            "transformer_num_layers": cfg.encoders.vision.transformer.num_hidden_layers,
                            "transformer_num_attention_heads": cfg.encoders.vision.transformer.num_attention_heads,
                            "transformer_dim_feedforward": cfg.encoders.vision.transformer.intermediate_size,
                            "transformer_dropout": cfg.encoders.vision.transformer.dropout,
                        }
                    ),
                    Encoder_MOSI_Audio(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        dataset=cfg.dataset,
                        params={
                            "transformer_num_layers": cfg.encoders.audio.transformer.num_hidden_layers,
                            "transformer_num_attention_heads": cfg.encoders.audio.transformer.num_attention_heads,
                            "transformer_dim_feedforward": cfg.encoders.audio.transformer.intermediate_size,
                            "transformer_dropout": cfg.encoders.audio.transformer.dropout,
                        }
                    )
                ]
            )
        )
    elif cfg.dataset == "vgg_sound":
        output_dim = 309
        n_modalities = 2

        encoders = Encoders(
            encoders = nn.ModuleList(
                [
                    Encoder_VGG_Video(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                    ),
                    Encoder_VGG_Audio(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                    ),
                ]
            )
        )
    elif cfg.dataset == "crema_d":
        output_dim = 6  # 24
        n_modalities = 2

        encoders = Encoders(
            encoders = nn.ModuleList(
                [
                    Encoder_CREMA_D_Video(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                    ),
                    Encoder_CREMA_D_Audio(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                    ),
                ]
            )
        )
    elif cfg.dataset == "vision_touch":
        output_dim = 1
        n_modalities = 3

        encoders = Encoders(
            encoders = nn.ModuleList(
                [
                    Encoder_VisionTouch_Vision(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        params={
                            "vit_dropout": cfg.encoders.vision.vit.dropout,
                        }
                    ),
                    Encoder_VisionTouch_Proprio(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                    ),
                    Encoder_VisionTouch_Force(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        params = {
                            "transformer_num_layers": cfg.encoders.force.transformer.num_hidden_layers,
                            "transformer_num_attention_heads": cfg.encoders.force.transformer.num_attention_heads,
                            "transformer_dim_feedforward": cfg.encoders.force.transformer.intermediate_size,
                            "transformer_dropout": cfg.encoders.force.transformer.dropout,
                        }
                    ),
                ]
            )
        )
    elif cfg.dataset == "ch_sims" or cfg.dataset == "ch_sims_v2":
        output_dim = 5
        n_modalities = 3

        encoders = Encoders(
            encoders = nn.ModuleList(
                [
                    Encoder_CHS_Vision(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        params={
                            "num_hidden_layers": cfg.encoders.vision.transformer.num_hidden_layers,
                            "num_attention_heads": cfg.encoders.vision.transformer.num_attention_heads,
                            "intermediate_size": cfg.encoders.vision.transformer.intermediate_size,
                        }
                    ),
                    Encoder_CHS_Audio(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        params={
                            "num_hidden_layers": cfg.encoders.audio.transformer.num_hidden_layers,
                            "num_attention_heads": cfg.encoders.audio.transformer.num_attention_heads,
                            "intermediate_size": cfg.encoders.audio.transformer.intermediate_size,
                        }
                    ),
                    Encoder_CHS_Language(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                    ),
                ]
            )
        )
    elif cfg.dataset == "kinetics_400" or cfg.dataset == "kinetics_600" or cfg.dataset == "kinetics_700":
        output_dim = int(cfg.dataset.split("_")[1])  # 400 / 600 / 700
        n_modalities = 2

        encoders = Encoders(
            encoders = nn.ModuleList(
                [
                    Encoder_Kinetics_Video(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                    ),
                    Encoder_Kinetics_Audio(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                    ),
                ]
            )
        )
    elif cfg.dataset == "inspect":
        output_dim = 1
        n_modalities = 2

        encoders = Encoders(
            encoders = nn.ModuleList(
                [
                    Encoder_INSPECT_Vision(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        params={
                            "hidden_dims": cfg.encoders.vision.mlp.hidden_dims,
                            "hidden_dropouts": cfg.encoders.vision.mlp.hidden_dropouts,
                        }
                    ),
                    Encoder_INSPECT_EHR(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        params={
                            "hidden_dims": cfg.encoders.ehr.mlp.hidden_dims,
                            "hidden_dropouts": cfg.encoders.ehr.mlp.hidden_dropouts,
                        }
                    ),
                ]
            )
        )
    else: 
        raise NotImplementedError("Dataset not implemented")
    
    if cfg.modelname.modelname == "transformer":
        multimodal_transformer = Multimodal_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            dim_output=output_dim
        )
    elif cfg.modelname.modelname == "imder":
        multimodal_transformer = IMDer(
            params_transformerhead={
                "d_model": cfg.modelname.head_transformer.d_model,
                "nhead": cfg.modelname.head_transformer.nhead,
                "dim_feedforward": cfg.modelname.head_transformer.dim_feedforward,
                "dropout": cfg.modelname.head_transformer.dropout,
                "num_layers": cfg.modelname.head_transformer.num_layers,
                "dim_output": output_dim,
            },
            params_ddpm={
                "sampling_iter": cfg.modelname.ddpms.sampling_iter,
                "n_steps": cfg.modelname.ddpms.n_steps,
                "n_modalities": n_modalities,
                "seq_lens": cfg.modelname.ddpms.seq_lens,
                "d_model": cfg.modelname.ddpms.d_model,
                "hidden_dim": cfg.modelname.ddpms.hidden_dim,
                "num_layers": cfg.modelname.ddpms.num_layers,
                "nhead": cfg.modelname.ddpms.nhead,
                "dropout": cfg.modelname.ddpms.dropout,
            },
            params_imder={
                "beta": cfg.modelname.imder.beta,
            }
        )
    elif cfg.modelname.modelname == "albef":
        multimodal_transformer = ALBEF(
            n_modalities = n_modalities,
            params_transformerhead={
                "d_model": cfg.modelname.head_transformer.d_model,
                "nhead": cfg.modelname.head_transformer.nhead,
                "dim_feedforward": cfg.modelname.head_transformer.dim_feedforward,
                "dropout": cfg.modelname.head_transformer.dropout,
                "num_layers": cfg.modelname.head_transformer.num_layers,
                "dim_output": output_dim,
            },
            embed_dim = cfg.modelname.itc.dim, 
            itc_temperature = cfg.modelname.itc.temperature,
            distill_temperature = cfg.modelname.distill.temperature,
            itc_weight = cfg.modelname.itc.weight,
            itm_weight = cfg.modelname.itm.weight,
            queue_size = cfg.modelname.itc.queue_size,
            momentum = cfg.modelname.itc.momentum,
        )
    elif cfg.modelname.modelname == "mult":
        multimodal_transformer = MulT(
            params_ca_transformerhead={
                "d_model": cfg.modelname.crossmodal_transformer.d_model,
                "nhead": cfg.modelname.crossmodal_transformer.nhead,
                "dim_feedforward": cfg.modelname.crossmodal_transformer.dim_feedforward,
                "dropout": cfg.modelname.crossmodal_transformer.dropout,
                "num_layers": cfg.modelname.crossmodal_transformer.num_layers,
            },
            params_sa_transformerhead={
                "d_model": cfg.modelname.head_transformer.d_model,
                "nhead": cfg.modelname.head_transformer.nhead,
                "dim_feedforward": cfg.modelname.head_transformer.dim_feedforward,
                "dropout": cfg.modelname.head_transformer.dropout,
                "num_layers": cfg.modelname.head_transformer.num_layers,
                "dim_output": output_dim,
            },
            num_modalities = n_modalities,
        )
    elif cfg.modelname.modelname == "ogm":
        multimodal_transformer = OGM(
            params_transformerhead={
                "d_model": cfg.modelname.head_transformer.d_model,
                "nhead": cfg.modelname.head_transformer.nhead,
                "dim_feedforward": cfg.modelname.head_transformer.dim_feedforward,
                "dropout": cfg.modelname.head_transformer.dropout,
                "num_layers": cfg.modelname.head_transformer.num_layers,
                "dim_output": output_dim,
            },
            num_modalities = n_modalities,
            input_dims = (cfg.modelname.head_transformer.d_model, cfg.modelname.head_transformer.d_model),
            alpha = cfg.modelname.ogm.alpha,
        )
    elif cfg.modelname.modelname == "regbn":
        encs = encoders.encoders

        encoders = Encoders_RegBN(
            encoders=encs,
            output_dim=cfg.modelname.head_transformer.d_model,
            regbn_params={
                "momentum": cfg.modelname.rbn.momentum,
                "sigma_THR": cfg.modelname.rbn.sigma_THR,
                "sigma_MIN": cfg.modelname.rbn.sigma_MIN,
                "normalize_input": cfg.modelname.rbn.normalize_input,
                "normalize_output": cfg.modelname.rbn.normalize_output,
                "affine": cfg.modelname.rbn.affine,
                "beta1": cfg.modelname.rbn.beta[0],
                "beta2": cfg.modelname.rbn.beta[1],
            },
            n_modalities=n_modalities,
            reference_modality_idx=cfg.modelname.rbn.reference_modality_idx,
        )
        
        multimodal_transformer = Multimodal_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            dim_output=output_dim
        )
    elif cfg.modelname.modelname == "mbt":
        multimodal_transformer = Multimodal_Bottleneck_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            num_modalities=n_modalities,
            dim_output=output_dim,
            mbt_params={
                "num_bottlenecks": cfg.modelname.bottleneck.num_bottlenecks,
                "num_layers_mbt": cfg.modelname.bottleneck.layers,
                "dropout": cfg.modelname.bottleneck.dropout,
                "nhead": cfg.modelname.bottleneck.nhead,
                "dim_feedforward": cfg.modelname.bottleneck.dim_feedforward,
            }
        )
    elif cfg.modelname.modelname == "arl":
        multimodal_transformer = Asymmetric_Representation_Learning_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            num_modalities=n_modalities,
            dim_output=output_dim,
            arl_temperature=cfg.modelname.arl.temperature,
        )
    elif cfg.modelname.modelname == "dgl":
        multimodal_transformer = Disentangled_Gradient_Learning_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            num_modalities=n_modalities,
            dim_output=output_dim,
        )
    elif cfg.modelname.modelname == "mcr":
        multimodal_transformer = MCR_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            num_modalities=n_modalities,
            dim_output=output_dim,
        )
    elif cfg.modelname.modelname == "mixup":
        multimodal_transformer = Modality_Mixup_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            num_modalities=n_modalities,
            dim_output=output_dim,
            mixup_alpha=cfg.modelname.mixup.mixup_alpha,
            modalities_to_mix=cfg.modelname.mixup.modalities_to_mix,
            consistency_type=cfg.modelname.mixup.consistency_loss_type,
        )
    elif cfg.modelname.modelname == "lmf":
        multimodal_transformer = Low_Rank_Matrix_Fusion_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            num_modalities=n_modalities,
            dim_output=output_dim,
            rank=cfg.modelname.low_rank_matrix_fusion.rank,
        )
    elif cfg.modelname.modelname == "mmpareto":
        multimodal_transformer = MMPareto_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            num_modalities=n_modalities,
            dim_output=output_dim,
        )
    elif cfg.modelname.modelname == "bmml":
        multimodal_transformer = Balanced_Multimodal_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            dim_output=output_dim,
            num_modalities=n_modalities,
            bmml_momentum=cfg.modelname.bmml.bmml_momentum,
        )
    elif cfg.modelname.modelname == "gblend":
        multimodal_transformer = GBlend_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            dim_output=output_dim,
            num_modalities=n_modalities,
        )
    elif cfg.modelname.modelname == "mmp":
        multimodal_transformer = Masked_Modality_Projection_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            dim_output=output_dim,
            num_modalities=n_modalities,
            params_proj_mlp={
                "hidden_dim": cfg.modelname.mmp.proj_mlp.hidden_dim,
                "dropout": cfg.modelname.mmp.proj_mlp.dropout,
            },
            params_attn_steps={
                "dropout": cfg.modelname.mmp.attn_steps.dropout,
                "nhead": cfg.modelname.mmp.attn_steps.nhead,
            },
            num_aggregated_tokens=cfg.modelname.mmp.num_aggregated_tokens,
            loss_alignment_alpha=cfg.modelname.mmp.loss_alignment_alpha,
        )
    elif cfg.modelname.modelname == "pdf":
        multimodal_transformer = Predictive_Dynamic_Fusion_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            dim_output=output_dim,
            num_modalities=get_num_modalities(cfg),
            task_type=get_task_type(cfg),
            p_head_params=cfg.modelname.pdf.p_head,
        )
    elif cfg.modelname.modelname == "pmr":
        multimodal_transformer = Prototypical_Modal_Rebalance_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            dim_output=output_dim,
            num_modalities=get_num_modalities(cfg),
            params_pmr={
                "alpha": cfg.modelname.pmr.alpha,
                "mu": cfg.modelname.pmr.mu,
                "regularization_epochs": cfg.modelname.pmr.regularization_epochs,
                "num_classes": 2 if cfg.dataset == "vision_touch" else get_output_dim(cfg),
                "is_multilabel": get_multilabel(cfg),
                "num_bins_regression": cfg.modelname.pmr.num_bins_regression,
                "epsilon": cfg.modelname.pmr.epsilon,
            },
        )
    elif cfg.modelname.modelname == "coupled_ssm":
        from codefiles.methods.coupled_ssm.coupled_ssm import Coupled_State_Space_Model
        multimodal_transformer = Coupled_State_Space_Model(
            d_model=cfg.modelname.head_transformer.d_model,
            n_layer=cfg.modelname.head_transformer.num_layers,
            d_state=cfg.modelname.head_transformer.d_state,
            d_conv=cfg.modelname.head_transformer.d_conv,
            expand=cfg.modelname.head_transformer.expand,
            num_modalities=get_num_modalities(cfg),
            dim_output=output_dim,
        )
    elif cfg.modelname.modelname == "placeholder":
        from codefiles.methods.placeholder.transformer import Multimodal_Placeholder_Transformer
        multimodal_transformer = Multimodal_Placeholder_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            dim_output=output_dim,
        )
    elif cfg.modelname.modelname == "shaspec":
        from codefiles.methods.shaspec.shaspec import Shared_Specific_Feature_Modelling_Transformer
        multimodal_transformer = Shared_Specific_Feature_Modelling_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            dim_output=output_dim,
            num_modalities=get_num_modalities(cfg),
            loss_alpha=cfg.modelname.shaspec.loss_alpha,
            loss_beta=cfg.modelname.shaspec.loss_beta,
        )
    elif cfg.modelname.modelname == "smil":
        from codefiles.methods.smil.smil import SMIL
        multimodal_transformer = SMIL(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            dim_output=output_dim,
            # num_modalities=get_num_modalities(cfg),
            # num_priors=cfg.modelname.smil.num_priors,
        )
    elif cfg.modelname.modelname == "ebr":
        from codefiles.methods.ebr.ebr import Explicit_Basis_Reallocation_Transformer
        multimodal_transformer = Explicit_Basis_Reallocation_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            dim_output=output_dim,
            max_modalities=get_num_modalities(cfg),
            ebr_params={
                "alpha_ebr": cfg.modelname.ebr.alpha_ebr,
                "hidden_dim": cfg.modelname.ebr.hidden_dim,
                "d_shared": cfg.modelname.ebr.d_shared,
            }
        )
    elif cfg.modelname.modelname == "omib":
        from codefiles.methods.omib.omib import Optimal_Multimodal_Information_Bottleneck_Transformer
        multimodal_transformer = Optimal_Multimodal_Information_Bottleneck_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            dim_output=output_dim,
            num_modalities=get_num_modalities(cfg),
            beta=cfg.modelname.omib.beta,
            task_type=get_task_type(cfg),
            params_cross_attn_network={
                "num_layers": cfg.modelname.omib.cross_attn_network.num_layers,
                "dropout": cfg.modelname.omib.cross_attn_network.dropout,
                "num_heads": cfg.modelname.omib.cross_attn_network.num_heads,
                "dim_feedforward": cfg.modelname.omib.cross_attn_network.dim_feedforward,
            },
        )
    elif cfg.modelname.modelname == "simmdg":
        from codefiles.methods.simmdg.simmdg import SiMMDG_Transformer
        multimodal_transformer = SiMMDG_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            dim_output=output_dim,
            proj_out_dim=cfg.modelname.simmdg.proj_out_dim,
            trans_hidden_dim=cfg.modelname.simmdg.trans_hidden_dim,
            temp=cfg.modelname.simmdg.temp,
            n_modalities=get_num_modalities(cfg),
            loss_contrastive=cfg.modelname.simmdg.loss_contrastive,
            loss_distance=cfg.modelname.simmdg.loss_distance,
            loss_translation=cfg.modelname.simmdg.loss_translation,
        )
    elif cfg.modelname.modelname == "simmlm":
        from codefiles.methods.simmlm.simmlm import Sim_MLM_Transformer
        multimodal_transformer = Sim_MLM_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            dim_output=output_dim,
            n_modalities=get_num_modalities(cfg),
            gating_hidden_dim=cfg.modelname.simmlm.gating_hidden_dim,
        )
    else:
        raise NotImplementedError("Model not implemented")

    model = Multimodal_Architecture(
        encoders=encoders,
        transformer=multimodal_transformer,
        full_seq=cfg.modelname.pipeline.full_seq,
    )

    return model

def build_lightningmodule(
        cfg: dict = {},
        model: Multimodal_Architecture = Multimodal_Architecture(),
) -> pl.LightningModule:
    
    manual_opt = manual_optimizer(cfg)

    if "arl" in cfg.modelname:
        params_arl = {
            "unimodal_loss_weight": cfg.modelname.arl.unimodal_loss_weight,
        }
    else:
        params_arl = {}
    if "dgl" in cfg.modelname:
        params_dgl = {
            "unimodal_loss_weight": cfg.modelname.dgl.unimodal_loss_weight,
        }
    else:
        params_dgl = {}
    if "mcr" in cfg.modelname:
        params_mcr = {
            "strategy": cfg.modelname.mcr.strategy,
            "loss_weights": cfg.modelname.mcr.loss_weights,
            "contrastive_temp": cfg.modelname.mcr.contrastive_temp,
            "ceb_reconstruction_head": cfg.modelname.mcr.ceb_reconstruction_head,
            "num_permutations": cfg.modelname.mcr.num_permutations,
            "d_model": cfg.modelname.head_transformer.d_model,
            "num_classes": get_output_dim(cfg),
            "num_modalities": get_num_modalities(cfg),
            "is_multilabel": get_multilabel(cfg),
        }
    else:
        params_mcr = {}
    if "mmpareto" in cfg.modelname:
        params_mmpareto = {
            "unimodal_loss_weight": cfg.modelname.mmpareto.unimodal_loss_weight,
            "gamma": cfg.modelname.mmpareto.gamma,
        }
    else:
        params_mmpareto = {}
    if "bmml" in cfg.modelname:
        params_bmml = {
            "unimodal_loss_weight": cfg.modelname.bmml.unimodal_loss_weight,
            "alpha": cfg.modelname.bmml.alpha,
            "q": cfg.modelname.bmml.q,
            "warmup_epochs": cfg.modelname.bmml.warmup_epochs,
            "num_modalities": get_num_modalities(cfg),
        }
    else:
        params_bmml = {}
    if "gblend" in cfg.modelname:
        params_gblend = {
            "mode": cfg.modelname.gblend.mode,
            "num_modalities": get_num_modalities(cfg),
            "lookahead_epochs": cfg.modelname.gblend.lookahead_epochs,
            "update_freq": cfg.modelname.gblend.update_freq,
        }
    else:
        params_gblend = {}
    if "pdf" in cfg.modelname:
        params_pdf = {
            "loss_weight": cfg.modelname.pdf.loss_weight,
            "unimodal_loss_weight": cfg.modelname.pdf.unimodal_loss_weight,
            "p_true_loss_fn": cfg.modelname.pdf.p_true_loss_fn,
        }
    else:
        params_pdf = {}
    if "pmr" in cfg.modelname:
        params_pmr = {
            "epsilon": cfg.modelname.pmr.epsilon,
        }
    else:
        params_pmr = {}
    if "omib" in cfg.modelname:
        params_omib = {
            "warmup_epochs": cfg.modelname.omib.warmup_epochs,
        }
    else:
        params_omib = {}
    if "smil" in cfg.modelname:
        params_smil = {
            "inner_lr": cfg.modelname.smil.inner_lr,
            "alpha": cfg.modelname.smil.alpha,
        }
    else:
        params_smil = {}
    if "mixup" in cfg.modelname:
        params_mixup = {
            "unimodal_loss_weights": cfg.modelname.mixup.unimodal_loss_weights,
            "consistency_loss_weights": cfg.modelname.mixup.consistency_loss_weights,
            "modalities_to_mix_keys": cfg.modelname.mixup.modalities_to_mix,
            "consistency_loss_type": cfg.modelname.mixup.consistency_loss_type,
        }
    else:
        params_mixup = {}
    if "ogm" in cfg.modelname:
        params_ogm = {
            "use_ge": cfg.modelname.ogm.use_ge,
        }
    else:
        params_ogm = {}
    if "ebr" in cfg.modelname:
        params_ebr = {
            "interleave_epochs": cfg.modelname.ebr.interleave_epochs,
        }
    else:
        params_ebr = {}
    if "simmlm" in cfg.modelname:
        params_simmlm = {
            "mofe_lambda": cfg.modelname.simmlm.mofe_lambda,
        }
    else:
        params_simmlm = {}

    if cfg.dataset == "mimic_symile" or cfg.dataset == "mimic_haim":
        lightningmodule = MIMIC_Lightning_Module(
            model=model,
            dataset=cfg.dataset,
            params_optimizer={key: value for key, value in cfg.modelname.optimizer.items()},
            manual_opt=manual_opt,
            params_ogm=params_ogm,
            params_arl=params_arl,
            params_dgl=params_dgl,
            params_mcr=params_mcr,
            params_mmpareto=params_mmpareto,
            params_bmml=params_bmml,
            params_gblend=params_gblend,
            params_pdf=params_pdf,
            params_pmr=params_pmr,
            params_smil=params_smil,
            params_avmc=params_mixup,
            params_ebr=params_ebr,
            params_simmlm=params_simmlm,
        )
    elif cfg.dataset == "mnist" or cfg.dataset == "fmnist":
        lightningmodule = Synthetic_Lightning_Module(
            model=model,
            dataset=cfg.dataset,
            params_optimizer={key: value for key, value in cfg.modelname.optimizer.items()},
            manual_opt=manual_opt,
            params_ogm=params_ogm,
            params_arl=params_arl,
            params_dgl=params_dgl,
            params_mcr=params_mcr,
            params_mmpareto=params_mmpareto,
            params_bmml=params_bmml,
            params_gblend=params_gblend,    
            params_pdf=params_pdf,
            params_pmr=params_pmr,
            params_smil=params_smil,
            params_avmc=params_mixup,
            params_ebr=params_ebr,
            params_simmlm=params_simmlm,
        )
    elif cfg.dataset == "mosi" or cfg.dataset == "mosei":
        lightningmodule = MOSI_Lightning_Module(
            model=model,
            dataset=cfg.dataset,
            params_optimizer={key: value for key, value in cfg.modelname.optimizer.items()},
            manual_opt=manual_opt,
            params_ogm=params_ogm,
            params_arl=params_arl,
            params_dgl=params_dgl,
            params_mcr=params_mcr,
            params_mmpareto=params_mmpareto,
            params_bmml=params_bmml,
            params_gblend=params_gblend,
            params_pdf=params_pdf,
            params_pmr=params_pmr,
            params_smil=params_smil,
            params_avmc=params_mixup,
            params_ebr=params_ebr,
            params_simmlm=params_simmlm,
        )
    elif cfg.dataset == "vgg_sound":
        lightningmodule = VGGSound_Lightning_Module(
            model=model,
            dataset=cfg.dataset,
            params_optimizer={key: value for key, value in cfg.modelname.optimizer.items()},
            manual_opt=manual_opt,
            params_ogm=params_ogm,
            params_arl=params_arl,
            params_dgl=params_dgl,
            params_mcr=params_mcr,
            params_mmpareto=params_mmpareto,    
            params_bmml=params_bmml,
            params_gblend=params_gblend,
            params_pdf=params_pdf,
            params_pmr=params_pmr,
            params_smil=params_smil,
            params_avmc=params_mixup,
            params_ebr=params_ebr,
            params_simmlm=params_simmlm,
        )
    elif cfg.dataset == "crema_d":
        lightningmodule = CREMAD_Lightning_Module(
            model=model,
            dataset=cfg.dataset,
            params_optimizer={key: value for key, value in cfg.modelname.optimizer.items()},
            manual_opt=manual_opt,
            params_arl=params_arl,
            params_dgl=params_dgl,
            params_mcr=params_mcr,
            params_mmpareto=params_mmpareto,
            params_bmml=params_bmml,
            params_gblend=params_gblend,
            params_pdf=params_pdf,
            params_pmr=params_pmr,
            params_smil=params_smil,
            params_avmc=params_mixup,
            params_ebr=params_ebr,
            params_simmlm=params_simmlm,
        )
    elif cfg.dataset == "ch_sims" or cfg.dataset == "ch_sims_v2":
        lightningmodule = CH_Sims_Lightning_Module(
            model=model,
            dataset=cfg.dataset,
            params_optimizer={key: value for key, value in cfg.modelname.optimizer.items()},
            manual_opt=manual_opt,
            params_ogm=params_ogm,
            params_arl=params_arl,
            params_dgl=params_dgl,
            params_mcr=params_mcr,
            params_mmpareto=params_mmpareto,    
            params_bmml=params_bmml,
            params_gblend=params_gblend,
            params_pdf=params_pdf,
            params_pmr=params_pmr,
            params_smil=params_smil,
            params_avmc=params_mixup,
            params_ebr=params_ebr,
            params_simmlm=params_simmlm,
        )
    elif cfg.dataset == "vision_touch":
        lightningmodule = VisionTouch_Lightning_Module(
            model=model,
            dataset=cfg.dataset,
            params_optimizer={key: value for key, value in cfg.modelname.optimizer.items()},
            manual_opt=manual_opt,
            params_ogm=params_ogm,
            params_arl=params_arl,
            params_dgl=params_dgl,
            params_mcr=params_mcr,
            params_mmpareto=params_mmpareto,
            params_bmml=params_bmml,
            params_gblend=params_gblend,
            params_pdf=params_pdf,
            params_pmr=params_pmr,
            params_smil=params_smil,
            params_avmc=params_mixup,
            params_ebr=params_ebr,
            params_simmlm=params_simmlm,
        )
    elif cfg.dataset == "kinetics_400" or cfg.dataset == "kinetics_600" or cfg.dataset == "kinetics_700":
        lightningmodule = Kinetics_Lightning_Module(
            model=model,
            dataset=cfg.dataset,
            params_optimizer={key: value for key, value in cfg.modelname.optimizer.items()},
            manual_opt=manual_opt,
            params_ogm=params_ogm,
            params_arl=params_arl,
            params_dgl=params_dgl,
            params_mcr=params_mcr,
            params_mmpareto=params_mmpareto,
            params_bmml=params_bmml,
            params_gblend=params_gblend,
            params_pdf=params_pdf,
            params_pmr=params_pmr,
            params_smil=params_smil,
            params_avmc=params_mixup,
            params_ebr=params_ebr,
            params_simmlm=params_simmlm,
        )
    elif cfg.dataset == "inspect":
        lightningmodule = INSPECT_Lightning_Module(
            model=model,
            dataset=cfg.dataset,
            params_optimizer={key: value for key, value in cfg.modelname.optimizer.items()},
            manual_opt=manual_opt,
            params_ogm=params_ogm,
            params_arl=params_arl,
            params_dgl=params_dgl,
            params_mcr=params_mcr,
            params_mmpareto=params_mmpareto,
            params_bmml=params_bmml,
            params_gblend=params_gblend,
            params_pdf=params_pdf,
            params_pmr=params_pmr,
            params_smil=params_smil,
            params_avmc=params_mixup,
            params_ebr=params_ebr,
            params_simmlm=params_simmlm,
        )
    else: 
        raise NotImplementedError("Dataset not implemented")
    
    return lightningmodule

def build_datamodule(
        cfg: dict = {},
        debug: bool = False
) -> pl.LightningDataModule:
    
    datamodule_params = {
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "missing": {key: value for key, value in cfg.missing.items()},
        "variant": cfg.modelname.modelname,
        "num_workers": cfg.num_workers,
        "split_nr": cfg.split_nr
    }
    
    if cfg.dataset == "mimic_symile":
        datamodule = MIMIC_Symile_Datamodule(**datamodule_params)
    elif cfg.dataset == "mimic_haim":
        datamodule_params["debug"] = debug
        datamodule = MIMIC_Haim_Datamodule(**datamodule_params)
    elif cfg.dataset == "mosi" or cfg.dataset == "mosei":
        datamodule_params["dataset"] = cfg.dataset
        datamodule = MOSI_MOSEI_Datamodule(**datamodule_params)
    elif cfg.dataset == "vgg_sound":
        datamodule = VGGSound_Datamodule(**datamodule_params)
    elif cfg.dataset == "crema_d":
        datamodule = CREMAD_Datamodule(**datamodule_params)
    elif cfg.dataset == "ch_sims" or cfg.dataset == "ch_sims_v2":
        datamodule_params["v2"] = True if cfg.dataset == "ch_sims_v2" else False
        datamodule = CH_Sims_Datamodule(**datamodule_params)
    elif cfg.dataset == "mnist" or cfg.dataset == "fmnist":
        datamodule_params["dataset"] = cfg.dataset
        datamodule = Halved_Fashion_or_Vanilla_MNIST_Datamodule(**datamodule_params)
    elif cfg.dataset == "vision_touch":
        datamodule = VisionTouch_Datamodule(**datamodule_params)
    elif cfg.dataset == "kinetics_400" or cfg.dataset == "kinetics_600" or cfg.dataset == "kinetics_700":
        datamodule_params["num_classes"] = int(cfg.dataset.split("_")[1])
        datamodule = Kinetics_Datamodule(**datamodule_params)
    elif cfg.dataset == "inspect":
        datamodule = INSPECT_Datamodule(**datamodule_params)
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