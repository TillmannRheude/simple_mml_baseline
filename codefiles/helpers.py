import torch 
import random 
import signal 
import sys

import numpy as np 
import torch.nn as nn 
import pytorch_lightning as pl

from codefiles.architecture import Multimodal_Architecture
from codefiles.encoders import (
    Encoders, 
    Encoder_MOSI_Language,
    Encoder_MOSI_Vision,
    Encoder_MOSI_Audio,
    Encoder_CHS_Language,
    Encoder_CHS_Vision,
    Encoder_CHS_Audio,
    Encoder_CREMA_D_Video,
    Encoder_CREMA_D_Audio,
    Encoder_HAIM_Vision,
    Encoder_HAIM_Sequential,
    Encoder_Symile_Sequential,
    Encoder_Symile_Tabular,
    Encoder_Symile_Vision,
    Encoder_UKB_Tabular,
    Encoder_MysteryMML_Tabular,
    Encoders_RegBN,
    Encoder_INSPECT_Vision, Encoder_INSPECT_EHR
)
from codefiles.transformer import Multimodal_Transformer
from codefiles.methods.mbt.mbt import Multimodal_Bottleneck_Transformer
from codefiles.methods.arl.asym_rep_learning import Asymmetric_Representation_Learning_Transformer
from codefiles.methods.dgl.disent_grad_learning import Disentangled_Gradient_Learning_Transformer
from codefiles.methods.lowrank.lowrank import Low_Rank_Matrix_Fusion_Transformer
from codefiles.methods.mmpareto.mmpareto import MMPareto_Transformer
from codefiles.methods.bmml.bmml import Balanced_Multimodal_Transformer
from codefiles.methods.gblend.gblend import GBlend_Transformer
from codefiles.methods.mmp.mmp import Masked_Modality_Projection_Transformer
from codefiles.methods.pdf.pdf import Predictive_Dynamic_Fusion_Transformer
from codefiles.methods.imder.imder import IMDer
from codefiles.methods.mult.multim_transf import MulT
from codefiles.methods.ogm.gradient_modulation import OGM
from codefiles.methods.aug.aug import AUG_Transformer
from codefiles.lightningmodules.mimic import MIMIC_Lightning_Module
from codefiles.lightningmodules.inspect import INSPECT_Lightning_Module
from codefiles.lightningmodules.mosi_mosei import MOSI_Lightning_Module
from codefiles.lightningmodules.ch_sims import CH_Sims_Lightning_Module
from codefiles.lightningmodules.crema_d import CREMAD_Lightning_Module
from codefiles.lightningmodules.ukb import UKB_Lightning_Module
from codefiles.lightningmodules.mysterymml import MysteryMML_Lightning_Module
from codefiles.lightningdatamodules.inspect import INSPECT_Datamodule
from codefiles.lightningdatamodules.mimic_symile import MIMIC_Symile_Datamodule
from codefiles.lightningdatamodules.mimic_haim import MIMIC_Haim_Datamodule
from codefiles.lightningdatamodules.mosi_mosei import MOSI_MOSEI_Datamodule
from codefiles.lightningdatamodules.ch_sims import CH_Sims_Datamodule
from codefiles.lightningdatamodules.crema_d import CREMAD_Datamodule
from codefiles.lightningdatamodules.mysterymml import MysteryMML_Datamodule

def get_output_dim(cfg: dict = {}) -> int:
    if cfg.dataset == "mimic_symile":
        return 10
    elif cfg.dataset == "mimic_haim":
        return 10
    elif cfg.dataset == "mosi" or cfg.dataset == "mosei":
        return 7
    elif cfg.dataset == "crema_d":
        return 6
    elif cfg.dataset == "ch_sims" or cfg.dataset == "ch_sims_v2":
        return 5
    elif cfg.dataset == "inspect":
        return 1
    elif cfg.dataset == "ukb":
        return 1
    elif cfg.dataset == "mystery_mml":
        return 1
    else:
        raise NotImplementedError("Dataset not implemented")
    
def get_num_modalities(cfg: dict = {}) -> int:
    if cfg.dataset == "mimic_symile":
        return 3
    elif cfg.dataset == "mimic_haim":
        return 2
    elif cfg.dataset == "mosi" or cfg.dataset == "mosei":
        return 3
    elif cfg.dataset == "crema_d":
        return 2
    elif cfg.dataset == "ch_sims" or cfg.dataset == "ch_sims_v2":
        return 3
    elif cfg.dataset == "inspect":
        return 2
    elif cfg.dataset == "ukb":
        if "datamodule" in cfg and "plugins" in cfg.datamodule:
            return len([mod for mod in cfg.datamodule.plugins.keys() if mod != "labels"])
        return 23
    elif cfg.dataset == "mystery_mml":
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
    elif cfg.dataset == "crema_d":
        return "ce"
    elif cfg.dataset == "ch_sims" or cfg.dataset == "ch_sims_v2":
        return "ce"
    elif cfg.dataset == "inspect":
        return "bce"
    elif cfg.dataset == "ukb":
        return "bce"
    elif cfg.dataset == "mystery_mml":
        return "bce"
    else:
        raise NotImplementedError("Dataset not implemented")
    
def get_multilabel(cfg: dict = {}) -> bool:
    if cfg.dataset == "mimic_symile":
        return True
    elif cfg.dataset == "mimic_haim":
        return True
    elif cfg.dataset == "mosi" or cfg.dataset == "mosei":
        return False
    elif cfg.dataset == "crema_d":
        return False
    elif cfg.dataset == "ch_sims" or cfg.dataset == "ch_sims_v2":
        return False
    elif cfg.dataset == "inspect":
        return False
    elif cfg.dataset == "ukb":
        return False
    elif cfg.dataset == "mystery_mml":
        return False
    else:
        raise NotImplementedError("Dataset not implemented")

def manual_optimizer(cfg: dict = {}) -> dict:
    if cfg.modelname.modelname == "bmml":
        return True
    elif cfg.modelname.modelname == "arl":
        return True
    elif cfg.modelname.modelname == "dgl":
        return True
    elif cfg.modelname.modelname == "ogm":
        return True
    elif cfg.modelname.modelname == "gblend":
        return True
    elif cfg.modelname.modelname == "aug":
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
    elif cfg.dataset == "crema_d":
        output_dim = 6  # 24
        n_modalities = 2

        if cfg.encoders["use_embeddings"]:
            encoders = Encoders(
                encoders = nn.ModuleList(
                    [
                        Encoder_CremaD_Video_Embeddings(
                            latent_dim=cfg.modelname.head_transformer.d_model,
                        ),
                        Encoder_CremaD_Audio_Embeddings(
                            latent_dim=cfg.modelname.head_transformer.d_model,
                        ),
                    ]
                )
            )
        else:
            encoders = Encoders(
                encoders = nn.ModuleList(
                    [
                        Encoder_CREMA_D_Video(
                            params={
                                "latent_dim": cfg.modelname.head_transformer.d_model,
                                "vit_dim": cfg.encoders.vision.vit.dim,
                                "vit_heads": cfg.encoders.vision.vit.heads,
                                "vit_depth": cfg.encoders.vision.vit.depth,
                                "vit_mlp_dim": cfg.encoders.vision.vit.mlp_dim,
                                "vit_dropout": cfg.encoders.vision.vit.dropout,
                            }
                        ),
                        Encoder_CREMA_D_Audio(
                            params={
                                "latent_dim": cfg.modelname.head_transformer.d_model,
                                "vit_dim": cfg.encoders.audio.vit.dim,
                                "vit_heads": cfg.encoders.audio.vit.heads,
                                "vit_depth": cfg.encoders.audio.vit.depth,
                                "vit_mlp_dim": cfg.encoders.audio.vit.mlp_dim,
                                "vit_dropout": cfg.encoders.audio.vit.dropout,
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
    elif cfg.dataset == "inspect":
        output_dim = 1
        n_modalities = 2

        encoders = Encoders(
            encoders = nn.ModuleList(
                [
                    Encoder_INSPECT_Vision(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        params={
                            "input_dim": 768,
                            "hidden_dims": cfg.encoders.vision.mlp.hidden_dims,
                            "hidden_dropouts": cfg.encoders.vision.mlp.hidden_dropouts,
                        }
                    ),
                    Encoder_INSPECT_EHR(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        params={
                            "input_dim": 769,
                            "hidden_dims": cfg.encoders.ehr.mlp.hidden_dims,
                            "hidden_dropouts": cfg.encoders.ehr.mlp.hidden_dropouts,
                        }
                    ),
                ]
            )
        )
    elif cfg.dataset == "ukb":
        output_dim = 1

        input_dims_modalities = {
            "nmr": 249,
            "ehr": 3584,
            "olink": 1463,
            "prs": 135,
            "bloodbio": 30,
            "baselinechars": 28,
            "localenvironment": 33,
            "arterialstiffness": 9,
            "anthropometry": 43,
            "bloodpressure": 12,
            "ecgduringexercise": 355,
            "eyemeasures": 310,
            "bonedensitometry": 27,
            "handgripstrength": 2,
            "spirometry": 29,
            "touchscreen": 119,
            "cognitivefunction": 27,
            "hearingtest": 68,
            "verbalinterview": 224,
            "bloodcount": 31,
            "urineassays": 4,
            "telomeres": 4,
            "infectiousdiseases": 66,
        }
        encoders = nn.ModuleList([])

        for mod in input_dims_modalities.keys():
            # if cfg.datamodule.plugins[mod] is not None
            if mod in cfg.datamodule.plugins:
                print(cfg.datamodule.plugins[mod])  
                encoders.append(
                    Encoder_UKB_Tabular(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                        params={
                            "input_dim": input_dims_modalities[mod],
                            "hidden_dims": cfg.encoders[mod].mlp.hidden_dims,
                            "hidden_dropouts": cfg.encoders[mod].mlp.hidden_dropouts,
                        }
                    )
                )
        n_modalities = len(encoders)
        encoders = Encoders(encoders=encoders)
    elif cfg.dataset == "mystery_mml":
        output_dim = 1
        n_modalities = 2

        encoders = Encoders(
            encoders = nn.ModuleList(
                [
                    Encoder_MysteryMML_Tabular(
                        latent_dim=cfg.modelname.head_transformer.d_model,
                    ),
                    Encoder_MysteryMML_Tabular(
                        latent_dim=cfg.modelname.head_transformer.d_model,
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
                "score_model": getattr(cfg.modelname.imder, "score_model", "dit"),
            }
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
    elif cfg.modelname.modelname == "aug":

        multimodal_transformer = AUG_Transformer(
            d_model=cfg.modelname.head_transformer.d_model,
            nhead=cfg.modelname.head_transformer.nhead,
            dim_feedforward=cfg.modelname.head_transformer.dim_feedforward,
            dropout=cfg.modelname.head_transformer.dropout,
            num_layers=cfg.modelname.head_transformer.num_layers,
            dim_output=output_dim,
            num_modalities=n_modalities,
            task_type=get_task_type(cfg),
            merge_alphas=cfg.modelname.aug.merge_alphas,
            lambda_smooth=cfg.modelname.aug.lambda_smooth
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
    else:
        raise NotImplementedError("Model not implemented")

    model = Multimodal_Architecture(
        encoders=encoders,
        transformer=multimodal_transformer,
        full_seq=cfg.modelname.pipeline.full_seq,
        unimodal=cfg.unimodal,
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
    if "omib" in cfg.modelname:
        params_omib = {
            "warmup_epochs": cfg.modelname.omib.warmup_epochs,
        }
    else:
        params_omib = {}
    if "ogm" in cfg.modelname:
        params_ogm = {
            "use_ge": cfg.modelname.ogm.use_ge,
        }
    else:
        params_ogm = {}
    if "aug" in cfg.modelname:
        params_aug = {
            "check_interval": cfg.modelname.aug.check_interval,
            "threshold": cfg.modelname.aug.threshold,
            "confidence_coeff": cfg.modelname.aug.confidence_coeff,
        }
    else:
        params_aug = {}

    if cfg.dataset == "mimic_symile" or cfg.dataset == "mimic_haim":
        lightningmodule = MIMIC_Lightning_Module(
            model=model,
            dataset=cfg.dataset,
            params_optimizer={key: value for key, value in cfg.modelname.optimizer.items()},
            manual_opt=manual_opt,
            params_ogm=params_ogm,
            params_arl=params_arl,
            params_dgl=params_dgl,
            params_mmpareto=params_mmpareto,
            params_bmml=params_bmml,
            params_gblend=params_gblend,
            params_pdf=params_pdf,
            params_omib=params_omib,
            params_aug=params_aug,
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
            params_mmpareto=params_mmpareto,
            params_bmml=params_bmml,
            params_gblend=params_gblend,
            params_pdf=params_pdf,
            params_omib=params_omib,
            params_aug=params_aug,
        )
    elif cfg.dataset == "crema_d":
        lightningmodule = CREMAD_Lightning_Module(
            model=model,
            dataset=cfg.dataset,
            params_optimizer={key: value for key, value in cfg.modelname.optimizer.items()},
            manual_opt=manual_opt,
            params_ogm=params_ogm,
            params_arl=params_arl,
            params_dgl=params_dgl,
            params_mmpareto=params_mmpareto,
            params_bmml=params_bmml,
            params_gblend=params_gblend,
            params_pdf=params_pdf,
            params_omib=params_omib,
            params_aug=params_aug,
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
            params_mmpareto=params_mmpareto,    
            params_bmml=params_bmml,
            params_gblend=params_gblend,
            params_pdf=params_pdf,
            params_omib=params_omib,
            params_aug=params_aug,
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
            params_mmpareto=params_mmpareto,
            params_bmml=params_bmml,
            params_gblend=params_gblend,
            params_pdf=params_pdf,
            params_omib=params_omib,
            params_aug=params_aug,
        )
    elif cfg.dataset == "ukb":
        lightningmodule = UKB_Lightning_Module(
            model=model,
            dataset=cfg.dataset,
            params_optimizer={key: value for key, value in cfg.modelname.optimizer.items()},
            manual_opt=manual_opt,
            params_ogm=params_ogm,
            params_arl=params_arl,
            params_dgl=params_dgl,
            params_mmpareto=params_mmpareto,
            params_bmml=params_bmml,
            params_gblend=params_gblend,
            params_pdf=params_pdf,
            params_omib=params_omib,
            params_aug=params_aug,
        )
    elif cfg.dataset == "mystery_mml":
        lightningmodule = MysteryMML_Lightning_Module(
            model=model,
            dataset=cfg.dataset,
            params_optimizer={key: value for key, value in cfg.modelname.optimizer.items()},
            manual_opt=manual_opt,
            params_ogm=params_ogm,
            params_arl=params_arl,
            params_dgl=params_dgl,
            params_mmpareto=params_mmpareto,
            params_bmml=params_bmml,
            params_gblend=params_gblend,
            params_pdf=params_pdf,
            params_omib=params_omib,
            params_aug=params_aug,
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
    elif cfg.dataset == "crema_d":
        # if corrupted_data_protocol in cfg.modelname 
        if "corrupted_data_protocol" in cfg.modelname:
            print(f"Warning: Train and Validation Set are concatenated, only test set for validation and test.")
            datamodule_params["corrupted_data_protocol"] = cfg.modelname.corrupted_data_protocol
        
        if "use_embeddings" in cfg.encoders:
            datamodule_params["use_embeddings"] = cfg.encoders.use_embeddings
        
        datamodule = CREMAD_Datamodule(**datamodule_params)
    elif cfg.dataset == "ch_sims" or cfg.dataset == "ch_sims_v2":
        datamodule_params["v2"] = True if cfg.dataset == "ch_sims_v2" else False
        datamodule = CH_Sims_Datamodule(**datamodule_params)
    elif cfg.dataset == "inspect":
        datamodule = INSPECT_Datamodule(**datamodule_params)
    elif cfg.dataset == "ukb":
        # lazy import if private UDM not installed 
        from udm.general_datamodule import GeneralDatamodule
        datamodule_params = {key: value for key, value in cfg.datamodule.items()}
        datamodule = GeneralDatamodule(**datamodule_params)
    elif cfg.dataset == "mystery_mml":
        datamodule = MysteryMML_Datamodule(**datamodule_params)
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
