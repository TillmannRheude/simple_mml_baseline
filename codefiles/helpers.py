"""Factories for the UKB-only SimBaMM training pipeline."""

from __future__ import annotations

import random

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from codefiles.architecture import SimBaMM
from codefiles.encoders import ModalityEncoders, TabularEncoder
from codefiles.lightningmodules.ukb import UKBLightningModule
from codefiles.transformer import SimBaMMTransformer


def get_modality_names(cfg: DictConfig) -> list[str]:
    """Return modalities in the same Hydra order used by the UKB datamodule."""
    return [name for name in cfg.data.plugins if name != "labels"]


def build_model(cfg: DictConfig) -> SimBaMM:
    modality_names = get_modality_names(cfg)
    input_dims = cfg.encoder.input_dims
    unknown = [name for name in modality_names if name not in input_dims]
    if unknown:
        raise ValueError(f"Missing encoder input dimensions for: {', '.join(unknown)}")

    encoders = ModalityEncoders(
        [
            TabularEncoder(
                input_dim=input_dims[name],
                latent_dim=cfg.model.transformer.d_model,
                hidden_dims=list(cfg.encoder.hidden_dims),
                hidden_dropouts=list(cfg.encoder.hidden_dropouts),
            )
            for name in modality_names
        ]
    )
    transformer = SimBaMMTransformer(
        d_model=cfg.model.transformer.d_model,
        nhead=cfg.model.transformer.nhead,
        dim_feedforward=cfg.model.transformer.dim_feedforward,
        dropout=cfg.model.transformer.dropout,
        num_layers=cfg.model.transformer.num_layers,
        output_dim=1,
    )
    return SimBaMM(encoders=encoders, transformer=transformer)


def build_lightning_module(cfg: DictConfig, model: SimBaMM) -> UKBLightningModule:
    return UKBLightningModule(
        model=model,
        modality_names=get_modality_names(cfg),
        optimizer_config=dict(cfg.model.optimizer),
    )


def build_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
    # UDM is an internal UKB dependency, so importing it lazily keeps model-only use possible.
    from udm.general_datamodule import GeneralDatamodule

    return GeneralDatamodule(**dict(cfg.data))


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
