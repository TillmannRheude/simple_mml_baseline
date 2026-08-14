"""Tabular modality encoders used by SimBaMM on UK Biobank."""

from __future__ import annotations

import torch
from torch import nn


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class TabularEncoder(nn.Module):
    """Map one UKB feature vector to one modality token."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        hidden_dropouts: list[float],
    ) -> None:
        super().__init__()
        if len(hidden_dims) != len(hidden_dropouts):
            raise ValueError("hidden_dims and hidden_dropouts must have equal length.")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim, dropout in zip(hidden_dims, hidden_dropouts):
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)
        self.apply(_init_weights)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features).unsqueeze(1)


class ModalityEncoders(nn.Module):
    """Apply one independent tabular encoder per configured modality."""

    def __init__(self, encoders: list[TabularEncoder]) -> None:
        super().__init__()
        if not encoders:
            raise ValueError("At least one modality encoder is required.")
        self.encoders = nn.ModuleList(encoders)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != len(self.encoders):
            raise ValueError(
                f"Expected {len(self.encoders)} modalities, received {len(inputs)}."
            )
        return torch.cat(
            [encoder(features) for encoder, features in zip(self.encoders, inputs)],
            dim=1,
        )
