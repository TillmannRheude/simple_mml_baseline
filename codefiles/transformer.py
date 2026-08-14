"""Transformer fusion head for SimBaMM."""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(1, max_len, d_model)
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.encoding.shape[1]:
            raise ValueError("Token sequence is longer than the positional encoding.")
        return tokens + self.encoding[:, : tokens.shape[1]]


class SimBaMMTransformer(nn.Module):
    """Fuse modality tokens and predict mortality from a learned CLS token."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, output_dim)
        # Match the paper implementation: PyTorch initializes the Transformer,
        # while the prediction head uses the shared Kaiming initialization.
        self.output.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        modality_tokens: torch.Tensor,
        missing_mask: torch.Tensor,
    ) -> torch.Tensor:
        cls_token = self.cls_token.expand(modality_tokens.shape[0], -1, -1)
        tokens = self.positional_encoding(torch.cat([cls_token, modality_tokens], dim=1))
        cls_mask = torch.zeros(
            missing_mask.shape[0], 1, dtype=torch.bool, device=missing_mask.device
        )
        padding_mask = torch.cat([cls_mask, missing_mask], dim=1)
        fused = self.transformer(tokens, src_key_padding_mask=padding_mask)
        return self.output(fused[:, 0])
