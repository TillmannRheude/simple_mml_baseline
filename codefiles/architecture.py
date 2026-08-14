"""SimBaMM: a simple late-fusion Transformer for multimodal UKB data."""

from __future__ import annotations

import torch
from torch import nn

from codefiles.encoders import ModalityEncoders
from codefiles.transformer import SimBaMMTransformer


class SimBaMM(nn.Module):
    """Encode each UKB modality independently and fuse its token with a Transformer."""

    def __init__(
        self,
        encoders: ModalityEncoders,
        transformer: SimBaMMTransformer,
    ) -> None:
        super().__init__()
        self.encoders = encoders
        self.transformer = transformer

    @staticmethod
    def _missing_modality_mask(inputs: list[torch.Tensor]) -> torch.Tensor:
        if not inputs:
            raise ValueError("SimBaMM requires at least one modality.")

        batch_size = inputs[0].shape[0]
        if any(tensor.shape[0] != batch_size for tensor in inputs):
            raise ValueError("All modalities must have the same batch size.")

        # True means that every feature for this modality is missing.
        return torch.stack(
            [torch.isnan(tensor).reshape(batch_size, -1).all(dim=1) for tensor in inputs],
            dim=1,
        )

    def forward(self, inputs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        missing_mask = self._missing_modality_mask(inputs)
        inputs = [torch.nan_to_num(tensor, nan=0.0) for tensor in inputs]
        modality_tokens = self.encoders(inputs)
        return {"logits": self.transformer(modality_tokens, missing_mask)}
