import math
import torch 
import torch.nn as nn 
import numpy as np
import warnings

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE

class Modality_Mixup_Transformer(nn.Module):

    """
    https://arxiv.org/pdf/2209.02604
    
    https://github.com/thuiar/ch-sims-v2 
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_layers: int = 4,
        num_modalities: int = 2,
        dim_output: int = 10,
        mixup_alpha: float = 0.2,
        modalities_to_mix: list[int] = None,
        consistency_type: str = "mse"
    ) -> None: 
        super().__init__()
        self.num_modalities = num_modalities
        self.mixup_alpha = mixup_alpha
        self.d_model = d_model
        self.modalities_to_mix = modalities_to_mix if modalities_to_mix is not None else []
        self.consistency_type = consistency_type

        self.fused_classifier = nn.Linear(d_model, dim_output)

        self.unimodal_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, dim_output)
            )
            for _ in range(num_modalities)
        ])

        self.apply(self._init_weights)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers
        )

        self.transformer_cls_sequence = nn.ModuleList([
            AddCLSToken(d_model),
            AddPE(d_model),
            self.transformer,
            ExtractCLSToken(),
            self.fused_classifier
        ])
        
    def _init_weights(
            self,
            m
        ) -> None: 
        if isinstance(m, (torch.nn.LayerNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def _add_cls_token_mask_to_src_mask(
            self, 
            src_mask: torch.Tensor
    ) -> torch.Tensor:
        assert src_mask.dtype == torch.bool
        src_mask = torch.cat(
            [
                torch.zeros(src_mask.shape[0], 1, dtype=torch.bool, device=src_mask.device), 
                src_mask
            ], dim=1
        ).to(dtype=torch.bool)
        return src_mask 

    def forward(
        self, 
        x: torch.Tensor, # Shape: (batch, n_modalities, d_model)
        src_mask: torch.Tensor,
        y: torch.Tensor,
    ) -> dict:
        
        # --- AV-MC Change: Get original predictions for creating consistency targets ---
        unimodal_logits_orig = {}
        with torch.no_grad(): # No gradients needed for the "teacher" path
            for i in range(self.num_modalities):
                if i in self.modalities_to_mix:
                    mod_key = str(i)
                    rep_i = x[:, i, :]
                    unimodal_logits_orig[mod_key] = self.unimodal_classifiers[i](rep_i)

        x_for_processing = x
        target_for_loss = y
        consistency_targets = {}

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        shuffled_indices = torch.randperm(x.size(0), device=x.device)

        # 1. Create mixed target for the main supervised loss (standard mixup).
        target_for_loss = lam * y.float() + (1 - lam) * y[shuffled_indices].float()

        # --- AV-MC Change: Create consistency targets from original predictions ---
        for i in range(self.num_modalities):
            if i in self.modalities_to_mix:
                mod_key = str(i)
                original_logits = unimodal_logits_orig[mod_key]
                
                if self.consistency_type == 'kldiv':
                    # For KL-divergence, we mix probabilities
                    original_probs = torch.softmax(original_logits, dim=-1)
                    consistency_targets[mod_key] = lam * original_probs + (1 - lam) * original_probs[shuffled_indices]
                else:
                    # For MSE or L1, we mix the logits directly
                    consistency_targets[mod_key] = lam * original_logits + (1 - lam) * original_logits[shuffled_indices]

        # 2. Create mixed input representations.
        x_mixed_list = []
        for i in range(self.num_modalities):
            rep_i = x[:, i, :]
            if i in self.modalities_to_mix:
                x_mixed_list.append(lam * rep_i + (1 - lam) * rep_i[shuffled_indices])
            else:
                x_mixed_list.append(rep_i)
        x_for_processing = torch.stack(x_mixed_list, dim=1)

        # 3. Get all unimodal logits from the (potentially mixed) representations
        unimodal_logits = {}
        for i in range(self.num_modalities):
            mod_key = str(i)
            rep_i = x_for_processing[:, i, :]
            unimodal_logits[mod_key] = self.unimodal_classifiers[i](rep_i)

        # 4. Fuse the (potentially mixed) representations for the final multimodal prediction
        fused_x = x_for_processing
        for layer in self.transformer_cls_sequence:
            if isinstance(layer, nn.TransformerEncoder) and src_mask is not None:
                src_mask = self._add_cls_token_mask_to_src_mask(src_mask)
                fused_x = layer(fused_x, src_key_padding_mask=src_mask)
            else:
                fused_x = layer(fused_x)

        return {
            "logits": fused_x,
            "unimodal_logits": unimodal_logits,
            "target": target_for_loss,
            "consistency_targets": consistency_targets
        }
