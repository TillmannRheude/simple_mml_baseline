import math
import torch 
import torch.nn as nn 

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE
from codefiles.methods.utils import mimetic_init_svd_

class MCR_Transformer(nn.Module):
    """
    https://arxiv.org/pdf/2411.07335

    No codebase available, yet.
    "Due to the extensive list of hyperparameters, we will provide detailed configurations
    for each experiment in our GitHub repository. The repository link will be included here following
    the double-blind review process."
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_layers: int = 4,
        num_modalities: int = 2,
        dim_output: int = 10
    ) -> None: 
        super().__init__()
        self.num_modalities = num_modalities
        
        self.linear_out = nn.Linear(d_model, dim_output)
        self.unimodal_heads = nn.ModuleList([
            nn.Linear(d_model, dim_output) for _ in range(num_modalities)
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

        self.transformer_cls = nn.ModuleList([
            AddCLSToken(d_model),
            AddPE(d_model),
            self.transformer,
            ExtractCLSToken(),
            self.linear_out
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

    def fusion(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        fused_x = x
        for layer in self.transformer_cls:
            if isinstance(layer, nn.TransformerEncoder) and src_mask is not None:
                current_mask = self._add_cls_token_mask_to_src_mask(src_mask)
                fused_x = layer(fused_x, src_key_padding_mask=current_mask)
            else:
                fused_x = layer(fused_x)
        return fused_x

    def forward(
        self, 
        x: torch.Tensor,
        src_mask: torch.Tensor = None,
        y: torch.Tensor = None
    ) -> dict:
        
        split_size = x.shape[1] // self.num_modalities
        unimodal_reps = list(torch.split(x, split_size, dim=1))

        fused_x = self.fusion(x, src_mask)

        unimodal_logits = []
        for i in range(self.num_modalities):
            summary_rep = torch.mean(unimodal_reps[i], dim=1)
            unimodal_logits.append(self.unimodal_heads[i](summary_rep))

        return {
            "logits": fused_x,
            "unimodal_reps": unimodal_reps, # e.g., [z1, z2]
            "unimodal_logits": unimodal_logits,
        }
