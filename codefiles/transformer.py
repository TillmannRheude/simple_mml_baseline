import math
import torch 
import torch.nn as nn 

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE

class Multimodal_Transformer(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_layers: int = 4,
        dim_output: int = 10,
    ) -> None: 
        super().__init__()

        self.linear_out = nn.Linear(d_model, dim_output)
        self.apply(self._init_weights)

        # Transformer Head 
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

    def forward(
        self, 
        x: torch.Tensor = torch.Tensor,
        src_mask: torch.Tensor = torch.Tensor,
        y: torch.Tensor = None
    ) -> dict:

        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        if isinstance(src_mask, list):
            src_mask = torch.cat(src_mask, dim=1)
        
        for layer in self.transformer_cls:
            if isinstance(layer, nn.TransformerEncoder) and src_mask is not None:
                src_mask = self._add_cls_token_mask_to_src_mask(src_mask)
                x = layer(x, src_key_padding_mask=src_mask)
            else:
                x = layer(x)
        
        return {
            "logits": x
        }
