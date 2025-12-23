import math
import torch 
import torch.nn as nn 

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE
from codefiles.methods.utils import mimetic_init_svd_

class GBlend_Transformer(nn.Module):

    """
    https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.pdf
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_layers: int = 4,
        dim_output: int = 10,
        num_modalities: int = 2
    ) -> None: 
        super().__init__()

        # G-Blend State
        # -1 = fused head (default), 0, 1, ... = unimodal heads
        self.gblend_active_head = -1
        self.num_modalities = num_modalities

        # Heads
        self.unimodal_classifiers = nn.ModuleList(
            [nn.Linear(d_model, dim_output) for _ in range(self.num_modalities)]
        )
        self.linear_out = nn.Linear(d_model, dim_output)

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
        
    def set_gblend_head_active(self, head_idx: int):
        """
        This is the command called by the LightningModule's on_fit_start hook
        to isolate a specific head for the lookahead calculation.
        """
        self.gblend_active_head = head_idx

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
        
        split_size = x.shape[1] // self.num_modalities
        x_list = list(torch.split(x, split_size, dim=1))
        
        if self.gblend_active_head >= 0:
            # Lookahead mode: a specific unimodal head is active.
            active_rep = x_list[self.gblend_active_head]
            unimodal_logits = self.unimodal_classifiers[self.gblend_active_head](active_rep)
            return {"logits": unimodal_logits}

        # Default mode (gblend_active_head == -1): main training step
        # Here, x_list is also expected to be raw encoder outputs.
        unimodal_logits_list = [self.unimodal_classifiers[i](x_list[i]) for i in range(self.num_modalities)]

        # Stack representations for the fusion transformer.
        x = torch.cat(x_list, dim=1)
        
        for layer in self.transformer_cls:
            if isinstance(layer, nn.TransformerEncoder) and src_mask is not None:
                src_mask = self._add_cls_token_mask_to_src_mask(src_mask)
                x = layer(x, src_key_padding_mask=src_mask)
            else:
                x = layer(x)

        return {
            "logits": x,
            "unimodal_logits": unimodal_logits_list,
            "gblend": True 
        }
