import math
import torch 
import torch.nn as nn 

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE

class Balanced_Multimodal_Transformer(nn.Module):

    """
    https://proceedings.mlr.press/v162/wu22d/wu22d.pdf

    https://github.com/nyukat/greedy_multimodal_learning 
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
        bmml_momentum: float = 0.9,
    ) -> None: 
        super().__init__()
        self.num_modalities = num_modalities
        self.d_model = d_model
        self.bmml_momentum = bmml_momentum
        self.linear_out = nn.Linear(d_model, dim_output)

        self.apply(self._init_weights)

        # Buffer to store running average of representations for re-balancing
        self.register_buffer('running_avg_reps', torch.zeros(num_modalities, 1, d_model))
        self.register_buffer('is_initialized', torch.tensor(False))

        self.bmml_rebalancing_mode = 'none'

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
        
        # Split concatenated input into per-modality representations
        split_size = x.shape[1] // self.num_modalities
        unimodal_reps = list(torch.split(x, split_size, dim=1))
        
        # Calculate unimodal logits from original representations for loss calculation
        unimodal_logits = [self.linear_out(rep.mean(dim=1)) for rep in unimodal_reps]

        reps_for_fusion = unimodal_reps
        if self.training:
            # Init running average buffer
            if not self.is_initialized:
                self.running_avg_reps = torch.stack(unimodal_reps, dim=0).mean(dim=1).detach()
                self.is_initialized = torch.tensor(True)

            if self.bmml_rebalancing_mode == 'none':
                # Regular step: update the running averages using EMA
                current_reps_stacked = torch.stack(unimodal_reps, dim=0).mean(dim=1).detach()
                self.running_avg_reps.data = self.bmml_momentum * self.running_avg_reps.data + (1 - self.bmml_momentum) * current_reps_stacked
            else:
                # Re-balancing step: substitute representations with the stored average
                boost_idx = int(self.bmml_rebalancing_mode[1:])
                temp_reps = []
                for i in range(self.num_modalities):
                    if i == boost_idx:
                        temp_reps.append(unimodal_reps[i]) # Use live representation
                    else:
                        # Use stored average, expanded to batch size
                        avg_rep = self.running_avg_reps[i].unsqueeze(0).expand_as(unimodal_reps[i])
                        temp_reps.append(avg_rep)
                reps_for_fusion = temp_reps
        
        x_fused = torch.cat(reps_for_fusion, dim=1)

        for layer in self.transformer_cls:
            if isinstance(layer, nn.TransformerEncoder) and src_mask is not None:
                src_mask = self._add_cls_token_mask_to_src_mask(src_mask)
                x_fused = layer(x_fused, src_key_padding_mask=src_mask)
            else:
                x_fused = layer(x_fused)

        return {
            "logits": x_fused,
            "unimodal_logits": unimodal_logits,
            "bmml": True 
        }
