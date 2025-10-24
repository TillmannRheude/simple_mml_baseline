import math
import torch 
import torch.nn as nn 

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE
from codefiles.methods.utils import mimetic_init_svd_

class Low_Rank_Matrix_Fusion_Transformer(nn.Module):

    """ 
    https://arxiv.org/abs/1806.00064
    
    https://github.com/Justin1904/Low-rank-Multimodal-Fusion
    
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_layers: int = 4,
        dim_output: int = 10,
        num_modalities: int = 3,
        rank: int = 4
    ) -> None: 
        super().__init__()

        self.d_model = d_model
        self.num_modalities = num_modalities
        self.rank = rank

        self.modality_factors = nn.ParameterList([
            nn.Parameter(torch.empty(self.rank, self.d_model + 1, self.d_model))
            for _ in range(self.num_modalities)
        ])
        self.fusion_weights = nn.Parameter(torch.empty(1, 1, self.rank, 1))
        self.fusion_bias = nn.Parameter(torch.empty(1, 1, self.d_model))

        self.linear_out = nn.Linear(d_model, dim_output)

        self.fusion_layernorm = nn.LayerNorm(d_model)

        self.apply(self._init_weights)

        for factor in self.modality_factors:
            nn.init.kaiming_uniform_(factor, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fusion_weights, a=math.sqrt(5))
        nn.init.zeros_(self.fusion_bias)

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

    def _low_rank_fusion(self, unimodal_features: list[torch.Tensor]) -> torch.Tensor:
        bs = unimodal_features[0].shape[0]
        
        unimodal_features_with_ones = []
        for feature in unimodal_features:
            ones = torch.ones(bs, feature.shape[1], 1, device=feature.device)
            unimodal_features_with_ones.append(torch.cat([feature, ones], dim=2))

        projected_features = []
        for i in range(self.num_modalities):
            # unimodal_features_with_ones[i]: (batch, seq_len, d_model+1)
            # self.modality_factors[i]:       (rank, d_model+1, d_model)
            proj = torch.einsum('bsd,rdh->bsrh', unimodal_features_with_ones[i], self.modality_factors[i])
            projected_features.append(proj)

        fused_projection = torch.ones_like(projected_features[0])
        for proj in projected_features:
            fused_projection = fused_projection * proj
        
        fused_representation = (fused_projection * self.fusion_weights).sum(dim=2)

        fused_representation = fused_representation + self.fusion_bias

        fused_representation = self.fusion_layernorm(fused_representation)

        return fused_representation
    
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
        unimodal_features = list(torch.split(x, split_size, dim=1))

        if src_mask is not None:
            unimodal_masks = list(torch.split(src_mask, split_size, dim=1))
            src_mask = torch.stack(unimodal_masks, dim=0).any(dim=0)

        # low-rank fusion
        x = self._low_rank_fusion(unimodal_features)
        
        for layer in self.transformer_cls:
            if isinstance(layer, nn.TransformerEncoder) and src_mask is not None:
                src_mask = self._add_cls_token_mask_to_src_mask(src_mask)
                x = layer(x, src_key_padding_mask=src_mask)
            else:
                x = layer(x)

        return {
            "logits": x
        }
