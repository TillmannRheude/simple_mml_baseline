import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from jaxtyping import Bool, Float
from typing import List, Dict, Any, Union, Callable, Optional
from torch import Tensor 

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10) / (half_dim - 1)  # 
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Diffusion_Transformer(nn.Module):
    def __init__(
        self, 
        input_dim: int = 256, 
        hidden_dim: int = 512, 
        output_dim: int = 256, 
        condition_dim: int = 256, 
        dropout: float = 0.1,
        nhead: int = 1,
        num_layers: int = 10
    ) -> None: 
        super().__init__()
        self.seq_len_x = None # will be set in imder.py

        # init projections
        self.init_proj_x = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
        )
        self.init_proj_c = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim)
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # final projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

        # Initialize weights
        self.apply(self._init_weights)  # Transformer + MLPs
        self.time_mlp.apply(self._init_time_mlp_weights)  # Special init for time MLP
        self.out_proj.apply(self._init_out_proj_weights)  # Ensure output starts at zero

        # Use the custom TransformerEncoderLayer
        dit = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=nhead, 
                dim_feedforward=512,
                batch_first=True,
                dropout=dropout,
            ), 
            num_layers=num_layers
        )

        self.dit = nn.ModuleList([
            #AddCLSToken(hidden_dim),
            AddPE(hidden_dim),
            dit,
            #ExtractCLSToken()
        ])

        
    def _init_weights(self, module):
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.constant_(module.weight, 0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.weight, 0)
            torch.nn.init.constant_(module.bias, 0)
        """ 
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)
        
    
    def _init_time_mlp_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #nn.init.constant_(module.weight, 0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def _init_out_proj_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.constant_(module.weight, 0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def _add_x_t_tokens_to_src_mask(
        self,
        src_mask: torch.Tensor,
        seq_len_x: int
    ) -> torch.Tensor: 
        assert src_mask.dtype == torch.bool
        mask_for_x = torch.zeros(src_mask.shape[0], seq_len_x, dtype=torch.bool, device=src_mask.device)
        mask_for_t = torch.zeros(src_mask.shape[0], 1, dtype=torch.bool, device=src_mask.device)
        src_mask = torch.cat(
            [
                mask_for_x,
                mask_for_t,
                src_mask
            ], dim=1
        ).to(dtype=torch.bool)
        return src_mask 

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
        x: Float[Tensor, "batch seq embdim"], 
        t: Float[Tensor, "batch embdim"],
        condition: Float[Tensor, "batch seq embdim"], 
        src_mask: Float[Tensor, "batch seq"] = None
    ) -> Float[Tensor, "batch seq embdim"]:

        # Projections (time, conditioning, noisy latent)
        t_proj = self.time_mlp(t).unsqueeze(1)
        c_proj = self.init_proj_c(condition)
        x_proj = self.init_proj_x(x)

        seq_len_x = x.shape[1]

        # DiT input incl. src_mask
        x = torch.cat([x_proj, t_proj, c_proj], dim=1)
        if src_mask is not None:
            src_mask = self._add_x_t_tokens_to_src_mask(src_mask, seq_len_x)
        
        # DiT processing 
        for layer in self.dit:
            if isinstance(layer, nn.TransformerEncoder) and src_mask is not None:
                x = layer(
                    x, 
                    src_key_padding_mask=src_mask
                )
            else:
                x = layer(x)
        
        # Projection out
        x = self.out_proj(x)

        # Extract the part of the output that corresponds to the original x
        x = x[:, :seq_len_x, :]

        return x

