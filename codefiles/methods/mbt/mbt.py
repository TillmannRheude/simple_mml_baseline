import math
import torch 
import torch.nn as nn 

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE
from codefiles.methods.utils import mimetic_init_svd_


class BottleneckFusionLayer(nn.Module):
    """
    A single layer that performs bi-directional fusion via bottlenecks, followed
    by unimodal self-attention and FFN blocks.
    https://github.com/NMS05/Multimodal-Fusion-with-Attention-Bottlenecks
    """
    def __init__(
            self, 
            d_model: int, 
            nhead: int, 
            dim_feedforward: int, 
            dropout: float, 
            num_bottlenecks: int, 
            num_modalities: int
    ) -> None:
        super().__init__()

        self.num_modalities = num_modalities

        self.fusion_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_modalities)])
        self.apply(self._init_weights)

        # Fusion parameters
        self.latents = nn.Parameter(torch.randn(1, num_bottlenecks, d_model))
        self.fusion_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.latent_to_mod_attns = nn.ModuleList([
             nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True) for _ in range(num_modalities)
        ])
        self.scales = nn.Parameter(torch.zeros(num_modalities))

        # Unimodal processing blocks
        self.unimodal_encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True,
            ) for _ in range(num_modalities)
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

    def forward(
        self, 
        modalities: list[torch.Tensor], 
        masks: list[torch.Tensor] = None
    ) -> list[torch.Tensor]:
        bs = modalities[0].size(0)
        
        # Modalities -> Latents
        full_context = torch.cat(modalities, dim=1)
        full_mask = torch.cat(masks, dim=1) if masks is not None else None
        
        latents_q = self.latents.repeat(bs, 1, 1)
        fused_latents, _ = self.fusion_attn(latents_q, full_context, full_context, key_padding_mask=full_mask)

        # Latents -> Modalities
        updated_modalities = []
        for i, mod in enumerate(modalities):
            norm_mod = self.fusion_norms[i](mod)
            update, _ = self.latent_to_mod_attns[i](norm_mod, fused_latents, fused_latents)
            updated_modalities.append(mod + self.scales[i] * update)

        # Unimodal Processing
        processed_modalities = []
        for i, mod in enumerate(updated_modalities):
            mask = masks[i] if masks is not None else None
            processed_modalities.append(self.unimodal_encoders[i](mod, src_key_padding_mask=mask))
        
        return processed_modalities


class Multimodal_Bottleneck_Transformer(nn.Module):

    """
    https://proceedings.neurips.cc/paper_files/paper/2021/file/76ba9f564ebbc35b1014ac498fafadd0-Paper.pdf
    
    https://github.com/NMS05/Multimodal-Fusion-with-Attention-Bottlenecks
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
        mbt_params: dict = {
            "num_bottlenecks": 4,
            "num_layers_mbt": 4,
            "dropout": 0.0,
            "nhead": 4,
            "dim_feedforward": 1024
        }
    ) -> None: 
        super().__init__()
        self.num_modalities = num_modalities

        self.add_cls_token = AddCLSToken(d_model)
        self.extract_cls_token = ExtractCLSToken()
        self.add_pe = AddPE(d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.linear_out = nn.Linear(d_model, dim_output)
        self.apply(self._init_weights)
        
        self.layers = nn.ModuleList([
            BottleneckFusionLayer(
                d_model, 
                mbt_params["nhead"], 
                mbt_params["dim_feedforward"], 
                mbt_params["dropout"], 
                mbt_params["num_bottlenecks"], 
                num_modalities
            )
            for _ in range(mbt_params["num_layers_mbt"])
        ])

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
        x: torch.Tensor,
        src_mask: torch.Tensor = None,
        y: torch.Tensor = None
    ) -> dict:
        
        # Split modalities and masks
        split_size = x.shape[1] // self.num_modalities
        modalities = list(torch.split(x, split_size, dim=1))
        masks = list(torch.split(src_mask, split_size, dim=1)) if src_mask is not None else [None] * self.num_modalities

        # CLS token to each modality and pos encodings
        for i in range(self.num_modalities):
            modalities[i] = self.add_cls_token(modalities[i])
            if masks[i] is not None:
                cls_mask = torch.zeros(src_mask.shape[0], 1, dtype=torch.bool, device=src_mask.device)
                masks[i] = torch.cat([cls_mask, masks[i]], dim=1)
            modalities[i] = self.add_pe(modalities[i])

        # Fusion layers
        for layer in self.layers:
            modalities = layer(modalities, masks)

        # CLS token from each modality + Normalize and get logits for each CLS token
        cls_tokens = [self.extract_cls_token(mod) for mod in modalities]
        cls_tokens = torch.stack(cls_tokens, dim=1)  # (bs, num_modalities, d_model)
        cls_tokens = self.norm(cls_tokens)

        x = cls_tokens
        
        # original paper:
        #logits_per_modality = self.linear_out(cls_tokens)  # (bs, num_modalities, dim_output)
        #x = torch.mean(logits_per_modality, dim=1)

        # transformer head:
        for layer in self.transformer_cls:
            if isinstance(layer, nn.TransformerEncoder) and src_mask is not None:
                src_mask = self._add_cls_token_mask_to_src_mask(src_mask)
                x = layer(x, src_key_padding_mask=src_mask)
            else:
                x = layer(x)

        return {
            "logits": x
        }
