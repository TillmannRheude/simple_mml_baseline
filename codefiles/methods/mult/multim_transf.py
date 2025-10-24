import torch 
import math 
import torch.nn.functional as F
import torch.nn as nn

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE
from codefiles.methods.utils import mimetic_init_svd_


class MultiHeadCrossmodalAttention(nn.Module):

    def __init__(
            self, 
            params_transformer: dict = {
                "d_model": 512,
                "nhead": 4,
                "dim_feedforward": 1024,
                "dropout": 0.0,
                "num_layers": 4,
                "dim_output": 10,
            },
        ) -> None: 
        super().__init__()
        
        d_model = params_transformer["d_model"]
        nhead = params_transformer["nhead"]
        dropout = params_transformer["dropout"]
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.apply(self._init_weights)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
    
    def _init_weights(
            self,
            m
        ) -> None: 
        if isinstance(m, (torch.nn.LayerNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Parameter):
            if m.dim() > 1:
                torch.nn.init.xavier_normal_(m)
            else:
                torch.nn.init.zeros
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(
            self, 
            query_input, 
            kv_input, 
            mask=None
        ) -> tuple:
        residual = query_input

        x, attn = self.attn(
            query=query_input, 
            key=kv_input, 
            value=kv_input, 
            attn_mask=mask, 
            need_weights=True
        )
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x, attn
    

class PositionwiseFeedForwardWithNorm(nn.Module):
    def __init__(
            self, 
            params_ffn: dict = {
                "d_model": 512,
                "dim_feedforward": 1024,
                "dropout": 0.0,
                "dim_output": 10,
            },
        ) -> None: 
        super().__init__()

        self.w_1 = nn.Linear(params_ffn["d_model"], params_ffn["dim_feedforward"])
        self.w_2 = nn.Linear(params_ffn["dim_feedforward"], params_ffn["d_model"])
        self.dropout = nn.Dropout(params_ffn["dropout"])
        self.layer_norm = nn.LayerNorm(params_ffn["d_model"])

        self.apply(self._init_weights)

    def _init_weights(
            self,
            m
        ) -> None: 
        if isinstance(m, (torch.nn.LayerNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Parameter):
            if m.dim() > 1:
                torch.nn.init.xavier_normal_(m)
            else:
                torch.nn.init.zeros
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        residual = x
        x = self.w_2(self.dropout(F.relu(self.w_1(x))))
        x += residual
        x = self.layer_norm(x)
        return x


class CrossmodalTransformerBlock(nn.Module):
    def __init__(
            self, 
            params_transformer: dict = {
                "d_model": 512,
                "nhead": 4,
                "dim_feedforward": 1024,
                "dropout": 0.0,
                "num_layers": 4,
                "dim_output": 10,
                "d_k": 128,
                "d_v": 128,
            },
        ) -> None: 
        super().__init__()

        self.cross_attn = MultiHeadCrossmodalAttention(params_transformer)
        self.pos_ffn = PositionwiseFeedForwardWithNorm(params_transformer)

    def forward(self, target_modality_input, source_modality_input_original, mask=None):
        cross_output, attn_weights = self.cross_attn(target_modality_input, source_modality_input_original, mask=mask)
        output = self.pos_ffn(cross_output)
        return output, attn_weights


class CrossmodalTransformer(nn.Module):
    
    def __init__(
            self, 
            params_transformer: dict = {
                "d_model": 512,
                "nhead": 4,
                "dim_feedforward": 1024,
                "dropout": 0.0,
                "num_layers": 4,
                "dim_output": 10,
                "d_k": 128,
                "d_v": 128,
            },
        ) -> None: 
        super().__init__()

        self.num_layers = params_transformer["num_layers"]
        self.layers = nn.ModuleList([
            CrossmodalTransformerBlock(params_transformer)
            for _ in range(params_transformer["num_layers"])
        ])

    def forward(self, target_input, source_input_original, mask=None):
        output = target_input
        all_attn_weights = []
        for i in range(self.num_layers):
            output, attn_weights = self.layers[i](output, source_input_original, mask=mask)
            all_attn_weights.append(attn_weights)
        return output, all_attn_weights
    

class CrossAttentionTransformers(nn.Module):

    def __init__(
            self,
            num_modalities: int = 2,
            params_transformer: dict = {
                "d_model": 512,
                "nhead": 4,
                "dim_feedforward": 1024,
                "dropout": 0.0,
                "num_layers": 4,
            }
        ):
        super().__init__()
        self.num_modalities = num_modalities

        params_transformer["d_k"] = params_transformer["d_model"] // params_transformer["nhead"]
        params_transformer["d_v"] = params_transformer["d_model"] // params_transformer["nhead"]

        self.cross_attn_transformers = nn.ModuleDict()
        for i in range(self.num_modalities): # Target modality index
            for j in range(self.num_modalities): # Source modality index
                if i != j:
                    self.cross_attn_transformers[f'target_{i}_source_{j}'] = CrossmodalTransformer(params_transformer)

    def forward(
        self,
        x: list[torch.Tensor, torch.Tensor],
        src_mask: torch.Tensor = None,
    ) -> list[torch.Tensor, torch.Tensor]:
        
        outputs_per_target = [[] for _ in range(self.num_modalities)]

        for i in range(self.num_modalities): # Target modality index
            target_mod_emb = x[i] 

            for j in range(self.num_modalities): # Source modality index
                if i == j: continue # Don't attend modality to itself here
                source_mod_emb = x[j]

                # Apply crossmodal transformer 
                output, _ = self.cross_attn_transformers[f'target_{i}_source_{j}'](
                    target_input=target_mod_emb,
                    source_input_original=source_mod_emb,
                    mask=None # Explicitly pass None for the token mask
                )

                # Masking
                if src_mask is not None:
                    is_source_missing = src_mask[:, j][:, None, None]
                    output = output * (~is_source_missing)

                    is_target_missing = src_mask[:, i][:, None, None]
                    output = output * (~is_target_missing)

                outputs_per_target[i].append(output)
            outputs_per_target[i] = torch.cat(outputs_per_target[i], dim=2)

        return outputs_per_target


class SelfAttentionTransformers(nn.Module):

    def __init__(
        self,
        params_transformer: dict = {
            "d_model": 512,
            "nhead": 4,
            "dim_feedforward": 1024,
            "dropout": 0.0,
            "num_layers": 4,
            "dim_output": 10,
        },
        num_modalities: int = 2,
    ) -> None:
        super().__init__()

        self.num_modalities = num_modalities
        separate_d_model = params_transformer["d_model"] * (num_modalities - 1)

        self.n_transformers = nn.ModuleList([ 
            nn.ModuleList([
                AddCLSToken(separate_d_model),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=separate_d_model, 
                        nhead=params_transformer["nhead"], 
                        dim_feedforward=params_transformer["dim_feedforward"], 
                        dropout=params_transformer["dropout"],
                        batch_first=True,
                    ),
                    num_layers=params_transformer["num_layers"]
                ),
                ExtractCLSToken(),
            ]) for _ in range(num_modalities)
        ])

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
        x: list = [torch.Tensor, torch.Tensor],  # outputs per target, i.e., modality
        src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        
        if src_mask is not None:
            src_mask = self._add_cls_token_mask_to_src_mask(src_mask)

        sa_outputs = []
        for i_mod in range(self.num_modalities):
            for layer in self.n_transformers[i_mod]:
                if isinstance(layer, nn.TransformerEncoder) and src_mask is not None:                
                    x[i_mod] = layer(
                        x[i_mod],
                        src_key_padding_mask=src_mask
                    )
                else:
                    x[i_mod] = layer(x[i_mod])
            sa_outputs.append(x[i_mod])

        sa_outputs = torch.cat(sa_outputs, dim=1)

        return sa_outputs


class MulT(nn.Module):

    """
    https://arxiv.org/abs/1906.00295 
    
    https://github.com/yaohungt/Multimodal-Transformer
    """
    def __init__(
            self, 
            params_ca_transformerhead: dict = {
                "d_model": 512,
                "nhead": 4,
                "dim_feedforward": 1024,
                "dropout": 0.0,
                "num_layers": 4,
            },
            params_sa_transformerhead: dict = {
                "d_model": 512,
                "nhead": 4,
                "dim_feedforward": 1024,
                "dropout": 0.0,
                "num_layers": 4,
                "dim_output": 10,
            },
            num_modalities: int = 2,
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        separate_d_model = params_sa_transformerhead["d_model"] * (num_modalities - 1)
        final_concat_dim = separate_d_model * self.num_modalities

        d_model = params_ca_transformerhead["d_model"]
        self.modality_projections = nn.ModuleList([
            nn.LazyConv1d(
                out_channels=d_model,
                kernel_size=3,
                padding=1,
                bias=False
            ) for _ in range(num_modalities)
        ])

        # PE 
        self.add_pe = AddPE(params_ca_transformerhead["d_model"])

        # Final projection
        final_proj = nn.Linear(final_concat_dim, params_sa_transformerhead["dim_output"])

        self.apply(self._init_weights)

        # Cross Attention Transformer 
        d_k = d_v_att = params_ca_transformerhead["d_model"] // params_ca_transformerhead["nhead"]
        params_ca_transformerhead["d_k"] = d_k
        params_ca_transformerhead["d_v"] = d_v_att
        cross_attn_transformers = CrossAttentionTransformers(
            params_transformer=params_ca_transformerhead,
            num_modalities=num_modalities,
        )
        
        # Self Attention Transformer 
        self_attn_transformers = SelfAttentionTransformers(
            params_transformer=params_sa_transformerhead,
            num_modalities=num_modalities,
        )

        # Final model
        self.transformer = nn.ModuleList([
            cross_attn_transformers,
            self_attn_transformers,
            final_proj
        ])

    def _init_weights(
            self,
            m
        ) -> None: 
        if isinstance(m, (torch.nn.LayerNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Parameter):
            if m.dim() > 1:
                torch.nn.init.xavier_normal_(m)
            else:
                torch.nn.init.zeros
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(
        self,
        x: list[torch.Tensor],
        src_mask: torch.Tensor = None,
        y: torch.Tensor = None
    ) -> dict:

        src_mask = None

        # Apply temporal convolutions and add positional embedding
        processed_x = []
        for i in range(self.num_modalities):
            x_mod = x[i].transpose(1, 2)
            x_mod = self.modality_projections[i](x_mod)
            x_mod = x_mod.transpose(1, 2)
            x_mod = self.add_pe(x_mod)
            processed_x.append(x_mod)
        x = processed_x

        # Apply crossmodal and self-attention transformers
        for layer in self.transformer:
            if isinstance(layer, CrossAttentionTransformers) and src_mask is not None:
                x = layer(
                    x,
                    src_mask=src_mask
                )  # list
            elif isinstance(layer, SelfAttentionTransformers) and src_mask is not None:
                x = layer(
                    x,  # list
                    src_mask=src_mask
                )  # tensor
            else:
                x = layer(x)

        return {
            "logits": x
        }