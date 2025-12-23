import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE
from codefiles.methods.utils import mimetic_init_svd_

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)
    
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
        elif isinstance(m, nn.Parameter):
            torch.nn.init.xavier_uniform_(m)

    def forward(self, x: torch.Tensor, output_seq_len: int) -> torch.Tensor:
        input_seq_len = x.shape[1]

        x = self.fc1(x)
        x = self.activation(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.norm2(x)
        x = self.dropout(x)

        if input_seq_len != output_seq_len:
            x = x.transpose(1, 2)
            x = F.interpolate(x, size=output_seq_len, mode='linear', align_corners=False)
            x = x.transpose(1, 2)
        
        return x

class Masked_Modality_Projection_Transformer(nn.Module):

    """ 
    https://arxiv.org/html/2410.03010v2
    
    https://drive.google.com/drive/folders/155IsgLD88Dt6Q9DJCKCENuDF7vjI9jI 
    Code was online deleted after download on 14th of August 2025, locally saved.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_layers: int = 4,
        dim_output: int = 10,
        num_modalities: int = 2,
        params_proj_mlp: dict = {
            "hidden_dim": 512,
            "dropout": 0.1
        },
        params_attn_steps: dict = {
            "dropout": 0.0,
            "nhead": 4
        },
        num_aggregated_tokens: int = 8,
        loss_alignment_alpha: float = 1.0
    ) -> None: 
        super().__init__()

        self.num_modalities = num_modalities
        self.d_model = d_model
        self.nhead = nhead
        self.num_aggregated_tokens = num_aggregated_tokens
        self.loss_alignment_alpha = loss_alignment_alpha

        # Learnable aggregated tokens
        self.aggregated_tokens = nn.Parameter(torch.randn(num_modalities, self.num_aggregated_tokens, d_model))

        # Pre-projection regularization and final MLPs
        in_dim = d_model
        self.pre_projection_norms = nn.ModuleList([nn.LayerNorm(in_dim) for _ in range(num_modalities)])
        self.pre_projection_dropouts = nn.ModuleList([nn.Dropout(params_proj_mlp["dropout"]) for _ in range(num_modalities)])
        
        self.projection_mlps = nn.ModuleList()
        for _ in range(self.num_modalities):
            mlp = ResidualMLP(in_dim, params_proj_mlp["hidden_dim"], d_model, dropout=params_proj_mlp["dropout"])
            self.projection_mlps.append(mlp)

        self.alignment_loss_fn = nn.SmoothL1Loss()
        self.linear_out = nn.Linear(d_model, dim_output)

        self.apply(self._init_weights)

        # Attention layers for projection mechanism
        self.attn_step1 = nn.ModuleList([
            nn.MultiheadAttention(d_model, params_attn_steps["nhead"], dropout=params_attn_steps["dropout"], batch_first=True)
            for _ in range(num_modalities)
        ])
        self.attn_step2 = nn.MultiheadAttention(d_model, params_attn_steps["nhead"], dropout=params_attn_steps["dropout"], batch_first=True)
        self.attn_step3 = nn.MultiheadAttention(d_model, params_attn_steps["nhead"], dropout=params_attn_steps["dropout"], batch_first=True)

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
        elif isinstance(m, nn.Parameter):
            torch.nn.init.xavier_uniform_(m)
    
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
        
        if not isinstance(x, list):
            bs = x.shape[0]
            split_size = x.shape[1] // self.num_modalities
            modalities_features = list(torch.split(x, split_size, dim=1))
            modality_is_missing = src_mask
            modality_is_available = ~src_mask
        else:
            bs = x[0].shape[0]
            modalities_features = x
            modality_is_missing = torch.cat(src_mask, dim=1)
            modality_is_available = ~modality_is_missing
            
        seq_lens = [f.shape[1] for f in modalities_features]
        real_features = [f.clone() for f in modalities_features]
       
        # Aggregated tokens for available modalities
        agg_tokens = self.aggregated_tokens.unsqueeze(0).expand(bs, -1, -1, -1)        
        
        attended_tokens_per_modality = []
        for j in range(self.num_modalities):
            mask = modality_is_available[:, j]
            original_tokens = agg_tokens[:, j]
            
            if not mask.any():
                attended_tokens_per_modality.append(original_tokens)
                continue

            # Cross Attention 1 for available samples
            query = agg_tokens[mask, j]
            key = value = real_features[j][mask]
            attended_output, _ = self.attn_step1[j](query, key, value)
            
            attended_full = torch.zeros_like(original_tokens)
            indices = mask.nonzero(as_tuple=False)
            if indices.numel() > 0:
                indices = indices.expand(-1, self.num_aggregated_tokens * self.d_model).view(attended_output.shape)
                attended_full = torch.scatter(attended_full, 0, indices, attended_output)

            broadcast_mask = mask.view(bs, 1, 1).expand_as(original_tokens)
            updated_tokens = torch.where(broadcast_mask, attended_full, original_tokens)
            attended_tokens_per_modality.append(updated_tokens)

        updated_agg_tokens = torch.stack(attended_tokens_per_modality, dim=1)

        # Final projections for each modality
        final_projections = [None] * self.num_modalities
        for target_idx in range(self.num_modalities):
            t_attended_list = []
            for source_idx in range(self.num_modalities):
                if source_idx == target_idx: continue

                # Cross Attention 2
                query2 = updated_agg_tokens[:, target_idx]
                key2 = value2 = updated_agg_tokens[:, source_idx]
                X_ij, _ = self.attn_step2(query2, key2, value2)

                # Cross Attention 3 (aligned with authors' code: Query=Tj, Key=Xij)
                query3 = real_features[source_idx]
                key3 = value3 = X_ij
                T_attended_j, _ = self.attn_step3(query3, key3, value3)
                t_attended_list.append(T_attended_j)
            
            # Projection
            concatenated_tokens = torch.cat(t_attended_list, dim=1)
            normed_tokens = self.pre_projection_norms[target_idx](concatenated_tokens)
            dropped_tokens = self.pre_projection_dropouts[target_idx](normed_tokens)
            target_seq_len = seq_lens[target_idx]
            final_projections[target_idx] = self.projection_mlps[target_idx](dropped_tokens, output_seq_len=target_seq_len)

        # Update features for missing modalities and calculate alignment loss
        total_alignment_loss = 0
        modalities_with_loss = 0
        for j in range(self.num_modalities):
            update_mask = modality_is_missing[:, j]
            
            if torch.any(update_mask):
                avg_proj_for_update = final_projections[j][update_mask]
                loss = self.alignment_loss_fn(avg_proj_for_update, real_features[j][update_mask])
                total_alignment_loss += loss
                modalities_with_loss += 1

            update_mask_expanded = update_mask.unsqueeze(-1).unsqueeze(-1).expand_as(modalities_features[j])
            modalities_features[j] = torch.where(update_mask_expanded, final_projections[j], modalities_features[j])

        losses = {}
        if modalities_with_loss > 0:
            losses['alignment_loss'] = self.loss_alignment_alpha * (total_alignment_loss / modalities_with_loss)

        x = torch.cat(modalities_features, dim=1)
        
        for layer in self.transformer_cls:
            x = layer(x)

        return {
            "logits": x,
            "losses": losses
        }
