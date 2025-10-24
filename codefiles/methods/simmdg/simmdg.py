import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE
# Assuming SupConLoss is in your codebase, e.g., from a losses file.
# You can use the implementation from the official SimMMDG repository.
from codefiles.losses.supconloss import SupConLoss

class ProjectHead(nn.Module):
    """Projection head for supervised contrastive learning."""
    def __init__(self, input_dim: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(x), dim=1)

class EncoderTrans(nn.Module):
    """MLP for cross-modal translation."""
    def __init__(self, input_dim: int, out_dim: int, hidden: int = 2048):
        super().__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_net(x)

class SiMMDG_Transformer(nn.Module):

    def __init__(
        self,
        n_modalities: int,
        d_model: int = 512,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_layers: int = 4,
        dim_output: int = 10,
        proj_out_dim: int = 128,
        trans_hidden_dim: int = 2048,
        temp: float = 0.1,
        loss_contrastive: float = 1.0,
        loss_distance: float = 1.0,
        loss_translation: float = 1.0,
    ) -> None: 
        super().__init__()
        assert d_model % 2 == 0, "d_model must be an even number for feature splitting."
        self.d_model = d_model
        self.n_modalities = n_modalities
        self.loss_contrastive = loss_contrastive
        self.loss_distance = loss_distance
        self.loss_translation = loss_translation

        self.project_head = ProjectHead(input_dim=d_model // 2, out_dim=proj_out_dim)
        
        translators = {}
        for i in range(n_modalities):
            for j in range(n_modalities):
                if i == j:
                    continue
                translators[f'{i}_to_{j}'] = EncoderTrans(
                    input_dim=d_model, out_dim=d_model, hidden=trans_hidden_dim
                )
        self.translators = nn.ModuleDict(translators)

        self.criterion_contrast = SupConLoss(temperature=temp)

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
        
        assert x.shape[1] == self.n_modalities, f"Input tensor has {x.shape[1]} modalities, but model was initialized with {self.n_modalities}."

        if not self.training and src_mask is not None and src_mask.any():
            bs, n_mod, d_mod = x.shape
            synthesized_sums = torch.zeros_like(x)
            synthesized_counts = torch.zeros(bs, n_mod, device=x.device)

            for source_idx in range(n_mod):
                for target_idx in range(n_mod):
                    if source_idx == target_idx:
                        continue
                    
                    # Find all samples that need this specific translation
                    # i.e., target is missing AND source is available.
                    mask = (src_mask[:, target_idx]) & (~src_mask[:, source_idx])

                    if not mask.any():
                        continue
                    
                    source_features = x[mask, source_idx]
                    
                    translator = self.translators[f'{source_idx}_to_{target_idx}']
                    translated = translator(source_features)
                    
                    synthesized_sums[mask, target_idx] += translated
                    synthesized_counts[mask, target_idx] += 1

            avg_synthesized = synthesized_sums / synthesized_counts.unsqueeze(-1).clamp(min=1)
            
            x = torch.where(src_mask.unsqueeze(-1), avg_synthesized, x)
        
        elif self.training and src_mask is not None and src_mask.any():
                raise ValueError("Missing modalities (src_mask) are not supported during training.")

        d_half = self.d_model // 2
        x_shared = x[..., :d_half]
        x_specific = x[..., d_half:]

        loss_dict = {}
        if y is not None:
            # Supervised Contrastive Loss
            bs, n_mod, _ = x_shared.shape
            features_for_contrast = self.project_head(x_shared.reshape(bs * n_mod, -1))
            features_for_contrast = features_for_contrast.view(bs, n_mod, -1)
            
            contrast_labels = None
            contrast_mask = None
            is_bce_task = y.dim() > 1 and y.shape[1] > 1

            if is_bce_task:
                # Multi-label case: two samples are a positive pair if they share at least one label.
                # NaNs are ignored by converting them to 0 for the check.
                y_clean = y.nan_to_num(0)
                # A dot product between the one-hot labels gives the number of shared classes.
                shared_label_matrix = torch.matmul(y_clean, y_clean.T)
                contrast_mask = (shared_label_matrix > 0).float()
            else:
                # Standard single-label CE case
                contrast_labels = y

            loss_dict["loss_contrastive"] = self.criterion_contrast(features_for_contrast, labels=contrast_labels, mask=contrast_mask)
            loss_dict["loss_contrastive"] = loss_dict["loss_contrastive"] * self.loss_contrastive

            # Distance Loss
            loss_dict["loss_distance"] = -F.mse_loss(x_shared, x_specific) * self.loss_distance

            # Cross-modal Translation Loss
            total_trans_loss = 0.0
            count = 0
            for i in range(self.n_modalities):
                for j in range(self.n_modalities):
                    if i == j:
                        continue
                    source_mod = x[:, i, :]
                    target_mod = x[:, j, :]
                    
                    translator = self.translators[f'{i}_to_{j}']
                    translated = translator(source_mod)

                    translated_norm = F.normalize(translated, p=2, dim=1)
                    target_norm = F.normalize(target_mod, p=2, dim=1)
                    
                    loss = torch.mean(torch.norm(translated_norm - target_norm, p=2, dim=1))
                    total_trans_loss += loss
                    count += 1
            loss_dict["loss_translation"] = total_trans_loss / count if count > 0 else torch.tensor(0.0, device=x.device)
            loss_dict["loss_translation"] = loss_dict["loss_translation"] * self.loss_translation

        for layer in self.transformer_cls:
            x = layer(x)

        return {
            "logits": x,
            "losses": loss_dict
        }
