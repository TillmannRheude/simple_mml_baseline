import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Function

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE
from codefiles.losses.nanbce import WeightedNaNBCEWithLogitsLoss

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

class Explicit_Basis_Reallocation_Transformer(nn.Module):

    """
    https://arxiv.org/abs/2505.22483

    As the authors did not publish a code repository (yet), 
    this implementation is based on the descriptions in the paper.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_layers: int = 4,
        dim_output: int = 10,
        max_modalities: int = 10,
        ebr_params: dict = {
            "alpha_ebr": 1.0,
            "hidden_dim": 1024,
            "d_shared": 512,
        },
        modality_ranking: torch.Tensor = None,
    ) -> None: 
        super().__init__()

        self.linear_out = nn.Linear(d_model, dim_output)

        # h and h-1 encoders/decoders (Sec 3.4) 
        self.h_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ebr_params["hidden_dim"]),
                nn.ReLU(),
                nn.Linear(ebr_params["hidden_dim"], ebr_params["d_shared"])
            ) for _ in range(max_modalities)
        ])
        self.h_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ebr_params["d_shared"], ebr_params["hidden_dim"]),
                nn.ReLU(),
                nn.Linear(ebr_params["hidden_dim"], d_model)
            ) for _ in range(max_modalities)
        ])

        # Classify which modality an input embedding belongs to (Appendix C.7).
        self.modality_classifier = nn.Sequential(
            nn.Linear(ebr_params["d_shared"], ebr_params["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(ebr_params["hidden_dim"], max_modalities)
        )
        self.alpha_ebr = ebr_params["alpha_ebr"]
        self.ebr_loss_fn = nn.CrossEntropyLoss() 

        # (n_modalities, n_modalities-1), where each row `i` indices of other modalities, 
        # ranked by their similarity to `i`. (substitution strategy for missing modalities 
        # at inference time (Sec. 4.4))
        self.register_buffer('modality_ranking', modality_ranking)

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

    def forward(
        self, 
        x: torch.Tensor = torch.Tensor,  
        src_mask: torch.Tensor = torch.Tensor,
        y: torch.Tensor = None
    ) -> dict:
        batch_size, n_modalities, d_model = x.shape

        # Project to shared embedding space
        g_list = [self.h_encoders[i](x[:, i, :]) for i in range(n_modalities)]
        g = torch.stack(g_list, dim=1)
        _, _, d_shared = g.shape

        # EBR Loss Calculation on g
        g_flat = g.view(-1, d_shared)
        labels = torch.arange(n_modalities, device=x.device).unsqueeze(0).expand(batch_size, -1).reshape(-1)
        if src_mask is not None:
            mask_flat = ~src_mask.view(-1)
            g_to_classify = g_flat[mask_flat]
            labels_to_classify = labels[mask_flat]
        else:
            g_to_classify = g_flat
            labels_to_classify = labels

        if g_to_classify.numel() > 0:
            reversed_g = grad_reverse(g_to_classify)
            modality_logits = self.modality_classifier(reversed_g)
            loss_ebr = self.ebr_loss_fn(modality_logits, labels_to_classify)
        else:
            loss_ebr = torch.tensor(0.0, device=x.device)

        # Project back to denoised modality-specific space
        f_list = [self.h_decoders[i](g[:, i, :]) for i in range(n_modalities)]
        f = torch.stack(f_list, dim=1)

        x_for_transformer = f

        # Inference-Time Missing Modality Substitution (Sec. 4.4)
        if not self.training:
            # Fill in missing data using the most similar available modality.
            if src_mask is not None and self.modality_ranking is not None:
                f_substituted = f.clone()
                new_src_mask = src_mask.clone()

                for i in range(batch_size):
                    missing_indices = torch.where(src_mask[i])[0]
                    available_indices = torch.where(~src_mask[i])[0]

                    if len(available_indices) > 0:
                        for missing_idx in missing_indices:
                            # Find the best substitute from the pre-computed ranked list
                            for substitute_candidate in self.modality_ranking[missing_idx]:
                                if substitute_candidate in available_indices:
                                    g_substitute = g[i, substitute_candidate]
                                    f_proxy = self.h_decoders[missing_idx](g_substitute)
                                    f_substituted[i, missing_idx] = f_proxy
                                    new_src_mask[i, missing_idx] = False
                                    break # Move to next missing modality
                
                x_for_transformer = f_substituted
                src_mask = new_src_mask

        x_processed = x_for_transformer
        for layer in self.transformer_cls:
            if isinstance(layer, nn.TransformerEncoder) and src_mask is not None:
                src_mask = self._add_cls_token_mask_to_src_mask(src_mask)
                x_processed = layer(x_processed, src_key_padding_mask=src_mask)
            else:
                x_processed = layer(x_processed)

        return {
            "shared_embeddings_g": g,
            "logits": x_processed,
            "losses": {
                "ebr": self.alpha_ebr * loss_ebr
            }
        }
