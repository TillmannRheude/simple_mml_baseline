import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE

class Predictive_Dynamic_Fusion_Transformer(nn.Module):

    """
    https://arxiv.org/html/2406.04802v3
    
    https://github.com/Yinan-Xia/PDF
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
        task_type: str = "ce",
        p_head_params: dict = {
            "hidden_dims": [128],
            "dropout": 0.1
        }
    ) -> None: 
        super().__init__()

        self.num_modalities = num_modalities
        self.task_type = task_type

        self.unimodal_classifiers = nn.ModuleList([
            nn.Linear(d_model, dim_output) for _ in range(num_modalities)
        ])

        self.p_heads = nn.ModuleList()
        for _ in range(num_modalities):
            layers = []
            input_dim = d_model
            if p_head_params and p_head_params["hidden_dims"]:
                for h_dim in p_head_params["hidden_dims"]:
                    layers.append(nn.Linear(input_dim, h_dim))
                    layers.append(nn.ReLU())
                    if p_head_params["dropout"] > 0:
                        layers.append(nn.Dropout(p_head_params["dropout"]))
                    input_dim = h_dim
            layers.append(nn.Linear(input_dim, 1))
            layers.append(nn.Sigmoid())
            self.p_heads.append(nn.Sequential(*layers))

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
    
    def forward(
        self, 
        x: torch.Tensor,
        src_mask: torch.Tensor = None,
        y: torch.Tensor = None
    ) -> dict:

        unimodal_logits = []
        p_trues = []
        for i in range(self.num_modalities):
            modality_rep = x[:, i, :]
            unimodal_logits.append(self.unimodal_classifiers[i](modality_rep))
            p_trues.append(self.p_heads[i](modality_rep.detach()))
        p_trues_tensor = torch.cat(p_trues, dim=1) + 1e-8

        # Co-Belief
        mono_confidence = p_trues_tensor
        log_p_trues = torch.log(p_trues_tensor)
        sum_log_p_trues = torch.sum(log_p_trues, dim=1, keepdim=True)
        
        # log(prod(p_j for j!=m)) = sum(log(p_j for j!=m)) = total_sum - log(p_m)
        holo_confidence = (sum_log_p_trues - log_p_trues) / (sum_log_p_trues + 1e-8)
        co_belief = mono_confidence + holo_confidence

        weights_stack = co_belief

        # Relative Calibration is only applied during evaluation, as per the official repository.
        if not self.training:
            # Relative Calibration based on Distribution Uniformity (DU)
            dus = []
            if self.task_type == "ce":
                unimodal_probs = [F.softmax(logits, dim=1) for logits in unimodal_logits]
                for probs in unimodal_probs:
                    num_classes = probs.shape[1]
                    # DU_m = mean(abs(p_i - 1/C))
                    du = torch.mean(torch.abs(probs - 1.0 / num_classes), dim=1, keepdim=True)
                    dus.append(du)
            else: # For BCE
                unimodal_probs = [torch.sigmoid(logits) for logits in unimodal_logits]
                for probs in unimodal_probs:
                    # DU for BCE measures confidence as distance from 0.5 (max uncertainty)
                    du = torch.mean(torch.abs(probs - 0.5) * 2, dim=1, keepdim=True)
                    dus.append(du)
            
            dus_tensor = torch.cat(dus, dim=1)
            sum_dus = torch.sum(dus_tensor, dim=1, keepdim=True)
            
            # RC_m = DU_m * (|M|-1) / sum(DU_j for j!=m)
            rc_denominators = (sum_dus - dus_tensor)
            rc_denominators[rc_denominators < 1e-8] = 1e-8
            rc = dus_tensor * (self.num_modalities - 1) / rc_denominators
            
            # Asymmetric calibration: k_m = min(RC_m, 1.0)
            k_m = torch.clamp(rc, max=1.0)
            
            weights_stack = co_belief * k_m
        
        weights = F.softmax(weights_stack, dim=1)

        stacked_logits = torch.stack(unimodal_logits, dim=1)
        weights_expanded = weights.unsqueeze(2)
        # Detach weights, as per the official repository
        fused_logits = (stacked_logits * weights_expanded.detach()).sum(dim=1)

        return {
            "logits": fused_logits,
            "unimodal_logits": unimodal_logits,
            "p_trues": p_trues,
            "weights": weights,
            "pdf": True 
        }
