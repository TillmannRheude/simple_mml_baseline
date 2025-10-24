import torch 
import torch.nn as nn 
import torch.nn.functional as F

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE

class Asymmetric_Representation_Learning_Transformer(nn.Module):

    """
    https://arxiv.org/abs/2507.10203 

    https://github.com/shicaiwei123/ICCV2025-ARL
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
        arl_temperature: float = 4.0,
    ) -> None: 
        super().__init__()
        self.num_modalities = num_modalities
        self.arl_temperature = arl_temperature
        self.eps = 1e-8
        
        # Auxiliary head for unimodal regularization
        self.auxiliary_classifier = nn.Linear(d_model, dim_output)

        self.apply(self._init_weights)

        # Main multimodal model 
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
        self.transformer_cls_pipeline = nn.ModuleList([
            AddCLSToken(d_model),
            AddPE(d_model),
            self.transformer,
            ExtractCLSToken(),
            nn.Linear(d_model, dim_output)
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

    def _run_transformer_pipeline(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        for layer in self.transformer_cls_pipeline:
            if isinstance(layer, nn.TransformerEncoder) and src_mask is not None:
                mask = self._add_cls_token_mask_to_src_mask(src_mask)
                x = layer(x, src_key_padding_mask=mask)
            else:
                x = layer(x)
        return x

    def forward(
        self, 
        x: torch.Tensor,
        src_mask: torch.Tensor = None,
        y: torch.Tensor = None
    ) -> dict:
        
        # Split modalities and masks
        split_size = x.shape[1] // self.num_modalities
        modalities_seqs = list(torch.split(x, split_size, dim=1))

        # Unimodal Regularization
        unimodal_logits = []
        for mod_seq in modalities_seqs:
            mod_rep = torch.mean(mod_seq, dim=1)
            unimodal_logits.append(self.auxiliary_classifier(mod_rep))

        # ARL coefficients
        q_vals = []
        d_vals = []
        
        # Check if the task is multi-label based on target shape
        is_multilabel = y.ndim > 1 and y.shape[1] > 1

        for p_m in unimodal_logits:
            # Calculate d (dependency) using the official repo's method:
            # Sum of mean absolute logits across classes.
            d_i = torch.mean(torch.abs(p_m), dim=0).sum()

            if is_multilabel:
                # Use sigmoid for multi-label probabilities
                probs = torch.sigmoid(p_m)
                
                # Eq. 17: Calculate q (inverse variance) via summed binary entropy
                binary_entropy = - (probs * torch.log(probs + self.eps) + (1-probs) * torch.log(1-probs + self.eps))
                entropy = torch.sum(binary_entropy, dim=-1)
                q_i = (1 / (entropy + self.eps)).mean()

            else: # Single-label classification case
                probs = F.softmax(p_m, dim=-1)
                
                # Eq. 17: Calculate q (inverse variance) via categorical entropy
                entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=-1)
                q_i = (1 / (entropy + self.eps)).mean()

            q_vals.append(q_i)
            d_vals.append(d_i)

        q_tensor = torch.stack(q_vals)
        d_tensor = torch.stack(d_vals)
        
        # Generalized version of the repository's logic to calculate modulation coefficients.
        # This is proportional to softmax(q_i / d_i), rewarding modalities where
        # inverse variance (q) is high relative to their dependency (d).
        ratios = q_tensor / (d_tensor + self.eps)
        arl_coeffs = F.softmax(ratios * self.arl_temperature, dim=0).tolist()

        # Main multimodal model
        main_logits = self._run_transformer_pipeline(x, src_mask)

        return {
            "logits": main_logits,
            "unimodal_logits": unimodal_logits,
            "arl_coeffs": arl_coeffs,
        }
