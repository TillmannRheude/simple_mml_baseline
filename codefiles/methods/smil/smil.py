import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE

class SMIL(nn.Module):

    """
    https://arxiv.org/abs/2103.05677

    https://github.com/deep-real/SMIL
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
        num_priors: int = 10
    ) -> None: 
        super().__init__()
        self.num_modalities = num_modalities
        self.num_priors = num_priors
        
        # This will be initialized randomly and then overwritten by the pre-computed priors.
        self.modality_priors = nn.Parameter(torch.randn(num_modalities, num_priors, d_model))

        # Reconstruction network (phi_c)
        self.reconstruction_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_modalities * num_priors)
        )

        # Regularization network (phi_r)
        self.regularization_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.linear_out = nn.Linear(d_model, dim_output)

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
        x: torch.Tensor,
        src_mask: torch.Tensor,
        y: torch.Tensor = None,
        return_pre_logits: bool = False
    ) -> dict:
        
        r = None
        if src_mask is not None and src_mask.any():
            # Reconstruction and regularization is only applied when modalities are missing. 
            # This follows the original author's implementation, where the 'encoder' is only used 
            # when mode='one'. See src/models/soundlenet5.py in their repository.
            present_mask = ~src_mask
            present_features = x * present_mask.unsqueeze(-1)
            num_present = present_mask.sum(dim=1, keepdim=True).clamp(min=1)
            avg_present_feature = present_features.sum(dim=1) / num_present

            # Reconstruction of missing modalities
            reconstruction_weights_mu = self.reconstruction_net(avg_present_feature)
            reconstruction_weights_mu = reconstruction_weights_mu.view(-1, self.num_modalities, self.num_priors)
            
            if self.training:
                reconstruction_weights = F.softplus(torch.randn_like(reconstruction_weights_mu) + reconstruction_weights_mu)
            else:
                reconstruction_weights = F.softplus(reconstruction_weights_mu)
            
            reconstruction_weights = reconstruction_weights.unsqueeze(-1)

            priors = self.modality_priors.unsqueeze(0)
            reconstructed_features = (priors * reconstruction_weights).sum(dim=2)
            reconstructed_features = reconstructed_features / reconstruction_weights.sum(dim=2).clamp(min=1e-9)

            x_reconstructed = torch.where(src_mask.unsqueeze(-1), reconstructed_features, x)

            # Regularization
            r_mu = self.regularization_net(avg_present_feature)
            if self.training:
                r = r_mu + torch.randn_like(r_mu)
            else:
                r = r_mu
            
            x = x_reconstructed
            src_mask = torch.zeros_like(src_mask, dtype=torch.bool)

        pre_logits = None
        for i, layer in enumerate(self.transformer_cls):
            if isinstance(layer, nn.TransformerEncoder):
                mask = None
                if src_mask is not None:
                    mask = self._add_cls_token_mask_to_src_mask(src_mask)
                x = layer(x, src_key_padding_mask=mask)
            else:
                if return_pre_logits and i == len(self.transformer_cls) - 1:
                    pre_logits = x
                
                x = layer(x)

                if isinstance(layer, ExtractCLSToken) and r is not None:
                    x = x + r

        output = {"logits": x}
        if return_pre_logits:
            output["pre_logits"] = pre_logits
        
        return output
