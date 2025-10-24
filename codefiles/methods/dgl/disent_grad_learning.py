import torch 
import torch.nn as nn 
import torch.nn.functional as F

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE

class Disentangled_Gradient_Learning_Transformer(nn.Module):

    """
    https://arxiv.org/abs/2507.10213

    https://github.com/shicaiwei123/ICCV2025-GDL
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
    ) -> None: 
        super().__init__()
        self.num_modalities = num_modalities
        
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
        
        # Split modalities to calculate unimodal logits
        split_size = x.shape[1] // self.num_modalities
        modalities_seqs = list(torch.split(x, split_size, dim=1))

        if src_mask is not None:
            modalities_masks = list(torch.split(src_mask, split_size, dim=1))
        else:
            # If no mask is provided, create a default mask of all False (nothing masked)
            modalities_masks = [
                torch.zeros(s.shape[0], s.shape[1], dtype=torch.bool, device=x.device) 
                for s in modalities_seqs
            ]

        # Unimodal path (for encoder optimization)
        unimodal_logits = []
        for i in range(self.num_modalities):
            # Create unimodal input by zeroing out other modalities, preserving order
            unimodal_x_list = [
                modalities_seqs[j] if i == j else torch.zeros_like(modalities_seqs[j])
                for j in range(self.num_modalities)
            ]
            mod_input = torch.cat(unimodal_x_list, dim=1)

            # Create the corresponding mask. True values mean the position is masked.
            # We use the original mask for the active modality and mask everything else.
            unimodal_mask_list = [
                modalities_masks[j] if i == j else torch.ones_like(modalities_masks[j])
                for j in range(self.num_modalities)
            ]
            unimodal_src_mask = torch.cat(unimodal_mask_list, dim=1)
            
            mod_rep = self._run_transformer_pipeline(mod_input, unimodal_src_mask)
            unimodal_logits.append(mod_rep)

        # Multimodal path (for fusion module optimization)
        # We detach 'x' to stop gradients from the multimodal loss flowing back to the encoders.
        main_logits = self._run_transformer_pipeline(x.detach(), src_mask)

        return {
            "logits": main_logits,
            "unimodal_logits": unimodal_logits,
            "dgl": True,
        }
