import math
import torch 
import torch.nn as nn 

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE

class Sim_MLM_Transformer(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_layers: int = 4,
        dim_output: int = 10,
        n_modalities: int = 3,
        gating_hidden_dim: int = 128
    ) -> None: 
        super().__init__()
        self.n_modalities = n_modalities

        # DMoME 
        self.experts = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(n_modalities)]
        )
        self.gating_network = nn.Sequential(
            nn.Linear(n_modalities * d_model, gating_hidden_dim),
            nn.ReLU(),
            nn.Linear(gating_hidden_dim, n_modalities)
        )

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

    def _create_fewer_modality_mask(self, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Creates a new mask for the "fewer" case by randomly dropping one available modality per sample.
        """
        new_mask = src_mask.clone()
        num_available = (~src_mask).sum(dim=1)
        
        # Iterate over samples that have more than one modality available to drop one.
        for i in (num_available > 1).nonzero(as_tuple=True)[0]:
            available_indices = (~src_mask[i]).nonzero(as_tuple=True)[0]
            drop_idx = torch.randint(len(available_indices), (1,)).item()
            modality_to_drop = available_indices[drop_idx]
            new_mask[i, modality_to_drop] = True
            
        return new_mask

    def _run_forward_pass(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # 1. DMoME Experts: Process each modality token with its dedicated expert.
        expert_outputs = [self.experts[i](x[:, i, :]) for i in range(self.n_modalities)]
        o = torch.stack(expert_outputs, dim=1)
        
        # 2. DMoME Gating: Calculate weights for each expert.
        # For the gating network, missing modality inputs are zeroed out as per the paper.
        gating_x = x.clone()
        gating_x.masked_fill_(src_mask.unsqueeze(-1), 0)
        gating_input = gating_x.flatten(start_dim=1)
        g = self.gating_network(gating_input)
        
        # Set gating scores for missing modalities to -inf to ensure their weight is 0 after softmax.
        g.masked_fill_(src_mask, -torch.inf)
        w = torch.softmax(g, dim=1)

        # 3. DMoME Mixture: Apply learned weights to expert outputs.
        # The result is a weighted sequence of modality tokens.
        o_weighted = o * w.unsqueeze(-1)

        # 4. Transformer Classifier: Process the weighted sequence.
        transformer_input = o_weighted
        transformer_mask = src_mask.clone()
        for layer in self.transformer_cls:
            if isinstance(layer, nn.TransformerEncoder) and transformer_mask is not None:
                transformer_mask = self._add_cls_token_mask_to_src_mask(transformer_mask)
                transformer_input = layer(transformer_input, src_key_padding_mask=transformer_mask)
            else:
                transformer_input = layer(transformer_input)
        
        return transformer_input

    def forward(
        self, 
        x: torch.Tensor,
        src_mask: torch.Tensor,
        y: torch.Tensor = None
    ) -> dict:
        # The original input (x, src_mask) serves as the "more modalities" (plus) case for MoFe loss.
        logits_plus = self._run_forward_pass(x, src_mask)
        
        if not self.training:
            return {"logits": logits_plus}
        
        # For MoFe loss, create a "fewer modalities" (minus) case by dropping one more modality.
        src_mask_minus = self._create_fewer_modality_mask(src_mask)
        
        # If no modality could be dropped, the "minus" case is the same as the "plus" case.
        # Detach to avoid gradients flowing through this path twice.
        if torch.equal(src_mask, src_mask_minus):
            logits_minus = logits_plus.detach()
        else:
            logits_minus = self._run_forward_pass(x, src_mask_minus)
        
        return {
            "logits": logits_plus, 
            "logits_plus": logits_plus,
            "logits_minus": logits_minus
        }
