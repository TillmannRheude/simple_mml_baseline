import math
import torch
import torch.nn as nn
from torch.autograd import Function

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE

class Shared_Specific_Feature_Modelling_Transformer(nn.Module):

    """
    https://arxiv.org/abs/2307.14126

    https://github.com/billhhh/ShaSpec/
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
        loss_alpha: float = 1.0,
        loss_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.d_model = d_model
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta

        # F projections for residual fusion
        self.f_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model), # Input is concatenation of r and s
                nn.ReLU(),
                nn.Linear(self.d_model, self.d_model),
            )
            for _ in range(num_modalities)
        ])

        # Domain Classifier for DC loss
        self.domain_classifier_dc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.num_modalities)
        )

        self.linear_out = nn.Linear(d_model, dim_output)
        
        self.apply(self._init_weights)
        
        # Shared Encoder
        self.shared_encoder = self._create_encoder_pipeline(d_model, nhead, dim_feedforward, dropout, num_layers)

        # Specific Encoders (one for each modality)
        self.specific_encoders = nn.ModuleList(
            [self._create_encoder_pipeline(d_model, nhead, dim_feedforward, dropout, num_layers) for _ in range(num_modalities)]
        )

        # Task Head / Decoder
        decoder_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers
        )
        self.decoder = nn.ModuleList([
            AddCLSToken(d_model),
            AddPE(d_model),
            decoder_transformer,
            ExtractCLSToken(),
            self.linear_out
        ])

    def _create_encoder_pipeline(self, d_model, nhead, dim_feedforward, dropout, num_layers):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        return nn.ModuleList([
            AddCLSToken(d_model),
            AddPE(d_model),
            nn.TransformerEncoder(encoder_layer, num_layers),
            ExtractCLSToken(),
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

    def _run_encoder(self, encoder: nn.ModuleList, x: torch.Tensor, src_mask: torch.Tensor):
        for layer in encoder:
            if isinstance(layer, nn.TransformerEncoder) and src_mask is not None:
                mask = self._add_cls_token_mask_to_src_mask(src_mask)
                x = layer(x, src_key_padding_mask=mask)
            else:
                x = layer(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor,
        y: torch.Tensor = None
    ) -> dict:
        
        x_list = torch.chunk(x, chunks=self.num_modalities, dim=1)
        mask_list = torch.chunk(src_mask, chunks=self.num_modalities, dim=1)
        
        available_r = [] # Shared features
        available_s = [] # Specific features
        available_indices = []

        for i in range(self.num_modalities):
            # A modality is considered missing if its mask is all True (all padded)
            if not torch.all(mask_list[i]):
                r_i = self._run_encoder(self.shared_encoder, x_list[i], mask_list[i])
                s_i = self._run_encoder(self.specific_encoders[i], x_list[i], mask_list[i])
                available_r.append(r_i)
                available_s.append(s_i)
                available_indices.append(i)

        # Generate features for the decoder
        r_fused = torch.mean(torch.stack(available_r, dim=0), dim=0)
        features_for_decoder = []
        
        available_idx_iterator = 0
        for i in range(self.num_modalities):
            if i in available_indices:
                r_i = available_r[available_idx_iterator]
                s_i = available_s[available_idx_iterator]
                # Residual Fusion for available modalities
                f_i = self.f_projections[i](torch.cat([r_i, s_i], dim=-1)) + r_i
                features_for_decoder.append(f_i)
                available_idx_iterator += 1
            else:
                # Feature Generation for missing modalities
                features_for_decoder.append(r_fused)

        # Decoder
        decoder_input = torch.stack(features_for_decoder, dim=1) # (B, N_modalities, D)
        logits = decoder_input
        for layer in self.decoder:
            #if isinstance(layer, nn.TransformerEncoder):
                # Create a padding mask for the decoder sequence.
                # A feature is masked if its original modality was missing.
            #    decoder_mask = torch.tensor(
            #        [not (i in available_indices) for i in range(self.num_modalities)],
            #        device=logits.device
            #    ).unsqueeze(0).expand(logits.size(0), -1)
                
            #    mask_with_cls = self._add_cls_token_mask_to_src_mask(decoder_mask)
            #     logits = layer(logits, src_key_padding_mask=mask_with_cls)
            #else:
            logits = layer(logits)

        output = {"logits": logits}

        # Auxiliary Losses
        # DA Loss: L1 distance between shared features
        da_loss = 0.0
        if len(available_r) > 1:
            for i in range(len(available_r)):
                for j in range(i + 1, len(available_r)):
                    da_loss += torch.mean(torch.abs(available_r[i] - available_r[j]))
            # Normalize by the number of pairs
            num_pairs = len(available_r) * (len(available_r) - 1) / 2
            da_loss = da_loss / num_pairs

        # DC Loss: Domain classification on specific features
        dc_logits = [self.domain_classifier_dc(s) for s in available_s]
        
        domain_labels = torch.tensor(available_indices, device=logits.device)
        
        dc_loss = 0.0
        for i, logit in enumerate(dc_logits):
            target = domain_labels[i].unsqueeze(0).expand(logit.shape[0])
            dc_loss += nn.CrossEntropyLoss()(logit, target)

        output["losses"] = {}
        output["losses"]["da_loss"] = self.loss_alpha * da_loss
        output["losses"]["dc_loss"] = self.loss_beta * (dc_loss / len(dc_logits) if dc_logits else 0.0)

        return output
