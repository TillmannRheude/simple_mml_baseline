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
        modality_is_available = ~src_mask

        shared_features = []
        specific_features = []

        for i in range(self.num_modalities):
            # Run the encoders for the complete batch. The padding mask prevents
            # missing modality tokens from contributing to the CLS embedding;
            # the per-sample availability mask below decides whether to use the
            # resulting feature or a generated shared feature.
            shared_features.append(
                self._run_encoder(self.shared_encoder, x_list[i], mask_list[i])
            )
            specific_features.append(
                self._run_encoder(self.specific_encoders[i], x_list[i], mask_list[i])
            )

        # Generate features for the decoder
        shared_stack = torch.stack(shared_features, dim=1)
        available_weights = modality_is_available.unsqueeze(-1).to(shared_stack.dtype)
        num_available = available_weights.sum(dim=1).clamp(min=1.0)
        r_fused = (shared_stack * available_weights).sum(dim=1) / num_available

        features_for_decoder = []
        for i in range(self.num_modalities):
            r_i = shared_features[i]
            s_i = specific_features[i]
            f_i = self.f_projections[i](torch.cat([r_i, s_i], dim=-1)) + r_i
            use_observed_feature = modality_is_available[:, i].unsqueeze(-1)
            features_for_decoder.append(
                torch.where(use_observed_feature, f_i, r_fused)
            )

        # Decoder
        decoder_input = torch.stack(features_for_decoder, dim=1) # (B, N_modalities, D)
        logits = decoder_input
        for layer in self.decoder:
            logits = layer(logits)

        output = {"logits": logits}

        # Auxiliary Losses
        # DA Loss: L1 distance between shared features
        da_losses = []
        for i in range(self.num_modalities):
            for j in range(i + 1, self.num_modalities):
                both_available = (
                    modality_is_available[:, i] & modality_is_available[:, j]
                )
                if both_available.any():
                    da_losses.append(
                        torch.mean(torch.abs(
                            shared_features[i][both_available]
                            - shared_features[j][both_available]
                        ))
                    )
        da_loss = (
            torch.stack(da_losses).mean()
            if da_losses
            else x.new_zeros(())
        )

        # DC Loss: Domain classification on specific features
        dc_losses = []
        for i, specific_feature in enumerate(specific_features):
            available = modality_is_available[:, i]
            if available.any():
                dc_logits = self.domain_classifier_dc(specific_feature[available])
                target = torch.full(
                    (dc_logits.shape[0],),
                    i,
                    device=logits.device,
                    dtype=torch.long,
                )
                dc_losses.append(nn.CrossEntropyLoss()(dc_logits, target))
        dc_loss = (
            torch.stack(dc_losses).mean()
            if dc_losses
            else x.new_zeros(())
        )

        output["losses"] = {}
        output["losses"]["da_loss"] = self.loss_alpha * da_loss
        output["losses"]["dc_loss"] = self.loss_beta * dc_loss

        return output
