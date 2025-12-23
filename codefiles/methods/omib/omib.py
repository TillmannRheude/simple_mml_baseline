import math
import torch 
import torch.nn as nn 
from torch.nn import functional as F
from typing import List, Union

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE
from codefiles.losses.nanbce import WeightedNaNBCEWithLogitsLoss

class MeanPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

class Optimal_Multimodal_Information_Bottleneck_Transformer(nn.Module):

    """
    https://openreview.net/pdf?id=5TUa2UXSpp
    https://arxiv.org/abs/2505.19996 
    
    no code repository published yet
    """

    def __init__(
        self,
        num_modalities: int,
        d_model: int = 512,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_layers: int = 4,
        dim_output: int = 10,
        beta: float = 1e-3,
        task_type: str = 'bce', 
        params_cross_attn_network: dict = {
            "num_layers": 4,
            "dropout": 0.0,
            "num_heads": 4,
            "dim_feedforward": 1024
        }
    ) -> None: 
        super().__init__()
        self.num_modalities = num_modalities
        self.d_model = d_model
        self.warmup_phase = True
        self.beta = beta
        self.task_type = task_type
        if self.task_type not in ['ce', 'bce']:
            raise ValueError("task_type must be 'ce' or 'bce'")
        self.task_loss_fn = nn.CrossEntropyLoss() if task_type == 'ce' else WeightedNaNBCEWithLogitsLoss()

        # Optimal Multimodal Fusion (OMF) Block
        self.vaes = nn.ModuleList([
            nn.Linear(d_model, 2 * d_model) for _ in range(num_modalities)
        ])

        # Task Relevance Branch (TRB) Blocks
        self.trb_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                MeanPool(),
                nn.Linear(d_model, dim_output)
            ) for _ in range(num_modalities)
        ])

        self.ddec_linear = nn.Linear(d_model, dim_output)

        self.apply(self._init_weights)

        cross_attn_network = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=params_cross_attn_network["num_heads"], 
                dim_feedforward=params_cross_attn_network["dim_feedforward"], 
                dropout=params_cross_attn_network["dropout"],
                batch_first=True,
            ),
            num_layers=params_cross_attn_network["num_layers"]
        )
        self.can_fusion = nn.Sequential(
            AddCLSToken(d_model),
            AddPE(d_model),
            cross_attn_network,
            ExtractCLSToken(),
        )

        # Decoder
        transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers
        )
        self.dDec = nn.ModuleList([
            AddCLSToken(d_model),
            AddPE(d_model),
            transformer,
            ExtractCLSToken(),
            self.ddec_linear
        ])

    def set_warmup_phase(self, is_warmup: bool):
        self.warmup_phase = is_warmup
        
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

    def create_concatenated_mask(
            self, 
            modality_mask: torch.Tensor, 
            unimodal_sequences: List[torch.Tensor]
    ) -> torch.Tensor:
        # modality_mask is (bs, num_modalities)
        # unimodal_sequences is a list of tensors (bs, seq_len_i, dim)
        masks = []
        for i, seq in enumerate(unimodal_sequences):
            seq_len = seq.shape[1]
            # modality_mask[:, i] is (bs,). We need (bs, seq_len_i)
            mask_i = modality_mask[:, i].unsqueeze(1).expand(-1, seq_len)
            masks.append(mask_i)
        
        return torch.cat(masks, dim=1) # shape: (bs, sum(seq_lens))

    def _kl_divergence_bernoulli_with_logits(self, l_p: torch.Tensor, l_q: torch.Tensor) -> torch.Tensor:
        """ 
        KL-divergence between two Bernoulli distributions parameterized by logits.
        KL(p || q) where p = sigmoid(l_p) and q = sigmoid(l_q).
        """
        p = torch.sigmoid(l_p)
        # Using log_sigmoid for stability is equivalent to log(p) and log(1-p)
        kl = p * (F.logsigmoid(l_p) - F.logsigmoid(l_q)) + \
             (1 - p) * (F.logsigmoid(-l_p) - F.logsigmoid(-l_q))
        return kl

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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
        x: Union[torch.Tensor, List[torch.Tensor]],
        src_mask: torch.Tensor = None,
        y: torch.Tensor = None
    ) -> dict:

        if isinstance(x, torch.Tensor):
            # Handle CLS token input format (bs, num_modalities, dim)
            # Convert to list of sequences with seq_len=1 to unify the code path
            unimodal_zs = [x[:, i, :].unsqueeze(1) for i in range(x.shape[1])]
        else:
            # Handle list of sequences input format
            unimodal_zs = x

        # Warmup Phase
        if self.warmup_phase:
            trb_logits = []
            # add Stochastic Gaussian Noise to the unimodal embeddings
            for i, z in enumerate(unimodal_zs):
                noise = torch.randn_like(z)
                inp = torch.cat([z, noise], dim=-1)
                trb_logits.append(self.trb_heads[i](inp))

            # Average the unimodal predictions as a proxy for model performance.
            ensembled_logits = torch.stack(trb_logits, dim=0).mean(dim=0)

            # TRB Losses
            losses = {}
            metrics_to_log = {}
            trb_losses = [self.task_loss_fn(logits, y) for logits in trb_logits]
            for i, curr_trb_loss in enumerate(trb_losses):
                losses[f"trb_loss_{i}"] = curr_trb_loss
            metrics_to_log["total_trb_loss"] = torch.sum(torch.stack(trb_losses))
                
            return {
                "logits": ensembled_logits,
                "losses": losses,
                "metrics_to_log": metrics_to_log
            }
    
        # Main training phase
        mus, logvars, unimodal_zetas = [], [], []
        for i, z in enumerate(unimodal_zs):
            # 1. Pass unimodal representations z_i through VAEs to get mu_i, logvar_i
            mu_logvar = self.vaes[i](z)
            mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
            zeta = self.reparameterize(mu, logvar)
            mus.append(mu)
            logvars.append(logvar)
            unimodal_zetas.append(zeta)
        
        # Concatenate sequences and create a combined mask for the transformer
        transformer_input = torch.cat(unimodal_zetas, dim=1)
        if src_mask is not None:
            concatenated_mask = self.create_concatenated_mask(src_mask, unimodal_zetas)
        else:
            concatenated_mask = None

        # 2. Pass all zeta_i through the Cross-Attention Network (CAN) to get the fused MIB (xi)
        xi = transformer_input
        for layer in self.can_fusion:
            if isinstance(layer, nn.TransformerEncoder) and concatenated_mask is not None:
                final_mask = self._add_cls_token_mask_to_src_mask(concatenated_mask)
                xi = layer(xi, src_key_padding_mask=final_mask)
            else:
                xi = layer(xi)

        # 3. Pass the fused MIB (xi) through the final prediction head (dDec)
        d_inp = xi.unsqueeze(1)
        for layer in self.dDec:
            d_inp = layer(d_inp)
        main_logits = d_inp
        
        trb_logits = []
        for i, z in enumerate(unimodal_zs):
            xi_expanded = xi.unsqueeze(1).expand(-1, z.shape[1], -1)
            inp = torch.cat([z, xi_expanded], dim=-1)
            trb_logits.append(self.trb_heads[i](inp))

        losses = {}
        metrics_to_log = {}
        # KL Divergence Losses
        kl_losses_omf = []
        for i in range(self.num_modalities):
            kl_loss = -0.5 * torch.sum(1 + logvars[i] - mus[i].pow(2) - logvars[i].exp(), dim=1)
            kl_losses_omf.append(kl_loss.mean())
        
        # Dynamic Regularization weights 'r'
        kls_y_hat = []
        main_logits_detached = main_logits.detach()

        for i in range(self.num_modalities):
            trb_logits_detached = trb_logits[i].detach()
            if self.task_type == 'ce':
                # KL(p_trb || p_main)
                kl = F.kl_div(
                    F.log_softmax(trb_logits_detached, dim=-1), 
                    F.softmax(main_logits_detached, dim=-1), 
                    reduction='batchmean', 
                    log_target=False
                )
            else: # bce
                kl = self._kl_divergence_bernoulli_with_logits(trb_logits_detached, main_logits_detached).mean()
            kls_y_hat.append(kl)

        rs = []
        # The paper's formulation is for r_2 relative to modality 1.
        # We generalize: r_i is relative to modality 0.
        for i in range(1, self.num_modalities):
            # The paper's formula r = 1-tanh(log(KL2/KL1)) is proportional to KL1/KL2.
            # Our ratio is KL_i / KL_0, so r_i will be proportional to KL_0/KL_i.
            ratio = kls_y_hat[i] / (kls_y_hat[0] + 1e-8)
            r = 1.0 - torch.tanh(torch.log(ratio + 1e-8))
            rs.append(r)

        weighted_kl_loss = kl_losses_omf[0]
        for i, r in enumerate(rs):
            weighted_kl_loss += r * kl_losses_omf[i+1]

        # TRB Losses (calculated after main trb_logits are available)
        trb_losses = [self.task_loss_fn(logits, y) for logits in trb_logits]
        
        for i, curr_trb_loss in enumerate(trb_losses):
            losses[f"loss_trb_{i}"] = curr_trb_loss
        losses["regularization_omf"] = self.beta * weighted_kl_loss
        
        metrics_to_log["loss_omf_y"] = self.task_loss_fn(main_logits, y)
        for i, curr_kl_loss in enumerate(kl_losses_omf):
            metrics_to_log[f"unweighted_kl_loss_{i}"] = curr_kl_loss
        for i, curr_r in enumerate(rs):
            metrics_to_log[f"r_{i+1}"] = curr_r

        return {
            "logits": main_logits,
            "losses": losses,
            "metrics_to_log": metrics_to_log,
        }
