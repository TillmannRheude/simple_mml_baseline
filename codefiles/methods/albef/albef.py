from selectors import BaseSelector
from termios import BSDLY
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from itertools import combinations

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE
from codefiles.methods.albef.losses import ITCLoss, ITMLoss
from codefiles.methods.albef.fusion import PairwiseCrossAttentionFusion, IncrementalCrossAttentionFusion

class ALBEF(nn.Module):
    """
    https://arxiv.org/abs/2107.07651
    
    https://github.com/salesforce/ALBEF?tab=readme-ov-file
    """

    def __init__(
            self, 
            embed_dim: int = 128, 
            itc_temperature: float = 0.07,
            distill_temperature: float = 0.1,
            queue_size: int = 65536,
            momentum: float = 0.995,
            alpha: float = 0.4,
            params_transformerhead: dict = {
                "d_model": 512,
                "nhead": 4,
                "dim_feedforward": 1024,
                "dropout": 0.0,
                "num_layers": 4,
                "dim_output": 10,
            },
            n_modalities: int = 2,
            itc_weight: float = 1.0,
            itm_weight: float = 1.0,
        ) -> None:
        super().__init__()

        self.num_modalities = n_modalities
        self.input_dims = [params_transformerhead["d_model"]] * n_modalities
        self.itc_weight = itc_weight
        self.itm_weight = itm_weight
        self.alpha = alpha
        self.distill_temp = distill_temperature

        self.add_cls_token = AddCLSToken(embdim=params_transformerhead["d_model"])

        # Projection Head
        self.linear_out = nn.Linear(params_transformerhead["d_model"], params_transformerhead["dim_output"])

        # ITC loss
        self.itc_loss_calculator = ITCLoss(
            embed_dim, 
            self.input_dims, 
            itc_temperature
        )

        # Pairwise Multimodal Fusion Encoder
        self.multimodal_encoder = IncrementalCrossAttentionFusion(  # PairwiseCrossAttentionFusion(
            d_model=params_transformerhead["d_model"],
            nhead=params_transformerhead["nhead"],
            num_layers=params_transformerhead["num_layers"],
            dim_feedforward=params_transformerhead["dim_feedforward"],
            dropout=params_transformerhead["dropout"],
            num_modalities=n_modalities,
        )

        # ITM loss 
        self.itm_loss = ITMLoss(
            params_transformerhead["d_model"],
            self.itc_loss_calculator.mod_projs, 
            self.itc_loss_calculator.logit_scale, 
            self.multimodal_encoder
        )

        self.momentum = momentum

        # Queues for ITC loss
        self.queue_size = queue_size
        for i in range(self.num_modalities):
            self.register_buffer(f"queue_{i}", F.normalize(torch.randn(embed_dim, self.queue_size), dim=0))
            self.register_buffer(f"queue_ptr_{i}", torch.zeros(1, dtype=torch.long))

        # Initialize weights
        self.apply(self._init_weights)

        # Create momentum model
        self.model_m = deepcopy(self)
        self._disable_grad(self.model_m)
      
    @staticmethod
    def _disable_grad(model: nn.Module):
        for param in model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for param, param_m in zip(self.parameters(), self.model_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, modality_embeds_m):
        for i in range(self.num_modalities):
            keys = self._gather_keys(modality_embeds_m[i])
            bs = keys.shape[0]
            
            queue = getattr(self, f"queue_{i}")
            ptr = int(getattr(self, f"queue_ptr_{i}"))

            # Replace the keys at ptr (dequeue and enqueue)
            if ptr + bs > self.queue_size:
                queue[:, ptr:] = keys[:self.queue_size - ptr].T
                queue[:, :ptr + bs - self.queue_size] = keys[self.queue_size - ptr:].T
            else:
                queue[:, ptr:ptr + bs] = keys.T

            ptr = (ptr + bs) % self.queue_size
            getattr(self, f"queue_ptr_{i}")[0] = ptr

    @torch.no_grad()
    def _gather_keys(self, keys):
        return keys

    def _init_weights(
            self,
            m
        ) -> None: 
        if isinstance(m, (torch.nn.LayerNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Parameter):
            if m.dim() > 1:
                torch.nn.init.xavier_normal_(m)
            else:
                torch.nn.init.zeros
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def compute_outputs(self, x_list):
        embeddings_cls = [emb[:, 0, :] for emb in x_list]
        projected_embeddings = []
        for i in range(self.num_modalities):
            embed = self.itc_loss_calculator.mod_projs[i](embeddings_cls[i])
            projected_embeddings.append(F.normalize(embed, dim=-1))
        
        fused_embedding = self.multimodal_encoder(x_list)
        logits = self.linear_out(fused_embedding)
        return logits, projected_embeddings

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor,
        y: torch.Tensor = None
    ) -> torch.Tensor:

        if not isinstance(x, list):
            # extract tensor (bs, n_seq*n_mod, dim) to list of tensors (bs, n_seq, dim)
            x_list = [x[:, i * x.shape[1] // self.num_modalities: (i + 1) * x.shape[1] // self.num_modalities, :] for i in range(self.num_modalities)]
        else:
            x_list = x
        
        # Add CLS token to each modality's sequence
        x_list = [self.add_cls_token(mod_emb) for mod_emb in x_list]
        
        # Compute output logits
        logits, projected_embeddings = self.compute_outputs(x_list)
        
        # ITM Loss -> fine align step 
        # Sample negative samples with ITC similarity and fuse positive and negative pairs
        # to predict whether its positive (1) or negative (0) -> i.e., standard CE loss
        itm_loss = self.itm_loss(x_list)
        
        # ITC Loss Calculation -> coarse align step 
        # Match image and text by getting positive and negative samples from the queue of the 
        # momentum model and the queue of the student model
        bs, device = x_list[0].shape[0], x_list[0].device
        labels = torch.arange(bs, dtype=torch.long, device=device)
        total_itc_loss = 0.0
        total_itc_distill_loss = 0.0
        num_pairs = 0
        
        with torch.no_grad():
            # Update momentum model and get projected embeddings of momentum model
            self._momentum_update()
            _, projected_embeddings_m = self.model_m.compute_outputs(x_list)

        # Dequeue and enqueue projected embeddings of momentum model
        queues = [getattr(self, f"queue_{i}") for i in range(self.num_modalities)]
        self._dequeue_and_enqueue(projected_embeddings_m)

        for i, j in combinations(range(self.num_modalities), 2):
            keys_j = torch.cat([projected_embeddings_m[j], queues[j].T], dim=0)
            keys_i = torch.cat([projected_embeddings_m[i], queues[i].T], dim=0)

            sim_i_j = self.itc_loss_calculator(projected_embeddings[i], keys_j)
            sim_j_i = self.itc_loss_calculator(projected_embeddings[j], keys_i)

            loss_i2j = F.cross_entropy(sim_i_j, labels)
            loss_j2i = F.cross_entropy(sim_j_i, labels)
            total_itc_loss += (loss_i2j + loss_j2i)
            num_pairs += 2

            with torch.no_grad():
                sim_i_j_m = self.model_m.itc_loss_calculator(projected_embeddings_m[i], keys_j)
                sim_j_i_m = self.model_m.itc_loss_calculator(projected_embeddings_m[j], keys_i)

            soft_target_i2j = F.softmax(sim_i_j_m / self.distill_temp, dim=1)
            soft_target_j2i = F.softmax(sim_j_i_m / self.distill_temp, dim=1)
            
            distill_loss_i2j = -torch.sum(F.log_softmax(sim_i_j / self.distill_temp, dim=1) * soft_target_i2j, dim=1).mean()
            distill_loss_j2i = -torch.sum(F.log_softmax(sim_j_i / self.distill_temp, dim=1) * soft_target_j2i, dim=1).mean()
            total_itc_distill_loss += (distill_loss_i2j + distill_loss_j2i)

        itc_hard_loss = total_itc_loss / num_pairs
        itc_distill_loss = total_itc_distill_loss / num_pairs
        itc_loss = (1 - self.alpha) * itc_hard_loss + self.alpha * itc_distill_loss
        
        return_dict = {
            "logits": logits,
            "losses": {
                "itc_loss": itc_loss * self.itc_weight,
                "itm_loss": itm_loss * self.itm_weight,
            }
        }

        with torch.no_grad():
            logits_m, _ = self.model_m.compute_outputs(x_list)
        return_dict['logits_m'] = logits_m
        
        return return_dict