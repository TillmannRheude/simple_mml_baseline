import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from itertools import combinations 

class ITCLoss(nn.Module):

    def __init__(
            self, 
            embed_dim: int = 128, 
            input_dims: tuple = (768, 768),
            temperature: float = 0.07
        ) -> None: 
        super().__init__()

        self.embed_dim = embed_dim
        self.num_modalities = len(input_dims)

        # Projection heads (g_v and g_w)
        self.mod_projs = nn.ModuleList()
        for modality_proj in input_dims:
            self.mod_projs.append(nn.Linear(modality_proj, embed_dim))

        # Learnable temperature parameter (logit_scale = 1 / tau)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(
            self, 
            queries: torch.Tensor,
            keys: torch.Tensor
        ) -> torch.Tensor:
        # Calculates the similarity matrix between a set of queries and keys.
        logit_scale = self.logit_scale.exp().clamp(max=100)
        sim_matrix = torch.matmul(queries, keys.T) * logit_scale
        return sim_matrix


class ITMLoss(nn.Module):

    def __init__(
        self,
        itm_head_input_dim: int, 
        modality_proj_layers: nn.ModuleList, 
        temperature_logit_scale: nn.Parameter,
        multimodal_encoder: nn.Module
    ) -> None:
        super().__init__()

        self.multimodal_encoder = multimodal_encoder
        
        self.itm_head = nn.Linear(itm_head_input_dim, 2)
        
        self.mod_projs = modality_proj_layers
        self.logit_scale = temperature_logit_scale
        self.num_modalities = len(modality_proj_layers)
        
    def forward(
            self,
            modality_embeddings: list = [torch.Tensor, torch.Tensor]
        ) -> None:
        bs, device = modality_embeddings[0].shape[0], modality_embeddings[0].device
        modality_embeddings_cls = [emb[:, 0, :] for emb in modality_embeddings]

        # Generate hard negative inputs for each modality with ITC similarity
        all_negative_inputs = []
        with torch.no_grad():
            projected_embeddings = [
                F.normalize(self.mod_projs[i](mod_cls), dim=-1) for i, mod_cls in enumerate(modality_embeddings_cls)
            ]
            logit_scale = self.logit_scale.exp().clamp(max=100)
            
            for i, j in combinations(range(self.num_modalities), 2):
                # Sample hard negative from modality j for modality i
                sim_i_j = torch.matmul(projected_embeddings[i], projected_embeddings[j].T) * logit_scale
                sim_i_j.fill_diagonal_(-float('inf'))
                weights_i_j = F.softmax(sim_i_j, dim=1)
                neg_indices_j = torch.multinomial(weights_i_j, num_samples=1).squeeze(-1)

                negative_input_ij = []
                for mod_idx in range(self.num_modalities):
                    if mod_idx == j:
                        negative_input_ij.append(modality_embeddings[j][neg_indices_j])
                    else:
                        negative_input_ij.append(modality_embeddings[mod_idx])
                all_negative_inputs.append(negative_input_ij)

                # Sample hard negative from modality i for modality j
                sim_j_i = torch.matmul(projected_embeddings[j], projected_embeddings[i].T) * logit_scale
                sim_j_i.fill_diagonal_(-float('inf'))
                weights_j_i = F.softmax(sim_j_i, dim=1)
                neg_indices_i = torch.multinomial(weights_j_i, num_samples=1).squeeze(-1)
                
                negative_input_ji = []
                for mod_idx in range(self.num_modalities):
                    if mod_idx == i:
                        negative_input_ji.append(modality_embeddings[i][neg_indices_i])
                    else:
                        negative_input_ji.append(modality_embeddings[mod_idx])
                all_negative_inputs.append(negative_input_ji)

        # Fuse positive and negative pairs
        positive_cls_fused = self.multimodal_encoder(modality_embeddings)
        negative_cls_fused_list = []
        for neg_input_list in all_negative_inputs:
            fused_neg = self.multimodal_encoder(neg_input_list)
            negative_cls_fused_list.append(fused_neg)

        # Calculate loss
        all_cls_fused = torch.cat([positive_cls_fused] + negative_cls_fused_list, dim=0) 

        pos_labels = torch.ones(bs, dtype=torch.long, device=device)
        num_neg_sets = len(all_negative_inputs)
        neg_labels = torch.zeros(bs * num_neg_sets, dtype=torch.long, device=device)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        logits = self.itm_head(all_cls_fused)

        loss = F.cross_entropy(logits, labels)

        return loss
