import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from collections import defaultdict

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE
from codefiles.methods.utils import mimetic_init_svd_
from codefiles.losses.nanbce import WeightedNaNBCEWithLogitsLoss

class Prototypical_Modal_Rebalance_Transformer(nn.Module):

    """
    https://openaccess.thecvf.com/content/CVPR2023/papers/Fan_PMR_Prototypical_Modal_Rebalance_for_Multimodal_Learning_CVPR_2023_paper.pdf

    https://github.com/fanyunfeng-bit/Modal-Imbalance-PMR
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
        params_pmr: dict = {
            "alpha": 1.0,
            "mu": 1e-3,
            "regularization_epochs": 10,
            "num_classes": 2,
            "is_multilabel": False,
            "num_bins_regression": 10,
        },
    ) -> None: 
        super().__init__()

        self.num_modalities = num_modalities
        self.d_model = d_model
        self.num_classes = params_pmr["num_classes"]
        self.is_multilabel = params_pmr["is_multilabel"]
        self.num_bins_regression = params_pmr.get("num_bins_regression", 10)

        self.alpha = params_pmr["alpha"]
        self.mu = params_pmr["mu"]
        self.regularization_epochs = params_pmr["regularization_epochs"]
        self.current_epoch = 0
        
        # For regression, num_classes for prototypes will be the number of bins
        self.prototype_num_classes = self.num_bins_regression if not self.is_multilabel and self.num_classes == 1 else self.num_classes
        # The representation dimension for EACH modality is d_model, not d_model / num_modalities
        rep_dim = self.d_model
        self.prototypes = nn.ParameterList([
            nn.Parameter(torch.zeros(self.prototype_num_classes, rep_dim), requires_grad=False)
            for _ in range(self.num_modalities)
        ])
        self.prototypes_initialized = False
        self.register_buffer('regression_y_min', torch.tensor(0.0))
        self.register_buffer('regression_y_max', torch.tensor(1.0))

        self.bce_loss = WeightedNaNBCEWithLogitsLoss()
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

    def update_prototypes(self, dataloader, encoders, get_input_fn, get_target_fn, dataset_name, device, epsilon=0.99):
        was_training = encoders.training
        encoders.eval()

        all_reps = [defaultdict(list) for _ in range(self.num_modalities)]
        all_y = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > max(10, int(len(dataloader) * 0.1)):
                    break  # Prototypes can be calculated on a subset of data (Sec. 4.1)
                
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                x = get_input_fn(dataset_name, batch)
                y = get_target_fn(dataset_name, batch)
                
                all_y.append(y)
                unimodal_reps_cat = encoders(x)
                split_reps = list(torch.split(unimodal_reps_cat, 1, dim=1))

                for i in range(self.num_modalities):
                    modality_i_reps = split_reps[i].squeeze(1)  # [bs, embedding_dim]
                    all_reps[i][batch_idx] = (modality_i_reps, y)

        # for regression, bin labels into bins
        all_y_tensor = torch.cat(all_y)
        is_regression = all_y_tensor.dtype in [torch.float32, torch.float64] and not self.is_multilabel and len(torch.unique(all_y_tensor)) > self.prototype_num_classes
        binned_all_reps = [defaultdict(list) for _ in range(self.num_modalities)]
        
        if is_regression:
            self.regression_y_min.data = torch.min(all_y_tensor)
            self.regression_y_max.data = torch.max(all_y_tensor)
            bins = torch.linspace(self.regression_y_min, self.regression_y_max, steps=self.num_bins_regression, device=device)

        for i in range(self.num_modalities):
            for batch_idx in all_reps[i]:
                reps_i, y = all_reps[i][batch_idx]
                y_class_indices = y
                if is_regression:
                    y_class_indices = torch.bucketize(y, bins, right=True).clamp(0, self.num_bins_regression - 1)

                if not self.is_multilabel and y_class_indices.ndim > 1 and y_class_indices.shape[1] == 1:
                    y_class_indices = y_class_indices.squeeze(1)

                for j in range(reps_i.shape[0]):
                    if self.is_multilabel:
                        for k in range(self.num_classes):
                            if y_class_indices.shape[1] > k and y_class_indices[j, k] == 1: 
                                binned_all_reps[i][k].append(reps_i[j])
                    else:
                        binned_all_reps[i][y_class_indices[j].item()].append(reps_i[j])
        
        new_prototypes = [torch.zeros_like(p) for p in self.prototypes]
        for i in range(self.num_modalities):
            for k in range(self.prototype_num_classes):
                if len(binned_all_reps[i][k]) > 0:
                    new_prototypes[i][k] = torch.stack(binned_all_reps[i][k]).mean(dim=0)

        if self.prototypes_initialized:
            for i in range(self.num_modalities):
                self.prototypes[i].data = epsilon * self.prototypes[i].data + (1 - epsilon) * new_prototypes[i]
        else:
            for i in range(self.num_modalities):
                self.prototypes[i].data = new_prototypes[i]
            self.prototypes_initialized = True

        if was_training:
            encoders.train()

    def forward(
        self, 
        x: torch.Tensor,
        src_mask: torch.Tensor = None,
        y: torch.Tensor = None
    ) -> dict:
        
        unimodal_reps = list(torch.split(x, 1, dim=1))

        fused_x = x
        for layer in self.transformer_cls:
            if isinstance(layer, nn.TransformerEncoder) and src_mask is not None:
                src_mask = self._add_cls_token_mask_to_src_mask(src_mask)
                fused_x = layer(fused_x, src_key_padding_mask=src_mask)
            else:
                fused_x = layer(fused_x)

        output_dict = {
            "logits": fused_x,
            "unimodal_logits": [self.linear_out(rep.squeeze(1)) for rep in unimodal_reps],
            "pmr": True
        }

        if self.prototypes_initialized:
            is_regression = y.dtype in [torch.float32, torch.float64] and not self.is_multilabel and self.num_classes == 1
            is_bce = (y.dtype in [torch.float32, torch.float64]) and not is_regression or self.is_multilabel
            
            y_class_indices = y
            if is_regression:
                bins = torch.linspace(self.regression_y_min, self.regression_y_max, steps=self.num_bins_regression, device=y.device)
                y_class_indices = torch.bucketize(y, bins, right=True).clamp(0, self.num_bins_regression - 1)

            losses = {}
            performance_metrics = []
            y_squeezed = y_class_indices.squeeze(1) if y_class_indices.ndim > 1 and not is_bce else y_class_indices

            for i in range(self.num_modalities):
                # Squeeze the modality dimension for cdist: [bs, 1, d_model] -> [bs, d_model]
                current_unimodal_rep = unimodal_reps[i].squeeze(1)
                proto_logits = -torch.cdist(current_unimodal_rep, self.prototypes[i])
                if is_bce:
                    # Use direct loss as the performance metric (higher loss = worse performance)
                    loss = self.bce_loss(proto_logits, y)
                    performance_metrics.append(loss)
                else:
                    # For CE, the paper/official code uses the sum of probabilities as the performance metric.
                    # A lower sum indicates worse performance. We use the negative sum to make it act like a loss 
                    probs = F.softmax(proto_logits, dim=1)
                    gt_probs = probs.gather(1, y_squeezed.unsqueeze(1))
                    performance_metrics.append(-gt_probs.sum())

            metrics_tensor = torch.stack(performance_metrics)
            
            # Higher metric value (higher loss or lower prob sum) means slower modality
            max_metric, min_metric = torch.max(metrics_tensor), torch.min(metrics_tensor)
            # PCE weights are for slow (high metric) modalities
            pce_weights = torch.clamp(metrics_tensor / (min_metric + 1e-8) - 1, min=0, max=1)
            # PER weights are for fast (low metric) modalities
            per_weights = torch.clamp(max_metric / (metrics_tensor + 1e-8) - 1, min=0, max=1)

            total_pce_loss, total_per_loss = 0.0, 0.0
            for i in range(self.num_modalities):
                current_unimodal_rep = unimodal_reps[i].squeeze(1)
                proto_logits = -torch.cdist(current_unimodal_rep, self.prototypes[i])
                
                # Consistently use the appropriate loss function
                if is_bce:
                    pce_loss_i = self.bce_loss(proto_logits, y)
                else:
                    pce_loss_i = F.cross_entropy(proto_logits, y_squeezed)

                total_pce_loss += pce_weights[i] * pce_loss_i

                if self.current_epoch < self.regularization_epochs:
                    if is_bce:
                        probs = torch.sigmoid(proto_logits)
                        entropy = -torch.sum(probs * torch.log2(probs + 1e-8) + (1-probs) * torch.log2(1-probs + 1e-8), dim=1).mean()
                    else:
                        probs = F.softmax(proto_logits, dim=1)
                        entropy = -torch.sum(probs * torch.log2(probs + 1e-8), dim=1).mean()
                    total_per_loss += per_weights[i] * entropy
            
            losses["loss_pmr_pce"] = self.alpha * total_pce_loss
            if self.current_epoch < self.regularization_epochs and torch.is_tensor(total_per_loss):
                losses["loss_pmr_per"] = -self.mu * total_per_loss
            
            output_dict["losses"] = losses

        return output_dict
