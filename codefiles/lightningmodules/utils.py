import torch 
import schedulefree

import torch.nn as nn 
import torch.nn.functional as F
import pytorch_lightning as pl 

from torchmetrics.functional import auroc
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

class LightningModuleParent(pl.LightningModule):
    def __init__(
            self,
            **kwargs
    ) -> None: 
        super().__init__()

        self.save_hyperparameters()
    
    def forward(
            self, 
            x: list = [torch.Tensor],
        ) -> torch.Tensor:
        return self.model(x)

    def on_train_start(self):
        self.optimizers().train()
        
    def on_validation_start(self):
        self.optimizers().eval()
        
    def on_validation_end(self):
        self.optimizers().train()

    def configure_optimizers(self):        
        optimizer = schedulefree.AdamWScheduleFree(
            self.parameters(), 
            lr=self.params_optimizer["lr"],
            weight_decay=self.params_optimizer["weight_decay"], 
            eps=self.params_optimizer["eps"],
            warmup_steps=self.params_optimizer["warmup_steps"],
            betas=self.params_optimizer["betas"]
        )
        return optimizer



class WeightedNaNBCEWithLogitsLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, weight_mode='dynamic'):
        """
        Initialize the weighted BCE loss that handles NaN values
        
        Args:
            label_smoothing (float): Amount of label smoothing to apply
            weight_mode (str): How to handle class weights:
                - 'dynamic': Calculate weights based on batch statistics
                - 'manual': Use manually specified weights
                - None: No weighting
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.label_smoothing = label_smoothing
        self.weight_mode = weight_mode
        self.manual_weights = None

    def set_manual_weights(self, weights):
        """Set manual weights for each class"""
        self.manual_weights = weights

    def calculate_weights(self, target, mask):
        valid_targets = target[mask]
        if len(valid_targets) == 0:
            return torch.ones_like(target)

        # Ensure valid_targets only contains 0s and 1s
        valid_mask = ((valid_targets == 0) | (valid_targets == 1))
        valid_targets = valid_targets[valid_mask]
        
        if len(valid_targets) == 0:
            return torch.ones_like(target)

        n_zeros = (valid_targets == 0).sum()
        n_ones = (valid_targets == 1).sum()
        
        # Avoid division by zero
        total = max(n_zeros + n_ones, 1)
        
        # Calculate inverse frequency weights
        w0 = total / (2 * max(n_zeros, 1))
        w1 = total / (2 * max(n_ones, 1))
        
        # Create weight tensor matching target shape
        weights = torch.ones_like(target)
        weights[target == 0] = w0
        weights[target == 1] = w1
        
        weights = torch.clamp(weights, min=1e-6, max=1e6)
        return weights

    def forward(self, input, target):
        # Create a mask for valid targets (not NaN)
        mask = ~torch.isnan(target)
        
        # Clone target to avoid in-place operations
        target = target.clone()
        
        # Apply label smoothing to valid targets
        if self.label_smoothing > 0:
            target = torch.where(target == 1, 1 - self.label_smoothing, target)
            target = torch.where(target == 0, self.label_smoothing, target)
        
        # Compute weights
        if self.weight_mode == 'dynamic':
            weights = self.calculate_weights(target, mask)
        elif self.weight_mode == 'manual' and self.manual_weights is not None:
            weights = torch.ones_like(target)
            weights[target == 0] = self.manual_weights[0]
            weights[target == 1] = self.manual_weights[1]
        else:
            weights = torch.ones_like(target)
        
        # Compute BCE loss only for valid targets
        loss = self.bce(input, target.nan_to_num(0))  # NaNs replaced with 0 but masked later
        loss = loss * weights * mask  # Apply weights and mask out NaN contributions
        
        # Normalize loss by the sum of weights of valid elements
        valid_weights_sum = (weights * mask).sum()
        if valid_weights_sum > 0:
            return loss.sum() / valid_weights_sum
        else:
            return torch.tensor(0.0, requires_grad=True).to(input.device)

class NaNMultilabelAUROC(Metric):
    
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        average: str = "macro",
    ):
        super().__init__()
        self.num_labels = num_labels
        self.average = average
        
        self.ignore_index = num_labels + 1
        
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

        #self.preds = []
        #self.target = []

    def update(self, preds, target) -> None:
        mask = torch.isnan(target)
        target = target.clone()

        target[mask] = self.ignore_index

        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)

        target = target.to(torch.long)

        return auroc(preds, target, task="multilabel", num_classes=self.num_labels, num_labels=self.num_labels, average=self.average, ignore_index=self.ignore_index)


def get_input(name_dataset, dataloader_output):
    if name_dataset == "mimic_haim":
        x = [dataloader_output["image"], dataloader_output["lab"]]
    elif name_dataset == "mimic_symile":
        x = [
            dataloader_output["image"],
            dataloader_output["lab"], 
            dataloader_output["ecg"]
        ]
    elif name_dataset == "fmnist" or name_dataset == "mnist":
        if len(dataloader_output[0]) == 2:
            x = [dataloader_output[0][0].to(torch.float32), 
                 dataloader_output[0][1].to(torch.float32)]
        if len(dataloader_output[0]) == 4:
            x = [dataloader_output[0][0].to(torch.float32),
                 dataloader_output[0][1].to(torch.float32),
                 dataloader_output[0][2].to(torch.float32),
                 dataloader_output[0][3].to(torch.float32)]
        x_missing = None 
    elif name_dataset == "mosi" or name_dataset == "mosei": 
        x = [dataloader_output[0][0],
             dataloader_output[0][1],
             dataloader_output[0][2]]
        return x
    else: 
        raise NotImplementedError

    return x

def get_target(name_dataset, dataloader_output):
    if name_dataset == "mimic_haim" or name_dataset == "mimic_symile":
        y = dataloader_output["target"]
    elif name_dataset == "fmnist" or name_dataset == "mnist":
        y = dataloader_output[1]  # .to(torch.float32)
    elif name_dataset == "mosi" or name_dataset == "mosei":
        y = dataloader_output[1]  # .long()
    else: 
        raise NotImplementedError

    return y