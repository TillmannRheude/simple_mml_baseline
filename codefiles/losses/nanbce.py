import torch 
import torch.nn as nn 


class WeightedNaNBCEWithLogitsLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, weight_mode='dynamic', reduction='none'):
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
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
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
        weights[target == 0] = w0.to(weights.dtype)
        weights[target == 1] = w1.to(weights.dtype)
        
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