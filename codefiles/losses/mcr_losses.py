import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    
    def __init__(
        self, 
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (list[torch.Tensor]): List of representations for each modality.
                Each tensor is of shape (batch_size, feature_dim).
            labels (torch.Tensor): Ground truth labels of shape (batch_size) for single-label
                or (batch_size, num_classes) for multi-label.
        Returns:
            torch.Tensor: The supervised contrastive loss.
        """
        bs = features[0].shape[0]
        device = features[0].device

        # Handle single-label vs multi-label cases to create the positive pair mask
        if labels.ndim == 1:
            labels = labels.view(-1, 1)
        
        is_multilabel = labels.ndim > 1 and labels.shape[1] > 1
        if is_multilabel:
            # Multi-label: positive pairs are those that share at least one class
            mask = (torch.matmul(labels.float(), labels.float().T) > 0).float()
        else:
            # Single-label: positive pairs are those with the same class id
            mask = torch.eq(labels, labels.T).float()

        features = [F.normalize(f, p=2, dim=1) for f in features]
        all_features = torch.cat(features, dim=0)  # (M * batch_size, feature_dim)
        
        anchor_dot_contrast = torch.div(
            torch.matmul(all_features, all_features.T),
            self.temperature
        )
        
        # Tile the mask for all modalities
        mask = mask.repeat(len(features), len(features))
        
        # Forbid loss on diagonal (instance's self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(bs * len(features)).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.view(len(features), bs).mean()
        return loss


class ConditionalEntropyBottleneck(nn.Module):

    def __init__(
        self, 
        d_model: int, 
        num_classes: int, 
        num_modalities: int,
        hidden_dim: int,
        num_layers: int,
        is_multilabel: bool = False,
    ):
        super().__init__()
        self.is_multilabel = is_multilabel
        
        layers = []
        output_dim = d_model * num_modalities

        if num_layers == 1:
            if self.is_multilabel:
                layers.append(nn.Linear(num_classes, output_dim))
            else:
                layers.append(nn.Embedding(num_classes, output_dim))
        else:
            # First layer
            if self.is_multilabel:
                layers.append(nn.Linear(num_classes, hidden_dim))
            else:
                layers.append(nn.Embedding(num_classes, hidden_dim))
            
            layers.append(nn.ReLU())
            
            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.reconstruction_head = nn.Sequential(*layers)

    def forward(self, unimodal_summary_reps: list[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            unimodal_summary_reps (list[torch.Tensor]): List of unfused unimodal summary representations.
            y (torch.Tensor): The ground truth labels. For multi-class, shape (B) or (B,1). For multi-label, shape (B, C).
        Returns:
            torch.Tensor: The CEB reconstruction loss.
        """
        target_reps = torch.cat(unimodal_summary_reps, dim=1)
        
        if self.is_multilabel:
            # For multi-label (BCE), y is a float vector. Replace NaNs with 0.0
            # so the linear layer can process it without error.
            y_safe = torch.nan_to_num(y, nan=0.0)
            input_for_head = y_safe.float()
        else:
            # For multi-class (CE), y contains class indices. nn.Embedding expects Long.
            if y.ndim > 1:
                y = y.squeeze(-1)
            input_for_head = y.long()

        reconstructed_reps = self.reconstruction_head(input_for_head)
        return F.mse_loss(reconstructed_reps, target_reps) 