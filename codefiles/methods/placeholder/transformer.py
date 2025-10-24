import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE

class PEN_Module(nn.Module):
    """
    A PyTorch module implementing the Perturbative Equilibrium Network (PEN) mechanism.
    
    This module takes concatenated multimodal embeddings and simulates a dynamic system
    reaching an equilibrium. During training, it applies a "stress" by masking one of
    the modalities and calculates a new, restored equilibrium.
    """
    def __init__(self, emb_dim: int, num_modalities: int, num_iterations: int = 6, 
                 num_perturb_steps: int = 2, perturb_prob: float = 0.5, nhead: int = 8):
        """
        Args:
            emb_dim (int): The embedding dimension of each modality.
            num_modalities (int): The number of input modalities (n).
            num_iterations (int): Number of steps to reach the initial equilibrium.
            num_perturb_steps (int): Number of steps to run after perturbation.
            perturb_prob (float): Probability of perturbing a modality during training.
            nhead (int): Number of heads for the interaction block (transformer layer).
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.num_perturb_steps = num_perturb_steps
        self.perturb_prob = perturb_prob
        self.num_modalities = num_modalities

        # The InteractionBlock is a standard Transformer Encoder Layer.
        # It allows all modalities to interact and update each other.
        interaction_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=nhead, 
            dim_feedforward=emb_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.interaction_block = nn.TransformerEncoder(interaction_layer, num_layers=1)

    def _perturb(self, x: torch.Tensor) -> torch.Tensor:
        """Applies a 'stress' by randomly masking modalities."""
        perturbed_x = x.clone()
        bs, n, _ = x.shape
        
        for i in range(bs):
            # Choose a single modality to mask for each item in the batch
            # This is a common strategy to ensure the system is not overly stressed
            modality_to_mask = torch.randint(0, self.num_modalities, (1,)).item()
            if torch.rand(1) < self.perturb_prob:
                perturbed_x[i, modality_to_mask, :] = 0.0
                
        return perturbed_x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass implementing the PEN mechanism.

        Args:
            x (torch.Tensor): The input tensor of shape (bs, n, emb_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - equilibrium_embedding (bs, n, emb_dim): The stable state.
                - new_equilibrium_embedding (bs, n, emb_dim): The restored state after perturbation.
                  (During inference, this is the same as the equilibrium_embedding).
        """
        # 1. Reach Initial Equilibrium
        system_state = x
        for _ in range(self.num_iterations):
            update = self.interaction_block(system_state)
            system_state = system_state + update
        
        equilibrium_embedding = system_state
        
        # 2. Apply Stress and Restore (only during training)
        if self.training:
            # Apply perturbation
            perturbed_state = self._perturb(equilibrium_embedding)
            
            # Allow the system to shift and find a new equilibrium
            new_equilibrium_state = perturbed_state
            for _ in range(self.num_perturb_steps):
                new_equilibrium_state = self.interaction_block(new_equilibrium_state)
            
            new_equilibrium_embedding = new_equilibrium_state
        else:
            # In inference mode, no perturbation is applied.
            new_equilibrium_embedding = equilibrium_embedding

        return equilibrium_embedding, new_equilibrium_embedding

class Multimodal_Placeholder_Transformer(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_layers: int = 4,
        dim_output: int = 10
    ) -> None: 
        super().__init__()

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

        self.pen_module = PEN_Module(d_model, 3)
        self.restore_loss_fn = nn.KLDivLoss(reduction='batchmean') 

        
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

    def forward(
        self, 
        x: torch.Tensor,
        src_mask: torch.Tensor = None,
        y: torch.Tensor = None,
    ) -> dict:
        
        # Get both equilibrium states from the PEN module
        equilibrium_state, new_equilibrium_state = self.pen_module(x)
        
        # --- Process Main Equilibrium State ---
        main_logits = equilibrium_state
        # Store the original mask if it exists, as it will be modified
        original_src_mask = src_mask.clone() if src_mask is not None else None

        for layer in self.transformer_cls:
            if isinstance(layer, nn.TransformerEncoder) and src_mask is not None:
                # Use the original mask for this path
                current_mask = self._add_cls_token_mask_to_src_mask(original_src_mask)
                main_logits = layer(main_logits, src_key_padding_mask=current_mask)
            else:
                main_logits = layer(main_logits)

        # --- Process Restored Equilibrium State (During Training) ---
        restored_logits = None
        loss_restore = torch.tensor(0.0, device=x.device) # Initialize loss as a tensor

        if self.training:
            restored_logits = new_equilibrium_state
            # Use a fresh copy of the original mask for the restoration path
            restored_src_mask = src_mask.clone() if src_mask is not None else None
            
            for layer in self.transformer_cls:
                if isinstance(layer, nn.TransformerEncoder) and restored_src_mask is not None:
                    current_mask = self._add_cls_token_mask_to_src_mask(restored_src_mask)
                    restored_logits = layer(restored_logits, src_key_padding_mask=current_mask)
                else:
                    restored_logits = layer(restored_logits)

            # --- CORRECT LOSS CALCULATION for BCE task---
            # Use MSE to compare the logits directly. Detach the main_logits so we only
            # backpropagate through the restoration path for this loss term.
            restore_loss_fn = nn.MSELoss()
            loss_restore = restore_loss_fn(restored_logits, main_logits.detach())
        

        return {
            "logits": main_logits,
            "losses": {
                "restore": 0.01 * loss_restore
            }
        }
