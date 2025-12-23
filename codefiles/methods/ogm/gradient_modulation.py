import torch 
import math
import torch.nn as nn 
import torch.nn.functional as F


class OGM(nn.Module):

    """ 
    https://arxiv.org/abs/2203.15332

    https://github.com/GeWu-Lab/OGM-GE_CVPR2022 
    """

    def __init__(
        self,
        params_transformerhead: dict = {
            "d_model": 512,
            "nhead": 4,
            "dim_feedforward": 1024,
            "dropout": 0.0,
            "num_layers": 4,
            "dim_output": 10,
        },
        num_modalities: int = 2,
        input_dims: tuple = (512, 512),
        alpha: float = 0.5,  # [0, 1]
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.input_dims = input_dims
        self.d_model = params_transformerhead["d_model"]
        dim_output = params_transformerhead["dim_output"]

        # Fusion layer: Concatenation followed by a Linear layer
        self.fusion_layer = nn.Linear(self.num_modalities * self.d_model, dim_output)

        # OGM
        self.alpha = alpha
        self.apply(self._init_weights)
        
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

    def _create_final_src_mask(
        self, 
        src_mask: torch.Tensor
    ) -> torch.Tensor:
        n_modalities = src_mask.shape[-1]
        seq_len = 1

        # Create mask that's False for sampled embeddings and first sequence of original, True for repeated sequences
        src_masks_per_mod = []
        for i in range(n_modalities):
            mod_mask = torch.where(src_mask[:, i][:, None].repeat(1, self.si * seq_len), False, True)  # Don't mask sampled embeddings
            mod_mask[:, :seq_len] = False  # Don't mask first sequence of original embeddings
            src_masks_per_mod.append(mod_mask)
        src_mask = torch.cat(src_masks_per_mod, dim=1)

        return src_mask

    def _extract_unimodal_embeddings_from_embeddings(
        self,
        x
    ) -> list:
        # if x.shape[1] == self.num_modalities: every sequence is a CLS token
        embeddings_unimodal = []
        if x.shape[1] == self.num_modalities:
            for i_mod in range(self.num_modalities):
                x_modality = x[:, i_mod, :][:, None, :]
                embeddings_unimodal.append(x_modality)
        # else if x.shape[1] == sum(self.input_dims): 0-self.input_dims = first modality, self.input_dims-2*self.input_dims = second modality, etc.
        elif x.shape[1] == sum(self.input_dims):
            for i_mod in range(self.num_modalities):
                x_modality = x[:, (i_mod) * self.input_dims[i_mod]: (i_mod + 1) * self.input_dims[i_mod], :]
                embeddings_unimodal.append(x_modality)
        return embeddings_unimodal
    
    def _calc_modulation_coefficients(
        self,
        unimodal_embeds: list, 
        y: torch.Tensor,
    ) -> list:
        if y.dim() == 1:
            y = y.unsqueeze(1)

        device = unimodal_embeds[0].device
        y = y.to(device)

        # Determine task type based on label dtype
        is_regression = y.dtype in [torch.float, torch.float16, torch.float32, torch.float64, torch.bfloat16]

        # Calculate approximate uni-modal predictions/logits by splitting the main classifier's weights
        unimodal_outputs = []
        for i in range(self.num_modalities):
            # Split the fusion layer's weights and bias for each modality
            weight_slice = self.fusion_layer.weight[:, i * self.d_model : (i + 1) * self.d_model]
            bias_slice = self.fusion_layer.bias / self.num_modalities if self.fusion_layer.bias is not None else 0
            
            # Calculate unimodal logits as per the official OGM-GE implementation
            output = torch.mm(unimodal_embeds[i], torch.transpose(weight_slice, 0, 1)) + bias_slice
            unimodal_outputs.append(output)

        s_modality_contributions = []

        if is_regression:
            # Contribution is inverse of L1 error. Invalid labels are NaNs.
            valid_mask = ~torch.isnan(y)

            for i in range(self.num_modalities):
                # Error is L1 distance
                error = torch.abs(unimodal_outputs[i] - y)
                
                # Contribution score is inverse of error
                contribution = 1.0 / (error + 1e-8)
                
                # Mask out invalid contributions
                masked_contribution = torch.where(
                    valid_mask,
                    contribution,
                    torch.zeros_like(contribution)
                )
                s_modality_contributions.append(masked_contribution)
        else:
            unimodal_logits = unimodal_outputs
            # Calculate contribution scores s_i^u (Equation 8)
            unimodal_probs = []
            for i in range(len(unimodal_logits)):
                probs = F.softmax(unimodal_logits[i], dim=1)
                unimodal_probs.append(probs)

            # A label is invalid if it is NaN or -1
            valid_mask = (~torch.isnan(y)) & (y != -1) # Identify valid targets

            # Create safe indices for gather, replacing invalid entries with 0
            y_safe_indices = torch.where(
                valid_mask,
                y.long(), # Use original target if valid
                torch.zeros_like(y, dtype=torch.long) # Use 0 if target was invalid
            )

            for i in range(len(unimodal_logits)):
                # Gather probabilities using safe indices
                gathered_probs = torch.gather(unimodal_probs[i], 1, y_safe_indices)

                # Zero out contributions where the original y was invalid
                masked_gathered_probs = torch.where(
                    valid_mask,
                    gathered_probs,
                    torch.zeros_like(gathered_probs) # Set contribution to 0 for invalid targets
                )
                s_modality_contributions.append(masked_gathered_probs)

        # Calculate sum S_modality using only valid contributions
        S_modality = []
        for i in range(len(s_modality_contributions)):
            S_modality.append(torch.sum(s_modality_contributions[i])) # Sum masked contributions

        # Calculate discrepancy ratio rho_t^u (Equation 9)
        eps = 1e-8
        rhos = []
        total_S = torch.sum(torch.stack(S_modality)) # Sum of all valid contributions

        # Handle case where all targets might be NaN (total_S is zero)
        if total_S.item() < eps:
            # Assign a default rho (e.g., 1.0 implying no discrepancy signal)
            rhos = [torch.tensor(1.0, device=device) for _ in S_modality]
        else:
            for i in range(len(S_modality)):
                # Original calculation, now safe due to total_S check and valid S_modality sums
                denominator = total_S - S_modality[i] + eps
                rho_i = S_modality[i] / denominator
                rhos.append(rho_i)

        # Calculate modulation coefficient k_t^u (Equation 10)
        k_modality = []
        for i in range(len(rhos)):
            k = torch.tensor(1.0, device=device)
            if rhos[i].item() > 1.0: 
                # Ensure alpha and rho are floats for math.tanh
                alpha_val = self.alpha if isinstance(self.alpha, (int, float)) else self.alpha.item()
                rho_val = F.relu(rhos[i]).item() # Apply relu to ratio, as in the official implementation.
                k = torch.tensor(1.0 - math.tanh(alpha_val * rho_val), device=device)
            k_modality.append(k)

        return k_modality

    def forward(
        self,
        x: torch.Tensor = torch.Tensor,
        src_mask: torch.Tensor = None,
        y: torch.Tensor = None,
        ) -> dict:

        src_mask = None

        phi_modality = self._extract_unimodal_embeddings_from_embeddings(x)

        # Get a single feature vector for each modality by averaging over the sequence dimension
        phi_modality_avg = [torch.mean(phi, dim=1) for phi in phi_modality]

        # Concatenate the averaged unimodal features
        fused_embedding = torch.cat(phi_modality_avg, dim=1)

        # Pass through the fusion layer to get final logits
        logits = self.fusion_layer(fused_embedding)

        modulation_coeffs = self._calc_modulation_coefficients(phi_modality_avg, y)
        
        return {
            "logits": logits,
            "ogm": {
                "modulation_coeffs": modulation_coeffs
            }
        }
