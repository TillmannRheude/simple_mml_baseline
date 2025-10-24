import torch 
import torch.nn as nn 
import numpy as np

def mimetic_init_svd_(
    module: nn.Module,
    alpha1: float = 0.7,
    beta1: float = 0.7,
    alpha2: float = 0.4,
    beta2: float = 0.4
) -> None:
    """
    Applies mimetic initialization using SVD factorization.

    [1] A. Trockman and J. Z. Kolter, “Mimetic initialization of self-attention layers” 
    """
    if isinstance(module, (nn.MultiheadAttention)):
        embed_dim = module.embed_dim
        device = module.in_proj_weight.device
        dtype = module.in_proj_weight.dtype
        num_heads = module.num_heads
        head_dim = embed_dim // num_heads

        with torch.no_grad():
            eye = torch.eye(embed_dim, device=device, dtype=dtype)
            Z1 = torch.randn(embed_dim, embed_dim, device=device, dtype=dtype) * (1/embed_dim)

            # W_Q and W_K source calculation (embed_dim, embed_dim)
            W_Q_W_KT = (alpha1 * Z1) - (beta1 * eye)
            U_1, S_1, V_1T = torch.linalg.svd(W_Q_W_KT, full_matrices=True) # U_1, V_1T are embed_dim x embed_dim
            S_1 = torch.diag(torch.sqrt(S_1))

            # Construct W_V and W_proj from SVD of W_Q_W_KT
            W_V = U_1 @ S_1
            W_proj = V_1T @ (S_1**0.5)

            # Process each head separately for Q and K with a new Z2
            W_Q = torch.zeros(embed_dim, embed_dim, device=device, dtype=dtype)
            W_K = torch.zeros(embed_dim, embed_dim, device=device, dtype=dtype)
            for h in range(num_heads):
                Z2 = torch.randn(embed_dim, embed_dim, device=device, dtype=dtype) * (1/embed_dim)
                W_V_W_Tproj = (alpha2 * Z2) + (beta2 * eye)
                U_2, S_2, V_2T = torch.linalg.svd(W_V_W_Tproj, full_matrices=False)
                S_2 = torch.diag(torch.sqrt(S_2))

                head_W_Q = U_2[:, :head_dim] @ (S_2[:head_dim, :head_dim]**0.5) # (d, k) @ (k, k) -> (d, k)
                head_W_K = V_2T.T[:, :head_dim] @ (S_2[:head_dim, :head_dim]**0.5) # (d, k) @ (k, k) -> (d, k)

                # Assign to the appropriate slice of the final weight matrices
                start_idx = h * head_dim
                end_idx = (h + 1) * head_dim
                W_Q[:, start_idx:end_idx] = head_W_Q
                W_K[:, start_idx:end_idx] = head_W_K

            # Assign concatenated/calculated module weights
            module.in_proj_weight.data[:embed_dim] = W_Q
            module.in_proj_weight.data[embed_dim:2*embed_dim] = W_K
            module.in_proj_weight.data[2*embed_dim:] = W_V
            if isinstance(module.out_proj, nn.Linear):
                module.out_proj.weight.data = W_proj
            else: 
                module.out_proj.data = W_proj

            # Zero Bias Initialization
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)


def pad_and_mask_modalities(modalities: list) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a list of modality tensors to the same sequence length and generates a corresponding padding mask.

    Args:
        modalities (list): A list of tensors, each with shape (B, L_i, D),
                         where L_i can be different for each modality.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - A single padded tensor of all modalities, with shape (B, M, L_max, D).
            - A boolean padding mask tensor with shape (B, M, L_max), where M is the number of modalities.
              The mask is True for real data and False for padding.
    """
    if not modalities:
        return torch.empty(0), torch.empty(0)

    batch_size = modalities[0].shape[0]
    device = modalities[0].device
    dtype = modalities[0].dtype

    seq_lengths = [mod.shape[1] for mod in modalities]
    max_len = max(seq_lengths)

    padded_modalities = []
    masks = []

    for i, mod in enumerate(modalities):
        pad_len = max_len - seq_lengths[i]
        if pad_len > 0:
            padding = torch.zeros(batch_size, pad_len, mod.shape[2], device=device, dtype=dtype)
            padded_mod = torch.cat([mod, padding], dim=1)
        else:
            padded_mod = mod
        padded_modalities.append(padded_mod)

        # Create mask: True for real data, False for padding
        mask = torch.arange(max_len, device=device)[None, :] < seq_lengths[i]
        masks.append(mask)

    # Stack into single tensors
    final_tensor = torch.stack(padded_modalities, dim=1)
    final_mask = torch.stack(masks, dim=1)

    return final_tensor, final_mask
