import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

class CoupledMambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, d_rank):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.d_rank = d_rank

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.d_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_rank, self.d_inner, bias=True)

    def forward(
        self, 
        x: torch.Tensor
    ):
        _, seq_len, _ = x.shape

        # Algorithm 1: line 4-5
        u_and_z = self.in_proj(x)
        u, z = u_and_z.chunk(2, dim=-1)
        
        # Algorithm 1: line 9
        u_conv = self.conv1d(u.transpose(1, 2))
        u_conv = u_conv[:, :, :seq_len].transpose(1, 2)
        u_prime = F.silu(u_conv) 
        
        # Algorithm 1: line 10-12
        delta_b_c = self.x_proj(u_prime)
        delta, B, C = torch.split(delta_b_c, [self.d_rank, self.d_state, self.d_state], dim=-1)
        
        # Algorithm 1: line 12
        delta = F.softplus(self.dt_proj(delta))
        
        return u_prime, z, delta, B, C

class CoupledMambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, d_rank, num_modalities):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.num_modalities = num_modalities

        self.blocks = nn.ModuleList([
            CoupledMambaBlock(d_model, d_state, d_conv, expand, d_rank) for _ in range(num_modalities)
        ])

        A = repeat(torch.arange(1, self.d_state + 1), "n -> d n", d=self.d_inner)
        self.A_logs = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.recurrent_norm = nn.LayerNorm(self.d_state)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_projs = nn.ModuleList([
            nn.Linear(self.d_inner, d_model, bias=False) for _ in range(num_modalities)
        ])

    def forward(
        self, 
        modal_features: list[torch.Tensor],
        modality_mask: torch.Tensor
    ):
        # init variables 
        _, seq_len, _ = modal_features[0].shape
        all_u_prime, all_z, all_delta, all_B, all_C = [], [], [], [], []

        # step through coupled mamba blocks
        for i in range(self.num_modalities):
            u_prime, z, delta, B, C = self.blocks[i](modal_features[i])
            all_u_prime.append(u_prime)
            all_z.append(z)
            all_delta.append(delta)
            all_B.append(B)
            all_C.append(C)
        all_delta = torch.stack(all_delta, dim=0)
        all_B = torch.stack(all_B, dim=0)
        all_C = torch.stack(all_C, dim=0)
        all_u_prime = torch.stack(all_u_prime, dim=0)

        # Reshape modality mask for broadcasting to zero out missing modalities.
        # Target shape: (M, B, 1, 1, 1)
        # invert the mask because True indicates a missing modality.
        mask = (1.0 - modality_mask.float()).transpose(0, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Algorithm 1: line 13-15
        A = -torch.exp(self.A_logs)
        # S_o = torch.exp(all_delta.unsqueeze(-1) * self.A_logs.unsqueeze(0).unsqueeze(1).unsqueeze(2))
        # numerical stability fix 
        S_o = torch.exp(all_delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(1).unsqueeze(2))
        B_o = all_delta.unsqueeze(-1) * all_B.unsqueeze(-2)

        S_o = S_o * mask
        B_o = B_o * mask

        # Equation 7
        P = torch.sum(S_o, dim=0)
        U = torch.sum(B_o * all_u_prime.unsqueeze(-1), dim=0)

        # 2nd half of Equation 7 and summation in 8
        h_sum_list = []
        h_sum_t = torch.zeros_like(U[:, 0])
        for i in range(seq_len):
            h_sum_t = P[:, i] * h_sum_t + U[:, i]
            h_sum_t = self.recurrent_norm(h_sum_t)  # numerical stability fix 
            h_sum_list.append(h_sum_t)
        h_sum = torch.stack(h_sum_list, dim=1)
        
        # get previous time step of h_sum
        h_sum_t_minus_1 = F.pad(h_sum[:, :-1], (0, 0, 0, 0, 1, 0), value=0)
        # Equation 6
        h = S_o * h_sum_t_minus_1.unsqueeze(0) + B_o * all_u_prime.unsqueeze(-1)

        # Standard Mamba update
        y = torch.einsum("m b l d n, m b l n -> m b l d", h, all_C)
        y = y + all_u_prime * self.D

        # Algorithm 1: line 19
        y = self.out_norm(y)
        y = y * F.silu(torch.stack(all_z, dim=0))

        # Algorithm 1: line 20-21
        outputs = []
        for i in range(self.num_modalities):
            output = self.out_projs[i](y[i])
            outputs.append(output + modal_features[i])

        return outputs


class Coupled_State_Space_Model(nn.Module):

    """
    https://proceedings.neurips.cc/paper_files/paper/2024/file/6e09c213ac18d6375704a4f3ea75c4f8-Paper-Conference.pdf

    https://github.com/hustcselwb/coupled-mamba
    -> But the codebase is not updated, yet comprehensive, see issues. 
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layer: int = 3,
        d_state: int = 16,
        d_rank: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        num_modalities: int = 3,
        dim_output: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.num_modalities = num_modalities
        self.dim_output = dim_output

        self.init_lns = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_modalities)
        ])

        self.layers = nn.ModuleList([
            CoupledMambaLayer(d_model, d_state, d_conv, expand, d_rank, num_modalities)
            for _ in range(n_layer)
        ])

        # Vectorized implementation of the inter-layer projections
        self.inter_layer_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_modalities * d_model, num_modalities * d_model),
                nn.GELU(),
                nn.LayerNorm(num_modalities * d_model)
            ) for _ in range(n_layer - 1)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.prediction_head = nn.Linear(d_model, dim_output)

        self.apply(self._init_weights)

    def _init_weights(self, m) -> None: 
        if isinstance(m, (torch.nn.LayerNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        
    def forward(
        self, 
        x: list = [torch.Tensor, torch.Tensor, torch.Tensor], 
        src_mask: list = [torch.Tensor, torch.Tensor, torch.Tensor], 
        y: torch.Tensor = None
    ) -> dict:
        # Pad all modalities to the same sequence length
        max_len = 0
        for tensor in x:
            if tensor.shape[1] > max_len:
                max_len = tensor.shape[1]

        padded_x = []
        for tensor in x:
            pad_len = max_len - tensor.shape[1]
            if pad_len > 0:
                # Pad sequence dimension (L) to the right
                padded_tensor = F.pad(tensor, (0, 0, 0, pad_len))
                padded_x.append(padded_tensor)
            else:
                padded_x.append(tensor)
        
        modal_features = padded_x

        src_mask_modalities = [curr_src_mask[:, 0][:, None] for curr_src_mask in src_mask]
        modality_mask = torch.cat(src_mask_modalities, dim=1) if isinstance(src_mask_modalities, list) else src_mask_modalities
        # Padding mask for different sequence lengths
        padding_mask = [(mod.abs().sum(dim=-1) != 0).float() for mod in modal_features]

        modal_features = [self.init_lns[i](modal_features[i]) for i in range(self.num_modalities)]
        for i, layer in enumerate(self.layers):
            modal_features = layer(modal_features, modality_mask)
            
            # Perform Cat-and-Project fusion between layers as shown in Figure 1.
            if i < self.n_layer - 1:
                # Vectorized version of the loop
                concatenated = torch.cat(modal_features, dim=-1)
                projected = self.inter_layer_projs[i](concatenated)
                
                # Split back into a list of tensors for the next layer
                next_modal_features_stacked = projected.view(
                    concatenated.shape[0], 
                    concatenated.shape[1], 
                    self.num_modalities, 
                    self.d_model
                ).transpose(1, 2)
                
                next_modal_features = [
                    next_modal_features_stacked[:, m] + modal_features[m] # Add residual
                    for m in range(self.num_modalities)
                ]
                modal_features = next_modal_features

        concatenated_features = torch.cat(modal_features, dim=1)
        concatenated_mask = torch.cat(padding_mask, dim=1)

        # Masked average pooling to handle variable sequence lengths
        pooled_features = (concatenated_features * concatenated_mask.unsqueeze(-1)).sum(dim=1)
        pooled_features = pooled_features / concatenated_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-9)
        
        pooled_features = self.norm(pooled_features)
        logits = self.prediction_head(pooled_features)

        return {
            "logits": logits
        }
