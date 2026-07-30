import math

import torch
import torch.nn as nn

from jaxtyping import Float
from torch import Tensor


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        num_groups_in = math.gcd(in_channels, 8) or 1
        num_groups_out = math.gcd(out_channels, 8) or 1

        self.norm1 = nn.GroupNorm(num_groups=num_groups_in, num_channels=in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.context_proj = nn.Linear(context_dim, out_channels * 2)

        self.norm2 = nn.GroupNorm(num_groups=num_groups_out, num_channels=out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        residual = self.skip(x)

        x = self.conv1(self.act1(self.norm1(x)))

        scale_shift = self.context_proj(context).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)
        x = self.norm2(x)
        x = x * (1 + scale) + shift
        x = self.conv2(self.dropout(self.act2(x)))

        return x + residual


class Downsample1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] <= 1:
            return x
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, target_len: int) -> Tensor:
        if x.shape[-1] != target_len:
            x = nn.functional.interpolate(x, size=target_len, mode="nearest")
        return self.conv(x)


class Diffusion_UNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 256,
        condition_dim: int = 256,
        dropout: float = 0.1,
        nhead: int = 1,
        num_layers: int = 10,
    ) -> None:
        super().__init__()
        del nhead

        self.seq_len_x = None
        self.init_proj_x = nn.Linear(input_dim, hidden_dim)
        self.init_proj_c = nn.Linear(condition_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        depth = max(1, num_layers // 2)
        channel_multipliers = [1]
        for level in range(1, depth):
            channel_multipliers.append(min(2 ** level, 4))
        channel_dims = [hidden_dim * mult for mult in channel_multipliers]

        self.in_conv = nn.Conv1d(hidden_dim, channel_dims[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_channels = channel_dims[0]
        for next_channels in channel_dims:
            self.down_blocks.append(
                ResidualBlock1D(
                    in_channels=current_channels,
                    out_channels=next_channels,
                    context_dim=hidden_dim,
                    dropout=dropout,
                )
            )
            self.downsamples.append(Downsample1D(next_channels))
            current_channels = next_channels

        self.mid_block_1 = ResidualBlock1D(
            in_channels=current_channels,
            out_channels=current_channels,
            context_dim=hidden_dim,
            dropout=dropout,
        )
        self.mid_block_2 = ResidualBlock1D(
            in_channels=current_channels,
            out_channels=current_channels,
            context_dim=hidden_dim,
            dropout=dropout,
        )

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for skip_channels in reversed(channel_dims):
            self.upsamples.append(Upsample1D(current_channels))
            self.up_blocks.append(
                ResidualBlock1D(
                    in_channels=current_channels + skip_channels,
                    out_channels=skip_channels,
                    context_dim=hidden_dim,
                    dropout=dropout,
                )
            )
            current_channels = skip_channels

        self.final_norm = nn.GroupNorm(num_groups=math.gcd(current_channels, 8) or 1, num_channels=current_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv1d(current_channels, hidden_dim, kernel_size=3, padding=1)

        self.apply(self._init_weights)
        self.time_mlp.apply(self._init_time_mlp_weights)
        self.out_proj.apply(self._init_out_proj_weights)

    def _init_weights(self, module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.GroupNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def _init_time_mlp_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_out_proj_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.constant_(module.weight, 0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _pool_condition(
        self,
        condition: Tensor,
        src_mask: Tensor = None,
    ) -> Tensor:
        if condition.shape[1] == 0:
            return torch.zeros(condition.shape[0], condition.shape[2], device=condition.device, dtype=condition.dtype)

        if src_mask is None:
            return condition.mean(dim=1)

        valid_mask = (~src_mask).unsqueeze(-1).to(dtype=condition.dtype)
        denom = valid_mask.sum(dim=1).clamp(min=1.0)
        pooled = (condition * valid_mask).sum(dim=1) / denom
        empty_rows = (valid_mask.sum(dim=1) == 0)
        if empty_rows.any():
            pooled = pooled.masked_fill(empty_rows, 0.0)
        return pooled

    def forward(
        self,
        x: Float[Tensor, "batch seq embdim"],
        t: Float[Tensor, "batch embdim"],
        condition: Float[Tensor, "batch seq embdim"],
        src_mask: Float[Tensor, "batch seq"] = None,
    ) -> Float[Tensor, "batch seq embdim"]:
        x = self.init_proj_x(x)
        condition = self.init_proj_c(condition)
        context = self.time_mlp(t) + self._pool_condition(condition, src_mask)

        x = x.transpose(1, 2)
        x = self.in_conv(x)

        skips = []
        target_lengths = []
        for block, downsample in zip(self.down_blocks, self.downsamples):
            x = block(x, context)
            skips.append(x)
            target_lengths.append(x.shape[-1])
            x = downsample(x)

        x = self.mid_block_1(x, context)
        x = self.mid_block_2(x, context)

        for upsample, block, skip, target_len in zip(self.upsamples, self.up_blocks, reversed(skips), reversed(target_lengths)):
            x = upsample(x, target_len)
            x = torch.cat([x, skip], dim=1)
            x = block(x, context)

        x = self.final_conv(self.final_act(self.final_norm(x)))
        x = x.transpose(1, 2)
        x = self.out_proj(x)
        return x
