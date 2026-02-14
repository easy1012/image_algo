import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple



class PositionalEncoding(nn.Module):
    """
    gamma(x) = [x, sin(2^0 pi x), cos(2^0 pi x), ..., sin(2^{L-1} pi x), cos(...)]
    """
    def __init__(self, in_dim: int, num_freqs: int, include_input: bool = True, log_sampling: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.include_input = include_input

        if log_sampling:
            self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, steps=num_freqs)
        else:
            self.freq_bands = torch.linspace(1.0, 2.0 ** (num_freqs - 1), steps=num_freqs)

    def out_dim(self) -> int:
        dim = 0
        if self.include_input:
            dim += self.in_dim
        dim += 2 * self.in_dim * self.num_freqs
        return dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_dim)
        """
        out = []
        if self.include_input:
            out.append(x)

        # Ensure freq_bands on same device/dtype
        freq_bands = self.freq_bands.to(device=x.device, dtype=x.dtype)

        for f in freq_bands:
            out.append(torch.sin(math.pi * f * x))
            out.append(torch.cos(math.pi * f * x))
        return torch.cat(out, dim=-1)




class NeRF_layer(nn.Module):
    """
    Original-ish NeRF MLP:
    - Input: encoded position (x) and encoded view direction (d)
    - Outputs: sigma (density) and rgb
    """
    def __init__(
        self,
        x_dim: int,
        d_dim: int,
        hidden: int = 256,
        depth: int = 8,
        skips=(4,)
    ):
        super().__init__()
        self.depth = depth
        self.skips = set(skips)

        self.pts_linears = nn.ModuleList()
        self.pts_linears.append(nn.Linear(x_dim, hidden))
        for i in range(1, depth):
            if i in self.skips:
                self.pts_linears.append(nn.Linear(hidden + x_dim, hidden))
            else:
                self.pts_linears.append(nn.Linear(hidden, hidden))

        # density head
        self.sigma_linear = nn.Linear(hidden, 1)

        # feature for rgb head
        self.feature_linear = nn.Linear(hidden, hidden)

        # view-dependent rgb head
        self.rgb_linear_1 = nn.Linear(hidden + d_dim, hidden // 2)
        self.rgb_linear_2 = nn.Linear(hidden // 2, 3)

    def forward(self, x_enc: torch.Tensor, d_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_enc: (N, x_dim)
        d_enc: (N, d_dim)
        returns rgb (N,3), sigma (N,1)
        """
        h = x_enc
        for i, layer in enumerate(self.pts_linears):
            if i in self.skips:
                h = torch.cat([h, x_enc], dim=-1)
            h = F.relu(layer(h))

        sigma = F.relu(self.sigma_linear(h))  # density >= 0
        feat = self.feature_linear(h)

        h_rgb = torch.cat([feat, d_enc], dim=-1)
        h_rgb = F.relu(self.rgb_linear_1(h_rgb))
        rgb = torch.sigmoid(self.rgb_linear_2(h_rgb))  # in [0,1]
        return rgb, sigma
