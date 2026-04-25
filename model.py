import torch
import torch.nn as nn

import math
import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim

        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

    def sinusoidal_embedding(self, t):
        half_dim = self.time_dim // 2
        device = t.device

        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / half_dim
        )

        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return emb

    def forward(self, t):
        emb = self.sinusoidal_embedding(t)
        emb = self.mlp(emb)
        return emb



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(8, 64)
        self.silu = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
        Tself.downsample1 = nn.Conv2()

    def forward(self, x):
        out = self.initial_conv(x)
        return out

    






