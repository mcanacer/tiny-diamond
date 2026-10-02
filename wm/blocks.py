"""U-Net building blocks. Conditioning (action + noise level) enters every ResBlock
through AdaGroupNorm: the cond vector predicts a per-channel scale and shift.

DIAMOND equivalent: src/models/blocks.py
"""
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def GroupNorm(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(max(1, c // 32) if c >= 32 else 1, c, eps=1e-5)


class AdaGroupNorm(nn.Module):
    def __init__(self, c: int, cond_dim: int):
        super().__init__()
        self.norm = GroupNorm(c)
        self.linear = nn.Linear(cond_dim, 2 * c)

    def forward(self, x, cond):
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return self.norm(x) * (1 + scale) + shift


class FourierFeatures(nn.Module):
    """Embeds the scalar noise level c_noise into a vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("w", torch.randn(1, dim // 2))

    def forward(self, x):  # x: (B,)
        f = 2 * math.pi * x[:, None] @ self.w
        return torch.cat([f.cos(), f.sin()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, cond_dim: int):
        super().__init__()
        self.norm1, self.conv1 = AdaGroupNorm(c_in, cond_dim), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2, self.conv2 = AdaGroupNorm(c_out, cond_dim), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        nn.init.zeros_(self.conv2.weight)  # each block starts as identity

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x, cond)))
        h = self.conv2(F.silu(self.norm2(h, cond)))
        return self.skip(x) + h


class UNet(nn.Module):
    """channels[i] = width at resolution level i; depth ResBlocks per level."""

    def __init__(self, cond_dim: int, channels: List[int], depth: int = 2):
        super().__init__()
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.downsample, self.upsample = nn.ModuleList(), nn.ModuleList()
        c_prev = channels[0]
        for i, c in enumerate(channels):
            self.down.append(nn.ModuleList(ResBlock(c_prev if j == 0 else c, c, cond_dim) for j in range(depth)))
            c_prev = c
            if i < len(channels) - 1:
                self.downsample.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
        self.mid = ResBlock(c_prev, c_prev, cond_dim)
        for i, c in reversed(list(enumerate(channels))):
            # each up block sees its skip connections concatenated
            self.up.append(nn.ModuleList(ResBlock((c_prev if j == 0 else c) + c, c, cond_dim) for j in range(depth)))
            c_prev = c
            if i > 0:
                self.upsample.append(nn.Conv2d(c, channels[i - 1], 3, padding=1))
                c_prev = channels[i - 1]

    def forward(self, x, cond):
        skips = []
        for i, blocks in enumerate(self.down):
            for b in blocks:
                x = b(x, cond)
                skips.append(x)
            if i < len(self.downsample):
                x = self.downsample[i](x)
        x = self.mid(x, cond)
        for i, blocks in enumerate(self.up):
            for b in blocks:
                x = b(torch.cat([x, skips.pop()], dim=1), cond)
            if i < len(self.upsample):
                x = self.upsample[i](F.interpolate(x, scale_factor=2, mode="nearest"))
        return x
