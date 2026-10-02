"""Layer 2 - Network: "the raw brain".

f(noisy_next, c_noise, past_frames, past_actions) -> tensor shaped like one frame.
It knows nothing about diffusion - it's a conditional image-to-image network.

  * past frames are CONCATENATED AS CHANNELS with the noisy target frame
  * action embeddings + noise-level embedding are SUMMED into one cond vector
    that modulates every ResBlock (AdaGroupNorm)

DIAMOND equivalent: src/models/diffusion/inner_model.py
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import FourierFeatures, GroupNorm, UNet


class InnerModel(nn.Module):
    def __init__(self, num_actions: int, K: int, img_channels: int = 3,
                 cond_dim: int = 256, channels: List[int] = (64, 64, 64), depth: int = 2,
                 ctx_noise: bool = False):
        super().__init__()
        assert cond_dim % K == 0
        self.K = K
        self.noise_emb = FourierFeatures(cond_dim)
        # GameNGen trick: the context frames may be noised; the net is told HOW noisy
        # so it learns how much to trust them. Only built when enabled, so old checkpoints load.
        self.ctx_noise_emb = FourierFeatures(cond_dim) if ctx_noise else None
        # each of the K actions gets cond_dim/K dims; flattened they make one cond_dim vector,
        # so the ORDER of actions is preserved (the last one matters most)
        self.act_emb = nn.Sequential(nn.Embedding(num_actions, cond_dim // K), nn.Flatten())
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.conv_in = nn.Conv2d((K + 1) * img_channels, channels[0], 3, padding=1)
        self.unet = UNet(cond_dim, list(channels), depth)
        self.norm_out = GroupNorm(channels[0])
        self.conv_out = nn.Conv2d(channels[0], img_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)  # start by predicting "nothing"
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, noisy_next, c_noise, obs, act, ctx_noise=None):
        # noisy_next: (B, C, H, W)   obs: (B, K*C, H, W)   act: (B, K)   c_noise, ctx_noise: (B,)
        cond = self.noise_emb(c_noise) + self.act_emb(act)
        if self.ctx_noise_emb is not None:
            cond = cond + self.ctx_noise_emb(ctx_noise)
        cond = self.cond_proj(cond)
        x = self.conv_in(torch.cat([obs, noisy_next], dim=1))
        x = self.unet(x, cond)
        return self.conv_out(F.silu(self.norm_out(x)))
