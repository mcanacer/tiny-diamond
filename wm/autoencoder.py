"""Frame autoencoder for latent-space world models (as in Oasis, Dreamer 4, Stable Diffusion).

Compresses each 64x64x3 frame to a 16x16x8 latent (6x fewer numbers). The world model then predicts
latents instead of pixels, which is what makes higher resolutions affordable; at 64x64 the benefit
is mostly speed and a smoother space to diffuse in.

Training is deterministic (no KL): L1 + MSE reconstruction, plus a small penalty that keeps latents
small. Two details that matter for world models:
  * latent normalisation: running per-channel mean/std are tracked during training, and
    `encode()` returns latents scaled to std 0.5 (the same scale as pixels in [-1, 1]),
    so the EDM settings (sigma_data = 0.5) carry over unchanged.
  * decoder robustness: the decoder is also trained on slightly noised latents, because at play
    time it decodes the world model's imperfect predictions, not true encodings.
"""
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AEConfig:
    channels: List[int] = field(default_factory=lambda: [64, 128, 128])  # one downsample between levels
    latent_channels: int = 8
    latent_noise: float = 0.1      # decoder robustness noise (in normalised latent units)
    latent_l2: float = 1e-4


def _gn(c):
    return nn.GroupNorm(min(32, c // 4) if c >= 8 else 1, c)


class _Res(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.n1, self.c1 = _gn(c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.n2, self.c2 = _gn(c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        return self.skip(x) + self.c2(F.silu(self.n2(h)))


class Autoencoder(nn.Module):
    def __init__(self, cfg: AEConfig = AEConfig(), img_channels: int = 3):
        super().__init__()
        self.cfg = cfg
        ch = cfg.channels
        enc = [nn.Conv2d(img_channels, ch[0], 3, padding=1)]
        for i, c in enumerate(ch):
            enc += [_Res(ch[max(i - 1, 0)] if i else ch[0], c), _Res(c, c)]
            if i < len(ch) - 1:
                enc.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
        enc += [_gn(ch[-1]), nn.SiLU(), nn.Conv2d(ch[-1], cfg.latent_channels, 3, padding=1)]
        self.encoder = nn.Sequential(*enc)

        dec = [nn.Conv2d(cfg.latent_channels, ch[-1], 3, padding=1)]
        rev = list(reversed(ch))
        for i, c in enumerate(rev):
            dec += [_Res(rev[max(i - 1, 0)] if i else rev[0], c), _Res(c, c)]
            if i < len(rev) - 1:
                dec += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)]
        dec += [_gn(ch[0]), nn.SiLU(), nn.Conv2d(ch[0], img_channels, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)

        # running latent statistics (per channel), used to normalise latents to std 0.5
        self.register_buffer("z_mean", torch.zeros(cfg.latent_channels))
        self.register_buffer("z_std", torch.ones(cfg.latent_channels))
        self.register_buffer("stats_ready", torch.zeros(()))

    @property
    def downsample(self):
        return 2 ** (len(self.cfg.channels) - 1)

    # ---- raw (unnormalised) -------------------------------------------------------------------
    def _enc(self, x):
        return self.encoder(x)

    def _dec(self, z):
        return self.decoder(z)

    # ---- normalised interface used by world models --------------------------------------------
    def _norm(self, z):
        return (z - self.z_mean[:, None, None]) / self.z_std[:, None, None] * 0.5

    def _denorm(self, zn):
        return zn / 0.5 * self.z_std[:, None, None] + self.z_mean[:, None, None]

    @torch.no_grad()
    def encode(self, x):
        """pixels (..., 3, H, W) in [-1, 1] -> normalised latents (..., C, H/d, W/d) with std ~0.5"""
        lead = x.shape[:-3]
        z = self._norm(self._enc(x.reshape(-1, *x.shape[-3:])))
        return z.reshape(*lead, *z.shape[-3:])

    @torch.no_grad()
    def decode(self, zn):
        lead = zn.shape[:-3]
        x = self._dec(self._denorm(zn.reshape(-1, *zn.shape[-3:]))).clamp(-1, 1)
        return x.reshape(*lead, *x.shape[-3:])

    # ---- training ------------------------------------------------------------------------------
    def loss(self, batch):
        x = batch.obs.flatten(0, 1)  # every frame of every window is a training image
        z = self._enc(x)
        with torch.no_grad():  # track latent statistics (EMA)
            m, s = z.float().mean((0, 2, 3)), z.float().std((0, 2, 3)) + 1e-4
            if self.stats_ready.item() == 0:
                self.z_mean.copy_(m), self.z_std.copy_(s), self.stats_ready.fill_(1)
            else:
                self.z_mean.lerp_(m, 0.01), self.z_std.lerp_(s, 0.01)
        # decoder robustness: half the time decode a slightly noised latent
        noise = torch.randn_like(z) * (self.cfg.latent_noise * 2 * self.z_std[:, None, None])
        noise = noise * (torch.rand(z.shape[0], 1, 1, 1, device=z.device) < 0.5)
        rec = self._dec(z + noise)
        loss = F.l1_loss(rec, x) + F.mse_loss(rec, x) + self.cfg.latent_l2 * z.pow(2).mean()
        return loss
