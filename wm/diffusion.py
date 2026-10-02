"""Layer 3 - Diffusion wrapper: "how to train the brain" (EDM, Karras et al. 2022).

Two public methods:
    loss(batch)                       -> training
    denoise(x_noisy, sigma, obs, act) -> inference (used by the sampler)

EDM idea: instead of asking the net to output the clean frame directly, write
    D(x; sigma) = c_skip(sigma) * x  +  c_out(sigma) * F(c_in(sigma) * x, c_noise(sigma))
so that at every noise level the net's input and target both have unit variance.
That's what keeps it stable with only 1-3 denoising steps.

Two OPTIONAL anti-drift tricks (both off by default; see README "Fixing drift"):

  ctx_noise_max > 0   GameNGen-style context noise augmentation.
                      During training, the K context frames get Gaussian noise of a random
                      level in [0, ctx_noise_max], and the net is told that level. The model
                      learns that past frames can be wrong and to rely on its own knowledge
                      of the game to fix them, instead of copying errors forward.

  rollout_steps R > 0 Self-rollout context (the idea behind DIAMOND's multi-step loss and
                      Self Forcing). Before computing the loss, the model generates R frames
                      itself (no gradient) starting from real context. Those generated frames
                      become the most recent context, and the target is still the REAL next
                      frame. So the model is trained on exactly the kind of imperfect input it
                      sees when playing, and is rewarded for steering back to reality.

DIAMOND equivalent: src/models/diffusion/denoiser.py
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import InnerModel


@dataclass
class DiffusionConfig:
    sigma_data: float = 0.5          # std of data in [-1, 1]
    sigma_offset_noise: float = 0.3  # extra per-channel noise (helps with brightness shifts)
    loc: float = -0.4                # log-normal training noise: log(sigma) ~ N(loc, scale^2)
    scale: float = 1.2
    sigma_min: float = 2e-3
    sigma_max: float = 20.0
    ctx_noise_max: float = 0.0       # >0 enables context noise augmentation (GameNGen used ~0.7)
    rollout_steps: int = 0           # >0 enables self-rollout context
    clamp: bool = True               # clamp predictions to [-1, 1]: right for pixels, wrong for latents
    rollout_denoise_steps: int = 3   # make rollout frames exactly like at play time (sampler steps)
    rollout_frac: float = 0.5        # fraction of each batch that gets the self-rollout context
                                     # (the rest keeps real context, so the clean case isn't forgotten)


def _b(x):  # (B,) -> (B, 1, 1, 1) for broadcasting over images
    return x[:, None, None, None]


class Denoiser(nn.Module):
    def __init__(self, net: InnerModel, cfg: DiffusionConfig = DiffusionConfig()):
        super().__init__()
        self.net, self.cfg = net, cfg
        self.ctx_noise_inference = 0.0  # context noise level used when sampling (0 = clean)

    # ---- EDM plumbing ---------------------------------------------------------------
    def conditioners(self, sigma):
        s = (sigma ** 2 + self.cfg.sigma_offset_noise ** 2).sqrt()
        sd = self.cfg.sigma_data
        c_in = 1 / (s ** 2 + sd ** 2).sqrt()
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * c_skip.sqrt()
        c_noise = s.log() / 4
        return _b(c_in), _b(c_skip), _b(c_out), c_noise

    def _noise_context(self, obs, level):
        """Add Gaussian noise of std `level` (B,) to the context frames. Returns (obs, cond value)."""
        if self.net.ctx_noise_emb is None:
            return obs, None
        obs = obs + torch.randn_like(obs) * _b(level)
        return obs, level / max(self.cfg.ctx_noise_max, 1e-8)  # normalised to [0, 1] for the embedding

    def _net(self, x_noisy, sigma, obs, act, ctx_level):
        c_in, c_skip, c_out, c_noise = self.conditioners(sigma)
        obs, ctx_cond = self._noise_context(obs, ctx_level)
        out = self.net(x_noisy * c_in, c_noise, obs / self.cfg.sigma_data, act, ctx_cond)
        return out, c_skip, c_out

    # ---- training -----------------------------------------------------------------------
    @torch.no_grad()
    def _generate(self, ctx, act):
        """Same Euler sampler as sampler.py (kept here to avoid a circular import)."""
        from .sampler import build_sigmas
        B, K, C, H, W = ctx.shape
        obs = ctx.reshape(B, K * C, H, W)
        sigmas = build_sigmas(self.cfg.rollout_denoise_steps, device=ctx.device)
        x = torch.randn(B, C, H, W, device=ctx.device) * sigmas[0]
        for s, s_next in zip(sigmas[:-1], sigmas[1:]):
            x0_hat = self.denoise(x, s.expand(B), obs, act)
            x = x + (x - x0_hat) / s * (s_next - s)
        return x

    @torch.no_grad()
    def _self_rollout(self, obs, act):
        """obs: (B, K+R+1, C, H, W) real frames, act: (B, K+R).
        Returns a context of K frames whose last R are the model's own predictions
        (for a `rollout_frac` share of the batch; the rest keeps real frames),
        the matching K actions, and the real target frame."""
        B = obs.shape[0]
        K, R = self.net.K, self.cfg.rollout_steps
        ctx = obs[:, :K]
        was_training = self.training
        self.eval()
        for r in range(R):
            pred = self._generate(ctx, act[:, r:r + K])
            ctx = torch.cat([ctx[:, 1:], pred[:, None]], dim=1)
        self.train(was_training)
        use_rollout = (torch.rand(B, device=obs.device) < self.cfg.rollout_frac)[:, None, None, None, None]
        ctx = torch.where(use_rollout, ctx, obs[:, R:R + K])   # same time window either way
        return ctx, act[:, R:R + K], obs[:, K + R]

    def loss(self, batch):
        cfg = self.cfg
        K = self.net.K
        if cfg.rollout_steps > 0:
            ctx, act, x0 = self._self_rollout(batch.obs, batch.act)
        else:
            ctx, act, x0 = batch.obs[:, :K], batch.act[:, :K], batch.obs[:, K]
        B, _, C, H, W = ctx.shape
        obs = ctx.reshape(B, K * C, H, W)   # K context frames, stacked on channels

        # 1) pick a random noise level per sample
        sigma = (torch.randn(B, device=x0.device) * cfg.scale + cfg.loc).exp().clamp(cfg.sigma_min, cfg.sigma_max)
        # 2) corrupt the target
        offset = cfg.sigma_offset_noise * torch.randn(B, C, 1, 1, device=x0.device)
        x_noisy = x0 + offset + torch.randn_like(x0) * _b(sigma)
        # 3) (optional) corrupt the context too, by a random amount the net is told about
        ctx_level = torch.rand(B, device=x0.device) * cfg.ctx_noise_max
        # 4) predict, and regress onto the EDM-scaled target
        out, c_skip, c_out = self._net(x_noisy, sigma, obs, act, ctx_level)
        target = (x0 - c_skip * x_noisy) / c_out
        return F.mse_loss(out, target)

    # ---- inference ----------------------------------------------------------------------
    @torch.no_grad()
    def denoise(self, x_noisy, sigma, obs, act, ctx_level=None):
        """Best guess of the clean frame given a noisy one."""
        if ctx_level is None:
            ctx_level = torch.full((x_noisy.shape[0],), self.ctx_noise_inference, device=x_noisy.device)
        out, c_skip, c_out = self._net(x_noisy, sigma, obs, act, ctx_level)
        d = c_skip * x_noisy + c_out * out
        return d.clamp(-1, 1) if self.cfg.clamp else d
