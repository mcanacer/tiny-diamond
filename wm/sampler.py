"""Layer 4 - Sampler: "how to use the brain once".

Start from pure noise at sigma_max and walk down a schedule of noise levels,
denoising a bit at each step (Euler method on the probability-flow ODE).

DIAMOND equivalent: src/models/diffusion/diffusion_sampler.py
"""
import torch

from .diffusion import Denoiser


def build_sigmas(n_steps: int, sigma_min=2e-3, sigma_max=5.0, rho=7, device="cpu"):
    """Karras schedule: dense near sigma_min, sparse near sigma_max. Ends with 0."""
    ramp = torch.linspace(0, 1, n_steps, device=device)
    lo, hi = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    sigmas = (hi + ramp * (lo - hi)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros(1)])


class Sampler:
    def __init__(self, denoiser: Denoiser, n_steps: int = 3):
        self.denoiser = denoiser
        self.n_steps = n_steps

    @torch.no_grad()
    def sample(self, obs, act):
        """obs: (B, K, C, H, W) context frames, act: (B, K). Returns next frame (B, C, H, W)."""
        B, K, C, H, W = obs.shape
        obs = obs.reshape(B, K * C, H, W)
        sigmas = build_sigmas(self.n_steps, device=obs.device)
        x = torch.randn(B, C, H, W, device=obs.device) * sigmas[0]
        for s, s_next in zip(sigmas[:-1], sigmas[1:]):
            x0_hat = self.denoiser.denoise(x, s.expand(B), obs, act)
            d = (x - x0_hat) / s              # direction pointing toward more noise
            x = x + d * (s_next - s)          # Euler step (s_next < s, so we move toward x0_hat)
        return x
