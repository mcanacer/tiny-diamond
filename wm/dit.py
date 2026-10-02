"""Causal diffusion transformer world model with Diffusion Forcing.

This is the architecture family behind the 2025-26 world models (Dreamer 4, Waypoint-1, Matrix-Game),
shrunk to fit a single GPU:

  * Each frame (pixels or autoencoder latents) is cut into patches -> tokens.
  * A window of T frames is one token sequence. Attention is BLOCK-CAUSAL: tokens of frame t see all
    tokens of frames <= t (full attention inside a frame, no peeking at the future).
  * Every frame carries its OWN noise level sigma_t and the action that led into it. Both condition
    the transformer through adaLN (per token, from that token's frame).

Training modes (cfg.train_mode):
  "df"    Diffusion Forcing (Chen et al. 2024). Every frame in the window gets an independent random
          noise level and every frame is denoised (loss on all frames). The model learns to predict
          a frame from any mix of clean/noisy history, which makes it robust to its own imperfect
          past at play time. This generalises the GameNGen context-noise trick that worked for us.
  "last"  The U-Net recipe on a transformer: context frames get small noise U(0, ctx_noise_max),
          only the last frame is denoised. Lets us separate "transformer" from "diffusion forcing".

Generation: the context frames are given at a fixed small noise level and the new frame is denoised
with the same 3-step Euler sampler as the U-Net. Because attention is causal, context tokens never
look at the new frame, so their keys/values are computed ONCE per generated frame and reused for
every denoising step (KV cache).

EDM preconditioning (c_in, c_skip, c_out, c_noise) is applied per frame, exactly as in diffusion.py.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import FourierFeatures


@dataclass
class DiTConfig:
    in_channels: int = 3
    frame_size: int = 64
    patch: int = 4
    dim: int = 384
    depth: int = 8
    heads: int = 6
    max_frames: int = 5          # window length T = K + 1
    num_actions: int = 5
    train_mode: str = "df"       # "df" or "last"
    ctx_noise_max: float = 0.0   # "last" mode: context noise level range
    sigma_data: float = 0.5
    sigma_offset_noise: float = 0.3
    loc: float = -0.4
    scale: float = 1.2
    sigma_min: float = 2e-3
    sigma_max: float = 20.0
    clamp: bool = True           # clamp predictions to [-1, 1] (pixels); off for latents


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(dim // heads, elementwise_affine=False)  # QK-norm: stabler training
        self.k_norm = nn.LayerNorm(dim // heads, elementwise_affine=False)

    def forward(self, x, mask=None, past_kv=None, return_kv=False):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        kv = (k, v)
        if past_kv is not None:  # new frame attends to cached context + itself
            k, v = torch.cat([past_kv[0], k], 2), torch.cat([past_kv[1], v], 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = self.proj(out.transpose(1, 2).reshape(B, L, D))
        return (out, kv) if return_kv else out


class Block(nn.Module):
    """DiT block with adaLN-Zero: the per-token condition sets shift/scale/gate of both sub-layers."""

    def __init__(self, dim, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight), nn.init.zeros_(self.ada.bias)  # each block starts as identity

    def forward(self, x, c, mask=None, past_kv=None, return_kv=False):
        s1, sc1, g1, s2, sc2, g2 = self.ada(F.silu(c)).chunk(6, dim=-1)
        a = self.attn(_modulate(self.n1(x), s1, sc1), mask, past_kv, return_kv)
        a, kv = a if return_kv else (a, None)
        x = x + g1 * a
        x = x + g2 * self.mlp(_modulate(self.n2(x), s2, sc2))
        return (x, kv) if return_kv else x


class CausalDiT(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        p, d = cfg.patch, cfg.dim
        assert cfg.frame_size % p == 0
        self.grid = cfg.frame_size // p
        self.n_tok = self.grid ** 2
        self.patch_in = nn.Conv2d(cfg.in_channels, d, p, stride=p)
        self.pos_space = nn.Parameter(torch.randn(1, 1, self.n_tok, d) * 0.02)
        self.pos_time = nn.Parameter(torch.randn(1, cfg.max_frames, 1, d) * 0.02)
        self.noise_emb = FourierFeatures(d)
        self.act_emb = nn.Embedding(cfg.num_actions + 1, d)  # last index = "no action" (first frame of a window)
        self.cond = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(Block(d, cfg.heads) for _ in range(cfg.depth))
        self.n_out = nn.LayerNorm(d, elementwise_affine=False)
        self.ada_out = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, p * p * cfg.in_channels)
        for m in (self.ada_out, self.out):
            nn.init.zeros_(m.weight), nn.init.zeros_(m.bias)

    def _embed(self, x, c_noise, act, t0=0):
        """x: (B, T, C, H, W), c_noise/act: (B, T). Returns tokens (B, T*N, D), cond (B, T*N, D)."""
        B, T = x.shape[:2]
        tok = self.patch_in(x.flatten(0, 1)).flatten(2).transpose(1, 2).view(B, T, self.n_tok, -1)
        tok = tok + self.pos_space + self.pos_time[:, t0:t0 + T]
        c = self.cond(self.noise_emb(c_noise.flatten()).view(B, T, -1) + self.act_emb(act))
        c = c[:, :, None, :].expand(-1, -1, self.n_tok, -1)
        return tok.flatten(1, 2), c.flatten(1, 2)

    def _unpatch(self, h, c, B, T):
        s, sc = self.ada_out(F.silu(c)).chunk(2, dim=-1)
        h = self.out(_modulate(self.n_out(h), s, sc))  # (B, T*N, p*p*C)
        p, g, C = self.cfg.patch, self.grid, self.cfg.in_channels
        h = h.view(B, T, g, g, C, p, p).permute(0, 1, 4, 2, 5, 3, 6)
        return h.reshape(B, T, C, g * p, g * p)

    def causal_mask(self, T, device):
        f = torch.arange(T, device=device).repeat_interleave(self.n_tok)
        return f[None, :] <= f[:, None]  # (T*N, T*N): query frame >= key frame

    def forward(self, x, c_noise, act):
        """Full window forward (training). x: (B, T, C, H, W) already scaled by c_in."""
        B, T = x.shape[:2]
        h, c = self._embed(x, c_noise, act)
        mask = self.causal_mask(T, x.device)
        for blk in self.blocks:
            h = blk(h, c, mask)
        return self._unpatch(h, c, B, T)

    # ---- KV-cached generation ------------------------------------------------------------------
    def context_cache(self, x_ctx, c_noise_ctx, act_ctx):
        """Run the K context frames once; return per-layer keys/values."""
        B, T = x_ctx.shape[:2]
        h, c = self._embed(x_ctx, c_noise_ctx, act_ctx)
        mask = self.causal_mask(T, x_ctx.device)
        cache = []
        for blk in self.blocks:
            h, kv = blk(h, c, mask, return_kv=True)
            cache.append(kv)
        return cache

    def forward_new(self, x_new, c_noise_new, act_new, cache, t_index):
        """Only the new frame's tokens, attending to cached context (+ themselves)."""
        B = x_new.shape[0]
        h, c = self._embed(x_new[:, None], c_noise_new[:, None], act_new[:, None], t0=t_index)
        for blk, kv in zip(self.blocks, cache):
            h = blk(h, c, None, past_kv=kv)  # no mask needed: everything in the cache is in the past
        return self._unpatch(h, c, B, 1)[:, 0]


def _b(x):  # (...,) -> (..., 1, 1, 1)
    return x[..., None, None, None]


class DiTDenoiser(nn.Module):
    """EDM wrapper around CausalDiT with per-frame noise levels."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.net = CausalDiT(cfg)
        self.K = cfg.max_frames - 1
        self.ctx_noise_inference = 0.0  # noise level the context is presented at when generating
        self.none_action = cfg.num_actions

    def conditioners(self, sigma):
        s = (sigma ** 2 + self.cfg.sigma_offset_noise ** 2).sqrt()
        sd = self.cfg.sigma_data
        c_in = 1 / (s ** 2 + sd ** 2).sqrt()
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * c_skip.sqrt()
        return _b(c_in), _b(c_skip), _b(c_out), s.log() / 4

    def _sample_sigma(self, shape, device):
        c = self.cfg
        return (torch.randn(shape, device=device) * c.scale + c.loc).exp().clamp(c.sigma_min, c.sigma_max)

    def _actions_in(self, act):
        """act (B, T-1): a_t leads from frame t to t+1 -> per-frame action that led INTO each frame."""
        none = torch.full_like(act[:, :1], self.none_action)
        return torch.cat([none, act], 1)

    def loss(self, batch):
        x0 = batch.obs                      # (B, T, C, H, W)
        B, T, C = x0.shape[:3]
        cfg = self.cfg
        act_in = self._actions_in(batch.act[:, :T - 1])
        if cfg.train_mode == "df":
            sigma = self._sample_sigma((B, T), x0.device)               # independent per frame
            weight = torch.ones(B, T, device=x0.device)
        else:  # "last": slightly noisy context, denoise only the final frame
            sigma = torch.rand(B, T, device=x0.device) * cfg.ctx_noise_max
            sigma[:, -1] = self._sample_sigma((B,), x0.device)
            weight = torch.zeros(B, T, device=x0.device)
            weight[:, -1] = 1
        offset = cfg.sigma_offset_noise * torch.randn(B, T, C, 1, 1, device=x0.device)
        x_noisy = x0 + offset + torch.randn_like(x0) * _b(sigma)
        c_in, c_skip, c_out, c_noise = self.conditioners(sigma)
        out = self.net(x_noisy * c_in, c_noise, act_in)
        target = (x0 - c_skip * x_noisy) / c_out
        per_frame = (out.float() - target.float()).pow(2).mean((2, 3, 4))
        return (per_frame * weight).sum() / weight.sum()

    @torch.no_grad()
    def generate(self, ctx, act, n_steps=3, sigma_min=2e-3, sigma_max=5.0, rho=7):
        """ctx: (B, K, C, H, W) clean context, act: (B, K) with act[:, -1] the action producing the new frame.
        Returns the new frame (B, C, H, W)."""
        from .sampler import build_sigmas
        B, K, C, H, W = ctx.shape
        dev = ctx.device
        act_in = self._actions_in(act)                       # (B, K+1)
        s_ctx = torch.full((B, K), max(self.ctx_noise_inference, 0.0), device=dev)
        x_ctx = ctx + torch.randn_like(ctx) * _b(s_ctx) if self.ctx_noise_inference > 0 else ctx
        c_in, _, _, c_noise = self.conditioners(s_ctx)
        cache = self.net.context_cache(x_ctx * c_in, c_noise, act_in[:, :K])
        sigmas = build_sigmas(n_steps, sigma_min, sigma_max, rho, dev)
        x = torch.randn(B, C, H, W, device=dev) * sigmas[0]
        for s, s_next in zip(sigmas[:-1], sigmas[1:]):
            sig = s.expand(B)
            c_in, c_skip, c_out, c_noise = self.conditioners(sig)
            out = self.net.forward_new(x * c_in, c_noise, act_in[:, K], cache, t_index=K)
            x0_hat = c_skip * x + c_out * out
            if self.cfg.clamp:
                x0_hat = x0_hat.clamp(-1, 1)
            x = x + (x - x0_hat) / s * (s_next - s)
        return x
