"""Causal diffusion transformer world model with Diffusion Forcing.

This is the architecture family behind the 2025-26 world models (Dreamer 4, Genie, Waypoint-1,
Matrix-Game), shrunk to fit a single GPU:

  * Each frame (pixels or autoencoder latents) is cut into patches -> N tokens per frame.
  * A window of T frames is processed together, CAUSALLY in time: frame t never sees frames > t.
  * Every frame carries its OWN noise level sigma_t and the action that led into it. Both condition
    the transformer through adaLN-Zero (per token, from that token's frame).

Attention (cfg.attn):
  "st"    space-time factorized (Genie, Dreamer 4): each block does spatial attention inside each frame,
          then causal temporal attention along each patch position (a patch attends to the same patch
          position in earlier frames), then an MLP. The temporal path makes "look at the same spot in the
          previous frame" the easiest thing to learn, which is what next-frame prediction mostly needs.
  "full"  block-causal attention over all T*N tokens (every patch can attend to every patch of the same or
          earlier frames). More general but slower to learn: on the toy game it still ignored its context
          after 2500 steps, so "st" is the default.

Training modes (cfg.train_mode):
  "df"    Diffusion Forcing (Chen et al. 2024). Every frame in the window gets an independent random
          noise level and every frame is denoised (loss on all frames). The model learns to predict
          a frame from any mix of clean/noisy history, which makes it robust to its own imperfect
          past at play time. This generalises the GameNGen context-noise trick that worked for us.
          Each frame's level is drawn from a mixture: "history" (nearly clean) half the time, EDM's
          log-normal otherwise. Pure log-normal per frame made the clean-past + noisy-present case
          (exactly the situation when generating) too rare in training.
  "last"  The U-Net recipe on a transformer: context frames get small noise U(0, ctx_noise_max),
          only the last frame is denoised. Lets us separate "transformer" from "diffusion forcing".

Generation: the context frames are given at a fixed small noise level and the new frame is denoised
with the same 3-step Euler sampler as the U-Net. Because attention is causal, context tokens never
look at the new frame, so their keys/values are computed ONCE per generated frame and reused for
every denoising step (KV cache). With "st" only the temporal attention looks at the past, so only its
keys/values are cached.

EDM preconditioning (c_in, c_skip, c_out, c_noise) is applied per frame, exactly as in diffusion.py.
"""
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
    attn: str = "st"             # "st" (space-time factorized) or "full" (block-causal over all tokens)
    train_mode: str = "df"       # "df" or "last"
    ctx_noise_max: float = 0.0   # "last" mode: context noise level range
    df_history_prob: float = 0.5   # "df" mode: chance that a frame is "history" (nearly clean) ...
    df_history_noise: float = 0.1  # ... with noise level U(0, this); otherwise the usual log-normal
    sigma_data: float = 0.5
    sigma_offset_noise: float = 0.3
    loc: float = -0.4
    scale: float = 1.2
    sigma_min: float = 2e-3
    sigma_max: float = 20.0
    clamp: bool = True           # clamp predictions to [-1, 1] (pixels); off for latents


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


def _causal(T, device):
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


class Attention(nn.Module):
    """Multi-head attention over sequences (S, L, D), with QK-norm and an optional key/value cache."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(dim // heads, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(dim // heads, elementwise_affine=False)

    def forward(self, x, mask=None, past_kv=None):
        S, L, D = x.shape
        q, k, v = self.qkv(x).view(S, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        kv = (k, v)
        if past_kv is not None:  # new frame attends to cached context + itself
            k, v = torch.cat([past_kv[0], k], 2), torch.cat([past_kv[1], v], 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(S, L, D)), kv


class Block(nn.Module):
    """DiT block with adaLN-Zero: the per-token condition sets shift/scale/gate of every sub-layer.
    x, c: (B, T, N, D). Returns the new x and the keys/values that later frames will need."""

    def __init__(self, dim, heads, attn):
        super().__init__()
        self.mode = attn
        self.norms = nn.ModuleList(nn.LayerNorm(dim, elementwise_affine=False) for _ in range(3 if attn == "st" else 2))
        if attn == "st":
            self.s_attn, self.t_attn = Attention(dim, heads), Attention(dim, heads)
        else:
            self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))
        self.ada = nn.Linear(dim, 3 * len(self.norms) * dim)
        nn.init.zeros_(self.ada.weight), nn.init.zeros_(self.ada.bias)  # each block starts as identity

    def forward(self, x, c, past_kv=None):
        B, T, N, D = x.shape
        mods = self.ada(F.silu(c)).chunk(3 * len(self.norms), dim=-1)
        if self.mode == "st":
            # spatial attention inside each frame
            h = _modulate(self.norms[0](x), mods[0], mods[1]).reshape(B * T, N, D)
            h, _ = self.s_attn(h)
            x = x + mods[2] * h.view(B, T, N, D)
            # causal temporal attention along each patch position
            h = _modulate(self.norms[1](x), mods[3], mods[4]).permute(0, 2, 1, 3).reshape(B * N, T, D)
            mask = None if past_kv is not None else _causal(T, x.device)
            h, kv = self.t_attn(h, mask, past_kv)
            x = x + mods[5] * h.view(B, N, T, D).permute(0, 2, 1, 3)
            s, sc, g = mods[6:9]
        else:
            h = _modulate(self.norms[0](x), mods[0], mods[1]).reshape(B, T * N, D)
            mask = None
            if past_kv is None:  # block-causal: query frame >= key frame
                f = torch.arange(T, device=x.device).repeat_interleave(N)
                mask = f[None, :] <= f[:, None]
            h, kv = self.attn(h, mask, past_kv)
            x = x + mods[2] * h.view(B, T, N, D)
            s, sc, g = mods[3:6]
        x = x + g * self.mlp(_modulate(self.norms[-1](x), s, sc))
        return x, kv


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
        self.blocks = nn.ModuleList(Block(d, cfg.heads, cfg.attn) for _ in range(cfg.depth))
        self.n_out = nn.LayerNorm(d, elementwise_affine=False)
        self.ada_out = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, p * p * cfg.in_channels)
        for m in (self.ada_out, self.out):
            nn.init.zeros_(m.weight), nn.init.zeros_(m.bias)

    def _embed(self, x, c_noise, act, t0=0):
        """x: (B, T, C, H, W), c_noise/act: (B, T) -> tokens and conditions, both (B, T, N, D)."""
        B, T = x.shape[:2]
        tok = self.patch_in(x.flatten(0, 1)).flatten(2).transpose(1, 2).view(B, T, self.n_tok, -1)
        tok = tok + self.pos_space + self.pos_time[:, t0:t0 + T]
        c = self.cond(self.noise_emb(c_noise.flatten()).view(B, T, -1) + self.act_emb(act))
        return tok, c[:, :, None, :].expand(-1, -1, self.n_tok, -1)

    def _unpatch(self, h, c):
        B, T = h.shape[:2]
        s, sc = self.ada_out(F.silu(c)).chunk(2, dim=-1)
        h = self.out(_modulate(self.n_out(h), s, sc))  # (B, T, N, p*p*C)
        p, g, C = self.cfg.patch, self.grid, self.cfg.in_channels
        h = h.view(B, T, g, g, C, p, p).permute(0, 1, 4, 2, 5, 3, 6)
        return h.reshape(B, T, C, g * p, g * p)

    def forward(self, x, c_noise, act):
        """Full window forward (training). x: (B, T, C, H, W) already scaled by c_in."""
        h, c = self._embed(x, c_noise, act)
        for blk in self.blocks:
            h, _ = blk(h, c)
        return self._unpatch(h, c)

    # ---- KV-cached generation ------------------------------------------------------------------
    def context_cache(self, x_ctx, c_noise_ctx, act_ctx):
        """Run the K context frames once; return per-layer keys/values for later frames."""
        h, c = self._embed(x_ctx, c_noise_ctx, act_ctx)
        cache = []
        for blk in self.blocks:
            h, kv = blk(h, c)
            cache.append(kv)
        return cache

    def forward_new(self, x_new, c_noise_new, act_new, cache, t_index):
        """Only the new frame's tokens, attending to cached context (+ themselves)."""
        h, c = self._embed(x_new[:, None], c_noise_new[:, None], act_new[:, None], t0=t_index)
        for blk, kv in zip(self.blocks, cache):
            h, _ = blk(h, c, past_kv=kv)
        return self._unpatch(h, c)[:, 0]


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
            # independent noise level per frame: "history" (nearly clean) or "being generated" (log-normal)
            sigma = self._sample_sigma((B, T), x0.device)
            history = torch.rand(B, T, device=x0.device) < cfg.df_history_prob
            sigma = torch.where(history, torch.rand(B, T, device=x0.device) * cfg.df_history_noise, sigma)
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
            c_in, c_skip, c_out, c_noise = self.conditioners(s.expand(B))
            out = self.net.forward_new(x * c_in, c_noise, act_in[:, K], cache, t_index=K)
            x0_hat = c_skip * x + c_out * out
            if self.cfg.clamp:
                x0_hat = x0_hat.clamp(-1, 1)
            x = x + (x - x0_hat) / s * (s_next - s)
        return x
