"""One interface for every trained world model, in pixel space.

Whatever is inside the checkpoint (DIAMOND-style U-Net or causal transformer, on pixels or on
autoencoder latents), scripts only ever see this:

    wm = WorldModel("outputs/crafter_dit_df.pt", device="cuda")
    next_px = wm.predict(ctx_px, act)          # one step from real context  (B, K, 3, H, W), (B, K)
    wm.reset(ctx_px, act_prev)                 # start a rollout             (B, K, 3, H, W), (B, K-1)
    px = wm.step(action)                       # then feed actions one by one (B,) -> (B, 3, H, W)

For latent models the rollout keeps its memory in latent space (decoding/re-encoding every frame
would add error); frames are decoded only for display. Latent world-model checkpoints carry their
autoencoder inside, so they work on any machine without extra files.
"""
import torch

from .autoencoder import AEConfig, Autoencoder
from .config import build_model, config_from_dict
from .envs import get_env
from .sampler import Sampler


def load_autoencoder(payload, device="cpu"):
    from .config import config_from_dict as _cfd
    cfg = _cfd(payload["cfg"])
    ae = Autoencoder(AEConfig(channels=cfg.ae_channels, latent_channels=cfg.ae_latent_channels))
    ae.load_state_dict(payload["model"])
    return ae.to(device).eval().requires_grad_(False)


class WorldModel:
    def __init__(self, ckpt, device="cpu", denoise_steps=None, ctx_noise=0.0):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg = cfg = config_from_dict(ck["cfg"])
        if cfg.model == "ae":
            raise ValueError(f"{ckpt} is an autoencoder, not a world model")
        self.step_count = ck.get("step", "?")
        self.spec = get_env(cfg.env)
        self.device = device
        self.K = cfg.K
        self.model = build_model(cfg)
        self.model.load_state_dict(ck["model"])
        self.model.to(device).eval().requires_grad_(False)
        self.model.ctx_noise_inference = ctx_noise
        self.ae = load_autoencoder(ck["ae"], device) if cfg.latent else None
        self._n_steps = denoise_steps or cfg.denoise_steps
        self.sampler = Sampler(self.model, self._n_steps) if cfg.model == "unet" else None

    @property
    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    @property
    def n_steps(self):
        return self._n_steps

    @n_steps.setter
    def n_steps(self, n):
        self._n_steps = n
        if self.sampler is not None:
            self.sampler.n_steps = n

    # ---- pixels <-> model space -----------------------------------------------------------------
    def encode(self, px):
        return self.ae.encode(px) if self.ae is not None else px

    def decode(self, z):
        return self.ae.decode(z) if self.ae is not None else z

    @torch.no_grad()
    def _next(self, ctx, act):
        if self.sampler is not None:
            return self.sampler.sample(ctx, act)
        return self.model.generate(ctx, act, n_steps=self._n_steps)

    # ---- public API ------------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, ctx_px, act):
        """One-step prediction from real context. ctx_px (B, K, 3, H, W) in [-1, 1], act (B, K)."""
        return self.decode(self._next(self.encode(ctx_px.to(self.device)), act.to(self.device)))

    @torch.no_grad()
    def reset(self, ctx_px, act_prev):
        """Start a rollout from K real frames and the K-1 actions between them."""
        self.buf = self.encode(ctx_px.to(self.device))
        act_prev = act_prev.to(self.device)
        self.act = torch.cat([act_prev, act_prev.new_zeros(act_prev.shape[0], 1)], 1)
        return ctx_px[:, -1]

    @torch.no_grad()
    def step(self, action):
        """Feed one action per rollout (B,), get the generated frame (B, 3, H, W) in [-1, 1]."""
        self.act[:, -1] = action.to(self.device)
        z = self._next(self.buf, self.act)
        self.buf = torch.cat([self.buf[:, 1:], z[:, None]], 1)
        self.act = torch.cat([self.act[:, 1:], self.act.new_zeros(self.act.shape[0], 1)], 1)
        return self.decode(z)
