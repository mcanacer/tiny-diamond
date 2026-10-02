"""All hyperparameters in one place. Per-environment defaults live in PRESETS."""
from dataclasses import dataclass, field, fields
from typing import List

from .diffusion import DiffusionConfig, Denoiser
from .envs import get_env
from .network import InnerModel


@dataclass
class Config:
    env: str = "toy"
    # data
    img_size: int = 32
    num_episodes: int = 200
    episode_len: int = 100
    K: int = 4                      # number of context frames
    # model
    cond_dim: int = 256
    channels: List[int] = field(default_factory=lambda: [32, 64, 64])
    depth: int = 2
    # training
    batch_size: int = 32
    lr: float = 1e-4
    steps: int = 20000
    grad_clip: float = 1.0
    # anti-drift (both 0 = original behaviour; see README "Fixing drift")
    ctx_noise_max: float = 0.0      # GameNGen context noise augmentation
    rollout_steps: int = 0          # self-rollout context
    # sampling
    denoise_steps: int = 3


# Per-environment defaults. Anything here can still be overridden from the command line.
PRESETS = {
    "toy": dict(img_size=32, channels=[32, 64, 64], num_episodes=200, episode_len=100,
                lr=3e-4, steps=6000),
    # Same U-Net size as DIAMOND's Atari world model (4 levels x 64 channels), 64x64 frames.
    # Random play dies after ~170 steps on average, so 1000 episodes ~ 170k frames (~2 GB in RAM).
    "crafter": dict(img_size=64, channels=[64, 64, 64, 64], num_episodes=1000, episode_len=500,
                    lr=2e-4, steps=40000, batch_size=32),
}


# Named experiments: `python train.py --exp crafter_big` instead of a long list of flags.
# Each one changes ONE thing relative to the baseline, and all keep batch 32 and 40k steps,
# so results are directly comparable (the toy experiments showed why that matters).
# Add your own here; `python train.py --list` prints them.
EXPERIMENTS = {
    "toy": dict(env="toy"),
    "crafter": dict(env="crafter"),  # baseline: 3.4M params (your first Colab run)
    "crafter_big": dict(env="crafter", channels=[64, 128, 128, 256]),  # 14.6M params
    "crafter_big_noise": dict(env="crafter", channels=[64, 128, 128, 256], ctx_noise_max=0.7),
    "crafter_big_k8": dict(env="crafter", channels=[64, 128, 128, 256], K=8),  # twice the memory
    "crafter_big_noise_k8": dict(env="crafter", channels=[64, 128, 128, 256], ctx_noise_max=0.7, K=8),
}


def make_config(env: str = "toy", **overrides) -> Config:
    values = dict(PRESETS.get(env, {}))
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Config(env=env, **values)


def config_from_dict(d: dict) -> Config:
    """Load a config saved in a checkpoint (older checkpoints lack newer fields; defaults fill in)."""
    known = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in d.items() if k in known})


def build_denoiser(cfg: Config) -> Denoiser:
    net = InnerModel(get_env(cfg.env).num_actions, cfg.K, 3, cfg.cond_dim, cfg.channels, cfg.depth,
                     ctx_noise=cfg.ctx_noise_max > 0)
    return Denoiser(net, DiffusionConfig(ctx_noise_max=cfg.ctx_noise_max, rollout_steps=cfg.rollout_steps))
