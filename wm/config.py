"""All hyperparameters in one place. Per-environment defaults live in PRESETS."""
from dataclasses import dataclass, field, fields
from typing import List

from .autoencoder import AEConfig, Autoencoder
from .diffusion import DiffusionConfig, Denoiser
from .dit import DiTConfig, DiTDenoiser
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
    # what to train: "unet" (DIAMOND-style), "dit" (causal transformer), "ae" (frame autoencoder)
    model: str = "unet"
    # latent space: name/path of a trained autoencoder; the world model then predicts its latents.
    # latent_channels / latent_size are filled in automatically from that autoencoder.
    latent: str = ""
    latent_channels: int = 0
    latent_size: int = 0
    # causal transformer (model="dit")
    dit_dim: int = 384
    dit_depth: int = 8
    dit_heads: int = 6
    patch: int = 2                  # patch size in (latent) pixels
    train_mode: str = "df"          # "df" = Diffusion Forcing, "last" = denoise only the last frame
    # autoencoder (model="ae")
    ae_channels: List[int] = field(default_factory=lambda: [64, 128, 128])
    ae_latent_channels: int = 8


# Per-environment defaults. Anything here can still be overridden from the command line.
PRESETS = {
    "toy": dict(img_size=32, channels=[32, 64, 64], num_episodes=200, episode_len=100,
                lr=3e-4, steps=6000),
    # Same U-Net size as DIAMOND's Atari world model (4 levels x 64 channels), 64x64 frames.
    # Random play dies after ~170 steps on average, so 1000 episodes ~ 170k frames (~2 GB in RAM).
    "crafter": dict(img_size=64, channels=[64, 64, 64, 64], num_episodes=1000, episode_len=500,
                    lr=2e-4, steps=40000, batch_size=32),
    # Atari: random play episodes are ~130 (Breakout) to 1800 (Boxing) steps; ~150-200k frames each.
    "atari_breakout": dict(img_size=64, channels=[64, 64, 64, 64], num_episodes=1500, episode_len=1000,
                           lr=2e-4, steps=40000, batch_size=32),
    "atari_pong": dict(img_size=64, channels=[64, 64, 64, 64], num_episodes=220, episode_len=1000,
                       lr=2e-4, steps=40000, batch_size=32),
    "atari_boxing": dict(img_size=64, channels=[64, 64, 64, 64], num_episodes=200, episode_len=1000,
                         lr=2e-4, steps=40000, batch_size=32),
    # Doom: maze episodes time out after 2100 tics = 525 steps; defend episodes end at death (~90 steps).
    "doom_maze": dict(img_size=64, channels=[64, 64, 64, 64], num_episodes=400, episode_len=525,
                      lr=2e-4, steps=40000, batch_size=32),
    "doom_defend": dict(img_size=64, channels=[64, 64, 64, 64], num_episodes=2000, episode_len=525,
                        lr=2e-4, steps=40000, batch_size=32),
}


# Named experiments: `python train.py --exp crafter_big` instead of a long list of flags.
# Each one changes ONE thing relative to the baseline, and all keep batch 32 and 40k steps,
# so results are directly comparable (the toy experiments showed why that matters).
# Add your own here; `python train.py --list` prints them.
_BIG = dict(channels=[64, 128, 128, 256])                       # 14.6M-param pixel U-Net
_AE = dict(model="ae", steps=20000, lr=3e-4)                    # 3.1M autoencoder, 64x64x3 -> 16x16x8
_LAT_UNET = dict(model="unet", channels=[96, 192, 256])          # 15.6M U-Net on 16x16 latents
_DIT = dict(model="dit", dit_dim=320, dit_depth=8, dit_heads=5, patch=2)  # 15.2M causal transformer

EXPERIMENTS = {
    "toy": dict(env="toy"),
    "toy_dit_df": dict(env="toy", model="dit", dit_dim=192, dit_depth=4, dit_heads=3, patch=4,
                       steps=6000, lr=3e-4),
    # ---- Crafter: each step of this ladder changes ONE thing -----------------------------------
    "crafter": dict(env="crafter"),                                             # DIAMOND-size baseline, 3.4M
    "crafter_big": dict(env="crafter", **_BIG),                                 # + wider U-Net
    "crafter_big_noise": dict(env="crafter", ctx_noise_max=0.7, **_BIG),        # + context noise (best U-Net)
    "crafter_big_k8": dict(env="crafter", K=8, **_BIG),
    "crafter_big_noise_k8": dict(env="crafter", ctx_noise_max=0.7, K=8, **_BIG),
    "crafter_ae": dict(env="crafter", **_AE),                                   # autoencoder (train first)
    "crafter_latent_unet": dict(env="crafter", latent="crafter_ae", ctx_noise_max=0.7, **_LAT_UNET),  # + latent space
    "crafter_dit_last": dict(env="crafter", latent="crafter_ae", train_mode="last", ctx_noise_max=0.7, **_DIT),  # + transformer
    "crafter_dit_df": dict(env="crafter", latent="crafter_ae", train_mode="df", **_DIT),         # + Diffusion Forcing
    "crafter_dit_df_k16": dict(env="crafter", latent="crafter_ae", train_mode="df", K=16, **_DIT),  # + 16 frames memory
}

# Same recipes for the other environments: <env> (DIAMOND-size U-Net), <env>_big_noise (best U-Net),
# <env>_ae (autoencoder) and <env>_dit_df (latent causal transformer with Diffusion Forcing).
for _env in ["atari_breakout", "atari_pong", "atari_boxing", "doom_maze", "doom_defend"]:
    EXPERIMENTS[_env] = dict(env=_env)
    EXPERIMENTS[f"{_env}_big_noise"] = dict(env=_env, ctx_noise_max=0.7, **_BIG)
    EXPERIMENTS[f"{_env}_ae"] = dict(env=_env, **_AE)
    EXPERIMENTS[f"{_env}_dit_df"] = dict(env=_env, latent=f"{_env}_ae", train_mode="df", **_DIT)



def make_config(env: str = "toy", **overrides) -> Config:
    values = dict(PRESETS.get(env, {}))
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Config(env=env, **values)


def config_from_dict(d: dict) -> Config:
    """Load a config saved in a checkpoint (older checkpoints lack newer fields; defaults fill in)."""
    known = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in d.items() if k in known})


def frame_shape(cfg: Config):
    """(channels, size) of what the world model predicts: pixels, or autoencoder latents."""
    if cfg.latent:
        assert cfg.latent_channels > 0, "latent world model built before its autoencoder was loaded"
        return cfg.latent_channels, cfg.latent_size
    return 3, cfg.img_size


def build_model(cfg: Config):
    """unet -> Denoiser (EDM + U-Net), dit -> DiTDenoiser (EDM + causal transformer), ae -> Autoencoder."""
    if cfg.model == "ae":
        return Autoencoder(AEConfig(channels=cfg.ae_channels, latent_channels=cfg.ae_latent_channels))
    n_act = get_env(cfg.env).num_actions
    ch, size = frame_shape(cfg)
    pixels = not cfg.latent
    if cfg.model == "dit":
        return DiTDenoiser(DiTConfig(in_channels=ch, frame_size=size, patch=cfg.patch, dim=cfg.dit_dim,
                                     depth=cfg.dit_depth, heads=cfg.dit_heads, max_frames=cfg.K + 1,
                                     num_actions=n_act, train_mode=cfg.train_mode,
                                     ctx_noise_max=cfg.ctx_noise_max, clamp=pixels))
    net = InnerModel(n_act, cfg.K, ch, cfg.cond_dim, cfg.channels, cfg.depth, ctx_noise=cfg.ctx_noise_max > 0)
    return Denoiser(net, DiffusionConfig(ctx_noise_max=cfg.ctx_noise_max, rollout_steps=cfg.rollout_steps,
                                         clamp=pixels))


build_denoiser = build_model  # older name, still used by the toy diagnostics
