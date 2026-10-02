"""Checkpoints 3-5: does the trained model actually work as a world model?

  3. one_step.png   real context -> predicted next frame vs ground truth
  4. actions.png    same starting frames, each action repeated -> the paddle must move differently
  5. rollout.gif    autoregressive rollout (predictions fed back in) next to ground truth,
                    plus per-step error printed so you can see compounding drift
  6. printed table  one-step error of the model vs the "copy the last frame" baseline.
                    In games where most of the screen stays put (Crafter), copying is a strong
                    baseline; a model that hasn't learned the dynamics yet will not beat it on
                    the frames that actually change.

    python evaluate.py --ckpt outputs/crafter.pt          # test data: data/<env>/test
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from viz import labeled_row, stack_rows, upscale
from wm.config import Config, build_denoiser, config_from_dict
from wm.data import load_episodes, to_tensor, to_uint8
from wm.envs import get_env
from wm.rollout import WorldModelEnv
from wm.sampler import Sampler

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="outputs/model.pt")
p.add_argument("--data", default=None, help="default: data/<env>/test")
p.add_argument("--out", default="outputs")
p.add_argument("--horizon", type=int, default=40)
p.add_argument("--denoise_steps", type=int, default=Config.denoise_steps)
p.add_argument("--ctx_noise", type=float, default=0.0, help="context noise at inference")
args = p.parse_args()

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cfg = config_from_dict(ck["cfg"])
spec = get_env(cfg.env)
ACTION_NAMES = spec.action_names
model = build_denoiser(cfg)
model.load_state_dict(ck["model"])
model.eval()
model.ctx_noise_inference = args.ctx_noise
sampler = Sampler(model, args.denoise_steps)
K = cfg.K
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
episodes = [e for e in load_episodes(args.data or f"data/{cfg.env}/test") if len(e[1]) >= K + 50]
torch.manual_seed(0)

# --- 3. one-step prediction ---------------------------------------------------
rows = []
for obs, act in episodes[:4]:
    t = 10
    ctx = to_tensor(obs[t:t + K])[None]
    a = torch.from_numpy(act[t:t + K])[None]
    pred = to_uint8(sampler.sample(ctx, a))[0]
    frames = list(obs[t:t + K]) + [obs[t + K], pred]
    labels = [f"ctx a={ACTION_NAMES[x]}" for x in act[t:t + K]] + ["truth", "PREDICTED"]
    rows.append(labeled_row(frames, labels))
stack_rows(rows).save(out / "one_step.png")

# --- 4. action test -------------------------------------------------------------
obs, act = episodes[0]
env = WorldModelEnv(sampler)
rows = []
for a in spec.showcase_actions:
    env.reset(to_tensor(obs[:K])[None], torch.from_numpy(act[:K - 1])[None])
    frames = [env.step(torch.tensor([a])) for _ in range(6)]
    rows.append(labeled_row([obs[K - 1]] + [to_uint8(f)[0] for f in frames],
                            ["start"] + [f"{ACTION_NAMES[a]} x{i + 1}" for i in range(6)]))
stack_rows(rows).save(out / "actions.png")

# --- 5. autoregressive rollout vs ground truth ------------------------------------
H = min(args.horizon, len(episodes[0][1]) - K)
errs = np.zeros(H)
gif_frames = []
for e, (obs, act) in enumerate(episodes[:8]):
    env.reset(to_tensor(obs[:K])[None], torch.from_numpy(act[:K - 1])[None])
    for i in range(H):
        pred = env.step(torch.tensor([act[K - 1 + i]]))       # replay the recorded actions
        truth = to_tensor(obs[K + i])[None]
        errs[i] += (pred - truth).pow(2).mean().item()
        if e == 0:
            both = np.concatenate([obs[K + i], np.full((obs.shape[1], 2, 3), 255, np.uint8), to_uint8(pred)[0]], 1)
            gif_frames.append(Image.fromarray(upscale(both, 6)))
errs /= min(8, len(episodes))
gif_frames[0].save(out / "rollout.gif", save_all=True, append_images=gif_frames[1:], duration=120, loop=0)

print("rollout MSE by step (left = truth, right = model in rollout.gif):")
for i in [0, 4, 9, 19, 29, 39]:
    if i < H:
        print(f"  step {i + 1:3d}: {errs[i]:.4f}")
print(f"saved {out}/one_step.png, actions.png, rollout.gif")

# --- 6. one-step accuracy vs the copy-last-frame baseline -----------------------------
rng = np.random.default_rng(0)
windows = [(e, int(rng.integers(0, len(a) - K + 1))) for e, (o, a) in enumerate(episodes) for _ in range(16)]
m_err, c_err, changed = [], [], []
for i in range(0, len(windows), 32):
    chunk = windows[i:i + 32]
    ctx = torch.stack([to_tensor(episodes[e][0][t:t + K]) for e, t in chunk])
    a = torch.stack([torch.from_numpy(episodes[e][1][t:t + K]) for e, t in chunk])
    truth = torch.stack([to_tensor(episodes[e][0][t + K]) for e, t in chunk])
    pred = sampler.sample(ctx, a)
    m_err += (pred - truth).pow(2).mean((1, 2, 3)).tolist()
    c_err += (ctx[:, -1] - truth).pow(2).mean((1, 2, 3)).tolist()
    changed += ((ctx[:, -1] - truth).abs().amax((1, 2, 3)) > 0.05).tolist()
m_err, c_err, changed = map(np.array, (m_err, c_err, changed))
print(f"\none-step MSE on {len(m_err)} test windows   model   copy-last-frame")
print(f"  frames that changed ({changed.mean():4.0%})        {m_err[changed].mean():.4f}   {c_err[changed].mean():.4f}")
print(f"  all frames                      {m_err.mean():.4f}   {c_err.mean():.4f}")
print("  (lower is better; the model should clearly beat copying on frames that changed)")
