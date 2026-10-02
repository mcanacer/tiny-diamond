"""Checkpoints 3-6: does the trained model actually work as a world model? Works for every model
type (U-Net or transformer, pixels or latents) and every environment.

  3. one_step.png   real context -> predicted next frame vs ground truth
  4. actions.png    same starting frames, each action repeated -> the moves must look different
  5. rollout.gif    autoregressive rollout (predictions fed back in) next to ground truth,
                    plus per-step error printed so you can see compounding drift
  6. printed table  one-step error of the model vs the "copy the last frame" baseline.
                    When most of the screen stays put, copying is a strong baseline; a model that
                    hasn't learned the dynamics will not beat it on the frames that actually change.

    python evaluate.py --ckpt outputs/crafter.pt          # test data: data/<env>/test
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from viz import labeled_row, stack_rows, upscale
from wm.data import load_episodes, to_tensor, to_uint8
from wm.world_model import WorldModel

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="outputs/model.pt")
p.add_argument("--data", default=None, help="default: data/<env>/test")
p.add_argument("--out", default="outputs")
p.add_argument("--horizon", type=int, default=40)
p.add_argument("--denoise_steps", type=int, default=None)
p.add_argument("--ctx_noise", type=float, default=0.0, help="context noise at inference")
p.add_argument("--device", default=None)
args = p.parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
wm = WorldModel(args.ckpt, device, args.denoise_steps, args.ctx_noise)
cfg, spec, K = wm.cfg, wm.spec, wm.K
ACTION_NAMES = spec.action_names
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
episodes = [e for e in load_episodes(args.data or f"data/{cfg.env}/test") if len(e[1]) >= K + 50]
assert episodes, "no test episodes long enough"
torch.manual_seed(0)
print(f"{Path(args.ckpt).stem}: model={cfg.model} space={'latent' if cfg.latent else 'pixels'} "
      f"params={wm.n_params / 1e6:.1f}M K={K} denoise_steps={wm.n_steps}")

# --- 3. one-step prediction ---------------------------------------------------
rows = []
for obs, act in episodes[:4]:
    t = 10
    pred = to_uint8(wm.predict(to_tensor(obs[t:t + K])[None], torch.from_numpy(act[t:t + K])[None]))[0]
    frames = list(obs[t:t + K]) + [obs[t + K], pred]
    labels = [f"ctx a={ACTION_NAMES[x]}" for x in act[t:t + K]] + ["truth", "PREDICTED"]
    rows.append(labeled_row(frames, labels))
stack_rows(rows).save(out / "one_step.png")

# --- 4. action test -------------------------------------------------------------
obs, act = episodes[0]
rows = []
for a in spec.showcase_actions:
    wm.reset(to_tensor(obs[:K])[None], torch.from_numpy(act[:K - 1])[None])
    frames = [wm.step(torch.tensor([a])) for _ in range(6)]
    rows.append(labeled_row([obs[K - 1]] + [to_uint8(f)[0] for f in frames],
                            ["start"] + [f"{ACTION_NAMES[a]} x{i + 1}" for i in range(6)]))
stack_rows(rows).save(out / "actions.png")

# --- 5. autoregressive rollout vs ground truth (8 episodes in one batch) -----------
eps = [e for e in episodes if len(e[1]) >= K + args.horizon][:8]
H = args.horizon
wm.reset(torch.stack([to_tensor(o[:K]) for o, a in eps]), torch.stack([torch.from_numpy(a[:K - 1]) for o, a in eps]))
errs, gif_frames = np.zeros(H), []
for i in range(H):
    pred = wm.step(torch.stack([torch.tensor(a[K - 1 + i]) for o, a in eps])).cpu()  # replay recorded actions
    truth = torch.stack([to_tensor(o[K + i]) for o, a in eps])
    errs[i] = (pred - truth).pow(2).mean().item()
    o0 = eps[0][0]
    both = np.concatenate([o0[K + i], np.full((o0.shape[1], 2, 3), 255, np.uint8), to_uint8(pred[0])], 1)
    gif_frames.append(Image.fromarray(upscale(both, 6)))
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
    pred = wm.predict(ctx, a).cpu()
    m_err += (pred - truth).pow(2).mean((1, 2, 3)).tolist()
    c_err += (ctx[:, -1] - truth).pow(2).mean((1, 2, 3)).tolist()
    changed += ((ctx[:, -1] - truth).abs().amax((1, 2, 3)) > 0.05).tolist()
m_err, c_err, changed = map(np.array, (m_err, c_err, changed))
print(f"\none-step MSE on {len(m_err)} test windows   model   copy-last-frame")
print(f"  frames that changed ({changed.mean():4.0%})        {m_err[changed].mean():.4f}   {c_err[changed].mean():.4f}")
print(f"  all frames                      {m_err.mean():.4f}   {c_err.mean():.4f}")
print("  (lower is better; the model should clearly beat copying on frames that changed)")
