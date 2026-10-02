"""Compare Crafter world models in one table (same test episodes, same seeds, same sampler).

    python diagnostics/compare_crafter.py outputs/crafter.pt outputs/crafter_big.pt outputs/crafter_big_noise.pt

Columns:
  params            model size
  1-step vs copy    one-step error on frames that changed, as a fraction of the "copy the last frame"
                    error (lower is better; 0.43 = 57% less error than copying)
  player@10/@40     rollout error on the player tile after 10 / 40 steps: should stay low (predictable)
  inv@40            rollout error on the inventory bar after 40 steps
  map@40            rollout error on the map after 40 steps (partly unavoidable: new terrain is invented)
  detail@40         map sharpness of the model's frames / the real frames after 40 steps.
                    1.0 = as detailed as the real game; well below 1 = the world is emptying out / blurring.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from wm.config import build_denoiser, config_from_dict
from wm.data import load_episodes, to_tensor
from wm.rollout import WorldModelEnv
from wm.sampler import Sampler

REGIONS = {"player": (slice(21, 29), slice(28, 36)), "inv": (slice(50, 64), slice(0, 64)), "map": (slice(0, 49), slice(0, 64))}


def lap(x):  # mean absolute Laplacian = amount of fine detail
    return (4 * x[..., 1:-1, 1:-1] - x[..., :-2, 1:-1] - x[..., 2:, 1:-1] - x[..., 1:-1, :-2] - x[..., 1:-1, 2:]).abs().mean().item()


def load(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ck["cfg"])
    model = build_denoiser(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, ck.get("step", "?")


@torch.no_grad()
def evaluate(model, cfg, episodes, device, horizon=40, n_windows=16, batch=64, seed=0):
    K = cfg.K
    sampler = Sampler(model, cfg.denoise_steps)
    # one-step vs copy-last-frame
    rng = np.random.default_rng(seed)  # which test windows
    windows = [(e, int(rng.integers(0, len(a) - K + 1))) for e, (o, a) in enumerate(episodes) for _ in range(n_windows)]
    m_err, c_err, changed = [], [], []
    torch.manual_seed(seed)  # sampler noise
    for i in range(0, len(windows), batch):
        chunk = windows[i:i + batch]
        ctx = torch.stack([to_tensor(episodes[e][0][t:t + K]) for e, t in chunk]).to(device)
        act = torch.stack([torch.from_numpy(episodes[e][1][t:t + K]) for e, t in chunk]).to(device)
        truth = torch.stack([to_tensor(episodes[e][0][t + K]) for e, t in chunk]).to(device)
        pred = sampler.sample(ctx, act)
        m_err += (pred - truth).pow(2).mean((1, 2, 3)).tolist()
        c_err += (ctx[:, -1] - truth).pow(2).mean((1, 2, 3)).tolist()
        changed += ((ctx[:, -1] - truth).abs().amax((1, 2, 3)) > 0.05).tolist()
    m_err, c_err, changed = map(np.array, (m_err, c_err, changed))
    out = {"1-step vs copy": m_err[changed].mean() / c_err[changed].mean()}

    # rollouts replaying the recorded actions
    eps = [e for e in episodes if len(e[1]) >= K + horizon]
    env = WorldModelEnv(sampler)
    torch.manual_seed(seed + 1000)
    env.reset(torch.stack([to_tensor(o[:K]) for o, a in eps]).to(device),
              torch.stack([torch.from_numpy(a[:K - 1]) for o, a in eps]).to(device))
    for t in range(horizon):
        pred = env.step(torch.stack([torch.tensor(a[K - 1 + t]) for o, a in eps]).to(device))
        if t + 1 in (10, horizon):
            truth = torch.stack([to_tensor(o[K + t]) for o, a in eps]).to(device)
            for r, (ys, xs) in REGIONS.items():
                out[f"{r}@{t + 1}"] = (pred[..., ys, xs] - truth[..., ys, xs]).pow(2).mean().item()
            out[f"detail@{t + 1}"] = lap(pred[..., :49, :]) / lap(truth[..., :49, :])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--data", default="data/crafter/test")
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--device", default=None)
    p.add_argument("--windows", type=int, default=16, help="one-step test windows per episode (lower = faster)")
    p.add_argument("--seeds", type=int, default=1, help="repeat with this many seeds and print mean±std "
                   "(differences smaller than the ± are not trustworthy)")
    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    episodes = load_episodes(args.data)
    cols = ["params", "steps", "1-step vs copy", "player@10", f"player@{args.horizon}", f"inv@{args.horizon}",
            f"map@{args.horizon}", f"detail@{args.horizon}"]
    print(f"{len(episodes)} test episodes, {args.seeds} seed(s)")
    print(f"{'model':22s}" + "".join(f"{c:>16s}" for c in cols))
    for ck in args.ckpts:
        if not Path(ck).exists():
            print(f"{Path(ck).stem:22s}  (not found)")
            continue
        model, cfg, step = load(ck, device)
        runs = [evaluate(model, cfg, episodes, device, args.horizon, args.windows, seed=s) for s in range(args.seeds)]
        r = {}
        for c in runs[0]:
            v = np.array([x[c] for x in runs])
            r[c] = f"{v.mean():.4f}" if args.seeds == 1 else f"{v.mean():.4f}±{v.std(ddof=1):.4f}"
        r["params"] = f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
        r["steps"] = str(step)
        print(f"{Path(ck).stem:22s}" + "".join(f"{r[c]:>16s}" for c in cols), flush=True)


if __name__ == "__main__":
    main()
