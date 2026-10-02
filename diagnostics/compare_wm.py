"""Compare world models in one table: any model type (U-Net / transformer, pixels / latents), any env.
All models are scored on the same test episodes with the same seeds.

    python diagnostics/compare_wm.py outputs/crafter.pt outputs/crafter_dit_df.pt --seeds 3

Columns:
  params, steps     model size (world model only, not its autoencoder) and training steps
  1-step vs copy    one-step error on frames that changed, as a fraction of the "copy the last frame" error
                    (lower is better; 0.35 = 65% less error than copying)
  roll@10, roll@40  rollout error on the whole frame after 10 / 40 steps (replaying the recorded actions).
                    Partly unavoidable when the game is random or reveals new areas, so compare models
                    with each other rather than with zero.
  <region>@N        the same, restricted to an env-specific region (Crafter: player tile, inventory bar)
  detail@40         image detail of the model's frames / the real frames after 40 steps (1.0 = as detailed
                    as the real game; well below 1 = the world blurs or empties out)
With --seeds > 1 every number is mean±std over seeds; differences smaller than the ± are not trustworthy.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from wm.data import load_episodes, to_tensor
from wm.world_model import WorldModel


def lap(x):  # mean absolute Laplacian = amount of fine detail
    return (4 * x[..., 1:-1, 1:-1] - x[..., :-2, 1:-1] - x[..., 2:, 1:-1] - x[..., 1:-1, :-2] - x[..., 1:-1, 2:]).abs().mean().item()


@torch.no_grad()
def evaluate(wm, episodes, horizon=40, n_windows=16, batch=64, seed=0, max_rollouts=32):
    K, regions = wm.K, wm.spec.eval_regions
    rng = np.random.default_rng(seed)
    windows = [(e, int(rng.integers(0, len(a) - K + 1))) for e, (o, a) in enumerate(episodes) if len(a) >= K
               for _ in range(n_windows)]
    m_err, c_err, changed = [], [], []
    torch.manual_seed(seed)
    for i in range(0, len(windows), batch):
        chunk = windows[i:i + batch]
        ctx = torch.stack([to_tensor(episodes[e][0][t:t + K]) for e, t in chunk])
        act = torch.stack([torch.from_numpy(episodes[e][1][t:t + K]) for e, t in chunk])
        truth = torch.stack([to_tensor(episodes[e][0][t + K]) for e, t in chunk])
        pred = wm.predict(ctx, act).cpu()
        m_err += (pred - truth).pow(2).mean((1, 2, 3)).tolist()
        c_err += (ctx[:, -1] - truth).pow(2).mean((1, 2, 3)).tolist()
        changed += ((ctx[:, -1] - truth).abs().amax((1, 2, 3)) > 0.05).tolist()
    m_err, c_err, changed = map(np.array, (m_err, c_err, changed))
    out = {"1-step vs copy": m_err[changed].mean() / max(c_err[changed].mean(), 1e-8)}

    eps = [e for e in episodes if len(e[1]) >= K + horizon][:max_rollouts]
    torch.manual_seed(seed + 1000)
    wm.reset(torch.stack([to_tensor(o[:K]) for o, a in eps]), torch.stack([torch.from_numpy(a[:K - 1]) for o, a in eps]))
    detail_rows = regions.get("map", (slice(None), slice(None)))
    for t in range(horizon):
        pred = wm.step(torch.stack([torch.tensor(a[K - 1 + t]) for o, a in eps])).cpu()
        if t + 1 in (10, horizon):
            truth = torch.stack([to_tensor(o[K + t]) for o, a in eps])
            out[f"roll@{t + 1}"] = (pred - truth).pow(2).mean().item()
            for r, (ys, xs) in regions.items():
                if r != "map":
                    out[f"{r}@{t + 1}"] = (pred[..., ys, xs] - truth[..., ys, xs]).pow(2).mean().item()
            ys, xs = detail_rows
            out[f"detail@{t + 1}"] = lap(pred[..., ys, xs]) / max(lap(truth[..., ys, xs]), 1e-8)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--data", default=None, help="default: data/<env>/test of the first model")
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--device", default=None)
    p.add_argument("--windows", type=int, default=16, help="one-step test windows per episode (lower = faster)")
    p.add_argument("--seeds", type=int, default=1, help="repeat with this many seeds and print mean±std")
    p.add_argument("--denoise_steps", type=int, default=None, help="override the sampler's step count")
    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    episodes, cols, data_env = None, None, None
    for ck in args.ckpts:
        if not Path(ck).exists():
            print(f"{Path(ck).stem:24s}  (not found)")
            continue
        wm = WorldModel(ck, device, args.denoise_steps)
        if episodes is None:
            data_env = wm.cfg.env
            episodes = load_episodes(args.data or f"data/{data_env}/test")
            print(f"env={data_env}: {len(episodes)} test episodes, {args.seeds} seed(s)")
        elif wm.cfg.env != data_env:
            print(f"{Path(ck).stem:24s}  (skipped: env {wm.cfg.env} != {data_env}; compare one env at a time)")
            continue
        runs = [evaluate(wm, episodes, args.horizon, args.windows, seed=s) for s in range(args.seeds)]
        if cols is None:
            cols = ["params", "steps", "model"] + list(runs[0])
            print(f"{'model':24s}" + "".join(f"{c:>15s}" for c in cols))
        r = {}
        for c in runs[0]:
            v = np.array([x[c] for x in runs])
            r[c] = f"{v.mean():.4f}" if args.seeds == 1 else f"{v.mean():.3f}±{v.std(ddof=1):.3f}"
        r["params"] = f"{wm.n_params / 1e6:.1f}M"
        r["steps"] = str(wm.step_count)
        r["model"] = wm.cfg.model + ("+latent" if wm.cfg.latent else "")
        print(f"{Path(ck).stem:24s}" + "".join(f"{r[c]:>15s}" for c in cols), flush=True)


if __name__ == "__main__":
    main()
