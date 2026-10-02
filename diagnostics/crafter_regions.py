"""Where does a Crafter rollout go wrong? Splits the rollout error by screen region.

Crafter's 64x64 frame: top ~49 rows = map (9x7 tiles, player in the centre tile),
bottom ~15 rows = inventory bar. New terrain at the map edges is unknowable, so map error vs the real game
grows even for a perfect model. The player tile and the inventory are (mostly) predictable,
so error there is a better signal of real drift. Sharpness compares image detail to real frames:
a blurring model loses detail over time even if its colours are right.

    python diagnostics/crafter_regions.py --ckpt outputs/crafter.pt
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from wm.config import build_denoiser, config_from_dict
from wm.data import load_episodes, to_tensor
from wm.rollout import WorldModelEnv
from wm.sampler import Sampler

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="outputs/crafter.pt")
p.add_argument("--data", default="data/crafter/test")
p.add_argument("--horizon", type=int, default=40)
p.add_argument("--episodes", type=int, default=16)
args = p.parse_args()

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cfg = config_from_dict(ck["cfg"]); K = cfg.K
model = build_denoiser(cfg); model.load_state_dict(ck["model"]); model.eval()
env = WorldModelEnv(Sampler(model, 3)); torch.manual_seed(0)
eps = [e for e in load_episodes(args.data) if len(e[1]) >= K + args.horizon][:args.episodes]

regions = {"player tile": (slice(21, 29), slice(28, 36)), "inventory bar": (slice(50, 64), slice(0, 64)),
           "map (all)": (slice(0, 49), slice(0, 64))}
obs0 = torch.stack([to_tensor(o[:K]) for o, a in eps]); act0 = torch.stack([torch.from_numpy(a[:K - 1]) for o, a in eps])
env.reset(obs0, act0)
H = args.horizon
err = {r: np.zeros(H) for r in regions}
sharp_pred, sharp_real = np.zeros(H), np.zeros(H)
lap = lambda x: (4 * x[..., 1:-1, 1:-1] - x[..., :-2, 1:-1] - x[..., 2:, 1:-1] - x[..., 1:-1, :-2] - x[..., 1:-1, 2:]).abs().mean().item()
for i in range(H):
    pred = env.step(torch.stack([torch.tensor(a[K - 1 + i]) for o, a in eps]))
    truth = torch.stack([to_tensor(o[K + i]) for o, a in eps])
    for r, (ys, xs) in regions.items():
        err[r][i] = (pred[..., ys, xs] - truth[..., ys, xs]).pow(2).mean().item()
    sharp_pred[i], sharp_real[i] = lap(pred[..., :49, :]), lap(truth[..., :49, :])

print(f"rollout MSE by region over {len(eps)} episodes (replaying the recorded actions)")
print("step  " + "  ".join(f"{r:>14s}" for r in regions) + "   map sharpness (model / real)")
for i in [0, 4, 9, 19, 29, 39]:
    if i < H:
        print(f"{i + 1:4d}  " + "  ".join(f"{err[r][i]:14.4f}" for r in regions) + f"   {sharp_pred[i]:.3f} / {sharp_real[i]:.3f}")
