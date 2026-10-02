"""Run long play sessions in the world model alongside the real game and find when/why the ball vanishes.

    python diagnostics/ball_tracker.py --ckpt outputs/model.pt --runs 16 --steps 300
    python diagnostics/ball_tracker.py --policy chase --save vanish.png   # steer into the ball a lot
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from wm.data import to_tensor, to_uint8
from wm.world_model import WorldModel
from wm.env import ToyGame, random_policy


def red_mask(img):   # ball pixels: strong red, weak green/blue
    return (img[..., 0] > 150) & (img[..., 1] < 140) & (img[..., 2] < 140)


def blue_mask(img):
    return (img[..., 2] > 150) & (img[..., 0] < 140)


def load_model(ckpt, ctx_noise=0.0):
    """Any toy-game world model (U-Net or transformer) behind the common WorldModel interface."""
    wm = WorldModel(ckpt, "cpu", None, ctx_noise)
    return wm, wm.cfg


def run_sessions(model, cfg, runs=32, steps=600, policy="random", seed=500, denoise_steps=3, keep_frames=False):
    """Plays `runs` sessions in parallel (one batch). Same seed -> same games and same moves
    for every model, so results are directly comparable.

    Returns dict of arrays shaped (steps, runs): ball_px, pos_err, contact, near_wall
    (+ frames if keep_frames)."""
    K = cfg.K
    env = model  # a WorldModel: reset(frames, actions) / step(action)
    env.n_steps = denoise_steps
    torch.manual_seed(seed)
    games = [ToyGame(cfg.img_size, seed=seed + i) for i in range(runs)]
    rngs = [np.random.default_rng(seed + i) for i in range(runs)]
    start = []
    for g in games:
        f = [g.reset()]
        for _ in range(K - 1):
            f.append(g.step(0))
        start.append(np.stack(f))
    env.reset(to_tensor(np.stack(start)), torch.zeros(runs, K - 1, dtype=torch.long))

    out = {k: np.zeros((steps, runs)) for k in ["ball_px", "pos_err"]}
    out.update({k: np.zeros((steps, runs), bool) for k in ["contact", "near_wall"]})
    frames = [[] for _ in range(runs)] if keep_frames else None
    acts = np.zeros(runs, dtype=int)
    for t in range(steps):
        for i, g in enumerate(games):
            if policy == "chase":  # steer the paddle toward the ball -> lots of collisions
                d = g.b - g.p
                acts[i] = (1 if d[0] < 0 else 2) if abs(d[0]) > abs(d[1]) else (3 if d[1] < 0 else 4)
                if rngs[i].random() < 0.3:
                    acts[i] = rngs[i].integers(5)
            else:
                acts[i] = random_policy(rngs[i], acts[i])
        pred = to_uint8(env.step(torch.from_numpy(acts).long()))
        for i, g in enumerate(games):
            real = g.step(acts[i])
            m = red_mask(pred[i])
            out["ball_px"][t, i] = m.sum()
            out["pos_err"][t, i] = np.abs(np.argwhere(m).mean(0) - (g.b + 0.5)).max() if m.any() else np.nan
            out["contact"][t, i] = np.abs((g.b + 1) - (g.p + 2)).max() <= 4
            out["near_wall"][t, i] = (g.b.min() <= 1) or (g.b.max() >= g.size - 3)
            if keep_frames:
                frames[i].append((pred[i], real))
    if keep_frames:
        out["frames"] = frames
    return out


def summarize(res, K=4):
    """Headline numbers for one model."""
    px, err = res["ball_px"], res["pos_err"]
    present = px >= 1
    correct = present & (np.nan_to_num(err, nan=99) <= 2)  # ball visible AND within 2 px of the truth
    lost = []
    for i in range(px.shape[1]):  # did the ball ever go missing for more than K frames in a row?
        streak, worst = 0, 0
        for v in ~present[:, i]:
            streak = streak + 1 if v else 0
            worst = max(worst, streak)
        lost.append(worst > K)
    return {
        "ball visible": present.mean(),
        "ball in right place": correct.mean(),
        "sessions ever losing ball": (~present).any(0).mean(),
        f"sessions losing it >{K} frames": np.mean(lost),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/model.pt")
    p.add_argument("--runs", type=int, default=16)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--denoise_steps", type=int, default=3)
    p.add_argument("--ctx_noise", type=float, default=0.0, help="context noise level at inference")
    p.add_argument("--policy", default="random", choices=["random", "chase"])
    p.add_argument("--seed", type=int, default=500)
    p.add_argument("--save", default="", help="save a strip of frames around the first vanish")
    args = p.parse_args()

    model, cfg = load_model(args.ckpt, args.ctx_noise)
    res = run_sessions(model, cfg, args.runs, args.steps, args.policy, args.seed, args.denoise_steps,
                       keep_frames=bool(args.save))
    for k, v in summarize(res, cfg.K).items():
        print(f"{k:32s} {v:6.1%}")

    present = res["ball_px"] >= 1
    for i in range(args.runs):
        if (~present[:, i]).any():
            t = int(np.argmax(~present[:, i]))
            w = slice(max(0, t - 5), t + 1)
            print(f"  run {i:2d}: first vanish at step {t:3d} | touching paddle just before: "
                  f"{res['contact'][w, i].any()} | near wall: {res['near_wall'][w, i].any()} | "
                  f"visible in last 100 steps: {present[-100:, i].mean():.0%}")

    if args.save:
        from PIL import Image
        gone = np.where((~present).any(0))[0]
        i = int(gone[0]) if len(gone) else 0
        t = int(np.argmax(~present[:, i])) if len(gone) else args.steps - 1
        tiles = []
        for k in range(max(0, t - 7), min(t + 2, args.steps)):
            pr, re = res["frames"][i][k]
            tiles.append(np.concatenate([re, np.full((re.shape[0], 1, 3), 255, np.uint8), pr], 1))
        strip = np.concatenate([np.pad(x, ((1, 1), (1, 1), (0, 0)), constant_values=255) for x in tiles], 1)
        Image.fromarray(strip.repeat(6, 0).repeat(6, 1)).save(args.save)
        print(f"saved {args.save} (run {i}; each tile = real | model)")


if __name__ == "__main__":
    main()
