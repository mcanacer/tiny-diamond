"""Checkpoint 1: look at training windows and confirm the action alignment by eye.

For each row: K context frames, then the target. The label above each frame is the
action taken AT that frame. The action above the last context frame must explain
the paddle's move into the target.

    python inspect_data.py --data data/train --out outputs/data_windows.png
"""
import argparse
from pathlib import Path

import torch

from viz import labeled_row, stack_rows
from wm.config import Config
from wm.data import WindowDataset, load_episodes, to_uint8
from wm.envs import get_env

p = argparse.ArgumentParser()
p.add_argument("--env", default="toy")
p.add_argument("--data", default=None)
p.add_argument("--out", default=None)
p.add_argument("--n", type=int, default=6)
args = p.parse_args()
ACTION_NAMES = get_env(args.env).action_names
args.data = args.data or f"data/{args.env}/train"
args.out = args.out or f"outputs/{args.env}_data_windows.png"

ds = WindowDataset(load_episodes(args.data), Config.K)
print(f"{len(ds)} training windows")
rows = []
for i in torch.randint(len(ds), (args.n,)).tolist():
    obs, act = ds[i]
    frames = list(obs.numpy())  # dataset returns raw uint8 frames (K+1, H, W, 3)
    labels = [f"a={ACTION_NAMES[a]}" for a in act.tolist()] + ["TARGET"]
    rows.append(labeled_row(frames, labels))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
stack_rows(rows).save(args.out)
print("saved", args.out)
