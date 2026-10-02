"""Compare several world models on identical play sessions.

    python diagnostics/compare.py outputs/model.pt outputs/model_A_ctxnoise.pt outputs/model_B_ctxnoise_rollout.pt

Every model plays the same games with the same moves (same seeds), under two policies:
  random  - the random-ish policy used to record the training data
  chase   - steers the paddle into the ball, i.e. lots of the rare contact cases
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from diagnostics.ball_tracker import load_model, run_sessions, summarize

p = argparse.ArgumentParser()
p.add_argument("ckpts", nargs="+")
p.add_argument("--runs", type=int, default=32)
p.add_argument("--steps", type=int, default=600)
p.add_argument("--ctx_noise", type=float, nargs="*", default=[0.0],
               help="inference context-noise levels to try (only matters for models trained with it)")
p.add_argument("--policies", nargs="+", default=["random", "chase"])
p.add_argument("--seed", type=int, default=500)
p.add_argument("--json", default="")
args = p.parse_args()

rows = []
for ck in args.ckpts:
    for lvl in args.ctx_noise:
        model, cfg = load_model(ck, lvl)
        if lvl > 0 and cfg.ctx_noise_max == 0:
            continue
        for policy in args.policies:
            s = summarize(run_sessions(model, cfg, args.runs, args.steps, policy, args.seed), cfg.K)
            name = Path(ck).stem + (f" @ctx{lvl}" if lvl else "")
            rows.append({"model": name, "policy": policy, **s})
            print(f"{name:34s} {policy:6s} " + "  ".join(f"{k}: {v:5.1%}" for k, v in s.items()), flush=True)

if args.json:
    Path(args.json).write_text(json.dumps(rows, indent=2))
