"""Step 0: record (frames, actions) from a game with a random-ish behaviour policy.

    python record.py --env toy     --out data/toy/train
    python record.py --env crafter --out data/crafter/train                     # 1000 episodes, ~170k frames
    python record.py --env crafter --out data/crafter/test --episodes 20 --seed 1

Episodes end early if the game ends (e.g. the player dies in Crafter).
Each episode is saved as soon as it's recorded, and existing files are skipped, so an
interrupted recording resumes where it stopped. Episodes are recorded in parallel on all
CPU cores (Crafter spends most of its time generating each new world).
"""
import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from wm.config import PRESETS
from wm.envs import get_env


def record_episode(job):
    env_name, out, e, seed, ep_len = job
    path = Path(out) / f"ep_{e:05d}.npz"
    if path.exists():
        return 0
    spec = get_env(env_name)
    rng = np.random.default_rng([seed, e])
    game = spec.make(seed=seed * 100_000 + e)  # a new world per episode
    frames, actions, a = [game.reset()], [], 0
    for t in range(ep_len):
        a = spec.random_policy(rng, a)
        frame, done = game.step(a)
        actions.append(a)        # act[t] is taken at frame t ...
        frames.append(frame)     # ... and produces frame t+1
        if done:
            break
    tmp = path.with_name("tmp_" + path.name)  # must NOT match ep_*.npz, which is what the loader reads
    np.savez_compressed(tmp, obs=np.stack(frames).astype(np.uint8), act=np.array(actions, dtype=np.int64))
    os.replace(tmp, path)
    return len(actions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="toy")
    p.add_argument("--out", default=None, help="default: data/<env>/train")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--len", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()

    spec = get_env(args.env)
    preset = PRESETS.get(args.env, {})
    n_eps = args.episodes or preset.get("num_episodes", 200)
    ep_len = args.len or preset.get("episode_len", 100)
    out = args.out or f"data/{args.env}/train"
    Path(out).mkdir(parents=True, exist_ok=True)

    jobs = [(args.env, out, e, args.seed, ep_len) for e in range(n_eps)]
    t0, n_frames, done = time.time(), 0, 0
    with Pool(args.workers) as pool:
        for n in pool.imap_unordered(record_episode, jobs):
            n_frames += n
            done += 1
            if done % max(1, n_eps // 10) == 0:
                print(f"  {done}/{n_eps} episodes, {n_frames} new frames, {time.time() - t0:.0f}s", flush=True)
    total = sum(np.load(f)["act"].shape[0] for f in Path(out).glob("ep_*.npz"))
    print(f"{out}: {n_eps} episodes, {total} frames ({total * spec.img_size ** 2 * 3 / 1e9:.2f} GB in RAM when training)")


if __name__ == "__main__":
    main()
