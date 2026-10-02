"""Layer 1 - Data: "what happened".

Storage convention (the most important thing to get right):
    obs: uint8 (T+1, H, W, 3)   frames o_0 ... o_T
    act: int64 (T,)             act[t] is the action taken AT frame t, producing o_{t+1}

A training window of K context frames gives:
    ctx_obs = o_t ... o_{t+K-1}        (K frames)
    ctx_act = a_t ... a_{t+K-1}        (K actions; the LAST one produces the target)
    target  = o_{t+K}

DIAMOND equivalent: src/data/episode.py, segment.py, dataset.py, utils.py
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Batch:
    obs: torch.Tensor  # (B, K+1, C, H, W) float in [-1, 1]; last frame is the target
    act: torch.Tensor  # (B, K) long

    def to(self, device):
        obs = self.obs.to(device, non_blocking=True)
        if obs.dtype == torch.uint8:  # frames travel as uint8 (4x less data), normalised on the device
            obs = obs.float().div(127.5).sub(1).movedim(-1, -3)
        return Batch(obs, self.act.to(device, non_blocking=True))

    def pin_memory(self):
        return Batch(self.obs.pin_memory(), self.act.pin_memory())


def to_tensor(frames_uint8: np.ndarray) -> torch.Tensor:
    """(..., H, W, 3) uint8 -> (..., 3, H, W) float in [-1, 1]."""
    x = torch.from_numpy(frames_uint8).float().div(127.5).sub(1)
    return x.movedim(-1, -3)


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """(..., 3, H, W) float in [-1, 1] -> (..., H, W, 3) uint8."""
    x = x.clamp(-1, 1).add(1).mul(127.5).round().byte()
    return x.movedim(-3, -1).cpu().numpy()


def save_episodes(path: Path, episodes):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for i, (obs, act) in enumerate(episodes):
        np.savez_compressed(path / f"ep_{i:05d}.npz", obs=obs, act=act)


def load_episodes(path: Path):
    files = sorted(Path(path).glob("ep_*.npz"))
    assert files, f"no episodes found in {path} - run record.py first"
    eps = [(d["obs"], d["act"]) for d in (np.load(f) for f in files)]
    return [(o, a) for o, a in eps if len(a) > 0]


class WindowDataset(torch.utils.data.Dataset):
    """Every valid (episode, start) pair is one sample of K context frames + 1 target."""

    def __init__(self, episodes, K: int):
        self.episodes, self.K = episodes, K
        # episodes shorter than K+1 frames (e.g. an early death in Crafter) simply contribute no windows
        self.index = [(e, t) for e, (obs, act) in enumerate(episodes) for t in range(len(act) - K + 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        """Returns uint8 frames (K+1, H, W, 3); Batch.to(device) converts them to floats."""
        e, t = self.index[i]
        obs, act = self.episodes[e]
        return torch.from_numpy(np.ascontiguousarray(obs[t:t + self.K + 1])), torch.from_numpy(act[t:t + self.K]).long()


def collate(samples) -> Batch:
    obs, act = zip(*samples)
    return Batch(torch.stack(obs), torch.stack(act))
