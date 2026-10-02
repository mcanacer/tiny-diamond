"""Layer 5 - Rollout: "how to use it as a simulator".

Keeps a rolling buffer of the last K frames and actions. Each step:
    push action -> sample next frame -> push frame
Predictions are fed back as inputs, so errors can compound: this is the real test.

DIAMOND equivalent: src/envs/world_model_env.py (minus the reward/done model)
"""
import torch

from .sampler import Sampler


class WorldModelEnv:
    def __init__(self, sampler: Sampler):
        self.sampler = sampler

    def reset(self, obs, act):
        """Seed with K real frames (B, K, C, H, W) and the K-1 actions between them (B, K-1)."""
        self.obs_buf = obs.clone()
        self.act_buf = torch.cat([act, act.new_zeros(act.shape[0], 1)], dim=1)  # last slot filled by step()
        return self.obs_buf[:, -1]

    @torch.no_grad()
    def step(self, action):
        """action: (B,) long. Returns the predicted next frame (B, C, H, W)."""
        self.act_buf[:, -1] = action
        frame = self.sampler.sample(self.obs_buf, self.act_buf)
        self.obs_buf = torch.cat([self.obs_buf[:, 1:], frame[:, None]], dim=1)
        self.act_buf = torch.cat([self.act_buf[:, 1:], self.act_buf.new_zeros(action.shape[0], 1)], dim=1)
        return frame
