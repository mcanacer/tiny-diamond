"""Toy game used as the "real world" we record data from.

A blue paddle (controlled by the action) and a red ball that bounces on its own.
This gives the world model two kinds of dynamics to learn:
  - action-driven motion (the paddle) -> tests action conditioning
  - autonomous motion (the ball)      -> tests that it uses the frame history (velocity)

Actions: 0 = noop, 1 = up, 2 = down, 3 = left, 4 = right
"""
import numpy as np

NUM_ACTIONS = 5
ACTION_NAMES = ["noop", "up", "down", "left", "right"]
_MOVES = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


class ToyGame:
    def __init__(self, size: int = 32, seed: int = 0):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.paddle = 4 * size // 32   # paddle side in pixels
        self.ball = 2 * size // 32     # ball side in pixels
        self.speed = 2 * size // 32    # pixels per step

    def reset(self) -> np.ndarray:
        s = self.size
        self.p = self.rng.integers(0, s - self.paddle, size=2)          # paddle (row, col)
        self.b = self.rng.integers(0, s - self.ball, size=2)            # ball (row, col)
        self.v = self.rng.choice([-1, 1], size=2) * self.speed          # ball velocity
        return self.render()

    def step(self, action: int) -> np.ndarray:
        s = self.size
        dr, dc = _MOVES[int(action)]
        self.p = np.clip(self.p + np.array([dr, dc]) * self.speed, 0, s - self.paddle)
        self.b = self.b + self.v
        for i in range(2):  # bounce off walls
            if self.b[i] < 0 or self.b[i] > s - self.ball:
                self.v[i] *= -1
                self.b[i] = np.clip(self.b[i], 0, s - self.ball)
        return self.render()

    def render(self) -> np.ndarray:
        """Returns uint8 image (H, W, 3)."""
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        img[..., :] = (20, 20, 30)  # dark background
        r, c = self.p
        img[r:r + self.paddle, c:c + self.paddle] = (60, 140, 255)
        r, c = self.b
        img[r:r + self.ball, c:c + self.ball] = (255, 70, 70)
        return img


def random_policy(rng: np.random.Generator, prev: int, sticky: float = 0.8) -> int:
    """Random actions that repeat for a while, so the paddle makes visible moves."""
    return prev if rng.random() < sticky else int(rng.integers(NUM_ACTIONS))
