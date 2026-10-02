"""Environment registry: everything game-specific lives here, the rest of the code is generic.

Each environment provides:
    make(seed)                 -> object with reset() -> frame and step(action) -> (frame, done)
    num_actions, action_names
    img_size                   frames are uint8 (img_size, img_size, 3)
    random_policy(rng, prev)   the behaviour policy used to record training data
    keymap                     pygame key name -> action index, for play.py
    showcase_actions           actions shown in evaluate.py's action test

Available: "toy" (paddle + bouncing ball, 32x32) and "crafter" (2D Minecraft, 64x64).
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np


@dataclass
class EnvSpec:
    name: str
    make: Callable
    action_names: List[str]
    img_size: int
    random_policy: Callable
    keymap: Dict[str, int]
    showcase_actions: List[int]
    notes: str = ""

    @property
    def num_actions(self) -> int:
        return len(self.action_names)


# ---- toy game ----------------------------------------------------------------------------
class _ToyWrapper:
    def __init__(self, seed):
        from .env import ToyGame
        self.game = ToyGame(32, seed=seed)

    def reset(self):
        return self.game.reset()

    def step(self, action):
        return self.game.step(action), False


def _toy_policy(rng, prev):
    from .env import random_policy
    return random_policy(rng, prev)


TOY = EnvSpec(
    name="toy",
    make=_ToyWrapper,
    action_names=["noop", "up", "down", "left", "right"],
    img_size=32,
    random_policy=_toy_policy,
    keymap={"up": 1, "w": 1, "down": 2, "s": 2, "left": 3, "a": 3, "right": 4, "d": 4},
    showcase_actions=[0, 1, 2, 3, 4],
)


# ---- Crafter -------------------------------------------------------------------------------
CRAFTER_ACTIONS = ["noop", "move_left", "move_right", "move_up", "move_down", "do", "sleep",
                   "place_stone", "place_table", "place_furnace", "place_plant",
                   "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
                   "make_wood_sword", "make_stone_sword", "make_iron_sword"]


class _CrafterWrapper:
    """Crafter (Hafner 2021): procedurally generated 2D survival game. The view scrolls as you
    walk, so the world model must invent plausible new terrain at the edges of the screen."""

    def __init__(self, seed):
        import crafter
        self.env = crafter.Env(size=(64, 64), seed=seed)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, _, done, _ = self.env.step(int(action))
        return obs, done


# Behaviour policy for recording: sticky actions, mostly walking and "do" (hit/collect),
# so the data shows movement, collecting wood/stone, fighting, and crafting attempts.
_CRAFTER_P = np.array([0.04, 0.15, 0.15, 0.15, 0.15, 0.22, 0.02] + [0.012] * 10)
_CRAFTER_P = _CRAFTER_P / _CRAFTER_P.sum()


def _crafter_policy(rng, prev, sticky=0.75):
    return prev if rng.random() < sticky else int(rng.choice(len(CRAFTER_ACTIONS), p=_CRAFTER_P))


CRAFTER = EnvSpec(
    name="crafter",
    make=_CrafterWrapper,
    action_names=CRAFTER_ACTIONS,
    img_size=64,
    random_policy=_crafter_policy,
    # same keys as Crafter's own GUI (crafter/run_gui.py)
    keymap={"a": 1, "d": 2, "w": 3, "s": 4, "space": 5, "tab": 6, "r": 7, "t": 8, "f": 9, "p": 10,
            "1": 11, "2": 12, "3": 13, "4": 14, "5": 15, "6": 16},
    showcase_actions=[0, 1, 2, 3, 4, 5],
    notes="pip install crafter",
)

ENVS = {"toy": TOY, "crafter": CRAFTER}


def get_env(name: str) -> EnvSpec:
    if name not in ENVS:
        raise ValueError(f"unknown env {name!r}; choose from {list(ENVS)}")
    return ENVS[name]
