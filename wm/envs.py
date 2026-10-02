"""Environment registry: everything game-specific lives here, the rest of the code is generic.

Each environment provides:
    make(seed)                 -> object with reset() -> frame and step(action) -> (frame, done)
    num_actions, action_names
    img_size                   frames are uint8 (img_size, img_size, 3)
    random_policy(rng, prev)   the behaviour policy used to record training data
    keymap                     key combo -> action index for play.py; "w+a" means both keys held.
                               play.py picks the longest combo that is currently held.
    showcase_actions           actions shown in evaluate.py's action test
    eval_regions               optional named screen regions for diagnostics/compare_wm.py

Available: toy (32x32), crafter (2D Minecraft), atari_breakout / atari_pong / atari_boxing
(DIAMOND's benchmark), doom_maze / doom_defend (3D, ViZDoom). All but toy are 64x64.
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
    eval_regions: Dict[str, tuple] = field(default_factory=dict)

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
    # (rows, cols) slices: player tile in the centre of the 9x7 map, inventory bar at the bottom
    eval_regions={"player": (slice(21, 29), slice(28, 36)), "inv": (slice(50, 64), slice(0, 64)),
                  "map": (slice(0, 49), slice(0, 64))},
)

def _resize(frame: np.ndarray, size: int = 64) -> np.ndarray:
    """Area-average downsampling (keeps 1-pixel things like Breakout's ball visible)."""
    from PIL import Image
    return np.asarray(Image.fromarray(frame).resize((size, size), Image.BOX), dtype=np.uint8)


def _sticky_policy(weights):
    p = np.asarray(weights, dtype=float)
    p = p / p.sum()

    def policy(rng, prev, sticky=0.75):
        return prev if rng.random() < sticky else int(rng.choice(len(p), p=p))
    return policy


# ---- Atari (DIAMOND's benchmark) -------------------------------------------------------------
class _AtariWrapper:
    """ALE via gymnasium: frameskip 4, no sticky actions (like DIAMOND's NoFrameskip-v4 + frame skip),
    full frame (210x160) area-resized to 64x64 as in DIAMOND."""

    def __init__(self, game, seed):
        import ale_py
        import gymnasium as gym
        gym.register_envs(ale_py)
        self.env = gym.make(f"ALE/{game}-v5", frameskip=4, repeat_action_probability=0.0)
        self.seed = seed

    def reset(self):
        obs, _ = self.env.reset(seed=self.seed)
        return _resize(obs)

    def step(self, action):
        obs, _, term, trunc, _ = self.env.step(int(action))
        return _resize(obs), bool(term or trunc)

    def close(self):
        self.env.close()


_ATARI_KEYS = {"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right", "FIRE": "space"}


def _atari_keymap(meanings):
    """'UPRIGHTFIRE' -> 'up+right+space' (arrows to move, space to fire)."""
    km = {}
    for a, m in enumerate(meanings):
        if m == "NOOP":
            continue
        keys, rest = [], m
        for token in ["UP", "DOWN", "LEFT", "RIGHT", "FIRE"]:
            if token in rest:
                keys.append(_ATARI_KEYS[token])
                rest = rest.replace(token, "", 1)
        km["+".join(keys)] = a
    return km


def _atari_spec(game, meanings, showcase):
    return EnvSpec(
        name=f"atari_{game.lower()}",
        make=lambda seed, g=game: _AtariWrapper(g, seed),
        action_names=[m.lower() for m in meanings],
        img_size=64,
        random_policy=_sticky_policy([1.0] * len(meanings)),
        keymap=_atari_keymap(meanings),
        showcase_actions=showcase,
        notes="pip install 'gymnasium[atari]'",
    )


ATARI_BREAKOUT = _atari_spec("Breakout", ["NOOP", "FIRE", "RIGHT", "LEFT"], [0, 1, 2, 3])
ATARI_PONG = _atari_spec("Pong", ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"], [0, 1, 2, 3])
ATARI_BOXING = _atari_spec("Boxing", ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT",
                                      "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
                                      "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"],
                           [0, 1, 2, 3, 4, 5])


# ---- ViZDoom (3D, the setting of GameNGen) ---------------------------------------------------
class _DoomWrapper:
    """ViZDoom scenario rendered at 160x120 without HUD, 4 game tics per step (like GameNGen),
    area-resized to 64x64. `combos` maps each discrete action to the buttons held during it."""

    def __init__(self, scenario, combos, seed):
        import os
        import vizdoom as vzd
        self.game = vzd.DoomGame()
        self.game.load_config(os.path.join(vzd.scenarios_path, scenario + ".cfg"))
        self.game.set_window_visible(False)
        self.game.set_sound_enabled(False)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_render_hud(False)
        self.game.set_seed(int(seed) % (2 ** 31))
        self.game.init()
        self.n_buttons = len(self.game.get_available_buttons())
        self.combos = combos
        self.last = None

    def _frame(self):
        st = self.game.get_state()
        if st is not None:
            self.last = _resize(st.screen_buffer)
        return self.last

    def reset(self):
        self.game.new_episode()
        return self._frame()

    def step(self, action):
        buttons = [0] * self.n_buttons
        for b in self.combos[int(action)]:
            buttons[b] = 1
        self.game.make_action(buttons, 4)
        done = self.game.is_episode_finished()
        return self._frame(), done  # at episode end there is no new screen: repeat the last one

    def close(self):
        self.game.close()


# my_way_home buttons: 0 TURN_LEFT, 1 TURN_RIGHT, 2 MOVE_FORWARD, 3 MOVE_LEFT, 4 MOVE_RIGHT
_MAZE_COMBOS = [(), (2,), (0,), (1,), (2, 0), (2, 1), (3,), (4,)]
DOOM_MAZE = EnvSpec(
    name="doom_maze",
    make=lambda seed: _DoomWrapper("my_way_home", _MAZE_COMBOS, seed),
    action_names=["noop", "forward", "turn_left", "turn_right", "forward_left", "forward_right",
                  "strafe_left", "strafe_right"],
    img_size=64,
    # mostly walk and turn, so the recordings explore the maze
    random_policy=_sticky_policy([0.05, 0.3, 0.12, 0.12, 0.13, 0.13, 0.075, 0.075]),
    keymap={"w": 1, "up": 1, "a": 2, "left": 2, "d": 3, "right": 3, "w+a": 4, "up+left": 4,
            "w+d": 5, "up+right": 5, "q": 6, "e": 7},
    showcase_actions=[0, 1, 2, 3, 6, 7],
    notes="pip install vizdoom",
)

# defend_the_center buttons: 0 TURN_LEFT, 1 TURN_RIGHT, 2 ATTACK
_DEFEND_COMBOS = [(), (0,), (1,), (2,), (0, 2), (1, 2)]
DOOM_DEFEND = EnvSpec(
    name="doom_defend",
    make=lambda seed: _DoomWrapper("defend_the_center", _DEFEND_COMBOS, seed),
    action_names=["noop", "turn_left", "turn_right", "attack", "left_attack", "right_attack"],
    img_size=64,
    random_policy=_sticky_policy([0.1, 0.25, 0.25, 0.2, 0.1, 0.1]),
    keymap={"a": 1, "left": 1, "d": 2, "right": 2, "space": 3, "a+space": 4, "left+space": 4,
            "d+space": 5, "right+space": 5},
    showcase_actions=[0, 1, 2, 3],
    notes="pip install vizdoom",
)

ENVS = {e.name: e for e in [TOY, CRAFTER, ATARI_BREAKOUT, ATARI_PONG, ATARI_BOXING, DOOM_MAZE, DOOM_DEFEND]}


def get_env(name: str) -> EnvSpec:
    if name not in ENVS:
        raise ValueError(f"unknown env {name!r}; choose from {list(ENVS)}")
    return ENVS[name]
