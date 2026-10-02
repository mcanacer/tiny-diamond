"""Builds colab/train_crafter.ipynb (run: python colab/make_notebook.py)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# tiny-diamond: world-model experiments on Colab

**Runtime:** `Runtime → Change runtime type →` **A100 GPU** (T4 works too: set `MAX_PARALLEL = 1`).

**Code comes from GitHub** (`mcanacer/tiny-diamond`): every session clones the latest version, so code changes
need no uploads. Open this notebook straight from GitHub to always get the newest version:
`colab.research.google.com/github/mcanacer/tiny-diamond/blob/main/colab/train_crafter.ipynb`

Checkpoints, logs and recorded data live in `MyDrive/tiny-diamond-runs/`. If Colab disconnects, reconnect and
**run all cells again**: finished runs are skipped and interrupted ones resume."""),

    md("""## ⚙️ Config: the only cell you need to edit

Experiments are named in `wm/config.py` (`EXPERIMENTS`; the list is printed in step 2). The new Crafter ladder
changes **one thing per step**, all at ~15M parameters, so each comparison isolates one idea:

| experiment | what it adds | compare with |
|---|---|---|
| `crafter_big_noise` | best U-Net so far (pixels + context noise), already trained | |
| `crafter_ae` | frame autoencoder 64×64×3 → 16×16×8 (trained automatically when needed) | |
| `crafter_latent_unet` | same U-Net recipe, but on autoencoder **latents** | `crafter_big_noise` |
| `crafter_dit_last` | **causal transformer** instead of U-Net (same training recipe) | `crafter_latent_unet` |
| `crafter_dit_df` | **Diffusion Forcing** (every frame its own noise level) | `crafter_dit_last` |
| `crafter_dit_df_k16` | Diffusion Forcing with **16 frames of memory** | `crafter_dit_df` |

New environments, each with `<env>` (DIAMOND-size U-Net), `<env>_big_noise`, `<env>_ae`, `<env>_dit_df`:
`atari_breakout`, `atari_pong`, `atari_boxing` (DIAMOND's benchmark) and `doom_maze`, `doom_defend` (3D, ViZDoom).

Autoencoders are added automatically: latent models start as soon as their autoencoder has finished.

**Cheap and fair:** with `STEPS = 10000` every model trains the same short budget and *finishes* its
learning-rate schedule. Comparing runs of different lengths, or runs stopped halfway (learning rate still high),
mostly measures how far each got, not which method is better."""),
    code("""# What to train in this session (autoencoders they need are added automatically).
EXPERIMENTS = ['crafter_big_noise', 'crafter_latent_unet', 'crafter_dit_last', 'crafter_dit_df']

# Training budget per world model. Every run gets exactly this many steps with a COMPLETE learning-rate
# schedule (warmup, then decay), saved as <name>_<N>k.pt, so cheap short runs are still a fair comparison.
# None = each experiment's own default length (40k steps). Autoencoders always use their own schedule.
STEPS = 10000

# What to compare at the end (same STEPS budget; grouped by environment automatically).
COMPARE = EXPERIMENTS

MAX_PARALLEL = 3     # runs at the same time on one GPU (A100: 3, T4: 1)
MAX_MINUTES = 170    # each run stops cleanly after this long; re-run the notebook to continue
SEEDS = 3            # evaluation repeats: results shown as mean±std"""),

    md("## 1. GPU check + Google Drive"),
    code("""!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')"""),

    md("## 2. Get the latest code from GitHub and install the game environments"),
    code("""import os
DRIVE = '/content/drive/MyDrive'
RUNS = f'{DRIVE}/tiny-diamond-runs'          # checkpoints, logs and cached data live here (persistent)
os.makedirs(RUNS, exist_ok=True)
REPO = 'https://github.com/mcanacer/tiny-diamond.git'
if os.path.exists('/content/tiny-diamond/.git'):
    !git -C /content/tiny-diamond pull --ff-only
else:
    !git clone -q {REPO} /content/tiny-diamond
%cd /content/tiny-diamond
!git log -1 --format='code version: %h  %s  (%cr)'
!pip install -q crafter vizdoom "gymnasium[atari]"
!python train.py --list"""),

    md("""## 3. Data: record once per game, then reuse from Drive
Each game is recorded once (~10-20 min) with a random-ish player and cached on Drive as `<env>_data.tar`;
later sessions unpack the cache in about a minute."""),
    code("""import sys
sys.path.insert(0, '/content/tiny-diamond')
from colab.runner import Runner, envs_needed
from wm.config import EXPERIMENTS as ALL_EXPERIMENTS

for env in envs_needed(EXPERIMENTS + COMPARE):
    cache = f'{RUNS}/{env}_data.tar'
    if os.path.exists(f'data/{env}/train'):
        print(f'{env}: data already here')
    elif os.path.exists(cache):
        !tar -xf {cache} -C /content/tiny-diamond
        print(f'{env}: data restored from Drive')
    else:
        print(f'{env}: recording...')
        !python record.py --env {env} --out data/{env}/train
        !python record.py --env {env} --out data/{env}/test --episodes 20 --seed 1
        !tar -cf {cache} data/{env}
        print(f'{env}: recorded and cached to Drive')"""),

    md("""## 4. Train

Starts every experiment (and the autoencoders they need) in the background, up to `MAX_PARALLEL` at once.
Latent models wait for their autoencoder. The cell returns immediately; use the next cell to watch progress."""),
    code("""runner = Runner(EXPERIMENTS, RUNS, max_parallel=MAX_PARALLEL, max_minutes=MAX_MINUTES, steps=STEPS)
runner.start()"""),

    md("""**Watch progress.** Re-run this cell whenever you like: state of each run, its last log line
(`step`, `loss`, `ms/step`, `eta`) and GPU use. If Colab disconnects the runs stop too: reconnect and run all
cells again; they resume."""),
    code("""runner.status()
!nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv"""),

    md("""**Wait until everything is done** (optional; blocks until all runs finish)."""),
    code("""runner.wait()"""),

    md("""## 5. Compare the models

One table per environment (same test episodes and seeds for every model):
- **1-step vs copy**: one-step error as a fraction of "copy the last frame" (lower is better)
- **roll@10 / roll@40**: rollout error after 10 / 40 steps; for Crafter also **player** and **inv** regions
- **detail@40**: how much detail the model's world keeps vs the real game (1.0 = same; higher is better)

Differences smaller than the ± are not trustworthy."""),
    code("""from collections import defaultdict
from colab.runner import run_name
by_env = defaultdict(list)
for n in COMPARE:
    by_env[ALL_EXPERIMENTS[n]['env']].append(f'{RUNS}/{run_name(n, STEPS)}.pt')
for env, ckpts in by_env.items():
    print(f'\\n=========== {env} ===========')
    !python diagnostics/compare_wm.py {' '.join(ckpts)} --seeds {SEEDS}"""),

    md("""**Look at one model.** One-step predictions, the action test, and a 40-step rollout (left truth, right model)."""),
    code("""LOOK_AT = run_name('crafter_dit_last', STEPS)   # file name of the run to look at
from IPython.display import Image, display
!python evaluate.py --ckpt {RUNS}/{LOOK_AT}.pt --out outputs/eval_{LOOK_AT} --horizon 40 | tail -5
display(Image(f'outputs/eval_{LOOK_AT}/actions.png', width=900))
display(Image(f'outputs/eval_{LOOK_AT}/one_step.png', width=900))
display(Image(f'outputs/eval_{LOOK_AT}/rollout.gif'))"""),

    md("""## 6. Play it on your Mac

1. Download `tiny-diamond-runs/<name>.pt` from Google Drive into `tiny-diamond/outputs/` on your Mac
   (latent models carry their autoencoder inside, so one file is enough).
2. In a terminal:
```bash
cd tiny-diamond && git pull
pip install torch pygame numpy pillow crafter vizdoom "gymnasium[atari]"
python play.py --ckpt outputs/crafter_dit_df.pt
```
Keys depend on the game (printed at the top of `play.py`): Crafter WASD/space/tab/1-6, Atari arrows + space,
Doom W/A/D (+Q/E strafe) or A/D + space. Uses the Mac's GPU automatically; `--denoise_steps 1` if slow."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = Path(__file__).with_name("train_crafter.ipynb")
nbf.write(nb, out)
print("wrote", out)
