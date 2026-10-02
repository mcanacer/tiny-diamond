"""Builds colab/train_crafter.ipynb (run: python colab/make_notebook.py)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# tiny-diamond: Crafter world-model experiments on Colab

**Runtime:** `Runtime → Change runtime type →` **A100 GPU** (T4 also works, just slower).
**Code comes from GitHub** (`mcanacer/tiny-diamond`): every session clones the latest version, so code
changes need no uploads. Open this notebook straight from GitHub to always get the newest version:
`colab.research.google.com/github/mcanacer/tiny-diamond/blob/main/colab/train_crafter.ipynb`

Everything you normally change is in the **Config** cell below. Checkpoints and logs go to
`MyDrive/tiny-diamond-runs/`, so a disconnect loses at most a few minutes: reconnect, run all cells again, and
training resumes."""),

    md("""## ⚙️ Config: the only cell you need to edit

Experiments are defined by name in `wm/config.py` (`EXPERIMENTS`). Each changes **one thing** relative to the
baseline and keeps batch 32 and 40k steps, so the results are comparable:

| name | what changes | params |
|---|---|---|
| `crafter` | baseline (your first run, already done) | 3.4M |
| `crafter_big` | wider U-Net: 64-128-128-256 channels | 14.6M |
| `crafter_big_noise` | wider U-Net + GameNGen context noise (0.7) | 14.6M |
| `crafter_big_k8` | wider U-Net + 8 frames of memory instead of 4 | 14.6M |
| `crafter_big_noise_k8` | wider U-Net + context noise + 8 frames of memory | 14.6M |

To add your own, add a line to `EXPERIMENTS` in `wm/config.py` (any `train.py` flag works, e.g. `lr=1e-4`)."""),
    code("""# Which experiments to train in this session.
# On an A100 they all run AT THE SAME TIME (each in the background); on a T4, use one at a time.
EXPERIMENTS = ['crafter_big_k8', 'crafter_big_noise_k8']

# Which trained models to compare at the end (the baseline is already on your Drive).
COMPARE = ['crafter', 'crafter_big', 'crafter_big_noise', 'crafter_big_k8', 'crafter_big_noise_k8']

# Evaluation repeats: results are shown as mean±std over this many seeds.
# A difference smaller than the ± is not trustworthy.
SEEDS = 3

# Stop each run cleanly after this many minutes (re-run the notebook to continue).
MAX_MINUTES = 170"""),

    md("## 1. GPU check + Google Drive"),
    code("""!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')"""),

    md("## 2. Get the latest code from GitHub and install Crafter"),
    code("""import os
DRIVE = '/content/drive/MyDrive'
RUNS = f'{DRIVE}/tiny-diamond-runs'          # checkpoints, logs and cached data live here (persistent)
os.makedirs(RUNS, exist_ok=True)
REPO = 'https://github.com/mcanacer/tiny-diamond.git'
if os.path.exists('/content/tiny-diamond/.git'):
    !git -C /content/tiny-diamond pull --ff-only
else:
    !git clone -q {REPO} /content/tiny-diamond
!git -C /content/tiny-diamond log -1 --format='code version: %h  %s  (%cr)'
%cd /content/tiny-diamond
!pip install -q crafter
!python train.py --list"""),

    md("""## 3. Data: record once, then reuse from Drive
~1000 episodes of random-ish play (~170k frames). ~15 min the first time; later sessions unpack the cache in ~1 min."""),
    code("""cache = f'{RUNS}/crafter_data.tar'
if os.path.exists(cache):
    !tar -xf {cache} -C /content/tiny-diamond
    print('data restored from Drive')
else:
    !python record.py --env crafter --out data/crafter/train
    !python record.py --env crafter --out data/crafter/test --episodes 20 --seed 1
    !tar -cf {cache} data/crafter
    print('data recorded and cached to Drive')
!ls data/crafter/train | wc -l"""),

    md("""## 4. Train

**Option A (A100): all experiments at once.** Each one runs as a background process with its own log file on
Drive. The cell returns immediately; use the next cell to watch progress. These models are small for an A100,
so running several side by side uses the GPU much better than running them one after another."""),
    code("""import subprocess, time
procs = {}
for name in EXPERIMENTS:
    log = open(f'{RUNS}/{name}.log', 'a')
    procs[name] = subprocess.Popen(
        ['python', 'train.py', '--exp', name, '--runs_dir', RUNS, '--save_every', '1000',
         '--log_every', '200', '--max_minutes', str(MAX_MINUTES)],
        stdout=log, stderr=subprocess.STDOUT)
    print(f'started {name}  (log: {RUNS}/{name}.log)')
    time.sleep(20)  # stagger start-up so data loading doesn't all happen at once"""),

    md("""**Watch progress.** Re-run this cell whenever you like. `ms/step` and `eta` tell you how long is left.
If Colab disconnects, the background runs stop too: reconnect, run all cells again, and they resume."""),
    code("""for name in EXPERIMENTS:
    status = 'running' if name in procs and procs[name].poll() is None else 'stopped'
    print(f'=== {name} ({status})')
    !grep -E "exp=|params=|step|done|resuming|Error|error" {RUNS}/{name}.log | tail -n 3
!nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv"""),

    md("""**Wait until it's done.** Optional: this cell blocks until every run finishes (handy before running the comparison)."""),
    code("""while any(p.poll() is None for p in procs.values()):
    time.sleep(60)
print('all runs finished')"""),

    md("""**Option B (T4 or one experiment): run in the foreground** with the log printed here. Edit the name and run
this *instead of* Option A."""),
    code("""# !python train.py --exp crafter_big --runs_dir {RUNS} --save_every 1000 --log_every 200 --max_minutes {MAX_MINUTES}"""),

    md("""## 5. Compare the models

One table for all models in `COMPARE` (same test episodes, same seeds):
- **1-step vs copy**: one-step error as a fraction of "copy the last frame" (lower is better; baseline ≈ 0.41)
- **player@10 / player@40, inv@40**: rollout error on the player tile and inventory bar (should stay low)
- **map@40**: rollout error on the map (partly unavoidable: new terrain is invented)
- **detail@40**: how much detail the model's world keeps vs the real game (1.0 = same; the baseline drops well
  below 1 because trees, cows and water fade into grass. This is the number we're trying to raise.)

You can run this mid-training too: it uses the latest saved checkpoint."""),
    code("""ckpts = ' '.join(f'{RUNS}/{n}.pt' for n in COMPARE)
!python diagnostics/compare_crafter.py {ckpts} --seeds {SEEDS}"""),

    md("""**Look at one model.** One-step predictions, the action test, and a 40-step rollout (left truth, right model)."""),
    code("""LOOK_AT = 'crafter_big'
from IPython.display import Image, display
!python evaluate.py --ckpt {RUNS}/{LOOK_AT}.pt --out outputs/eval_{LOOK_AT} --horizon 40 | tail -4
display(Image(f'outputs/eval_{LOOK_AT}/actions.png', width=900))
display(Image(f'outputs/eval_{LOOK_AT}/one_step.png', width=900))
display(Image(f'outputs/eval_{LOOK_AT}/rollout.gif'))"""),

    md("""## 6. Play it on your Mac

1. Download `tiny-diamond-runs/<name>.pt` from Google Drive into `tiny-diamond/outputs/` on your Mac.
2. In a terminal:
```bash
cd tiny-diamond
pip install torch pygame crafter numpy pillow
python play.py --ckpt outputs/crafter_big.pt
```
WASD walk, space hit/collect, tab sleep, 1-6 craft. Uses the Mac's GPU automatically; `--denoise_steps 1` if slow.
Models trained with context noise also accept `--ctx_noise 0.1` (try with and without)."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = Path(__file__).with_name("train_crafter.ipynb")
nbf.write(nb, out)
print("wrote", out)
