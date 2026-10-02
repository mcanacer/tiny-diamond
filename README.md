# tiny-diamond

A from-scratch, minimal diffusion world model in the style of
[DIAMOND](https://github.com/eloialonso/diamond) (~600 lines), built to learn from.
It learns to predict the next frame of a toy game given the last K frames and actions,
then runs as a simulator by feeding its own predictions back in.

## Quick start (toy game, runs on a laptop CPU)

```bash
pip install -r requirements.txt

python record.py --env toy                                   # 0. record (frames, actions) -> data/toy/train
python record.py --env toy --out data/toy/test --episodes 20 --seed 1
python inspect_data.py --env toy                             # 1. check action alignment by eye
python train.py --env toy --overfit --steps 600              # 2. sanity check: loss must fall sharply
python train.py --env toy                                    # train (6k steps, ~1 h on CPU) -> outputs/toy.pt
python evaluate.py --ckpt outputs/toy.pt                     # 3-5. one-step, actions, rollout
python play.py --ckpt outputs/toy.pt                         # play inside the world model
```

A trained toy model is included: `python play.py --ckpt outputs/model_base_plus1500.pt`.

## Crafter (2D Minecraft, 64x64): train on free Colab, play on your Mac

[Crafter](https://github.com/danijar/crafter) is a much harder world: the view scrolls as you walk, so the
model has to invent new terrain at the screen edges; there's day and night, creatures, and an inventory bar
whose numbers must stay consistent.

1. Open the notebook straight from GitHub: [colab/train_crafter.ipynb](https://colab.research.google.com/github/mcanacer/tiny-diamond/blob/main/colab/train_crafter.ipynb).
   It clones the latest code from this repository every session, so there is nothing to upload.
2. Choose a GPU runtime (A100, or T4), set the experiments in the Config cell, and run all cells.
   It records ~170k frames once (cached on Drive), then trains 40k steps with checkpoints saved to Drive.
   If Colab disconnects, run all cells again and it resumes.
3. Download `MyDrive/tiny-diamond-runs/crafter.pt` into `outputs/` on your Mac, then:

```bash
pip install torch pygame crafter numpy pillow
python play.py --ckpt outputs/crafter.pt      # WASD walk, space hit/collect, tab sleep, 1-6 craft
```

The same commands work anywhere: `python record.py --env crafter`, `python train.py --env crafter`.

**What to expect.** A 1000-step CPU run (batch 16, ~0.1 epoch) already learns the layout: grass, the player
sprite, a crisp inventory bar, and the start of scrolling, with new terrain appearing at the edge it walks
toward. It is still blurry, ignores most actions, and only slightly beats "copy the last frame"
(`evaluate.py` prints this comparison). Judge the Colab model with the same numbers: on frames that changed,
it should clearly beat copying, and in `actions.png` each move should scroll the world a different way.

### First Colab result (40k steps, 3.4M params, 89 min on a T4)

| | model | copy last frame |
|---|---|---|
| one-step error, frames that changed | **0.0108** | 0.0254 |

Rollout error by region (`diagnostics/crafter_regions.py`, 16 test episodes):

| step | player tile | inventory | map | map sharpness model / real |
|---|---|---|---|---|
| 1 | 0.0001 | 0.0000 | 0.008 | 0.28 / 0.29 |
| 10 | 0.045 | 0.004 | 0.102 | 0.28 / 0.37 |
| 40 | 0.096 | 0.069 | 0.137 | 0.23 / 0.38 |

Movement is learned well: each direction scrolls correctly, the player turns, new terrain is invented at the
edge, and hitting grass can give a sapling. The main failure is that **long rollouts empty out**:
cows, trees and water fade into grass (the most common thing in the data), so the model's frames lose
detail while the real game's gain it. The next experiments are in the notebook: a bigger U-Net, then a bigger
U-Net with context noise.

### Named experiments

Instead of long flag lists, experiments are named in `EXPERIMENTS` in `wm/config.py`:

```bash
python train.py --list                                  # show them
python train.py --exp crafter_big --runs_dir outputs    # -> outputs/crafter_big.pt
python diagnostics/compare_crafter.py outputs/crafter.pt outputs/crafter_big.pt   # one comparison table
```

Explicit flags still override an experiment (`--exp crafter_big --lr 1e-4`). On GPUs that support it
(A100, L4, H100) training uses bf16 automatically; on a T4 it uses fp16 with loss scaling.

### Play controls (all games)

Enter = new world, Delete/Backspace = re-sync the model to the real game, `=`/`-` = more/fewer denoising
steps, `.` = pause, Esc = quit, `--autoplay` = random moves. The world model needs K real frames to start
from (its memory), so `play.py` takes them from the real game; after that, every frame on the left is generated.
`play.py` uses an NVIDIA GPU, Apple Silicon GPU (MPS) or CPU automatically.

### Training is interruption-proof

`train.py` saves its full state every `--save_every` steps and resumes automatically when you re-run the same
command. `--max_minutes` stops cleanly before a session limit. The learning rate warms up and then decays
(cosine, to 10%); the toy experiments below showed the low-learning-rate phase matters a lot.

## The five layers

```
world model = f(past K frames, past K actions) -> next frame
```

| Layer | File | Job | DIAMOND file |
|---|---|---|---|
| 1. Data | `wm/data.py` | episodes -> windows of K+1 frames + K actions | `src/data/*` |
| 2. Network | `wm/network.py`, `wm/blocks.py` | conditional U-Net: frames concatenated as channels, action + noise level via AdaGroupNorm | `models/diffusion/inner_model.py`, `models/blocks.py` |
| 3. Diffusion | `wm/diffusion.py` | EDM noise sampling, preconditioning, loss; `denoise()` | `models/diffusion/denoiser.py` |
| 4. Sampler | `wm/sampler.py` | noise -> frame in 3 Euler steps | `models/diffusion/diffusion_sampler.py` |
| 5. Rollout | `wm/rollout.py` | rolling buffer, autoregressive `step(action)` | `envs/world_model_env.py` |
| Trainer | `train.py` | the loop | `trainer.py` (`train_component`) |

The "real world" is `wm/env.py`: a blue paddle you control and a red ball that bounces
on its own. The paddle tests **action conditioning**; the ball tests that the model uses
**frame history** (it has to infer velocity from past frames).

## The data convention (read this twice)

```
obs[t]  = frame at time t              shape (T+1, H, W, 3)
act[t]  = action taken AT frame t      shape (T,)
          -> produces obs[t+1]

window:   context obs[t : t+K],  actions act[t : t+K],  target obs[t+K]
                                              ^ the last action produces the target
```

An off-by-one error here does not crash anything. The model just learns to ignore actions.
`inspect_data.py` and the action test in `evaluate.py` exist to catch it.

## Build-order checkpoints

1. **Data**: `outputs/data_windows.png`. The action above the last context frame explains the paddle move into TARGET.
2. **Overfit one batch**: loss falls sharply (1.7 -> ~0.07 in 600 steps). If not, the bug is in layers 2-3.
3. **One-step**: `outputs/one_step.png`. The prediction matches the truth given real context.
4. **Actions**: `outputs/actions.png`. Same start, each action repeated 6 times; the paddle must go 5 different ways.
5. **Rollout**: `outputs/rollout.gif` (left truth, right model) and per-step MSE. Watch for drift and blur over time.

## Fixing drift: the vanishing-ball experiment

The first model sometimes lost the ball when it touched the paddle near a wall. Once the ball was
missing from all K=4 context frames, it was gone for good. Two anti-drift options exist in
`train.py` (both off by default):

- `--ctx_noise_max 0.7`: GameNGen-style noise on the context frames during training, with the noise level given to the model.
- `--rollout_steps 2 --init <ckpt>`: self-rollout fine-tuning. Context includes frames the model generated itself;
  the target is still the real frame. Half of each batch keeps real context.

All models below were scored by `diagnostics/compare.py` on identical sessions (32 games x 600 steps,
same seeds and moves). "Right place" means the ball is visible and within 2 px of the real ball.
"Lost" means the share of sessions where the ball went missing for more than 4 frames in a row.

| Model | Training | Random play: right place / lost | Chase play: right place / lost |
|---|---|---|---|
| `model.pt` (baseline) | 4000 steps, lr 3e-4 | 88.7% / 15.6% | 82.8% / 12.5% |
| `model_A_ctxnoise` | 4000 steps + context noise | 76.0% / 0% | 78.7% / 0% |
| `model_B_...` (not shipped) | 4000 steps, self-rollout from scratch | 0.4% / 100% | 0.7% / 100% |
| `model_base_plus1500` | baseline + 1500 steps, lr 1e-4 | **100% / 0%** | **96.9%** / 3.1% |
| `model_A_plus1500` | A + 1500 steps, lr 1e-4 | **100% / 0%** | 94.4% / 0% |
| `model_C_A_then_rollout` | A + 1500 self-rollout steps, lr 1e-4 | 88.7% / 3.1% | 86.5% / 0% |

Stress test (64 chase games, new seeds): base+1500 98.5%, A+1500 98.4%, C 93.3% right place; none lost the ball.

**What this shows**

1. **The main cause was undertraining.** 1500 more steps at a lower learning rate fixed the baseline.
   Before blaming the method, train longer and decay the learning rate.
2. **Context noise works as advertised but has a cost.** At 4000 steps it stopped the ball from ever being lost
   (15.6% -> 0% of sessions), but the ball's position was less precise. Once both models were trained longer,
   there was no measurable difference on this toy game. It matters more in large, visually complex games
   with long rollouts, which is where GameNGen found it essential.
3. **Self-rollout training is easy to get wrong.** From scratch (model B), it ruined the model: early on its own
   frames are garbage, so it learned to ignore its recent context, which is exactly where the ball's velocity lives.
   Starting from a trained model (C) worked, but it was still slightly worse than the plain control here.
   Self Forcing and Matrix-Game use it at much larger scale with different losses, starting from a strong pretrained model.
4. **Always include a control trained for the same extra steps.** Without `model_A_plus1500` and
   `model_base_plus1500`, C would have looked like a big improvement over the 4000-step models.

## Differences from DIAMOND (good next exercises)

- **Multi-step training loss.** DIAMOND's `Denoiser.forward` loops over a sequence and feeds its own
  denoised predictions back as context during training (`num_autoregressive_steps: 1` in its Atari config).
  `--rollout_steps` here is a related but different version; see the experiment above.
- **Reward / done model.** DIAMOND adds `rew_end_model.py` so an RL agent can train inside the world model.
- **RL agent.** `actor_critic.py` and `agent.py` train a policy in imagination.
- **Scale.** CS:GO uses a 381M-param model at low resolution plus a separate upsampler model.

Try these one at a time: increase `K`, change `denoise_steps` in `evaluate.py` (1 vs 3 vs 10),
remove `sigma_offset_noise`, or make the ball's speed random so the future becomes uncertain.
