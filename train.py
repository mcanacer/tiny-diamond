"""Training loop: sample a batch, compute the diffusion loss, step the optimizer.

    python train.py --env toy                                   # toy game, ~6k steps
    python train.py --env crafter --out outputs/crafter.pt      # Crafter, 40k steps (use a GPU)
    python train.py --overfit --steps 600                       # checkpoint 2: one batch, loss must fall sharply

Built for interrupted sessions (Colab, laptops):
  * saves the full training state every --save_every steps (atomically, so a crash can't corrupt it)
  * re-running the SAME command resumes from that file automatically
  * --max_minutes stops cleanly (with a save) before a session limit hits

Learning rate: linear warmup, then cosine decay to 10% (our toy experiments showed the
low-learning-rate phase at the end matters a lot).

DIAMOND equivalent: src/trainer.py -> train_component("denoiser", ...)
"""
import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from wm.config import EXPERIMENTS, build_model, config_from_dict, make_config
from wm.data import Batch, WindowDataset, collate, load_episodes
from wm.world_model import load_autoencoder

p = argparse.ArgumentParser()
p.add_argument("--exp", default=None, help="named experiment from wm/config.py EXPERIMENTS (sets env, model, out)")
p.add_argument("--list", action="store_true", help="print the named experiments and exit")
p.add_argument("--runs_dir", default="outputs", help="where --exp saves its checkpoint (<runs_dir>/<exp>.pt)")
p.add_argument("--env", default=None)
p.add_argument("--data", default=None, help="default: data/<env>/train")
p.add_argument("--out", default=None, help="default: outputs/<env>.pt")
p.add_argument("--steps", type=int, default=None)
p.add_argument("--batch_size", type=int, default=None)
p.add_argument("--lr", type=float, default=None)
p.add_argument("--K", type=int, default=None, help="context frames (memory); must divide 256, e.g. 4 or 8")
p.add_argument("--model", default=None, choices=["unet", "dit", "ae"], help="what to train (default unet)")
p.add_argument("--latent", default=None, help="autoencoder (experiment name or .pt path) to work in latent space")
p.add_argument("--channels", type=int, nargs="+", default=None, help="U-Net widths per level, e.g. 64 128 128 256")
p.add_argument("--ctx_noise_max", type=float, default=None, help="context noise augmentation (GameNGen); try 0.7")
p.add_argument("--rollout_steps", type=int, default=None, help="self-rollout context (needs --init)")
p.add_argument("--init", default="", help="start from this checkpoint's weights (fine-tuning)")
p.add_argument("--overfit", action="store_true", help="train on a single fixed batch")
p.add_argument("--save_every", type=int, default=1000)
p.add_argument("--log_every", type=int, default=100)
p.add_argument("--max_minutes", type=float, default=0, help="stop (and save) after this long; re-run to resume")
p.add_argument("--no_resume", action="store_true", help="ignore an existing checkpoint at --out")
p.add_argument("--workers", type=int, default=None, help="DataLoader workers (default: 2 on CUDA, 0 otherwise)")
p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
p.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
args = p.parse_args()

if args.list:
    for name, e in EXPERIMENTS.items():
        print(f"{name:20s} {e}")
    raise SystemExit
exp = {}
if args.exp:
    if args.exp not in EXPERIMENTS:
        raise SystemExit(f"unknown --exp {args.exp!r}; choose from {list(EXPERIMENTS)}")
    exp = dict(EXPERIMENTS[args.exp])
args.env = args.env or exp.pop("env", None) or "toy"
exp.pop("env", None)
if args.exp and not args.out:
    args.out = f"{args.runs_dir}/{args.exp}.pt"

out = Path(args.out or f"outputs/{args.env}.pt")
data_dir = args.data or f"data/{args.env}/train"
device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
if args.threads:
    torch.set_num_threads(args.threads)

# ---- config: resume > fresh -----------------------------------------------------------------
state = None
if out.exists() and not args.no_resume and not args.overfit:
    state = torch.load(out, map_location="cpu", weights_only=False)
    if "opt" not in state:
        raise SystemExit(f"{out} is a finished/legacy checkpoint without optimizer state; "
                         f"use --init {out} with a new --out to fine-tune from it.")
    cfg = config_from_dict(state["cfg"])
    if args.steps:
        cfg.steps = args.steps  # allow extending a run
    print(f"resuming {out} from step {state['step']}/{cfg.steps}")
else:
    flags = dict(steps=args.steps, batch_size=args.batch_size, lr=args.lr, channels=args.channels, K=args.K,
                 ctx_noise_max=args.ctx_noise_max, rollout_steps=args.rollout_steps, model=args.model,
                 latent=args.latent)
    exp.update({k: v for k, v in flags.items() if v is not None})  # explicit flags override the experiment
    cfg = make_config(args.env, **exp)

# ---- latent space: load the (finished) autoencoder; it is frozen and saved inside our checkpoint ----
ae, ae_payload = None, None
if cfg.model != "ae" and cfg.latent:
    if state is not None and "ae" in state:
        ae_payload = state["ae"]
    else:
        ae_path = Path(cfg.latent)
        if not ae_path.exists():
            ae_path = Path(args.runs_dir) / f"{cfg.latent}.pt"
        if not ae_path.exists():
            raise SystemExit(f"autoencoder {cfg.latent!r} not found (looked for {ae_path}). "
                             f"Train it first: python train.py --exp {cfg.latent} --runs_dir {args.runs_dir}")
        ae_ck = torch.load(ae_path, map_location="cpu", weights_only=False)
        ae_cfg = config_from_dict(ae_ck["cfg"])
        if ae_ck.get("step", 0) < ae_cfg.steps:
            raise SystemExit(f"autoencoder {ae_path} has not finished training ({ae_ck.get('step')}/{ae_cfg.steps} steps)")
        ae_payload = {"model": ae_ck["model"], "cfg": ae_ck["cfg"]}
    ae = load_autoencoder(ae_payload, device)
    cfg.latent_channels = ae.cfg.latent_channels
    cfg.latent_size = cfg.img_size // ae.downsample

torch.manual_seed(0)
episodes = load_episodes(data_dir)
ds = WindowDataset(episodes, cfg.K + cfg.rollout_steps)  # K context + R self-generated + 1 target
workers = args.workers if args.workers is not None else (2 if device == "cuda" else 0)
loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                                     drop_last=True, num_workers=workers, pin_memory=device == "cuda",
                                     persistent_workers=workers > 0)

model = build_model(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)
warmup = min(1000, cfg.steps // 10)


def lr_lambda(step):
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, cfg.steps - warmup))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))  # cosine from 1.0 down to 0.1


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
use_amp = device == "cuda"
# Mixed precision: bf16 where the GPU supports it (A100, L4, H100): same range as fp32, no loss scaling.
# Older GPUs (T4) use fp16, which needs a GradScaler to avoid underflow.
amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True  # free speed on A100-class GPUs
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
step = 0

if state is not None:
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    if scaler.is_enabled() and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    step = state["step"]
elif args.init:
    init = torch.load(args.init, map_location="cpu", weights_only=False)
    model.load_state_dict(init["model"])
    print("initialised from", args.init)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
gpu = torch.cuda.get_device_name() if device == "cuda" else device
print(f"exp={args.exp or '-'}  out={out}  gpu={gpu}  precision={str(amp_dtype).split('.')[-1] if use_amp else 'fp32'}")
print(f"env={cfg.env}  device={device}  params={n_params:.2f}M  windows={len(ds)}  "
      f"batch={cfg.batch_size}  lr={cfg.lr}  steps={cfg.steps}")
space = f"latent {cfg.latent_channels}x{cfg.latent_size}x{cfg.latent_size} (ae={cfg.latent})" if ae else "pixels"
print(f"model={cfg.model}  space={space}  K={cfg.K}  ctx_noise_max={cfg.ctx_noise_max}  "
      f"rollout_steps={cfg.rollout_steps}" + (f"  train_mode={cfg.train_mode}" if cfg.model == "dit" else ""))


def save(path: Path, final=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "cfg": asdict(cfg), "step": step}
    if ae_payload is not None:
        payload["ae"] = ae_payload  # latent world models carry their autoencoder
    if not final:
        payload.update(opt=opt.state_dict(), sched=sched.state_dict(),
                       scaler=scaler.state_dict() if scaler.is_enabled() else None)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save never leaves a broken checkpoint
    # small status file next to the checkpoint (the Colab runner polls this instead of loading the .pt)
    status = {"exp": args.exp, "step": step, "steps": cfg.steps, "done": step >= cfg.steps, "model": cfg.model}
    path.with_suffix(".json").write_text(json.dumps(status))


def prepare(b):
    b = b.to(device)
    return Batch(ae.encode(b.obs), b.act) if ae is not None else b  # frames -> latents on the GPU


def batches():
    if args.overfit:
        b = prepare(next(iter(loader)))
        while True:
            yield b
    while True:
        for b in loader:
            yield prepare(b)


model.train()
it, running, t0, t_last = batches(), 0.0, time.time(), time.time()
start_step = step
stopped_early = False
while step < cfg.steps:
    with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
        loss = model.loss(next(it))
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(opt)
    scaler.update()
    sched.step()
    step += 1

    running += loss.item()
    if step % args.log_every == 0:
        dt = (time.time() - t_last) / args.log_every
        eta = dt * (cfg.steps - step) / 60
        print(f"step {step:6d}  loss {running / args.log_every:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
              f"{dt * 1000:.0f} ms/step  eta {eta:.0f} min", flush=True)
        running, t_last = 0.0, time.time()
    if not args.overfit and step % args.save_every == 0:
        save(out)
    if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
        print(f"time limit reached at step {step}; saving. Re-run the same command to resume.")
        stopped_early = True
        break

if args.overfit:
    save(out.with_name(out.stem + "_overfit.pt"), final=True)
else:
    save(out)  # keeps optimizer state so the run can be resumed or extended with --steps
    if not stopped_early:
        print(f"done: {step} steps")
print("saved", out)
