"""Runs several experiments side by side on one GPU, respecting dependencies.

A latent world model (e.g. crafter_dit_df) needs its autoencoder (crafter_ae) to be fully trained
first. The runner adds missing autoencoders automatically, starts everything that is ready (up to
`max_parallel` at once), and starts the rest as soon as their dependencies finish. Runs that are
already complete are skipped, and interrupted ones resume (train.py resumes from its checkpoint).

    from colab.runner import Runner
    r = Runner(["crafter_dit_df", "crafter_latent_unet"], runs_dir, max_parallel=3)
    r.start()      # returns immediately; scheduling continues in a background thread
    r.status()     # table of what is waiting / running / done, with the last log line of each
    r.wait()       # block until everything has finished
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wm.config import EXPERIMENTS  # noqa: E402


def dependencies(name):
    latent = EXPERIMENTS[name].get("latent", "")
    return [latent] if latent else []


def expand(names):
    """Add missing dependencies (autoencoders) in front of the experiments that need them."""
    out = []
    for n in names:
        if n not in EXPERIMENTS:
            raise ValueError(f"unknown experiment {n!r}")
        for d in dependencies(n):
            if d not in out:
                out.append(d)
        if n not in out:
            out.append(n)
    return out


def envs_needed(names):
    return sorted({EXPERIMENTS[n]["env"] for n in expand(names)})


class Runner:
    def __init__(self, names, runs_dir, max_parallel=3, max_minutes=170, extra_args=(), poll=20):
        self.names = expand(names)
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.max_parallel, self.max_minutes, self.extra_args, self.poll = max_parallel, max_minutes, list(extra_args), poll
        self.procs, self.failed = {}, set()
        self.thread = None

    # ---- state -----------------------------------------------------------------------------------
    def _status_file(self, n):
        f = self.runs_dir / f"{n}.json"
        return json.loads(f.read_text()) if f.exists() else None

    def is_done(self, n):
        s = self._status_file(n)
        return bool(s and s.get("done"))

    def state(self, n):
        if n in self.failed:
            return "FAILED"
        if self.is_done(n):
            return "done"
        if n in self.procs and self.procs[n].poll() is None:
            return "running"
        if n in self.procs:  # exited without finishing: time limit reached, or crashed
            return "stopped" if self.procs[n].returncode == 0 else "FAILED"
        if any(not self.is_done(d) for d in dependencies(n)):
            return "waiting for " + ",".join(dependencies(n))
        return "queued"

    # ---- scheduling --------------------------------------------------------------------------------
    def _launch(self, n):
        log = open(self.runs_dir / f"{n}.log", "a")
        cmd = [sys.executable, "train.py", "--exp", n, "--runs_dir", str(self.runs_dir), "--save_every", "1000",
               "--log_every", "200", "--max_minutes", str(self.max_minutes)] + self.extra_args
        self.procs[n] = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    def _tick(self):
        for n, p in self.procs.items():
            if p.poll() is not None and p.returncode != 0:
                self.failed.add(n)
        running = sum(p.poll() is None for p in self.procs.values())
        for n in self.names:
            if running >= self.max_parallel:
                break
            if n in self.procs or n in self.failed or self.is_done(n):
                continue
            deps = dependencies(n)
            if any(d in self.failed for d in deps):
                self.failed.add(n)  # its autoencoder failed: nothing to wait for
                continue
            if all(self.is_done(d) for d in deps):
                self._launch(n)
                running += 1
                time.sleep(15)  # stagger start-up (data loading)

    def _active(self):
        return any(self.state(n) in ("running", "queued") or self.state(n).startswith("waiting") for n in self.names)

    def _loop(self):
        while True:
            self._tick()
            if not self._active():
                return
            time.sleep(self.poll)

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        time.sleep(1)
        self.status()

    def wait(self):
        while self.thread is not None and self.thread.is_alive():
            time.sleep(30)
        self.status()

    def status(self):
        for n in self.names:
            last = ""
            log = self.runs_dir / f"{n}.log"
            if log.exists():
                lines = [l for l in log.read_text(errors="ignore").splitlines()
                         if l.startswith(("step", "done", "resuming", "time limit")) or "Error" in l or "error" in l]
                last = lines[-1].strip() if lines else ""
            print(f"{n:24s} {self.state(n):34s} {last}")
