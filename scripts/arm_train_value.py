"""Train V_theta(x; psi) for the Franka tasks by fitted value iteration.

Usage: python scripts/arm_train_value.py [--quick]
Outputs: results/arm/value.pt, results/arm/fvi_history.json, results/arm/states.npz
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from l2rvf.arm import env as E  # noqa: E402
from l2rvf.arm import fvi  # noqa: E402
from l2rvf.value import ValueNet, fit_value  # noqa: E402

OUT = ROOT / "results" / "arm"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--quick", action="store_true")
  ap.add_argument("--rounds", type=int, default=8)
  ap.add_argument("--n_per_round", type=int, default=3000)
  ap.add_argument("--n_synth", type=int, default=12000)
  ap.add_argument("--episodes_per_task", type=int, default=4)
  ap.add_argument("--T", type=int, default=25, help="target horizon (knots of 0.04 s)")
  ap.add_argument("--K", type=int, default=48)
  ap.add_argument("--iters", type=int, default=3)
  ap.add_argument("--chunk", type=int, default=48)
  ap.add_argument("--epochs", type=int, default=120)
  ap.add_argument("--hidden", type=int, default=256)
  ap.add_argument("--reuse_states", action="store_true")
  args = ap.parse_args()
  if args.quick:
    args.rounds, args.n_per_round, args.n_synth, args.episodes_per_task = 2, 300, 1000, 1
    args.epochs = 40
  OUT.mkdir(parents=True, exist_ok=True)
  logf = open(OUT / "train_log.txt", "a", encoding="utf-8")

  def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()

  log(f"=== {time.ctime()} args {vars(args)}")
  torch.set_num_threads(4)
  rng = np.random.default_rng(0)
  env = E.ArmEnv()

  st_path = OUT / "states.npz"
  if args.reuse_states and st_path.exists():
    z = np.load(st_path)
    visited, vfam, synth = z["visited"], z["vfam"], z["synth"]
    log(f"reused states: visited {len(visited)} synth {len(synth)}")
  else:
    t0 = time.time()
    visited, vfam = fvi.collect_visited_states(env, rng, args.episodes_per_task, log=log)
    arm_pool = env.arm_q(visited)
    synth = fvi.synthetic_states(env, args.n_synth, rng, arm_pool)
    np.savez_compressed(st_path, visited=visited, vfam=vfam, synth=synth)
    log(f"states: visited {len(visited)} synth {len(synth)} ({time.time()-t0:.0f}s)")

  # Held-out set: fixed states/psi to measure prediction error each round.
  def draw(n, rng):
    n_vis = n // 2
    iv = rng.integers(len(visited), size=n_vis)
    isy = rng.integers(len(synth), size=n - n_vis)
    st = np.concatenate([visited[iv], synth[isy]])
    fam = np.concatenate([vfam[iv], rng.choice(list("ABC"), size=n - n_vis)])
    # Mostly pair states with their own task family, sometimes cross families.
    fam = np.where(rng.uniform(size=n) < 0.2, rng.choice(list("ABC"), size=n), fam)
    ps = np.stack([fvi.sample_psi(f, rng) for f in fam])
    return st, ps

  hold_rng = np.random.default_rng(999)
  Xh, Ph = draw(min(400, args.n_per_round // 4), hold_rng)
  Zh = fvi.features_of_states(env, Xh, Ph)

  net, history = None, []
  Z_all, Y_all = [], []
  for k in range(args.rounds):
    t0 = time.time()
    X, P = draw(args.n_per_round, rng)
    vfn = None if net is None else net.predict
    y = fvi.compute_targets(env, X, P, vfn, args.T, args.K, args.iters, args.chunk,
                            seed=1000 * k)
    t_tgt = time.time() - t0
    Z = fvi.features_of_states(env, X, P)
    new = ValueNet(E.VALUE_IN_DIM, hidden=args.hidden)
    if net is not None:
      new.load_state_dict(net.state_dict())
    metrics = fit_value(new, Z, y, epochs=args.epochs, seed=k)
    yh = fvi.compute_targets(env, Xh, Ph, vfn, args.T, args.K, args.iters, args.chunk,
                             seed=777)
    ph = new.predict(Zh)
    metrics["heldout_rmse"] = float(np.sqrt(np.mean((ph - yh) ** 2)))
    metrics["heldout_rel"] = float(np.mean(np.abs(ph - yh)) / np.mean(np.abs(yh)))
    metrics.update(round=k, target_time_s=t_tgt, y_mean=float(y.mean()),
                   y_std=float(y.std()))
    history.append(metrics)
    log(f"[FVI round {k}] targets {t_tgt:.0f}s  y {y.mean():.3f}±{y.std():.3f}  "
        f"train rel {metrics['train_rel']:.3f}  val rel {metrics['val_rel']:.3f}  "
        f"held-out rel {metrics['heldout_rel']:.3f}")
    net = new
    torch.save(net.state_dict(), OUT / "value.pt")
    torch.save(net.state_dict(), OUT / f"value_round{k}.pt")
    json.dump({"args": vars(args), "history": history},
              open(OUT / "fvi_history.json", "w"), indent=2)
  env.close()


if __name__ == "__main__":
  main()
