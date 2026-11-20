"""Step 1: key-and-door toy problem.

1. Fitted value iteration (Eq. 2 of the proposal) with MPPI-solved targets.
2. Closed-loop comparison of the four controllers of the proposal.
3. Horizon sweep with and without the learned terminal value.

Usage:  python scripts/toy_experiment.py [--quick]
Outputs go to results/toy/.
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

from l2rvf.mppi import MPPI, MPPIConfig  # noqa: E402
from l2rvf.toy import keydoor as kd  # noqa: E402
from l2rvf.value import ValueNet, fit_value  # noqa: E402

OUT = ROOT / "results" / "toy"


def mppi_cfg(H, K, iters):
  return MPPIConfig(
      horizon=H,
      num_samples=K,
      noise_sigma=0.8,
      temperature=0.05,
      n_iters=iters,
      u_low=-1.0,
      u_high=1.0,
      explore_frac=0.25,
      noise_knot_stride=4,
  )


# --------------------------------------------------------------------------
# Fitted value iteration
# --------------------------------------------------------------------------
def compute_targets(X, P, value_fn, T, K, iters, chunk, seed):
  ys = []
  for i in range(0, len(X), chunk):
    x0, ps = X[i : i + chunk], P[i : i + chunk]
    m = MPPI(mppi_cfg(T, K, iters), kd.NU, batch=len(x0), seed=seed + i)
    out = m.optimize(lambda U: kd.rollout_cost(x0, ps, U, value_fn))
    ys.append(out["best_cost"])
  return np.concatenate(ys)


def train_value(args, log):
  rng = np.random.default_rng(0)
  net = None
  history = []
  # Held-out evaluation set (fixed across rounds).
  Xh, Ph = kd.sample_states(1000, rng), kd.sample_psi(1000, rng)
  for k in range(args.rounds):
    t0 = time.time()
    X, P = kd.sample_states(args.n_states, rng), kd.sample_psi(args.n_states, rng)
    vfn = None if net is None else net.predict
    y = compute_targets(X, P, vfn, args.T, args.K_target, args.iters_target,
                        args.chunk, seed=1000 * k)
    t_tgt = time.time() - t0
    new = ValueNet(kd.VALUE_IN_DIM, hidden=args.hidden)
    if net is not None:  # warm start the regression
      new.load_state_dict(net.state_dict())
    metrics = fit_value(new, kd.value_features(X, P), y, epochs=args.epochs,
                        seed=k)
    # Held-out error against fresh targets computed with the *previous* V
    # (i.e. the Bellman residual of the new fit on unseen states).
    yh = compute_targets(Xh, Ph, vfn, args.T, args.K_target, args.iters_target,
                         args.chunk, seed=999)
    ph = new.predict(kd.value_features(Xh, Ph))
    metrics["heldout_rmse"] = float(np.sqrt(np.mean((ph - yh) ** 2)))
    metrics["heldout_rel"] = float(np.mean(np.abs(ph - yh)) / np.mean(np.abs(yh)))
    metrics.update(round=k, target_time_s=t_tgt, y_mean=float(y.mean()),
                   y_max=float(y.max()))
    history.append(metrics)
    log(f"[FVI round {k}] targets {t_tgt:.1f}s  mean y {y.mean():.3f}  "
        f"train rel {metrics['train_rel']:.3f}  held-out rel "
        f"{metrics['heldout_rel']:.3f}")
    net = new
  return net, history


# --------------------------------------------------------------------------
# Closed-loop evaluation
# --------------------------------------------------------------------------
def run_mpc(x0, psi, H, K, iters, value_fn=None, shaping_w=0.0, steps=150,
            seed=0, record=False):
  """Runs B closed-loop episodes in parallel. Returns final states etc."""
  B = len(x0)
  m = MPPI(mppi_cfg(H, K, iters), kd.NU, batch=B, seed=seed)
  x = x0.copy()
  traj = [x.copy()]
  reached = np.zeros(B, bool)
  t_reach = np.full(B, np.nan)
  t0 = time.perf_counter()
  for t in range(steps):
    m.optimize(lambda U: kd.rollout_cost(x, psi, U, value_fn, shaping_w))
    u = m.action()
    m.shift()
    x = kd.step(x, u)
    ok = kd.success(x, psi)
    t_reach = np.where(ok & ~reached, (t + 1) * kd.DT, t_reach)
    reached |= ok
    if record:
      traj.append(x.copy())
  wall = time.perf_counter() - t0
  return {
      "x": x,
      "reached": reached,
      "final_success": kd.success(x, psi),
      "key": x[:, 2] > 0.5,
      "t_reach": t_reach,
      "traj": np.stack(traj, 1) if record else None,
      "wall_per_step_batch": wall / steps,
  }


def time_per_step(H, K, iters, value_fn, shaping_w, n=60):
  """Wall time of one control step for a single problem (B = 1)."""
  rng = np.random.default_rng(5)
  x0, psi = kd.sample_starts(1, rng), kd.sample_psi(1, rng)
  m = MPPI(mppi_cfg(H, K, iters), kd.NU, batch=1, seed=0)
  x = x0
  ts = []
  for _ in range(n):
    t0 = time.perf_counter()
    m.optimize(lambda U: kd.rollout_cost(x, psi, U, value_fn, shaping_w))
    ts.append(time.perf_counter() - t0)
    x = kd.step(x, m.action())
    m.shift()
  return float(np.median(ts))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--quick", action="store_true")
  ap.add_argument("--rounds", type=int, default=8)
  ap.add_argument("--n_states", type=int, default=8192)
  ap.add_argument("--T", type=int, default=20)
  ap.add_argument("--K_target", type=int, default=256)
  ap.add_argument("--iters_target", type=int, default=4)
  ap.add_argument("--chunk", type=int, default=512)
  ap.add_argument("--epochs", type=int, default=150)
  ap.add_argument("--hidden", type=int, default=128)
  ap.add_argument("--H", type=int, default=8)
  ap.add_argument("--H_long", type=int, default=40)
  ap.add_argument("--K", type=int, default=128)
  ap.add_argument("--iters", type=int, default=2)
  ap.add_argument("--episodes", type=int, default=100)
  args = ap.parse_args()
  if args.quick:
    args.rounds, args.n_states, args.epochs, args.episodes = 3, 2048, 60, 30

  OUT.mkdir(parents=True, exist_ok=True)
  logf = open(OUT / "log.txt", "w", encoding="utf-8")

  def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()

  log(f"args: {vars(args)}")
  torch.set_num_threads(8)
  net, history = train_value(args, log)
  torch.save(net.state_dict(), OUT / "value.pt")

  rng = np.random.default_rng(123)
  x0 = kd.sample_starts(args.episodes, rng)
  psi = kd.sample_psi(args.episodes, rng)
  controllers = {
      "L2R (reward only)": dict(H=args.H, value_fn=None, shaping_w=0.0),
      "L2R + hand shaping": dict(H=args.H, value_fn=None, shaping_w=1.0),
      "L2R + learned V (ours)": dict(H=args.H, value_fn=net.predict, shaping_w=0.0),
      f"Long horizon H={args.H_long}": dict(H=args.H_long, value_fn=None, shaping_w=0.0),
  }
  results = {}
  trajs = {}
  for name, c in controllers.items():
    r = run_mpc(x0, psi, c["H"], args.K, args.iters, c["value_fn"],
                c["shaping_w"], seed=7, record=True)
    tps = time_per_step(c["H"], args.K, args.iters, c["value_fn"], c["shaping_w"])
    results[name] = {
        "H": c["H"],
        "success_rate": float(r["reached"].mean()),
        "final_success_rate": float(r["final_success"].mean()),
        "key_rate": float(r["key"].mean()),
        "mean_time_to_goal_s": float(np.nanmean(r["t_reach"])) if r["reached"].any() else None,
        "ms_per_step": 1000 * tps,
    }
    trajs[name] = r["traj"]
    log(f"{name:28s} H={c['H']:3d}  success {results[name]['success_rate']:.2f}"
        f"  key {results[name]['key_rate']:.2f}  {1000*tps:.2f} ms/step")

  # Horizon sweep.
  sweep = {"H": [], "no_value": [], "value": [], "ms_no_value": [], "ms_value": []}
  for H in [2, 4, 8, 16, 24, 32, 48]:
    r0 = run_mpc(x0, psi, H, args.K, args.iters, None, 0.0, seed=11)
    r1 = run_mpc(x0, psi, H, args.K, args.iters, net.predict, 0.0, seed=11)
    sweep["H"].append(H)
    sweep["no_value"].append(float(r0["reached"].mean()))
    sweep["value"].append(float(r1["reached"].mean()))
    sweep["ms_no_value"].append(1000 * time_per_step(H, args.K, args.iters, None, 0.0, n=20))
    sweep["ms_value"].append(1000 * time_per_step(H, args.K, args.iters, net.predict, 0.0, n=20))
    log(f"sweep H={H:3d}: no V {sweep['no_value'][-1]:.2f}  with V {sweep['value'][-1]:.2f}")

  json.dump({"args": vars(args), "fvi": history, "controllers": results,
             "sweep": sweep}, open(OUT / "results.json", "w"), indent=2)
  np.savez_compressed(OUT / "trajs.npz", x0=x0, psi=psi,
                      **{k.replace(" ", "_"): v for k, v in trajs.items()})

  from l2rvf.toy import plots  # matplotlib only needed here
  plots.make_all(OUT, net, x0, psi, trajs, results, sweep, history)
  log(f"wrote {OUT}")


if __name__ == "__main__":
  main()
