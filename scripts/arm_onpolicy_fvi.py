"""On-policy fitted value iteration (data aggregation) + pessimistic ensemble.

Motivation (found in the first evaluation): a value net trained only on
exploration/synthetic states is accurate there, but the short-horizon MPPI
steers into states where V is wrongly optimistic (e.g. the hand on the target
while the block has not moved). Each round we therefore

  1. run the V-MPC itself on tasks A-C with psi drawn from the reward prior
     (never the stored LLM psi), and add the visited (x, psi) to the pool;
  2. recompute MPPI targets y = min sum l dt + V_k(x_T) for the pool;
  3. refit an ensemble of value nets from scratch; MPC uses mean + kappa*std.

Usage: python scripts/arm_onpolicy_fvi.py [--rounds 6]
Output: results/arm/ensemble.pt, results/arm/onpolicy_history.json
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
from l2rvf.arm import mpc as M  # noqa: E402
from l2rvf.arm import tasks as T  # noqa: E402
from l2rvf.arm import reward as R  # noqa: E402
from l2rvf.value import ValueEnsemble, ValueNet, fit_value  # noqa: E402

OUT = ROOT / "results" / "arm"


def collect_onpolicy(env, value_fn, rng, episodes, every, H, K, log):
  states, psis, stats = [], [], []
  for task in "ABC":
    for ep in range(episodes):
      psi = fvi.sample_psi(task, rng)
      if task == "A" and ep % 2 == 0:
        psi[R.BLOCK_TGT] = T.TARGET_A
      r = M.run_episode(env, task, psi, H=H, K=K, iters=1, value_fn=value_fn,
                        seed=int(rng.integers(1 << 30)), rng=rng)
      st = r["states"][::every]
      states.append(st)
      psis.append(np.repeat(psi[None], len(st), 0))
      stats.append((task, r["success"], r["stages"], r["n_stages"]))
  succ = {t: np.mean([s[1] for s in stats if s[0] == t]) for t in "ABC"}
  log(f"  on-policy episodes: success A {succ['A']:.2f} B {succ['B']:.2f} C {succ['C']:.2f}")
  return np.concatenate(states), np.concatenate(psis), succ


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--rounds", type=int, default=6)
  ap.add_argument("--episodes", type=int, default=4, help="per task per round")
  ap.add_argument("--every", type=int, default=3)
  ap.add_argument("--n_base", type=int, default=2500)
  ap.add_argument("--max_onpolicy", type=int, default=6000)
  ap.add_argument("--members", type=int, default=5)
  ap.add_argument("--kappa", type=float, default=1.0)
  ap.add_argument("--H", type=int, default=10)
  ap.add_argument("--K_mpc", type=int, default=64)
  ap.add_argument("--T", type=int, default=25)
  ap.add_argument("--K", type=int, default=48)
  ap.add_argument("--iters", type=int, default=3)
  ap.add_argument("--chunk", type=int, default=48)
  ap.add_argument("--epochs", type=int, default=150)
  ap.add_argument("--init", default=str(OUT / "value.pt"))
  ap.add_argument("--quick", action="store_true")
  ap.add_argument("--resume", action="store_true",
                  help="continue from --init ensemble and the saved on-policy pool")
  args = ap.parse_args()
  if args.quick:
    args.rounds, args.episodes, args.n_base, args.members, args.epochs = 1, 1, 200, 2, 20

  logf = open(OUT / "onpolicy_log.txt", "a", encoding="utf-8")

  def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()

  log(f"=== {time.ctime()} args {vars(args)}")
  torch.set_num_threads(4)
  rng = np.random.default_rng(1)
  env = E.ArmEnv()
  z = np.load(OUT / "states.npz")
  base_states = np.concatenate([z["visited"], z["synth"]])
  base_fam = np.concatenate([z["vfam"], rng.choice(list("ABC"), size=len(z["synth"]))])

  blob = torch.load(args.init, weights_only=True)
  if "members" in blob:  # continue from an ensemble (and its on-policy pool)
    ens = ValueEnsemble.load(args.init, E.VALUE_IN_DIM, kappa=args.kappa)
  else:
    init = ValueNet(E.VALUE_IN_DIM, hidden=256)
    init.load_state_dict(blob)
    init.eval()
    ens = ValueEnsemble([init], kappa=args.kappa)

  pool = OUT / "onpolicy_pool.npz"
  on_X = np.zeros((0, env.nstate))
  on_P = np.zeros((0, R.NPSI))
  history = []
  if args.resume:
    history = json.loads((OUT / "onpolicy_history.json").read_text())["history"]
    if pool.exists():
      zp = np.load(pool)
      on_X, on_P = zp["X"], zp["P"]
  k0 = len(history)
  rng = np.random.default_rng(1 + k0)
  for k in range(k0, k0 + args.rounds):
    t0 = time.time()
    X_new, P_new, succ = collect_onpolicy(env, ens.predict, rng, args.episodes,
                                          args.every, args.H, args.K_mpc, log)
    t_col = time.time() - t0
    # Optimism gap on the fresh on-policy states, before refitting.
    y_new = fvi.compute_targets(env, X_new, P_new, ens.mean, args.T, args.K,
                                args.iters, args.chunk, seed=50_000 + k)
    v_new = ens.mean(fvi.features_of_states(env, X_new, P_new))
    gap = float(np.mean(v_new - y_new))
    rel = float(np.mean(np.abs(v_new - y_new)) / np.mean(np.abs(y_new)))
    log(f"  optimism gap on new on-policy states: mean(V - y) = {gap:+.3f} "
        f"(mean y {y_new.mean():.3f}, rel err {rel:.3f})")

    on_X = np.concatenate([on_X, X_new])[-args.max_onpolicy:]
    on_P = np.concatenate([on_P, P_new])[-args.max_onpolicy:]
    ib = rng.integers(len(base_states), size=args.n_base)
    fam = np.where(rng.uniform(size=args.n_base) < 0.2,
                   rng.choice(list("ABC"), size=args.n_base), base_fam[ib])
    X = np.concatenate([on_X, base_states[ib]])
    P = np.concatenate([on_P, np.stack([fvi.sample_psi(f, rng) for f in fam])])
    t1 = time.time()
    y = fvi.compute_targets(env, X, P, ens.mean, args.T, args.K, args.iters,
                            args.chunk, seed=1000 * k)
    t_tgt = time.time() - t1
    Z = fvi.features_of_states(env, X, P)
    nets, fits = [], []
    for j in range(args.members):
      net = ValueNet(E.VALUE_IN_DIM, hidden=256)
      torch.manual_seed(100 * k + j)
      boot = rng.integers(len(Z), size=len(Z))  # bootstrap for diversity
      fits.append(fit_value(net, Z[boot], y[boot], epochs=args.epochs, seed=j))
      nets.append(net)
    ens = ValueEnsemble(nets, kappa=args.kappa)
    np.savez_compressed(pool, X=on_X, P=on_P)
    torch.save(ens.state_dicts(), OUT / "ensemble.pt")
    torch.save(ens.state_dicts(), OUT / f"ensemble_round{k}.pt")
    rec = {"round": k, "onpolicy_success": succ, "optimism_gap": gap,
           "onpolicy_rel_err": rel, "n_train": len(X), "n_onpolicy": len(on_X),
           "y_mean": float(y.mean()), "collect_s": t_col, "target_s": t_tgt,
           "val_rel": float(np.mean([f["val_rel"] for f in fits]))}
    history.append(rec)
    log(f"[round {k}] n={len(X)} (on-policy {len(on_X)})  targets {t_tgt:.0f}s  "
        f"val rel {rec['val_rel']:.3f}")
    json.dump({"args": vars(args), "history": history},
              open(OUT / "onpolicy_history.json", "w"), indent=2)
  env.close()


if __name__ == "__main__":
  main()
