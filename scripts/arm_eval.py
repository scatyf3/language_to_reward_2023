"""Step 2: controller comparison on tasks A-C with the stored LLM rewards.

Controllers (same number of samples K per step):
  l2r      MPPI, horizon H, LLM reward only (L2R baseline)
  shaping  MPPI, horizon H, LLM reward + hand-written shaping term
  value    MPPI, horizon H, LLM reward + learned terminal value V_theta (ours)
  long     MPPI, horizon H_long, LLM reward only (reference)

Usage:
  python scripts/arm_eval.py                         # full comparison
  python scripts/arm_eval.py --sweep C               # horizon sweep on task C
  python scripts/arm_eval.py --video                 # also save videos
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
from l2rvf.arm import mpc as M  # noqa: E402
from l2rvf.arm import reward as R  # noqa: E402
from l2rvf.arm import tasks as T  # noqa: E402
from l2rvf.value import ValueEnsemble, ValueNet  # noqa: E402

OUT = ROOT / "results" / "arm"


def load_value(path, kappa=None):
  blob = torch.load(path, weights_only=True)
  if isinstance(blob, dict) and "members" in blob:  # pessimistic ensemble
    return ValueEnsemble.load(path, E.VALUE_IN_DIM, kappa=kappa)
  net = ValueNet(E.VALUE_IN_DIM, hidden=256)
  net.load_state_dict(blob)
  net.eval()
  return net


def load_psis(task, n):
  cache = json.loads((ROOT / "data" / "psi_cache.json").read_text(encoding="utf-8"))
  ents = [e for e in cache["entries"] if e["task"] == task][:n]
  return [(e["instruction"], np.array(e["psi"])) for e in ents]


def controller_setup(name, psi, task, args, net):
  if name == "l2r":
    return psi, args.H, None
  if name == "shaping":
    return T.apply_shaping(psi, task), args.H, None
  if name == "value":
    return psi, args.H, net.predict
  if name == "long":
    return psi, args.H_long, None
  raise ValueError(name)


def run(env, task, psi, ctrl, args, net, seed, recorder=None):
  p, H, vfn = controller_setup(ctrl, psi, task, args, net)
  r = M.run_episode(env, task, p, H=H, K=args.K, iters=1, value_fn=vfn,
                    seed=seed, rng=np.random.default_rng(seed),
                    record_frames=recorder)
  flags = T.stage_flags(task, r["sens"])[:, -1]
  t_succ = float(np.argmax(flags) * E.CTRL_DT) if flags.any() else None
  return {"success": bool(r["success"]), "stages": int(r["stages"]),
          "n_stages": r["n_stages"], "ms_per_step": r["ms_per_step"], "H": H,
          "ever_success": bool(flags.any()), "t_first_success": t_succ}


def summarize(rows):
  out = {}
  for task in sorted({r["task"] for r in rows}):
    for ctrl in dict.fromkeys(r["ctrl"] for r in rows):
      rs = [r for r in rows if r["task"] == task and r["ctrl"] == ctrl]
      if not rs:
        continue
      out[f"{task}/{ctrl}"] = {
          "task": task, "ctrl": ctrl, "H": rs[0]["H"], "n": len(rs),
          "success": float(np.mean([r["success"] for r in rs])),
          "ever_success": float(np.mean([r["ever_success"] for r in rs])),
          "stages": float(np.mean([r["stages"] for r in rs])),
          "n_stages": rs[0]["n_stages"],
          "ms_per_step": float(np.median([r["ms_per_step"] for r in rs])),
      }
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tasks", default="ABC")
  ap.add_argument("--controllers", default="l2r,shaping,value,long")
  ap.add_argument("--n_psi", type=int, default=10)
  ap.add_argument("--seeds", type=int, default=1, help="episodes per psi")
  ap.add_argument("--H", type=int, default=10)
  ap.add_argument("--H_long", type=int, default=40)
  ap.add_argument("--K", type=int, default=64)
  ap.add_argument("--value", default=str(OUT / "ensemble.pt"))
  ap.add_argument("--kappa", type=float, default=None)
  ap.add_argument("--sweep", default=None, help="task for a horizon sweep")
  ap.add_argument("--sweep_H", default="3,5,10,20,40")
  ap.add_argument("--video", action="store_true")
  ap.add_argument("--tag", default="")
  args = ap.parse_args()

  OUT.mkdir(parents=True, exist_ok=True)
  torch.set_num_threads(1)
  env = E.ArmEnv()
  net = load_value(args.value, args.kappa) if pathlib.Path(args.value).exists() else None
  ctrls = args.controllers.split(",")
  if net is None and "value" in ctrls:
    print("no value network found; skipping 'value' controller")
    ctrls.remove("value")

  if args.sweep:
    task = args.sweep
    rows = []
    for H in map(int, args.sweep_H.split(",")):
      for i, (ins, psi) in enumerate(load_psis(task, args.n_psi)):
        for ctrl in ["l2r", "value"] if net is not None else ["l2r"]:
          a = argparse.Namespace(**{**vars(args), "H": H})
          r = run(env, task, psi, ctrl, a, net, seed=i)
          rows.append({"task": task, "ctrl": ctrl, "psi_idx": i, **r})
          print(f"sweep {task} H={H:3d} {ctrl:8s} psi{i} success={r['success']} "
                f"stages={r['stages']}/{r['n_stages']} {r['ms_per_step']:.1f}ms", flush=True)
    json.dump(rows, open(OUT / f"sweep_{task}{args.tag}.json", "w"), indent=1)
    env.close()
    return

  rows = []
  t0 = time.time()
  for task in args.tasks:
    for i, (ins, psi) in enumerate(load_psis(task, args.n_psi)):
     for sd in range(args.seeds):
      for ctrl in ctrls:
        rec = None
        if args.video and i == 0 and sd == 0:
          from l2rvf.arm.render import FrameRecorder
          rec = FrameRecorder(env, every=3)
        r = run(env, task, psi, ctrl, args, net, seed=i + 1000 * sd, recorder=rec)
        rows.append({"task": task, "ctrl": ctrl, "psi_idx": i, "seed": sd, "instruction": ins,
                     "psi": R.describe(psi), **r})
        print(f"[{task}] {ctrl:8s} psi{i} '{ins[:30]}' success={r['success']} "
              f"stages={r['stages']}/{r['n_stages']} {r['ms_per_step']:.1f}ms "
              f"({time.time()-t0:.0f}s)", flush=True)
        if rec is not None:
            vid_dir = OUT / "videos"
            vid_dir.mkdir(exist_ok=True)
            rec.save_video(vid_dir / f"{task}_{ctrl}.mp4")
            import imageio.v3 as iio
            iio.imwrite(vid_dir / f"{task}_{ctrl}_sheet.png", rec.contact_sheet(6))
  summary = summarize(rows)
  json.dump({"args": vars(args), "rows": rows, "summary": summary},
            open(OUT / f"eval{args.tag}.json", "w"), indent=1)
  print("\n| task | controller | H | success | ever reached | stages | ms/step |")
  print("|---|---|---|---|---|---|---|")
  for s in summary.values():
    print(f"| {s['task']} | {s['ctrl']} | {s['H']} | {s['success']:.0%} | "
          f"{s['ever_success']:.0%} | {s['stages']:.1f}/{s['n_stages']} | {s['ms_per_step']:.1f} |")
  env.close()


if __name__ == "__main__":
  main()
