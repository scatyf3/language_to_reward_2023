"""Reproduce the L2R manipulation experiments on our Franka scene.

Uses the paper's prompts (l2rvf/repro/prompts.py), reward API and Table 4
weights (l2rvf/repro/reward.py), and the tasks in l2rvf/repro/tasks.py.
LLM responses are read from data/repro_l2r_responses.jsonl.

Paper protocol: 10 LLM responses per task x 50 MJPC runs each, Predictive
Sampling planner. Here: the stored responses (one per prompt) x --seeds runs,
with our MPPI at horizon H (default: 10 = real-time setting of the project,
and 40 = long horizon).

Usage:
  python scripts/repro_l2r.py                       # all single + multi-turn
  python scripts/repro_l2r.py --seeds 3 --H 10      # quick smoke test
  python scripts/repro_l2r.py --video               # also save seed-0 videos
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from l2rvf.arm import env as E  # noqa: E402
from l2rvf.llm import reward_api  # noqa: E402
from l2rvf.repro import reward as RR  # noqa: E402
from l2rvf.repro import tasks as RT  # noqa: E402

OUT = ROOT / "results" / "repro"
RESP = ROOT / "data" / "repro_l2r_responses.jsonl"


def load_specs():
  single, multi = {}, []
  for line in RESP.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    r = json.loads(line)
    spec = RR.execute(reward_api.process_code_block(r["coder_response"]))
    if r["kind"] == "single":
      single[r["key"]] = (r, spec)
    else:
      multi.append((r["turn"], r, spec))
  multi = [(r, s) for _, r, s in sorted(multi, key=lambda x: x[0])]
  return single, multi


def recorder_for(env, enabled):
  if not enabled:
    return None
  from l2rvf.arm.render import FrameRecorder
  return FrameRecorder(env, every=3)


def save_video(rec, name):
  if rec is None:
    return
  import imageio.v3 as iio
  d = OUT / "videos"
  d.mkdir(parents=True, exist_ok=True)
  rec.save_video(d / f"{name}.mp4")
  iio.imwrite(d / f"{name}_sheet.png", rec.contact_sheet(6))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--single", default="touch,move,open_drawer,block_in_drawer")
  ap.add_argument("--multi", default="strict,patient", help="user models; '' to skip")
  ap.add_argument("--H", default="10,40")
  ap.add_argument("--K", type=int, default=64)
  ap.add_argument("--seeds", type=int, default=20)
  ap.add_argument("--max_turn_s", type=float, default=10.0)
  ap.add_argument("--video", action="store_true")
  ap.add_argument("--tag", default="")
  args = ap.parse_args()
  OUT.mkdir(parents=True, exist_ok=True)

  single, multi = load_specs()
  env = E.ArmEnv()
  rows, t0 = [], time.time()
  Hs = [int(h) for h in args.H.split(",")]

  for key in [k for k in args.single.split(",") if k]:
    r, spec = single[key]
    for H in Hs:
      for sd in range(args.seeds):
        rec = recorder_for(env, args.video and sd == 0)
        out = RT.run_single(env, key, spec, H, args.K, seed=sd, recorder=rec)
        save_video(rec, f"single_{key}_H{H}")
        rows.append({"kind": "single", "task": key, "H": H, "seed": sd,
                     "instruction": r["instruction"], "reward": spec.describe(), **out})
        print(f"[single {key:16s}] H={H:2d} seed={sd:2d} success={out['success']!s:5} "
              f"ever={out['ever_success']!s:5} {out['ms_per_step']:.1f}ms ({time.time()-t0:.0f}s)",
              flush=True)

  specs = [s for _, s in multi]
  for user in [u for u in args.multi.split(",") if u]:
    for H in Hs:
      for sd in range(args.seeds):
        rec = recorder_for(env, args.video and sd == 0)
        out = RT.run_multi(env, specs, H, args.K, seed=sd, user=user,
                           max_turn_s=args.max_turn_s, recorder=rec)
        save_video(rec, f"multi_{user}_H{H}")
        rows.append({"kind": "multi", "task": f"interactive/{user}", "H": H, "seed": sd,
                     "rewards": [s.describe() for s in specs], **out})
        print(f"[multi {user:8s}] H={H:2d} seed={sd:2d} success={out['success']!s:5} "
              f"turns_done={[int(t['done']) for t in out['turns']]} "
              f"t={[round(t['seconds'],1) for t in out['turns']]} ({time.time()-t0:.0f}s)",
              flush=True)
  env.close()

  summary = {}
  for row in rows:
    k = f"{row['task']}/H{row['H']}"
    summary.setdefault(k, []).append(row)
  table = []
  for k, rs in summary.items():
    ent = {"task": rs[0]["task"], "H": rs[0]["H"], "n": len(rs),
           "success": float(np.mean([r["success"] for r in rs])),
           "ms_per_step": float(np.median([r["ms_per_step"] for r in rs]))}
    if rs[0]["kind"] == "single":
      ent["ever_success"] = float(np.mean([r["ever_success"] for r in rs]))
      ent["reward"] = rs[0]["reward"]
    else:
      ent["turn_done_rate"] = np.mean([[t["done"] for t in r["turns"]] for r in rs], 0).tolist()
      ent["mean_total_s"] = float(np.mean([r["total_s"] for r in rs]))
    table.append(ent)
  json.dump({"args": vars(args), "source": "data/repro_l2r_responses.jsonl", "rows": rows,
             "summary": table}, open(OUT / f"repro{args.tag}.json", "w"), indent=1)

  print("\n| task | H | n | success | ever / per-turn done | ms/step |")
  print("|---|---|---|---|---|---|")
  for e in table:
    extra = (f"{e['ever_success']:.0%}" if "ever_success" in e
             else " ".join(f"{x:.0%}" for x in e["turn_done_rate"]))
    print(f"| {e['task']} | {e['H']} | {e['n']} | {e['success']:.0%} | {extra} | {e['ms_per_step']:.1f} |")


if __name__ == "__main__":
  main()
