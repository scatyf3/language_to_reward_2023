"""Figures for the preliminary report from results/arm/*.json."""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARM = ROOT / "results" / "arm"
NAMES = {"l2r": "L2R reward only", "shaping": "+ hand shaping",
         "value": "+ learned $V_\\theta$ (ours)", "long": "long horizon"}
COLORS = {"l2r": "#d62728", "shaping": "#ff7f0e", "value": "#1f77b4", "long": "#7f7f7f"}


def comparison(tag="_final"):
  path = ARM / f"eval{tag}.json"
  if not path.exists():
    return
  summ = json.loads(path.read_text())["summary"]
  tasks = sorted({s["task"] for s in summ.values()})
  ctrls = [c for c in NAMES if any(s["ctrl"] == c for s in summ.values())]
  fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
  w = 0.8 / len(ctrls)
  for j, c in enumerate(ctrls):
    succ = [summ.get(f"{t}/{c}", {}).get("success", np.nan) for t in tasks]
    stages = [summ.get(f"{t}/{c}", {}).get("stages", np.nan) /
              summ.get(f"{t}/{c}", {}).get("n_stages", 1) for t in tasks]
    H = next(s["H"] for s in summ.values() if s["ctrl"] == c)
    x = np.arange(len(tasks)) + (j - (len(ctrls) - 1) / 2) * w
    axes[0].bar(x, succ, w, color=COLORS[c], label=f"{NAMES[c]} (H={H})")
    axes[1].bar(x, stages, w, color=COLORS[c])
  for ax, yl in zip(axes, ["success rate", "fraction of stages completed"]):
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([f"Task {t}" for t in tasks])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(yl)
    ax.grid(axis="y", alpha=0.3)
  axes[0].legend(fontsize=7, loc="upper right")
  fig.tight_layout()
  fig.savefig(ARM / "comparison.png", dpi=150)
  plt.close(fig)


def sweep(task="C"):
  path = ARM / f"sweep_{task}.json"
  if not path.exists():
    return
  rows = json.loads(path.read_text())
  fig, ax = plt.subplots(figsize=(4.2, 3.2))
  for c in ["l2r", "value"]:
    Hs = sorted({r["H"] for r in rows if r["ctrl"] == c})
    if not Hs:
      continue
    s = [np.mean([r["stages"] / r["n_stages"] for r in rows if r["ctrl"] == c and r["H"] == H]) for H in Hs]
    ax.plot(Hs, s, "o-", color=COLORS[c], label=NAMES[c])
  ax.set_xlabel("MPPI horizon H (knots of 0.04 s)")
  ax.set_ylabel(f"task {task}: fraction of stages")
  ax.set_ylim(-0.05, 1.05)
  ax.grid(alpha=0.3)
  ax.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(ARM / f"horizon_sweep_{task}.png", dpi=150)
  plt.close(fig)


def fvi_curve():
  path = ARM / "fvi_history.json"
  if not path.exists():
    return
  h = json.loads(path.read_text())["history"]
  fig, ax = plt.subplots(figsize=(4.2, 3.2))
  k = [r["round"] for r in h]
  ax.plot(k, [r["y_mean"] for r in h], "o-", label="mean target")
  ax.set_xlabel("FVI round k")
  ax.set_ylabel("mean target $y$")
  ax2 = ax.twinx()
  ax2.plot(k, [r["heldout_rel"] for r in h], "s--", color="C3")
  ax2.set_ylabel("held-out relative error", color="C3")
  fig.tight_layout()
  fig.savefig(ARM / "fvi.png", dpi=150)
  plt.close(fig)


def onpolicy_curve():
  path = ARM / "onpolicy_history.json"
  if not path.exists():
    return
  h = json.loads(path.read_text())["history"]
  k = [r["round"] for r in h]
  fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
  axes[0].plot(k, [-r["optimism_gap"] for r in h], "o-", color="C3", label="$-$mean$(V-y)$")
  axes[0].plot(k, [r["onpolicy_rel_err"] for r in h], "s--", color="C0", label="relative error")
  axes[0].set_xlabel("on-policy round")
  axes[0].set_title("error of $V$ on states the V-MPC visits", fontsize=9)
  axes[0].legend(fontsize=8)
  axes[0].grid(alpha=0.3)
  for t, c in zip("ABC", ["C2", "C1", "C4"]):
    axes[1].plot(k, [r["onpolicy_success"][t] for r in h], "o-", color=c, label=f"task {t}")
  axes[1].set_xlabel("on-policy round")
  axes[1].set_title("V-MPC success on training $\\psi$ (prior)", fontsize=9)
  axes[1].set_ylim(-0.05, 1.05)
  axes[1].legend(fontsize=8)
  axes[1].grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(ARM / "onpolicy.png", dpi=150)
  plt.close(fig)


if __name__ == "__main__":
  onpolicy_curve()
  comparison()
  sweep("C")
  fvi_curve()
  print("figures written to", ARM)
