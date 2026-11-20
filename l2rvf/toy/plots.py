"""Figures for the key-and-door toy problem."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from l2rvf.toy import keydoor as kd  # noqa: E402

COLORS = ["#d62728", "#ff7f0e", "#1f77b4", "#7f7f7f"]


def draw_world(ax):
  ax.plot([0, 0], [-1, -kd.DOOR_HALF], "k-", lw=3)
  ax.plot([0, 0], [kd.DOOR_HALF, 1], "k-", lw=3)
  ax.plot([0, 0], [-kd.DOOR_HALF, kd.DOOR_HALF], color="0.6", lw=3, ls=":")
  ax.add_patch(plt.Circle(kd.SWITCH, kd.SWITCH_R, color="gold", alpha=0.8))
  ax.text(kd.SWITCH[0], kd.SWITCH[1] - 0.2, "switch", ha="center", fontsize=8)
  ax.set_xlim(-1, 1)
  ax.set_ylim(-1, 1)
  ax.set_aspect("equal")
  ax.set_xticks([])
  ax.set_yticks([])


def make_all(out, net, x0, psi, trajs, results, sweep, history):
  # 1. Trajectories of each controller for a few episodes.
  names = list(trajs)
  fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4))
  for ax, name, c in zip(axes, names, COLORS):
    draw_world(ax)
    tr = trajs[name]
    for i in range(min(12, len(tr))):
      ax.plot(tr[i, :, 0], tr[i, :, 1], color=c, alpha=0.7, lw=1)
      ax.plot(*psi[i], "g*", ms=7)
      ax.plot(*x0[i, :2], "ko", ms=3)
    r = results[name]
    ax.set_title(f"{name}\nsuccess {r['success_rate']:.0%}", fontsize=9)
  fig.tight_layout()
  fig.savefig(out / "trajectories.png", dpi=150)
  plt.close(fig)

  # 2. Learned value, key off / key on, for one goal.
  g = np.array([0.7, 0.0])
  xs = np.linspace(-1, 1, 121)
  XX, YY = np.meshgrid(xs, xs)
  fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
  for ax, key in zip(axes, [0.0, 1.0]):
    X = np.stack([XX, YY, np.full_like(XX, key)], -1)
    P = np.broadcast_to(g, X.shape[:-1] + (2,))
    V = net.predict(kd.value_features(X, P))
    im = ax.imshow(V, origin="lower", extent=[-1, 1, -1, 1], cmap="viridis")
    draw_world(ax)
    ax.plot(*g, "r*", ms=10)
    ax.set_title(f"$V_\\theta(x;\\psi)$, key={'on' if key else 'off'}")
    fig.colorbar(im, ax=ax, fraction=0.046)
  fig.tight_layout()
  fig.savefig(out / "value.png", dpi=150)
  plt.close(fig)

  # 3. Horizon sweep.
  fig, ax = plt.subplots(figsize=(4.2, 3.2))
  ax.plot(sweep["H"], sweep["no_value"], "o-", color=COLORS[0], label="reward only")
  ax.plot(sweep["H"], sweep["value"], "o-", color=COLORS[2], label="+ learned $V_\\theta$")
  ax.set_xlabel("MPPI horizon H (steps of 0.1 s)")
  ax.set_ylabel("success rate")
  ax.set_ylim(-0.05, 1.05)
  ax.legend()
  ax.grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(out / "horizon_sweep.png", dpi=150)
  plt.close(fig)

  # 4. FVI learning curve.
  fig, ax = plt.subplots(figsize=(4.2, 3.2))
  ks = [h["round"] for h in history]
  ax.plot(ks, [h["y_mean"] for h in history], "o-", label="mean target $y$")
  ax2 = ax.twinx()
  ax2.plot(ks, [h["heldout_rel"] for h in history], "s--", color="C3",
           label="held-out rel. error")
  ax.set_xlabel("FVI round k")
  ax.set_ylabel("mean target")
  ax2.set_ylabel("held-out relative error", color="C3")
  fig.tight_layout()
  fig.savefig(out / "fvi.png", dpi=150)
  plt.close(fig)
