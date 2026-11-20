"""Tasks A-C of the proposal: initial conditions, stages and success."""

from __future__ import annotations

import dataclasses

import numpy as np

from l2rvf.arm import reward as R

TARGET_A = np.array([0.45, 0.05, 0.02])  # green marker for task A
BLOCK_A = np.array([0.55, -0.22, 0.02])
BLOCK_PARKED = np.array([0.30, -0.40, 0.02])  # out of the way for task B
BLOCK_C = np.array([0.55, 0.36, 0.20])  # on the cabinet top, near its front edge

OPEN_FRAC = 0.8  # drawer counts as open above this fraction


@dataclasses.dataclass
class Task:
  name: str
  description: str
  block_pos: np.ndarray
  stage_names: tuple
  episode_s: float
  # Hand-written shaping baseline: weights added on top of the LLM psi.
  shaping: dict
  target_pos: np.ndarray | None = None

  def init(self, rng: np.random.Generator | None = None):
    b = self.block_pos.copy()
    if rng is not None:
      b[:2] += rng.uniform(-0.02, 0.02, size=2)
    return dict(block_pos=b, drawer_q=0.0, target_pos=self.target_pos)


def stage_flags(task: str, s: np.ndarray) -> np.ndarray:
  """Boolean flags (..., n_stages) of the instantaneous stage conditions."""
  palm, block = s[..., R.S_PALM], s[..., R.S_BLOCK]
  frac = s[..., R.S_DRAWER_Q] / R.DRAWER_MAX
  if task == "A":
    ok = np.linalg.norm(block[..., :2] - TARGET_A[:2], axis=-1) < 0.03
    return ok[..., None]
  if task == "B":
    near = np.linalg.norm(palm - s[..., R.S_HANDLE], axis=-1) < 0.10
    return np.stack([near, frac > OPEN_FRAC], -1)
  if task == "C":
    opened = frac > OPEN_FRAC
    touched = np.linalg.norm(palm - block, axis=-1) < 0.08
    return np.stack([opened, touched, block_in_drawer(s)], -1)
  raise ValueError(task)


def block_in_drawer(s: np.ndarray) -> np.ndarray:
  rel = s[..., R.S_BLOCK] - s[..., R.S_INSIDE]
  return (np.abs(rel[..., 0]) < 0.11) & (rel[..., 1] > -0.09) & (rel[..., 1] < 0.18) & (
      s[..., R.S_BLOCK][..., 2] < 0.09
  )


def stages_completed(task: str, sens_traj: np.ndarray) -> int:
  """Number of stages completed in order along a trajectory (T, nsens)."""
  flags = stage_flags(task, sens_traj)  # (T, n)
  n, t0 = 0, 0
  for j in range(flags.shape[1]):
    hits = np.nonzero(flags[t0:, j])[0]
    if len(hits) == 0:
      break
    n += 1
    t0 += hits[0]
  return n


def success(task: str, sens_final: np.ndarray) -> bool:
  return bool(stage_flags(task, sens_final)[..., -1])


TASKS = {
    "A": Task(
        name="A",
        description="move the block to the green target",
        block_pos=BLOCK_A,
        target_pos=TARGET_A,
        stage_names=("block at target",),
        episode_s=8.0,
        shaping={R.W_PALM_BLOCK: 1.0},
    ),
    "B": Task(
        name="B",
        description="open the drawer",
        block_pos=BLOCK_PARKED,
        stage_names=("hand at handle", "drawer open"),
        episode_s=8.0,
        shaping={R.W_PALM_HANDLE: 1.0},
    ),
    "C": Task(
        name="C",
        description="put the block in the drawer",
        block_pos=BLOCK_C,
        stage_names=("drawer open", "hand at block", "block in drawer"),
        episode_s=14.0,
        shaping={R.W_PALM_HANDLE: 0.5, R.W_PALM_BLOCK: 0.5},
    ),
}


def apply_shaping(psi: np.ndarray, task: str) -> np.ndarray:
  psi = psi.copy()
  for i, w in TASKS[task].shaping.items():
    psi[i] = max(psi[i], w)
  return psi
