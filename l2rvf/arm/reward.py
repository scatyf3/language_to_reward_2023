"""Reward parameters psi for the Franka tasks and the stage cost l_psi.

psi is a fixed-length vector of (weights, targets) that the L2R-style reward
API (``l2rvf/llm/reward_api.py``) fills from LLM-written code. All residuals
are in metres (or metre-equivalents) so that weights are comparable.

Index layout (NPSI = 13):
  0  w_palm_block     weight of ||palm - block||
  1  w_palm_handle    weight of ||palm - drawer handle||
  2  w_block_drawer   weight of ||block - drawer inside||
  3  w_block_pos      weight of ||block - (bx, by, bz)||
  4-6  bx, by, bz     block target position
  7  w_drawer         weight of |drawer_open_fraction - f| * DRAWER_SCALE
  8  f                drawer target open fraction in [0, 1]
  9  w_palm_pos       weight of ||palm - (px, py, pz)||
  10-12 px, py, pz    palm target position
"""

from __future__ import annotations

import numpy as np

NPSI = 13
W_PALM_BLOCK, W_PALM_HANDLE, W_BLOCK_DRAWER, W_BLOCK_POS = 0, 1, 2, 3
BLOCK_TGT = slice(4, 7)
W_DRAWER, DRAWER_FRAC = 7, 8
W_PALM_POS = 9
PALM_TGT = slice(10, 13)

DRAWER_MAX = 0.18  # metres of travel (joint range)
DRAWER_SCALE = 0.3  # metres-equivalent of a fully wrong drawer fraction

# Sensor layout (see l2r_scene.xml <sensor>).
S_PALM = slice(0, 3)
S_PALM_Z = slice(3, 6)
S_BLOCK = slice(6, 9)
S_HANDLE = slice(9, 12)
S_INSIDE = slice(12, 15)
S_DRAWER_Q = 15
S_FINGER = 16
S_BLOCK_VEL = slice(17, 20)
S_PALM_X = slice(20, 23)
S_BLOCK_X = slice(23, 26)  # used by the L2R reproduction (l2rvf/repro)
S_BLOCK_Z = slice(26, 29)
S_TARGET = slice(29, 32)

# Fixed (task-independent) regularisation, as in MJPC's L2R tasks. These are
# not exposed to the language model.
W_ORIENT = 0.3  # keep the gripper pointing down
W_SMOOTH = 0.05  # penalise joint-target changes between knots


def _dist(a, b):
  return np.sqrt(np.sum((a - b) ** 2, axis=-1) + 1e-6)


def residual_terms(s: np.ndarray, psi: np.ndarray) -> np.ndarray:
  """Per-term residuals (..., 6) given sensors s (..., nsens), psi (..., NPSI)."""
  palm, block = s[..., S_PALM], s[..., S_BLOCK]
  drawer_frac = s[..., S_DRAWER_Q] / DRAWER_MAX
  return np.stack(
      [
          _dist(palm, block),
          _dist(palm, s[..., S_HANDLE]),
          _dist(block, s[..., S_INSIDE]),
          _dist(block, psi[..., BLOCK_TGT]),
          np.abs(drawer_frac - psi[..., DRAWER_FRAC]) * DRAWER_SCALE,
          _dist(palm, psi[..., PALM_TGT]),
      ],
      axis=-1,
  )


def weights(psi: np.ndarray) -> np.ndarray:
  return np.stack(
      [
          psi[..., W_PALM_BLOCK],
          psi[..., W_PALM_HANDLE],
          psi[..., W_BLOCK_DRAWER],
          psi[..., W_BLOCK_POS],
          psi[..., W_DRAWER],
          psi[..., W_PALM_POS],
      ],
      axis=-1,
  )


def stage_cost(s: np.ndarray, psi: np.ndarray) -> np.ndarray:
  """l_psi per knot (without dt): language terms + fixed regularisation."""
  lang = np.sum(weights(psi) * residual_terms(s, psi), axis=-1)
  z = s[..., S_PALM_Z]
  orient = np.sum((z - np.array([0.0, 0.0, -1.0])) ** 2, axis=-1)
  return lang + W_ORIENT * orient


def empty_psi() -> np.ndarray:
  psi = np.zeros(NPSI)
  psi[BLOCK_TGT] = [0.5, 0.0, 0.02]
  psi[PALM_TGT] = [0.5, 0.0, 0.3]
  return psi


def describe(psi: np.ndarray) -> str:
  parts = []
  names = ["palm-block", "palm-handle", "block-drawer"]
  for i, n in enumerate(names):
    if psi[i] > 0:
      parts.append(f"{psi[i]:.2g}*|{n}|")
  if psi[W_BLOCK_POS] > 0:
    parts.append(f"{psi[W_BLOCK_POS]:.2g}*|block-{np.round(psi[BLOCK_TGT], 3).tolist()}|")
  if psi[W_DRAWER] > 0:
    parts.append(f"{psi[W_DRAWER]:.2g}*|drawer-{psi[DRAWER_FRAC]:.2f}|")
  if psi[W_PALM_POS] > 0:
    parts.append(f"{psi[W_PALM_POS]:.2g}*|palm-{np.round(psi[PALM_TGT], 3).tolist()}|")
  return " + ".join(parts) if parts else "(empty)"
