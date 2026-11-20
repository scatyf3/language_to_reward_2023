"""Step 1 of the proposal: the key-and-door toy problem.

A 2D point (single integrator, |u|_inf <= 1, dt = 0.1, speed 1 m/s) lives in
[-1, 1]^2. A wall at x = 0 separates the left half (start) from the right
half (goal). The wall has a door at |y| < DOOR_HALF that only opens once the
point has touched the switch at SWITCH (the "key").

Reward parameters psi = (g_x, g_y): the goal. The stage cost only says *what*
to achieve -- the distance to the goal -- and says nothing about the switch:

    l_psi(x, u) = dt * (||p - g|| + c_u ||u||^2)

State x = (p_x, p_y, key) with key in {0, 1}. Everything is vectorised over
arbitrary leading dimensions.
"""

from __future__ import annotations

import numpy as np

DT = 0.1
DOOR_HALF = 0.2
SWITCH = np.array([-0.6, 0.8])
SWITCH_R = 0.12
GOAL_TOL = 0.1
C_U = 0.01
NX, NU, NPSI = 3, 2, 2


def step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
  """x: (..., 3), u: (..., 2) -> next state (..., 3)."""
  p = x[..., :2]
  key = x[..., 2]
  u = np.clip(u, -1.0, 1.0)
  pn = np.clip(p + DT * u, -1.0, 1.0)
  crossed = np.sign(p[..., 0]) != np.sign(pn[..., 0])
  # y-coordinate where the segment crosses x = 0.
  dx = pn[..., 0] - p[..., 0]
  safe_dx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)
  a = np.clip(-p[..., 0] / safe_dx, 0.0, 1.0)
  y_cross = p[..., 1] + a * (pn[..., 1] - p[..., 1])
  through_door = (key > 0.5) & (np.abs(y_cross) < DOOR_HALF)
  blocked = crossed & ~through_door
  side = np.where(p[..., 0] < 0, -1e-3, 1e-3)
  pn_x = np.where(blocked, side, pn[..., 0])
  pn = np.stack([pn_x, pn[..., 1]], axis=-1)
  touched = np.linalg.norm(pn - SWITCH, axis=-1) < SWITCH_R
  key_n = np.maximum(key, touched.astype(x.dtype))
  return np.concatenate([pn, key_n[..., None]], axis=-1)


def stage_cost(x: np.ndarray, u: np.ndarray, psi: np.ndarray) -> np.ndarray:
  """Language-level cost: distance to goal (+ tiny control effort)."""
  d = np.linalg.norm(x[..., :2] - psi[..., :2], axis=-1)
  return DT * (d + C_U * np.sum(u * u, axis=-1))


def shaping_cost(x: np.ndarray) -> np.ndarray:
  """Hand-written shaping baseline: go to the switch while the key is off."""
  d_sw = np.linalg.norm(x[..., :2] - SWITCH, axis=-1)
  return DT * (1.0 - x[..., 2]) * (d_sw + 1.0)


def value_features(x: np.ndarray, psi: np.ndarray) -> np.ndarray:
  """Input of V_theta: [p_x, p_y, key, g_x, g_y, p - g]."""
  return np.concatenate([x, psi, x[..., :2] - psi[..., :2]], axis=-1)


VALUE_IN_DIM = 7


def rollout_cost(
    x0: np.ndarray,
    psi: np.ndarray,
    U: np.ndarray,
    value_fn=None,
    shaping_w: float = 0.0,
) -> np.ndarray:
  """Total cost of control sequences.

  x0: (B, 3), psi: (B, 2), U: (B, K, H, 2) -> (B, K).
  value_fn: optional callable on value features -> terminal cost.
  """
  B, K, H, _ = U.shape
  x = np.broadcast_to(x0[:, None], (B, K, NX)).copy()
  ps = np.broadcast_to(psi[:, None], (B, K, NPSI))
  total = np.zeros((B, K))
  for t in range(H):
    x = step(x, U[:, :, t])
    total += stage_cost(x, U[:, :, t], ps)
    if shaping_w:
      total += shaping_w * shaping_cost(x)
  if value_fn is not None:
    total += value_fn(value_features(x, ps))
  return total


def sample_states(n: int, rng: np.random.Generator) -> np.ndarray:
  p = rng.uniform(-1.0, 1.0, size=(n, 2))
  p[:, 0] = np.where(np.abs(p[:, 0]) < 1e-2, -0.05, p[:, 0])
  key = (rng.uniform(size=n) < 0.5).astype(np.float64)
  # States on the right side are only reachable with the key on.
  key = np.where(p[:, 0] > 0, 1.0, key)
  near = np.linalg.norm(p - SWITCH, axis=-1) < SWITCH_R
  key = np.where(near, 1.0, key)
  return np.concatenate([p, key[:, None]], axis=-1)


def sample_psi(n: int, rng: np.random.Generator) -> np.ndarray:
  return np.stack(
      [rng.uniform(0.3, 0.9, size=n), rng.uniform(-0.8, 0.8, size=n)], axis=-1
  )


def sample_starts(n: int, rng: np.random.Generator) -> np.ndarray:
  p = np.stack(
      [rng.uniform(-0.85, -0.5, size=n), rng.uniform(-0.6, 0.4, size=n)], -1
  )
  return np.concatenate([p, np.zeros((n, 1))], axis=-1)


def success(x: np.ndarray, psi: np.ndarray) -> np.ndarray:
  return np.linalg.norm(x[..., :2] - psi[..., :2], axis=-1) < GOAL_TOL
