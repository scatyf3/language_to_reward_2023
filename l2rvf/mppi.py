"""Batched MPPI (Williams et al. 2016) in numpy.

The solver is agnostic to the dynamics: the caller passes an ``evaluate``
function that maps a batch of control sequences to total trajectory costs.

Shapes
------
B : number of independent problems solved in parallel (1 for closed-loop MPC,
    many for value-iteration targets).
K : number of sampled control sequences per problem.
H : horizon (number of control knots).
nu: control dimension.

``evaluate(U)`` receives ``U`` with shape (B, K, H, nu) and must return costs of
shape (B, K) (stage costs + terminal cost, whatever the caller wants).
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np


@dataclasses.dataclass
class MPPIConfig:
  horizon: int = 10
  num_samples: int = 256
  noise_sigma: float | np.ndarray = 0.5
  temperature: float = 0.1  # relative to the cost spread of each batch
  n_iters: int = 1
  u_low: float | np.ndarray = -1.0
  u_high: float | np.ndarray = 1.0
  # Fraction of samples drawn around zero-mean noise instead of the nominal
  # (helps escape a bad warm start).
  explore_frac: float = 0.0
  # Temporal smoothing of the noise: number of knots between independent
  # samples (noise is linearly interpolated between them). 1 = white noise.
  noise_knot_stride: int = 1


class MPPI:
  """Batched MPPI with warm starting."""

  def __init__(self, cfg: MPPIConfig, nu: int, batch: int = 1, seed: int = 0):
    self.cfg = cfg
    self.nu = nu
    self.batch = batch
    self.rng = np.random.default_rng(seed)
    self.u_low = np.broadcast_to(np.asarray(cfg.u_low, np.float64), (nu,))
    self.u_high = np.broadcast_to(np.asarray(cfg.u_high, np.float64), (nu,))
    self.sigma = np.broadcast_to(np.asarray(cfg.noise_sigma, np.float64), (nu,))
    self.reset()

  def reset(self, u_nominal: np.ndarray | None = None):
    H = self.cfg.horizon
    if u_nominal is None:
      self.U = np.zeros((self.batch, H, self.nu))
    else:
      self.U = np.broadcast_to(u_nominal, (self.batch, H, self.nu)).copy()

  def shift(self, steps: int = 1):
    """Warm start for the next control step (receding horizon)."""
    if steps <= 0:
      return
    self.U = np.concatenate(
        [self.U[:, steps:], np.repeat(self.U[:, -1:], steps, axis=1)], axis=1
    )

  def _noise(self, B, K, H):
    stride = max(1, self.cfg.noise_knot_stride)
    if stride == 1:
      eps = self.rng.standard_normal((B, K, H, self.nu))
    else:
      n_knots = (H - 1) // stride + 2
      knots = self.rng.standard_normal((B, K, n_knots, self.nu))
      t = np.arange(H) / stride
      i0 = np.floor(t).astype(int)
      a = (t - i0)[None, None, :, None]
      eps = (1 - a) * knots[:, :, i0] + a * knots[:, :, i0 + 1]
      eps *= np.sqrt(stride)  # keep a comparable per-knot magnitude
    return eps * self.sigma

  def optimize(self, evaluate: Callable[[np.ndarray], np.ndarray]):
    """Runs ``n_iters`` MPPI updates. Returns dict with best costs/sequences."""
    B, K, H = self.batch, self.cfg.num_samples, self.cfg.horizon
    best_cost = np.full(B, np.inf)
    best_U = self.U.copy()
    for _ in range(self.cfg.n_iters):
      eps = self._noise(B, K, H)
      U = self.U[:, None] + eps
      n_explore = int(self.cfg.explore_frac * K)
      if n_explore > 0:
        U[:, 1 : 1 + n_explore] = eps[:, 1 : 1 + n_explore]
      U[:, 0] = self.U  # always evaluate the current nominal sequence
      U = np.clip(U, self.u_low, self.u_high)
      costs = np.asarray(evaluate(U), np.float64)  # (B, K)
      costs = np.where(np.isfinite(costs), costs, 1e10)

      # Track the best sequence found so far (used for value targets).
      idx = np.argmin(costs, axis=1)
      c_min = costs[np.arange(B), idx]
      improved = c_min < best_cost
      best_cost = np.where(improved, c_min, best_cost)
      best_U[improved] = U[np.arange(B), idx][improved]

      # Path-integral weights with a per-problem normalised temperature.
      spread = costs.max(axis=1, keepdims=True) - c_min[:, None] + 1e-9
      w = np.exp(-(costs - c_min[:, None]) / (self.cfg.temperature * spread))
      w /= w.sum(axis=1, keepdims=True)
      self.U = np.einsum("bk,bkhu->bhu", w, U)
    return {"best_cost": best_cost, "best_U": best_U}

  def action(self) -> np.ndarray:
    return self.U[:, 0].copy()
