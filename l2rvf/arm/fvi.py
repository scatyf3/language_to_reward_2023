"""Fitted value iteration for V_theta(x; psi) on the Franka tasks (Eq. 2).

    y_i = min_{u_0:T-1} sum_t l_psi_i(x_t, u_t) dt + V_k(x_T; psi_i),  x_0 = x_i
    V_{k+1} = argmin_theta sum_i (V_theta(x_i; psi_i) - y_i)^2

Targets are solved with batched MPPI (B states in parallel, K samples each).
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from l2rvf.arm import env as E
from l2rvf.arm import mpc as M
from l2rvf.arm import reward as R
from l2rvf.arm import tasks as T
from l2rvf.mppi import MPPI


# --------------------------------------------------------------------------
# Reward-parameter prior (what the reward API can express)
# --------------------------------------------------------------------------
def sample_psi(family: str, rng: np.random.Generator) -> np.ndarray:
  psi = R.empty_psi()
  if family == "A":
    psi[R.W_BLOCK_POS] = rng.uniform(0.5, 2.5)
    psi[R.BLOCK_TGT] = [rng.uniform(0.35, 0.65), rng.uniform(-0.3, 0.15), 0.02]
    if rng.uniform() < 0.4:
      psi[R.W_PALM_BLOCK] = rng.uniform(0.5, 1.5)
  elif family == "B":
    psi[R.W_DRAWER] = rng.uniform(0.5, 2.5)
    psi[R.DRAWER_FRAC] = 1.0 if rng.uniform() < 0.7 else rng.uniform(0.0, 1.0)
    if rng.uniform() < 0.4:
      psi[R.W_PALM_HANDLE] = rng.uniform(0.5, 1.5)
  elif family == "C":
    psi[R.W_BLOCK_DRAWER] = rng.uniform(0.5, 2.5)
    if rng.uniform() < 0.7:
      psi[R.W_DRAWER] = rng.uniform(0.5, 1.5)
      psi[R.DRAWER_FRAC] = 1.0
    if rng.uniform() < 0.3:
      psi[R.W_PALM_BLOCK] = rng.uniform(0.3, 1.0)
  else:
    raise ValueError(family)
  return psi


# --------------------------------------------------------------------------
# State distribution
# --------------------------------------------------------------------------
def collect_visited_states(env, rng, episodes_per_task=4, every=4, H=12, K=48,
                           log=print):
  """States visited by MPC with shaped rewards / long horizon (exploration)."""
  states, fams = [], []
  for task in "ABC":
    for ep in range(episodes_per_task):
      psi = sample_psi(task, rng)
      if task == "A":
        psi[R.BLOCK_TGT] = T.TARGET_A if ep % 2 == 0 else psi[R.BLOCK_TGT]
      psi_sh = T.apply_shaping(psi, task)
      h = H if ep % 2 == 0 else 3 * H
      t0 = time.time()
      r = M.run_episode(env, task, psi_sh, H=h, K=K, iters=1, seed=ep, rng=rng)
      states.append(r["states"][::every])
      fams += [task] * len(r["states"][::every])
      log(f"  visited [{task}] ep {ep} H={h} stages {r['stages']}/{r['n_stages']}"
          f" ({time.time()-t0:.0f}s)")
  return np.concatenate(states), np.array(fams)


def synthetic_states(env, n, rng, arm_pool, max_tries=None):
  """Randomised but physically consistent states."""
  m = env.model
  d = mujoco.MjData(m)
  out = []
  tries = 0
  while len(out) < n and (max_tries is None or tries < max_tries):
    tries += 1
    mujoco.mj_resetData(m, d)
    q = arm_pool[rng.integers(len(arm_pool))] + rng.normal(0, 0.08, 7)
    q = np.clip(q, env.u_low, env.u_high)
    d.qpos[:7] = q
    drawer = rng.uniform(0, R.DRAWER_MAX) if rng.uniform() < 0.7 else rng.choice([0.0, R.DRAWER_MAX])
    d.qpos[env.drawer_qadr] = drawer
    mode = rng.choice(["floor", "top", "drawer", "edge"], p=[0.4, 0.25, 0.2, 0.15])
    if mode == "floor":
      b = [rng.uniform(0.3, 0.7), rng.uniform(-0.42, 0.2), 0.02]
    elif mode == "top":
      b = [rng.uniform(0.43, 0.67), rng.uniform(0.32, 0.58), 0.20]
    elif mode == "edge":  # floor in front of the cabinet (fell without drawer)
      b = [rng.uniform(0.42, 0.68), rng.uniform(0.18, 0.28), 0.02]
    else:
      yc = 0.46 - drawer
      b = [rng.uniform(0.45, 0.65), rng.uniform(yc - 0.1, yc + 0.1), 0.05]
    a = env.block_qadr
    d.qpos[a : a + 3] = b
    d.qpos[a + 3 : a + 7] = [1, 0, 0, 0]
    d.ctrl[:7] = q
    d.ctrl[7] = E.GRIPPER_CLOSED
    mujoco.mj_forward(m, d)
    if d.ncon and np.min([d.contact[i].dist for i in range(d.ncon)]) < -0.01:
      continue
    for _ in range(30):
      mujoco.mj_step(m, d)
    if not np.all(np.isfinite(d.qpos)) or d.qpos[a + 2] < 0.0:
      continue
    d.time = 0.0
    out.append(env.get_state(d))
  return np.array(out)


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------
def compute_targets(env, states, psis, value_fn, T_knots, K, iters, chunk,
                    seed=0, sigma=0.15):
  ys = np.empty(len(states))
  for i in range(0, len(states), chunk):
    st, ps = states[i : i + chunk], psis[i : i + chunk]
    cfg = M.arm_mppi_cfg(T_knots, K, iters, env, sigma=sigma, stride=4)
    m = MPPI(cfg, E.NU_ARM, batch=len(st), seed=seed + i)
    m.U = np.repeat(env.arm_q(st)[:, None], T_knots, axis=1)  # hold position
    out = m.optimize(lambda U: M.trajectory_cost(env, st, ps, U, value_fn))
    ys[i : i + chunk] = out["best_cost"]
  return ys


def features_of_states(env, states, psis):
  sens = env.sensors_from_states(states)
  return E.value_features(states, sens, psis)
