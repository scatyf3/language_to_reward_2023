"""MPPI controller for the arm: min sum_t l_psi(x_t) dt + V_theta(x_H; psi)."""

from __future__ import annotations

import time

import numpy as np

from l2rvf.arm import env as E
from l2rvf.arm import reward as R
from l2rvf.arm import tasks as T
from l2rvf.mppi import MPPI, MPPIConfig


def arm_mppi_cfg(H, K, iters, env, sigma=0.12, stride=3):
  return MPPIConfig(
      horizon=H,
      num_samples=K,
      noise_sigma=sigma,
      temperature=0.05,
      n_iters=iters,
      u_low=env.u_low,
      u_high=env.u_high,
      explore_frac=0.0,
      noise_knot_stride=stride,
  )


def trajectory_cost(env, states, psi, U, value_fn=None):
  """Costs (B, K) of knot sequences U (B, K, H, 7) from states (B, nstate).

  psi: (B, NPSI). value_fn: callable(features) -> terminal cost, or None.
  """
  sens, final = env.rollout(states, U)
  B, K, H, _ = U.shape
  ps = np.broadcast_to(psi[:, None, None], (B, K, H, R.NPSI))
  c = R.stage_cost(sens, ps).sum(-1) * E.CTRL_DT
  dU = np.diff(U, axis=2)
  c += R.W_SMOOTH * np.sum(dU * dU, axis=(-1, -2))
  if value_fn is not None:
    feats = E.value_features(final, sens[:, :, -1], ps[:, :, -1])
    c += value_fn(feats)
  return c


class ArmMPC:
  def __init__(self, env, psi, H, K, iters=1, value_fn=None, sigma=0.12,
               seed=0):
    self.env = env
    self.psi = np.asarray(psi, np.float64)
    self.value_fn = value_fn
    self.mppi = MPPI(arm_mppi_cfg(H, K, iters, env, sigma), E.NU_ARM, batch=1,
                     seed=seed)
    self.mppi.reset(env.arm_q(env.get_state()))

  def act(self, state):
    self.mppi.optimize(
        lambda U: trajectory_cost(self.env, state[None], self.psi[None], U,
                                  self.value_fn)
    )
    u = self.mppi.action()[0]
    self.mppi.shift()
    return u


def run_episode(env, task_name, psi, H, K, iters=1, value_fn=None, seed=0,
                episode_s=None, record_frames=None, rng=None, verbose=False):
  """Closed-loop episode. Returns metrics and the sensor trajectory."""
  task = T.TASKS[task_name]
  env.reset(**task.init(rng))
  ctl = ArmMPC(env, psi, H, K, iters, value_fn, seed=seed)
  n_steps = int(round((episode_s or task.episode_s) / E.CTRL_DT))
  sens_traj, states = [env.sensors()], [env.get_state()]
  times = []
  for t in range(n_steps):
    st = env.get_state()
    t0 = time.perf_counter()
    u = ctl.act(st)
    times.append(time.perf_counter() - t0)
    env.step(u)
    sens_traj.append(env.sensors())
    states.append(env.get_state())
    if record_frames is not None:
      record_frames(env)
    if verbose and t % 25 == 0:
      s = sens_traj[-1]
      print(f"  t={t*E.CTRL_DT:5.2f}s palm={np.round(s[R.S_PALM],3)} "
            f"block={np.round(s[R.S_BLOCK],3)} drawer={s[R.S_DRAWER_Q]:.3f} "
            f"flags={T.stage_flags(task_name, s).astype(int)}")
  sens_traj = np.array(sens_traj)
  return {
      "success": T.success(task_name, sens_traj[-1]),
      "stages": T.stages_completed(task_name, sens_traj),
      "n_stages": len(task.stage_names),
      "ms_per_step": 1000 * float(np.median(times)),
      "sens": sens_traj,
      "states": np.array(states),
  }
