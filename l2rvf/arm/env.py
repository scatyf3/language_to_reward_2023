"""MuJoCo Franka Panda environment with batched, multi-threaded rollouts.

Control parametrisation: MPPI optimises H knots of absolute joint-position
targets for the 7 arm actuators (zero-order hold, CTRL_DT seconds per knot).
The gripper is kept closed: none of the tasks needs grasping, and closed
fingers make a good hook for the drawer handle.
"""

from __future__ import annotations

import os
import pathlib
import shutil

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout

from l2rvf.arm import reward as R

ROOT = pathlib.Path(__file__).resolve().parents[2]
MENAGERIE_PANDA = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda"
SCENE_SRC = ROOT / "l2rvf" / "assets" / "l2r_scene.xml"

HOME_Q = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853])
NU_ARM = 7
CTRL_DT = 0.04
GRIPPER_CLOSED = 0.0


def load_model() -> mujoco.MjModel:
  if not (MENAGERIE_PANDA / "mjx_panda.xml").exists():
    raise FileNotFoundError(
        f"{MENAGERIE_PANDA} not found. Run scripts/fetch_assets.sh first."
    )
  dst = MENAGERIE_PANDA / "l2r_scene.xml"
  shutil.copyfile(SCENE_SRC, dst)
  return mujoco.MjModel.from_xml_path(str(dst))


class ArmEnv:
  """Holds the model, a 'real' MjData and a thread pool for rollouts."""

  STATE_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS

  def __init__(self, nthread: int | None = None):
    self.model = load_model()
    self.data = mujoco.MjData(self.model)
    self.substeps = int(round(CTRL_DT / self.model.opt.timestep))
    self.nthread = nthread or max(1, (os.cpu_count() or 2) - 2)
    self._datas = [mujoco.MjData(self.model) for _ in range(self.nthread)]
    self._pool = mj_rollout.Rollout(nthread=self.nthread)
    self.nstate = mujoco.mj_stateSize(self.model, self.STATE_SPEC)
    self.nsens = self.model.nsensordata
    self.u_low = self.model.actuator_ctrlrange[:NU_ARM, 0].copy()
    self.u_high = self.model.actuator_ctrlrange[:NU_ARM, 1].copy()
    jid = self.model.joint("block").id
    self.block_qadr = self.model.jnt_qposadr[jid]
    self.drawer_qadr = self.model.jnt_qposadr[self.model.joint("drawer").id]
    self.target_mocap = self.model.body("target").mocapid[0]

  def close(self):
    self._pool.close()

  # ------------------------------------------------------------------ state
  def reset(self, block_pos, drawer_q=0.0, arm_q=HOME_Q, target_pos=None):
    m, d = self.model, self.data
    mujoco.mj_resetData(m, d)
    d.qpos[:7] = arm_q
    d.qpos[7:9] = 0.0
    d.qpos[self.drawer_qadr] = drawer_q
    a = self.block_qadr
    d.qpos[a : a + 3] = block_pos
    d.qpos[a + 3 : a + 7] = [1, 0, 0, 0]
    d.ctrl[:7] = arm_q
    d.ctrl[7] = GRIPPER_CLOSED
    if target_pos is not None:
      d.mocap_pos[self.target_mocap] = target_pos
    mujoco.mj_forward(m, d)
    # Let the block settle.
    for _ in range(40):
      mujoco.mj_step(m, d)
    d.time = 0.0
    return self.get_state()

  def get_state(self, data=None) -> np.ndarray:
    data = data or self.data
    s = np.empty(self.nstate)
    mujoco.mj_getState(self.model, data, s, self.STATE_SPEC)
    return s

  def set_state(self, s: np.ndarray, data=None):
    data = data or self.data
    mujoco.mj_setState(self.model, data, s, self.STATE_SPEC)
    mujoco.mj_forward(self.model, data)

  def sensors(self, data=None) -> np.ndarray:
    return (data or self.data).sensordata.copy()

  def arm_q(self, state: np.ndarray) -> np.ndarray:
    return state[..., 1 : 1 + NU_ARM]  # skip time

  # --------------------------------------------------------------- dynamics
  def _expand_controls(self, U: np.ndarray) -> np.ndarray:
    """(N, H, 7) knots -> (N, H*substeps, nu) actuator controls."""
    N, H, _ = U.shape
    ctrl = np.empty((N, H * self.substeps, self.model.nu))
    ctrl[..., :NU_ARM] = np.repeat(U, self.substeps, axis=1)
    ctrl[..., NU_ARM:] = GRIPPER_CLOSED
    return ctrl

  def rollout(self, states: np.ndarray, U: np.ndarray):
    """Batched open-loop rollouts.

    states: (B, nstate), U: (B, K, H, 7)
    Returns sensordata at each knot (B, K, H, nsens) and final states (B, K, nstate).
    """
    B, K, H, _ = U.shape
    init = np.repeat(states, K, axis=0)
    ctrl = self._expand_controls(U.reshape(B * K, H, NU_ARM))
    st, sd = self._pool.rollout(
        [self.model], self._datas, init, ctrl, initial_warmstart=None,
        skip_checks=False,
    )
    sub = self.substeps
    sens = sd[:, sub - 1 :: sub].reshape(B, K, H, self.nsens)
    final = st[:, -1].reshape(B, K, self.nstate)
    return sens, final

  def step(self, u: np.ndarray):
    """Applies one knot of joint targets to the real simulation."""
    self.data.ctrl[:NU_ARM] = np.clip(u, self.u_low, self.u_high)
    self.data.ctrl[NU_ARM:] = GRIPPER_CLOSED
    for _ in range(self.substeps):
      mujoco.mj_step(self.model, self.data)

  def sensors_from_states(self, states: np.ndarray) -> np.ndarray:
    """Sensor readings for a batch of states (single-threaded forward)."""
    d = self._datas[0]
    out = np.empty((len(states), self.nsens))
    for i, s in enumerate(states):
      mujoco.mj_setState(self.model, d, s, self.STATE_SPEC)
      mujoco.mj_forward(self.model, d)
      out[i] = d.sensordata
    return out


# --------------------------------------------------------------------------
# Value-function features
# --------------------------------------------------------------------------
def value_features(state: np.ndarray, sens: np.ndarray, psi: np.ndarray) -> np.ndarray:
  """Features of (x, psi) for V_theta.

  state: (..., nstate) FULLPHYSICS (time, qpos[17], qvel[16]).
  sens:  (..., nsens) sensor readings of the same state.
  """
  qpos = state[..., 1:18]
  qvel = state[..., 18:34]
  arm_q = qpos[..., :7]
  arm_v = qvel[..., :7]
  drawer = qpos[..., 9:10]
  drawer_v = qvel[..., 9:10]
  palm = sens[..., R.S_PALM]
  palm_z = sens[..., R.S_PALM_Z]
  block = sens[..., R.S_BLOCK]
  block_v = sens[..., R.S_BLOCK_VEL]
  handle = sens[..., R.S_HANDLE]
  inside = sens[..., R.S_INSIDE]
  res = R.residual_terms(sens, psi)
  return np.concatenate(
      [
          arm_q, 0.1 * arm_v, palm, palm_z, block, block_v, drawer, drawer_v,
          palm - block, palm - handle, block - inside,
          block - psi[..., R.BLOCK_TGT], palm - psi[..., R.PALM_TGT],
          res, psi,
      ],
      axis=-1,
  )


VALUE_IN_DIM = 7 + 7 + 3 + 3 + 3 + 3 + 1 + 1 + 3 * 5 + 6 + R.NPSI
