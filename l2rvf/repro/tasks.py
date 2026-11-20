"""L2R paper tasks reproduced on our Franka scene, and the closed-loop runner.

Single-turn tasks (paper Table 2, the ones a closed parallel gripper can do):
  touch         "Touch the block."              block in contact with hand/fingers
  move          "Move the block to the target." |block - target| < 0.1 m
  open_drawer   "Open the drawer."              drawer pulled fully open (>= 90 %)
plus one single-turn version of the multi-stage task, for comparison with the
interactive run:
  block_in_drawer "Put the block in the drawer."  block inside the drawer

Not reproducible here: lift / lift two / upright / flip (need grasping or a
dexterous hand; the cube is symmetric) and faucet (no faucet in the scene).

Multi-turn task (paper Appendix A.9, "place an apple in a drawer"): four user
instructions, each translated into a new reward that replaces the previous
one (reset_reward). Our block starts on the cabinet top, so the hint in
instruction 2 is adapted from "Assume drawer is on the floor so no need to
lift up the apple" to the equivalent fact about our scene.

Success criteria follow Table 2 and are checked at the end of the episode.
"""

from __future__ import annotations

import dataclasses
import re
import time

import numpy as np

from l2rvf.arm import env as E
from l2rvf.arm import mpc as M
from l2rvf.arm import reward as R
from l2rvf.arm import tasks as T
from l2rvf.mppi import MPPI
from l2rvf.repro import reward as RR

OPEN_FULL = 0.9  # "pulled fully open"
CLOSED = 0.1


@dataclasses.dataclass
class SingleTask:
  key: str
  instruction: str
  block_pos: np.ndarray
  episode_s: float
  target_pos: np.ndarray | None = None


SINGLE = {
    "touch": SingleTask("touch", "Touch the block.", T.BLOCK_A, 8.0, T.TARGET_A),
    "move": SingleTask("move", "Move the block to the target.", T.BLOCK_A, 8.0, T.TARGET_A),
    "open_drawer": SingleTask("open_drawer", "Open the drawer.", T.BLOCK_PARKED, 8.0),
    "block_in_drawer": SingleTask("block_in_drawer", "Put the block in the drawer.",
                                  T.BLOCK_C, 16.0),
}

MULTI_INSTRUCTIONS = [
    "Open the drawer.",
    "Good, now put the block inside the drawer while keep it open. The block is on top of the "
    "cabinet, so no need to lift it up.",
    "Good, now release the block and move hand away.",
    "Now close the drawer.",
]


def frac(s):
  return s[..., R.S_DRAWER_Q] / R.DRAWER_MAX


def touching(env) -> bool:
  """True if the block geom touches any geom of the hand or fingers."""
  m, d = env.model, env.data
  if not hasattr(env, "_hand_geoms"):
    bodies = {m.body(b).id for b in ("hand", "left_finger", "right_finger")}
    env._hand_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies}
    env._block_geom = m.geom("block").id
  for i in range(d.ncon):
    g1, g2 = d.contact[i].geom1, d.contact[i].geom2
    if (g1 == env._block_geom and g2 in env._hand_geoms) or (
        g2 == env._block_geom and g1 in env._hand_geoms):
      return True
  return False


def single_success(key: str, env, s: np.ndarray) -> bool:
  if key == "touch":
    return touching(env)
  if key == "move":
    return bool(np.linalg.norm(s[R.S_BLOCK] - s[R.S_TARGET]) < 0.1)
  if key == "open_drawer":
    return bool(frac(s) >= OPEN_FULL)
  if key == "block_in_drawer":
    return bool(T.block_in_drawer(s))
  raise ValueError(key)


def turn_done(turn: int, s: np.ndarray) -> bool:
  """What the user in A.9 waits to see before saying "Good, now ..."."""
  if turn == 0:
    return bool(frac(s) >= OPEN_FULL)
  if turn == 1:
    return bool(T.block_in_drawer(s))
  if turn == 2:
    return bool(np.linalg.norm(s[R.S_PALM] - RR.REST_POSITION) < 0.08)
  if turn == 3:
    return bool(frac(s) <= CLOSED)
  raise ValueError(turn)


def multi_success(s: np.ndarray) -> bool:
  """Apple-in-drawer outcome: block inside and drawer closed."""
  return bool(T.block_in_drawer(s) and frac(s) <= CLOSED)


def extract_plan(response: str) -> str:
  """Motion Descriptor output; accepts [start of plan] and [start of description]."""
  m = re.search(r"start of (?:plan|description)\]?(.*?)\[?end of (?:plan|description)",
                response, flags=re.IGNORECASE | re.DOTALL)
  return m.group(1).strip() if m else response


# ---------------------------------------------------------------------------
# Closed-loop control with a RewardSpec
# ---------------------------------------------------------------------------
class SpecMPC:
  """MPPI on sum_t l_spec(x_t) dt. The spec can be swapped between turns."""

  def __init__(self, env, H, K, iters=1, seed=0):
    self.env = env
    self.mppi = MPPI(M.arm_mppi_cfg(H, K, iters, env), E.NU_ARM, batch=1, seed=seed)
    self.mppi.reset(env.arm_q(env.get_state()))
    self.spec = None

  def _cost(self, state, U):
    sens, _ = self.env.rollout(state[None], U)
    c = RR.stage_cost(sens, self.spec).sum(-1) * E.CTRL_DT
    dU = np.diff(U, axis=2)
    return c + R.W_SMOOTH * np.sum(dU * dU, axis=(-1, -2))

  def act(self, state):
    self.mppi.optimize(lambda U: self._cost(state, U))
    u = self.mppi.action()[0]
    self.mppi.shift()
    return u


def run_single(env, key, spec, H, K, seed, recorder=None):
  task = SINGLE[key]
  rng = np.random.default_rng(seed)
  b = task.block_pos.copy()
  b[:2] += rng.uniform(-0.02, 0.02, size=2)
  env.reset(block_pos=b, drawer_q=0.0, target_pos=task.target_pos)
  ctl = SpecMPC(env, H, K, seed=seed)
  ctl.spec = spec
  n = int(round(task.episode_s / E.CTRL_DT))
  ever, times = False, []
  for _ in range(n):
    st = env.get_state()
    t0 = time.perf_counter()
    u = ctl.act(st)
    times.append(time.perf_counter() - t0)
    env.step(u)
    ever = ever or single_success(key, env, env.sensors())
    if recorder is not None:
      recorder(env)
  return {"success": single_success(key, env, env.sensors()), "ever_success": bool(ever),
          "ms_per_step": 1000 * float(np.median(times))}


def run_multi(env, specs, H, K, seed, user="patient", max_turn_s=10.0, recorder=None):
  """Runs the A.9 sequence.

  user="strict":  each turn runs for exactly its execute_plan duration, then the
                  next instruction is given regardless of the outcome.
  user="patient": each turn runs at least its execute_plan duration and until the
                  user sees the turn's goal achieved (turn_done), up to max_turn_s;
                  this models the human in A.9 who says "Good, now ..." only after
                  watching the result. It uses ground-truth state, like the human.
  """
  rng = np.random.default_rng(seed)
  b = T.BLOCK_C.copy()
  b[:2] += rng.uniform(-0.02, 0.02, size=2)
  env.reset(block_pos=b, drawer_q=0.0)
  ctl = SpecMPC(env, H, K, seed=seed)
  turns, times = [], []
  for k, spec in enumerate(specs):
    ctl.spec = spec
    min_steps = int(round(spec.duration / E.CTRL_DT))
    max_steps = min_steps if user == "strict" else max(min_steps, int(round(max_turn_s / E.CTRL_DT)))
    steps, done = 0, False
    while steps < max_steps:
      st = env.get_state()
      t0 = time.perf_counter()
      u = ctl.act(st)
      times.append(time.perf_counter() - t0)
      env.step(u)
      steps += 1
      if recorder is not None:
        recorder(env)
      done = turn_done(k, env.sensors())
      if user == "patient" and steps >= min_steps and done:
        break
    turns.append({"turn": k, "done": bool(done), "seconds": steps * E.CTRL_DT})
  s = env.sensors()
  return {"success": multi_success(s), "turns": turns,
          "turns_done": int(sum(t["done"] for t in turns)),
          "total_s": float(sum(t["seconds"] for t in turns)),
          "ms_per_step": 1000 * float(np.median(times))}
