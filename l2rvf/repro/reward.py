"""L2R manipulation reward API (paper Appendix A.5 v) and its stage cost.

The LLM-written code fills a RewardSpec; ``stage_cost`` evaluates it on sensor
readings. Residual terms and default weights follow the paper's Table 4:

  term                              residual             paper weight   here
  move obj1 close to obj2           |c1 - c2|            5              1.0
  move obj to target height         |c_z - c_z*|         10             2.0
  move obj to target orientation    |o - o*|             (illegible)    1.0
  move joint to target value        |q - q*|             10             2.0

Weights are the paper's divided by 5 so that the task-independent regularisers
shared with the rest of l2rvf (gripper pointing down, smooth targets) keep the
same relative strength. MPPI normalises its temperature by the cost spread, so
the overall scale does not matter, only the ratios. The orientation weight is
not legible in the paper's Table 4; 1.0 is our choice.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from l2rvf.arm import reward as R
from l2rvf.llm import reward_api as base

W_L2, W_Z, W_ORIENT_OBJ, W_JOINT = 1.0, 2.0, 1.0, 2.0
# Default palm position "when it's holding in the air": the palm at HOME_Q.
REST_POSITION = np.array([0.667, 0.0, 0.331])

POINT_OBJECTS = ("palm", "block", "target", "drawer_handle", "drawer_center", "rest_position")


@dataclasses.dataclass
class RewardSpec:
  l2: list = dataclasses.field(default_factory=list)  # [(objA, objB)]
  z: dict = dataclasses.field(default_factory=dict)  # obj -> height
  orient: dict = dataclasses.field(default_factory=dict)  # obj -> x-rotation (rad)
  joint: dict = dataclasses.field(default_factory=dict)  # 'drawer' -> fraction
  duration: float | None = None
  calls: list = dataclasses.field(default_factory=list)

  def describe(self) -> str:
    parts = [f"|{a}-{b}|" for a, b in self.l2]
    parts += [f"|{o}.z-{h:.2f}|" for o, h in self.z.items()]
    parts += [f"|{o}.rot_x-{np.rad2deg(r):.0f}deg|" for o, r in self.orient.items()]
    parts += [f"|{j}-{f:.2f}|" for j, f in self.joint.items()]
    return " + ".join(parts) if parts else "(empty)"


def _pos(s: np.ndarray, name: str) -> np.ndarray:
  if name == "palm":
    return s[..., R.S_PALM]
  if name == "block":
    return s[..., R.S_BLOCK]
  if name == "target":
    return s[..., R.S_TARGET]
  if name == "drawer_handle":
    return s[..., R.S_HANDLE]
  if name == "drawer_center":
    return s[..., R.S_INSIDE]
  if name == "rest_position":
    return np.broadcast_to(REST_POSITION, s.shape[:-1] + (3,))
  raise base.RewardAPIError(f"unknown object {name}")


def execute(code: str) -> RewardSpec:
  """Runs LLM reward code in the restricted namespace of l2rvf.llm.reward_api."""
  spec = RewardSpec()

  def _check(name):
    if name not in POINT_OBJECTS:
      raise base.RewardAPIError(f"unknown object {name!r}")

  def reset_reward():
    spec.l2.clear(), spec.z.clear(), spec.orient.clear(), spec.joint.clear()
    spec.calls.append("reset_reward()")

  def set_l2_distance_reward(name_obj_A, name_obj_B):
    _check(name_obj_A), _check(name_obj_B)
    spec.l2.append((name_obj_A, name_obj_B))
    spec.calls.append(f"l2({name_obj_A},{name_obj_B})")

  def set_obj_orientation_reward(name_obj, x_axis_rotation_radians):
    if name_obj != "block":
      raise base.RewardAPIError(f"orientation of {name_obj!r} is not controllable")
    spec.orient[name_obj] = float(x_axis_rotation_radians)
    spec.calls.append(f"orient({name_obj},{float(x_axis_rotation_radians):.3f})")

  def set_obj_z_position_reward(name_obj, z_height):
    _check(name_obj)
    spec.z[name_obj] = float(z_height)
    spec.calls.append(f"z({name_obj},{float(z_height)})")

  def set_joint_fraction_reward(name_joint, fraction):
    if name_joint != "drawer":
      raise base.RewardAPIError(f"unknown joint {name_joint!r}")
    spec.joint[name_joint] = float(np.clip(fraction, 0.0, 1.0))
    spec.calls.append(f"joint({name_joint},{fraction})")

  def execute_plan(duration=2):
    spec.duration = float(duration)
    spec.calls.append(f"execute_plan({duration})")

  env = {
      "__builtins__": {**base._SAFE_BUILTINS, "__import__": base._safe_import},
      "np": np,
      "reset_reward": reset_reward,
      "set_l2_distance_reward": set_l2_distance_reward,
      "set_obj_orientation_reward": set_obj_orientation_reward,
      "set_obj_z_position_reward": set_obj_z_position_reward,
      "set_joint_fraction_reward": set_joint_fraction_reward,
      "execute_plan": execute_plan,
  }
  try:
    exec(compile(code, "<llm>", "exec"), env)  # noqa: S102 - restricted namespace
  except base.RewardAPIError:
    raise
  except Exception as e:  # pylint: disable=broad-except
    raise base.RewardAPIError(f"reward code failed: {e!r}") from e
  if spec.duration is None:
    raise base.RewardAPIError("execute_plan was not called")
  return spec


def stage_cost(s: np.ndarray, spec: RewardSpec) -> np.ndarray:
  """Cost per knot (without dt) for sensors s (..., nsens)."""
  c = np.zeros(s.shape[:-1])
  for a, b in spec.l2:
    c = c + W_L2 * R._dist(_pos(s, a), _pos(s, b))
  for o, h in spec.z.items():
    c = c + W_Z * np.abs(_pos(s, o)[..., 2] - h)
  for o, rot in spec.orient.items():  # only the block: z-axis rotated about world x
    tgt = np.array([0.0, -np.sin(rot), np.cos(rot)])
    c = c + W_ORIENT_OBJ * R._dist(s[..., R.S_BLOCK_Z], tgt)
  for _, f in spec.joint.items():
    q = s[..., R.S_DRAWER_Q]
    c = c + W_JOINT * np.abs(q - f * R.DRAWER_MAX)
  # Task-independent regulariser shared with l2rvf.arm.reward (not exposed to the LLM).
  z = s[..., R.S_PALM_Z]
  return c + R.W_ORIENT * np.sum((z - np.array([0.0, 0.0, -1.0])) ** 2, axis=-1)
