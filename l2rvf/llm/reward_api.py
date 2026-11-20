"""Executes LLM-written reward code against the reward API and returns psi.

The code is untrusted. It runs with a whitelist of builtins, numpy, and the
reward-API functions only (the original repo uses a subprocess sandbox for the
same purpose; see src/language_to_reward_2023/safe_executor.py).
"""

from __future__ import annotations

import re

import numpy as np

from l2rvf.arm import reward as R


class RewardAPIError(ValueError):
  pass


def process_code_block(text: str) -> str:
  """Same logic as src/language_to_reward_2023/platforms/process_code.py."""
  matches = re.findall(r"```(python)?\n?([\s\S]*?)```", text)
  code = "\n".join(m[1] for m in matches) if matches else text
  if "np." in code and "import numpy as np" not in code:
    code = "import numpy as np\n" + code
  return code


def extract_description(response: str) -> str:
  try:
    body = re.split("start of description", response, flags=re.IGNORECASE)[1]
    body = re.split("end of description", body, flags=re.IGNORECASE)[0]
    return body.strip("[]` \n")
  except IndexError:
    return response


_SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in ["abs", "min", "max", "float", "int", "round", "range", "len", "print",
              "True", "False", "None", "list", "tuple", "dict", "sum"]
}


def _safe_import(name, *args, **kwargs):
  if name != "numpy":
    raise RewardAPIError(f"import of '{name}' is not allowed")
  return np


def execute(code: str) -> dict:
  """Runs reward code; returns {'psi': ndarray, 'duration': float, 'calls': [...]}."""
  psi = R.empty_psi()
  calls = []
  out = {"duration": None}
  pairs = {
      frozenset(["palm", "block"]): R.W_PALM_BLOCK,
      frozenset(["palm", "drawer_handle"]): R.W_PALM_HANDLE,
      frozenset(["block", "drawer_inside"]): R.W_BLOCK_DRAWER,
  }

  def reset_reward():
    psi[:] = R.empty_psi()
    calls.append("reset_reward()")

  def set_l2_distance_reward(name_obj_A, name_obj_B, weight=1.0):
    key = frozenset([name_obj_A, name_obj_B])
    if key not in pairs:
      raise RewardAPIError(f"unsupported pair {name_obj_A}, {name_obj_B}")
    psi[pairs[key]] = float(weight)
    calls.append(f"l2({name_obj_A},{name_obj_B},{weight})")

  def set_obj_position_reward(name_obj, x, y, z, weight=1.0):
    if name_obj == "block":
      psi[R.W_BLOCK_POS] = float(weight)
      psi[R.BLOCK_TGT] = [x, y, z]
    elif name_obj == "palm":
      psi[R.W_PALM_POS] = float(weight)
      psi[R.PALM_TGT] = [x, y, z]
    else:
      raise RewardAPIError(f"unsupported object {name_obj}")
    calls.append(f"pos({name_obj},{x},{y},{z},{weight})")

  def set_joint_fraction_reward(name_joint, fraction, weight=1.0):
    if name_joint != "drawer":
      raise RewardAPIError(f"unsupported joint {name_joint}")
    psi[R.W_DRAWER] = float(weight)
    psi[R.DRAWER_FRAC] = float(np.clip(fraction, 0.0, 1.0))
    calls.append(f"joint({name_joint},{fraction},{weight})")

  def execute_plan(duration=2):
    out["duration"] = float(duration)
    calls.append(f"execute_plan({duration})")

  env = {
      "__builtins__": {**_SAFE_BUILTINS, "__import__": _safe_import},
      "np": np,
      "reset_reward": reset_reward,
      "set_l2_distance_reward": set_l2_distance_reward,
      "set_obj_position_reward": set_obj_position_reward,
      "set_joint_fraction_reward": set_joint_fraction_reward,
      "execute_plan": execute_plan,
  }
  try:
    exec(compile(code, "<llm>", "exec"), env)  # noqa: S102 - restricted namespace
  except RewardAPIError:
    raise
  except Exception as e:  # pylint: disable=broad-except
    raise RewardAPIError(f"reward code failed: {e!r}") from e
  if np.any(psi[[R.W_PALM_BLOCK, R.W_PALM_HANDLE, R.W_BLOCK_DRAWER, R.W_BLOCK_POS,
                 R.W_DRAWER, R.W_PALM_POS]] < 0):
    raise RewardAPIError("negative weight")
  return {"psi": psi, "duration": out["duration"], "calls": calls}
