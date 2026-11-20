"""Fast unit tests for l2rvf (no MuJoCo rollouts). Run: python -m pytest tests -q"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from l2rvf.arm import reward as R  # noqa: E402
from l2rvf.llm import reward_api  # noqa: E402
from l2rvf.mppi import MPPI, MPPIConfig  # noqa: E402
from l2rvf.toy import keydoor as kd  # noqa: E402


def test_reward_api_fills_psi():
  code = reward_api.process_code_block(
      "```python\nreset_reward()\nset_l2_distance_reward('palm', 'block')\n"
      "set_obj_position_reward('block', 0.4, 0.1, 0.02, weight=2.0)\n"
      "set_joint_fraction_reward('drawer', 1.0)\nexecute_plan(3)\n```")
  out = reward_api.execute(code)
  psi = out["psi"]
  assert psi[R.W_PALM_BLOCK] == 1.0 and psi[R.W_BLOCK_POS] == 2.0
  np.testing.assert_allclose(psi[R.BLOCK_TGT], [0.4, 0.1, 0.02])
  assert psi[R.W_DRAWER] == 1.0 and psi[R.DRAWER_FRAC] == 1.0
  assert out["duration"] == 3.0


@pytest.mark.parametrize("bad", ["import os", "open('x')", "set_l2_distance_reward('palm', 'robot')",
                                 "__import__('subprocess')"])
def test_reward_api_rejects_unsafe_or_invalid(bad):
  with pytest.raises(reward_api.RewardAPIError):
    reward_api.execute(bad)


def test_keydoor_door_blocks_without_key():
  x = np.array([[-0.05, 0.0, 0.0]])
  for _ in range(5):
    x = kd.step(x, np.array([[1.0, 0.0]]))
  assert x[0, 0] < 0  # wall holds
  x = np.array([[-0.05, 0.0, 1.0]])
  for _ in range(5):
    x = kd.step(x, np.array([[1.0, 0.0]]))
  assert x[0, 0] > 0.3  # door open with key


def test_keydoor_switch_sets_key():
  x = np.array([[kd.SWITCH[0] - 0.15, kd.SWITCH[1], 0.0]])
  x = kd.step(x, np.array([[1.0, 0.0]]))
  assert x[0, 2] == 1.0


def test_mppi_solves_quadratic():
  cfg = MPPIConfig(horizon=5, num_samples=256, noise_sigma=0.5, temperature=0.05, n_iters=20)
  m = MPPI(cfg, nu=2, batch=3, seed=0)
  goal = np.array([0.3, -0.2])
  out = m.optimize(lambda U: np.sum((U - goal) ** 2, axis=(-1, -2)))
  # Fixed-sigma MPPI hovers around the optimum with noise of order sigma / sqrt(n_eff).
  np.testing.assert_allclose(m.U.mean(axis=1), np.broadcast_to(goal, (3, 2)), atol=0.1)
  assert np.all(out["best_cost"] < 0.1 * np.sum(goal ** 2) * cfg.horizon)
