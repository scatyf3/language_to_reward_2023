# l2rvf — Language to Rewards with a Learned Value Function

Code for the ROB-GY 6323 project *"Language to Rewards with a Learned Value
Function: Short-Horizon MPC for Multi-Stage Manipulation"* (see `proposal.pdf`).
It sits next to the original L2R code (`src/language_to_reward_2023`, Barkour
quadruped + C++ MJPC) and re-implements the parts the proposal needs in pure
Python so that it runs on any OS without building MJPC:

| Proposal component | File |
|---|---|
| MPPI (Williams et al. 2016), batched | `l2rvf/mppi.py` |
| Value network V_θ(x; ψ) + regression | `l2rvf/value.py` |
| Step 1: key-and-door toy | `l2rvf/toy/keydoor.py`, `scripts/toy_experiment.py` |
| Franka Panda scene (MuJoCo Menagerie MJX model), tasks A–C | `l2rvf/assets/l2r_scene.xml`, `l2rvf/arm/{env,tasks}.py` |
| L2R reward API → ψ, stage cost ℓ_ψ | `l2rvf/arm/reward.py`, `l2rvf/llm/reward_api.py` |
| LLM translator (Motion Descriptor + Reward Coder, as in the original repo) | `l2rvf/llm/{prompts,translator}.py`, `scripts/translate_instructions.py` |
| Fitted value iteration with MPPI targets (Eq. 2) | `l2rvf/arm/fvi.py`, `scripts/arm_train_value.py` |
| On-policy FVI + pessimistic ensemble (needed on the arm, see report) | `scripts/arm_onpolicy_fvi.py`, `l2rvf/value.py:ValueEnsemble` |
| Controller comparison, horizon study, videos | `scripts/arm_eval.py` |
| Results write-up | `docs/preliminary_report.md` |

## Setup

```sh
python -m venv .venv && . .venv/Scripts/activate     # (Linux/macOS: .venv/bin/activate)
pip install -r requirements-l2rvf.txt
bash scripts/fetch_assets.sh                          # Franka model from MuJoCo Menagerie
```

## Running

```sh
python scripts/toy_experiment.py                      # Step 1, ~8 min on 16 CPU cores
python scripts/translate_instructions.py --provider anthropic   # needs ANTHROPIC_API_KEY
python scripts/translate_instructions.py --provider replay      # re-execute stored responses
python scripts/arm_train_value.py                     # FVI on exploration states, ~40 min
python scripts/arm_onpolicy_fvi.py --rounds 6         # on-policy FVI + pessimistic ensemble
python scripts/arm_onpolicy_fvi.py --resume --init results/arm/ensemble.pt --rounds 4 --episodes 8
python scripts/arm_eval.py --video --seeds 2 --tag _final   # controller comparison A–C
python scripts/arm_eval.py --sweep C                  # horizon study on task C
python scripts/make_report_figures.py                 # figures for the report
```

## Design notes

* **Control.** MPPI samples H knots (0.04 s each, zero-order hold) of absolute
  joint-position targets for the 7 arm actuators; noise is linearly
  interpolated between every 3rd knot. Rollouts use `mujoco.rollout` with one
  `MjData` per thread. The gripper is kept closed (no task needs grasping; the
  closed fingers hook the drawer handle).
* **Tasks.** A: push the block onto the green target (1 stage). B: open the
  drawer (reach handle, pull). C: put the block in the drawer — the block starts
  on the cabinet top, so the robot must open the drawer, reach the block and
  push it off the front edge into the open drawer (3 stages, no grasping).
* **ψ (13 numbers).** Weights and targets of six residuals: palm–block,
  palm–handle, block–drawer-inside, block–position, drawer-fraction,
  palm–position. Fixed, task-independent regularisers (gripper pointing down,
  smooth targets) are not exposed to the LLM, as in MJPC.
* **Value targets.** For each sampled (x, ψ): MPPI over T = 25 knots (1 s),
  K = 48 samples, 3 iterations, warm-started at "hold position"; the target is
  the best sampled cost plus V_k at the end state. States come from MPC
  episodes with shaped rewards (exploration) plus randomised synthetic states
  (block on floor / cabinet top / in drawer, random drawer opening). ψ is drawn
  from a prior over what the reward API can express; the stored LLM ψ are *not*
  used in training, so the evaluation tests generalisation to LLM-written rewards.
* **Stored LLM rewards.** `data/llm_responses.jsonl` holds the raw Motion
  Descriptor / Reward Coder responses for the 30 instructions in
  `data/instructions.json`, and `data/psi_cache.json` the resulting ψ. The
  `source` field records where each response came from.
