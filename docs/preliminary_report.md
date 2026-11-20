# Preliminary Report — Language to Rewards with a Learned Value Function

ROB-GY 6323 · Team: Yifei Feng (yf3005), [Member 2], [Member 3], [Member 4], [Member 5] · Draft of 2026-11-14

**Summary.** We implemented the full pipeline of the proposal: an L2R-style
LLM translator that turns instructions into reward parameters ψ, batched MPPI
on MuJoCo, and a value function V_θ(x; ψ) trained by fitted value iteration
(FVI) with MPPI-solved targets, used as the MPC terminal cost. On the toy
problem, short-horizon MPPI with V_θ succeeds in 99% of episodes, against 0%
without it. On the Franka arm, the first V_θ made things *worse*, because the
MPC steered into states where V_θ was wrongly optimistic. Collecting the
V-MPC's own states (on-policy FVI) and using a pessimistic ensemble fixed this.
With the fix, H = 10 MPPI with V_θ solves the three-stage task C in 70% of
episodes. The same horizon without V_θ succeeds in 0%, and H = 40 MPPI without
V_θ succeeds in 60% while costing 3.2× more computation per step. Task A
(single-stage pushing) is still unsolved by our method.

## 1. What was built

| Component | Implementation |
|---|---|
| Simulator / MPC | MuJoCo 4.0 Python, `mujoco.rollout` with 14 threads; MPPI over H knots of 0.04 s of joint-position targets (zero-order hold, temporally correlated noise), K = 64 samples, one iteration per control step, warm start |
| Robot and tasks | Franka Panda (MuJoCo Menagerie, MJX variant with primitive collisions); A: push block to target (1 stage); B: open drawer (reach handle, pull; 2 stages); C: put block in drawer — the block starts on the cabinet top, so the robot must open the drawer, reach the block and push it off the edge into the drawer (3 stages, no grasping) |
| Reward API (ψ, 13 numbers) | `set_l2_distance_reward(palm/block/drawer_handle/drawer_inside)`, `set_obj_position_reward`, `set_joint_fraction_reward('drawer', f)`; stage cost ℓ_ψ = weighted sum of residuals + fixed regularisers (gripper down, smooth targets) that the LLM does not see |
| LLM translator | Same two-stage structure as the original L2R code (Motion Descriptor → Reward Coder), adapted to the arm; code runs in a restricted namespace; 10 paraphrased instructions per task, translated once and stored |
| Value learning | FVI, Eq. (2): targets y = best MPPI cost over T = 25 knots (1 s), K = 48, 3 iterations, + V_k(x_T). MLP 62 → 256 × 3 → 1 (softplus) on arm/object features, relative vectors, residuals and ψ. ψ is sampled from a prior over the reward API, so the stored LLM ψ are never used in training |

> **Caveat on the stored LLM rewards.** The machine used for these runs had no
> LLM API key, so the 30 stored Motion Descriptor / Reward Coder responses
> (`data/llm_responses.jsonl`) were written in-session by Claude (Opus 5.5)
> following the fixed prompts, not obtained by API calls. Each record says so
> in its `source` field. Re-run `python scripts/translate_instructions.py
> --provider anthropic` before the final report; all other numbers below can
> then be recomputed with the same scripts.

The stored ψ follow L2R's habit of describing outcomes. 7/10 task-A rewards
only ask for the block to be at the target. 8/10 task-B rewards only ask for
the drawer to be open. No task-C reward mentions the handle.

## 2. Step 1 — key-and-door toy problem

The point mass must touch a switch to open a door; the cost is only the
distance to the goal. 100 episodes with random starts and goals, K = 128.

| Controller | H | Success | ms / step |
|---|---|---|---|
| reward only (L2R) | 8 | 0% | 1.2 |
| + hand-written shaping | 8 | 86% | 1.4 |
| **+ learned V_θ (ours)** | 8 | **99%** | 2.2 |
| long horizon, no V | 40 | 8% | 5.3 |

With V_θ, MPPI succeeds in 97% of episodes even at H = 2, while without V_θ no
horizon up to 48 exceeds 23%. Success with V_θ drops slightly for long horizons
(81–85% at H ≥ 16). Our explanation is that MPPI sampling gets worse as H
grows, and that is also why the long-horizon baseline fails. FVI converged in
8 rounds, with held-out relative error 2.2%.
Figures: `results/toy/{trajectories,value,horizon_sweep,fvi}.png`.

## 3. Step 2 — Franka tasks A–C

### 3.1 A negative result first: optimism exploitation

After 8 rounds of FVI on exploration and synthetic states (held-out error
12%), the V-MPC did worse than the baseline on every task (A 0%, B 30%, C 0%).
Diagnosis on task A: along a successful trajectory V_θ behaved correctly, and
moving the block onto the target lowered V_θ from 0.43 to 0.11. The V-MPC,
however, moved the *hand* to the target while the block was still 0.30 m away,
reaching states with V_θ = 0.04. On the states the V-MPC visits, V_θ
underestimated the MPPI targets by 0.60 on average (relative error 87%).

**Fix: on-policy FVI + pessimism.** Each round, (i) the current V-MPC runs on
tasks A–C with prior-sampled ψ, and its visited (x, ψ) are added to a pool
(capped at 6000) together with 2500 exploration states; (ii) MPPI targets are
recomputed; (iii) an ensemble of 5 bootstrapped networks is refit from
scratch, and the MPC uses mean + κ·std (κ = 1, fixed in advance). Over 10
rounds the error on fresh on-policy states fell from 87% to 14%
(`results/arm/onpolicy.png`). The model evaluated below is the last round's
ensemble, chosen by that rule in advance; it was not selected on the test
instructions.

### 3.2 Controller comparison (stored LLM ψ, 10 instructions × 2 seeds per task)

| Task | Controller | H | Success | Stages done | ms / step |
|---|---|---|---|---|---|
| A | reward only (L2R) | 10 | 30% | 0.3 / 1 | 10.6 |
| A | + hand shaping | 10 | **60%** | 0.9 / 1 | 13.6 |
| A | + V_θ (ours) | 10 | 25% | 0.4 / 1 | 12.8 |
| A | long horizon | 40 | 25% | 0.4 / 1 | 42.9 |
| B | reward only (L2R) | 10 | 45% | 0.9 / 2 | 10.6 |
| B | + hand shaping | 10 | 95% | 2.0 / 2 | 13.1 |
| B | + V_θ (ours) | 10 | **100%** | 2.0 / 2 | 12.7 |
| B | long horizon | 40 | 100% | 2.0 / 2 | 42.8 |
| C | reward only (L2R) | 10 | 0% | 0.2 / 3 | 11.0 |
| C | + hand shaping | 10 | 0% | 0.6 / 3 | 13.6 |
| C | + V_θ (ours) | 10 | **70%** | 2.1 / 3 | 13.8 |
| C | long horizon | 40 | 60% | 1.8 / 3 | 44.7 |

Success means the final stage condition holds at the end of the episode (8 s
for A and B, 14 s for C); 20 episodes per cell. In the successful task-C
episodes, the V-MPC completes the stages in the intended order: it opens the
drawer, reaches the block on the cabinet top, and pushes it into the drawer.
A single fixed shaping term (palm–handle + palm–block) does not help on C,
because the two attractors conflict.

**Horizon study (task C, 10 instructions × 1 seed per cell).**

| H | 3 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|
| reward only: success / stages | 0% / 0.0 | 0% / 0.0 | 0% / 0.3 | 30% / 1.3 | 50% / 1.8 |
| + V_θ: success / stages | 20% / 1.8 | 10% / 1.5 | 60% / 1.9 | 80% / 2.2 | 80% / 2.2 |
| ms / step (reward only / + V_θ) | 3.5 / 5.7 | 5.6 / 8.1 | 11.0 / 14.0 | 21.9 / 29.0 | 44.4 / 47.9 |

With V_θ, H = 10 already does better than H = 40 without it, at about a third
of the computation. Even at H = 3 the V-MPC opens the drawer (1.8 of 3 stages
on average); without V_θ, short horizons do not move the drawer at all.

**Reproducibility note.** Episodes are not bit-reproducible: MuJoCo's solver
warm-start and the contact dynamics make runs with the same seed diverge. For
example, task-C instruction 0 succeeded with seeds 0 and 1000 in the main
evaluation but failed in a re-run for the video. All numbers here are rates
over 10–20 episodes and carry roughly ±10–15 percentage points of sampling
noise. Videos: `results/arm/videos/` (`C_value_side.mp4` uses a camera that
shows the drawer interior).

### 3.3 Reading of the results so far

* **Multi-stage tasks are where V_θ pays off.** On C, V_θ turns a 0% short-horizon controller into 70%, and it beats MPPI with 4× the horizon. On B, it matches hand shaping without the LLM having to mention the handle.
* **Single-stage pushing (A) is not solved.** Hand shaping is best there (60%). The likely reason is that pushing needs precise contact, while V_θ's error on A-states is still large: on-policy success for A during training stayed between 0 and 50%. A plausible fix is more on-policy rounds for A, or adding the palm–block term to V_θ's input. This is the "simple shaping works as well" outcome anticipated in the proposal's risk list, and we will report it.
* **Generalisation to LLM ψ.** V_θ never saw the stored LLM ψ during training, and on B and C it still did well on them. We have not yet needed the proposal's fallback of adding LLM ψ to the training data.

## 4. Deviations from the proposal

1. **No C++ MuJoCo MPC.** The original repository's MJPC/gRPC build only ships the Barkour quadruped, so we wrote MPPI in Python on `mujoco.rollout`. Planning still runs in real time: 11–14 ms per 40 ms control step at H = 10.
2. **Gripper fixed closed; task C redesigned to need no grasp.** We expect sampling-based MPC to be unreliable at grasping. The redesigned C still has three stages that must happen in order.
3. **On-policy FVI and the pessimistic ensemble were added.** The proposal's plain FVI failed on the arm, for the reason analysed in Sec. 3.1.

## 5. Next steps (Nov 19 – Dec 3)

1. Replace the in-session LLM responses with real API responses, then re-run the evaluation.
2. Task A: more on-policy rounds, focused on A; ablate κ and ensemble size.
3. Horizon study on all tasks; report V_θ prediction error on held-out on-policy states for each task.
4. Videos with a camera that shows the drawer interior.
5. Optional: compare against a PPO policy trained on the same reward prior.

## Reproduction

```sh
pip install -r requirements-l2rvf.txt && bash scripts/fetch_assets.sh
python scripts/toy_experiment.py
python scripts/translate_instructions.py --provider replay   # or --provider anthropic
python scripts/arm_train_value.py
python scripts/arm_onpolicy_fvi.py --rounds 6
python scripts/arm_onpolicy_fvi.py --resume --init results/arm/ensemble.pt --rounds 4 --episodes 8
python scripts/arm_eval.py --video --seeds 2 --tag _final
python scripts/arm_eval.py --sweep C
python scripts/make_report_figures.py
```
