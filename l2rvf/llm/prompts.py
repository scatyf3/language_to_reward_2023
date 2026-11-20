"""Fixed L2R-style prompts for the Franka manipulation reward API.

Mirrors the two-LLM structure of the original repo
(src/language_to_reward_2023/platforms/barkour/prompts/prompt_thinker_coder.py):
a Motion Descriptor ("thinker") fills a structured template and a Reward Coder
turns that description into calls to the reward API.
"""

SCENE = """
The scene: a Franka Panda robot arm with a parallel gripper stands at the origin
and faces the +x direction (+y is to its left, +z is up; units are metres).
* A small red block (4 cm cube). Its current position is given with each task.
* A green target marker on the floor at (x=0.45, y=0.05, z=0.0).
* A cabinet with one drawer to the robot's left. The cabinet occupies
  x in [0.40, 0.70], y in [0.30, 0.62]; its top surface is at z=0.18.
  The drawer opens towards the robot (along -y) by up to 0.18 m; its handle is
  at (x=0.55, y=0.27, z=0.10) when the drawer is closed.
"""

prompt_thinker = (
    """
We have a robot arm and want you to describe what the robot should achieve,
using the following form:

[start of description]
* This {CHOICE: [is, is not]} a new task.
* [optional] The palm of the robot should be close to {CHOICE: [the block, the drawer handle]}.
* [optional] The block should be close to {CHOICE: [the inside of the drawer]}.
* [optional] The block should be at position (x=[NUM: 0.0], y=[NUM: 0.0], z=[NUM: 0.0]).
* [optional] The drawer should be [NUM: 0.0]% open.
* [optional] The palm should be at position (x=[NUM: 0.0], y=[NUM: 0.0], z=[NUM: 0.0]).
[end of description]
"""
    + SCENE
    + """
Rules:
1. If you see phrases like [NUM: default_value], replace the entire phrase with a numerical value.
2. If you see phrases like {CHOICE: [choice1, choice2, ...]}, replace the entire phrase with one of the choices listed.
3. A bullet point may be repeated if more than one choice applies.
4. I will tell you a task that I want the robot to perform and you will provide the full description. Always start the description with [start of description] and end it with [end of description].
5. We can assume that the robot has a good low-level controller that can execute the description, and that the gripper stays closed.
6. Do not add additional descriptions not shown above. Only use the bullet points given in the template.
7. If a bullet point is marked [optional], do NOT add it unless it's absolutely needed.
8. Use as few bullet points as possible. Be concise.
"""
)

prompt_coder = """
We have a description of a robot's motion and we want you to turn that into the corresponding program with following functions:
```
def set_l2_distance_reward(name_obj_A, name_obj_B, weight=1.0)
```
where name_obj_A and name_obj_B are selected from ['palm', 'block', 'drawer_handle', 'drawer_inside']. This term rewards the two objects for being close to each other.

```
def set_obj_position_reward(name_obj, x, y, z, weight=1.0)
```
where name_obj is selected from ['palm', 'block']. This term rewards the object for being at position (x, y, z).

```
def set_joint_fraction_reward(name_joint, fraction, weight=1.0)
```
where name_joint is 'drawer' and fraction is between 0 (closed) and 1 (fully open).

```
def reset_reward()
```
This function resets the reward to default values.

```
def execute_plan(duration=2)
```
This function sends the parameters to the robot and executes the plan for `duration` seconds, default to be 2.

Example answer code:
```
import numpy as np  # import numpy because we are using it below

reset_reward()  # This is a new task so reset reward; otherwise we don't need it
set_l2_distance_reward('palm', 'block')
set_obj_position_reward('block', 0.3, -0.2, 0.02, weight=2.0)

execute_plan(4)
```

Remember:
1. Always format the code in code blocks.
2. Do not invent new functions or classes. The only allowed functions you can call are the ones listed above. Do not leave unimplemented code blocks in your response.
3. The only allowed library is numpy. Do not import or use any other library.
4. If you are not sure what value to use, just use your best judge. Do not use None for anything.
5. Do not calculate the position or direction of any object (except for the ones provided above). Just use a number directly based on your best guess.
6. Remember to call reset_reward() for a new task and execute_plan at the end.
"""


def task_context(task: str) -> str:
  """Per-task state information appended to the user command."""
  return {
      "A": "The block is on the floor at (x=0.55, y=-0.22, z=0.02). The drawer is closed.",
      "B": "The block is on the floor at (x=0.30, y=-0.40, z=0.02). The drawer is closed.",
      "C": "The block is on top of the cabinet at (x=0.55, y=0.36, z=0.20). The drawer is closed.",
  }[task]
