"""L2R manipulation prompts, copied from Yu et al. 2023, Appendix A.5 iv-v.

Text is verbatim from the paper (arXiv:2306.08647v2, pp. 18-19) except for the
object lists, which are replaced by the objects in our Franka scene:

  paper                                         here
  apple, banana, box, bowl                      block, target
  drawer_handle, drawer_center, rest_position   (same)
  faucet_handle, faucet                         (removed: no faucet in the scene)

Rule 5 of the Motion Descriptor ("The bowl is large enough ...") is replaced by
the equivalent fact about our scene. Everything else, including the paper's
own typos (e.g. set_obj_z_position_reward "encourages the orientation ... to be
close to the height", and rule 6 asking for "[start of plan]" while the
template says "[start of description]"), is kept as in the paper.

Unlike our project prompts (l2rvf/llm/prompts.py), the palm line is mandatory,
no weights are exposed to the LLM (Table 4 default weights are used), and no
scene coordinates are given.
"""

PALM_CHOICES = "block, target, drawer_handle, drawer_center, rest_position"
OBJ1_CHOICES = "block, target, drawer_handle, drawer_center"
OBJ2_CHOICES = "block, target, drawer_handle, drawer_center, nothing"
CODER_NAMES = '["palm", "block", "target", "drawer_handle", "drawer_center", "rest_position"]'

prompt_thinker = f"""
We have a dexterous manipulator and we want you to help plan how it should move to perform tasks
using the following template:
[start of description]
To perform this task, the manipulator's palm should move close to {{CHOICE: {PALM_CHOICES}}}.
object1={{CHOICE: {OBJ1_CHOICES}}} should be close to object2={{CHOICE: {OBJ2_CHOICES}}}.
[optional] object1 needs to be rotated by [NUM: 0.0] degrees along x axis.
[optional] object2 needs to be rotated by [NUM: 0.0] degrees along x axis.
[optional] object1 needs to be lifted to a height of [NUM: 0.0]m at the end.
[optional] object2 needs to be lifted to a height of [NUM: 0.0]m at the end.
[optional] object3={{CHOICE: drawer}} needs to be {{CHOICE: open, closed}}.
[end of description]
Rules:
1. If you see phrases like [NUM: default_value], replace the entire phrase with a numerical value.
2. If you see phrases like {{CHOICE: choice1, choice2, ...}}, it means you should replace the entire
phrase with one of the choices listed.
3. If you see [optional], it means you only add that line if necessary for the task, otherwise remove that
line.
4. The environment contains block, target, drawer_handle. Do not invent
new objects not listed here.
5. The target is a flat marker on the floor; objects can be placed on it.
6. I will tell you a behavior/skill/task that I want the manipulator to perform and you will provide the
full plan, even if you may only need to change a few lines. Always start the description with [start of
plan] and end it with [end of plan].
7. You can assume that the robot is capable of doing anything, even for the most challenging task.
8. Your plan should be as close to the provided template as possible. Do not include additional details.
"""

prompt_coder = f"""
We have a plan of a robot arm with palm to manipulate objects and we want you to turn that into the
corresponding program with following functions:

```
def set_l2_distance_reward(name_obj_A, name_obj_B)
```
where name_obj_A and name_obj_B are selected from {CODER_NAMES}. This term sets a reward for
minimizing l2 distance between name_obj_A and name_obj_B so they get closer to each other.
rest_position is the default position for the palm when it's holding in the air.

```
def set_obj_orientation_reward(name_obj, x_axis_rotation_radians)
```
this term encourages the orientation of name_obj to be close to the target (specified by
x_axis_rotation_radians).

```
def execute_plan(duration=2)
```
This function sends the parameters to the robot and execute the plan for "duration" seconds, default to
be 2.

```
def set_joint_fraction_reward(name_joint, fraction)
```
This function sets the joint to a certain value between 0 and 1. 0 means close and 1 means open.
name_joint needs to be select from ['drawer']

```
def set_obj_z_position_reward(name_obj, z_height)
```
this term encourages the orientation of name_obj to be close to the height (specified by z_height).

```
def reset_reward()
```
This function resets the reward to default values.

Example plan: To perform this task, the manipulator's palm should move close to object1=block.
object1 should be close to object2=target. object2 needs to be rotated by 30 degrees along x axis.
object2 needs to be lifted to a height of 1.0.
This is the first plan for a new task.
Example answer code:
```
import numpy as np

reset_reward()
# This is a new task so reset reward; otherwise we don't need it

set_l2_distance_reward("palm", "block")
set_l2_distance_reward("block", "target")
set_obj_orientation_reward("target", np.deg2rad(30))
set_obj_z_position_reward("target", 1.0)

execute_plan(4)
```

Remember:
1. Always format the code in code blocks. In your response execute_plan should be called exactly once
at the end.
2. Do not invent new functions or classes. The only allowed functions you can call are the ones listed
above. Do not leave unimplemented code blocks in your response.
3. The only allowed library is numpy. Do not import or use any other library.
4. If you are not sure what value to use, just use your best judge. Do not use None for anything.
5. Do not calculate the position or direction of any object (except for the ones provided above). Just
use a number directly based on your best guess.
6. You do not need to make the robot do extra things not mentioned in the plan such as stopping the
robot.
"""
