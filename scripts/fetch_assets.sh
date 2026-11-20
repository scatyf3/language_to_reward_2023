#!/usr/bin/env bash
# Fetches the Franka Panda model from MuJoCo Menagerie into third_party/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
if [ ! -d third_party/mujoco_menagerie ]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
fi
git -C third_party/mujoco_menagerie sparse-checkout set franka_emika_panda
ls third_party/mujoco_menagerie/franka_emika_panda/mjx_panda.xml
