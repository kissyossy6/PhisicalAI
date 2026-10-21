#!/usr/bin/env bash
# Python env + Franka Panda model (MuJoCo Menagerie, Apache-2.0)
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
if [ ! -d third_party/mujoco_menagerie ]; then
  mkdir -p third_party
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
  git -C third_party/mujoco_menagerie sparse-checkout set franka_emika_panda
fi
[ -f .env ] || cp .env.example .env
echo "Done. Put your key in .env (GEMINI_API_KEY=...)"
