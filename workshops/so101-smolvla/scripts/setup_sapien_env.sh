#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source_dir="${SAPIEN_SOURCE:-/tmp/Xbotics-SO101-Sim}"
commit="239b38e076387b11d5c0b0da29487edcbde5f905"

if [[ ! -d "$source_dir/.git" ]]; then
  git clone https://github.com/Xbotics-Embodied-AI-club/Xbotics-SO101-Sim.git "$source_dir"
fi

if [[ -n "$(git -C "$source_dir" status --short)" ]]; then
  echo "SAPIEN source has local changes; refusing to change revisions: $source_dir" >&2
  exit 1
fi

git -C "$source_dir" fetch origin "$commit"
git -C "$source_dir" checkout --detach "$commit"

if [[ ! -x .venv-sapien/bin/python ]]; then
  uv venv --python 3.12 .venv-sapien
fi
uv pip install --python .venv-sapien/bin/python -e "$source_dir"

.venv-sapien/bin/python -c \
  "import mani_skill, sapien, so101_sim; print(so101_sim.__file__)"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$commit"
