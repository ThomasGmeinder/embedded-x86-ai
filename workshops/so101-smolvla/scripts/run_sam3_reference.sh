#!/usr/bin/env bash
# Runs Python of .venv-sam3 inside the checkout of the fast SAM 3 reference
# (sam3-tracker-rocm), with the environment that the reference's container
# launcher gives its commands. No Docker.
#
#   scripts/run_sam3_reference.sh <arguments for python>
#   scripts/run_sam3_reference.sh --print <key>      prints one setting and exits
#
# The command runs with the checkout as working directory.
set -euo pipefail

workshop="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/sam3_reference_settings.sh
source "$workshop/scripts/sam3_reference_settings.sh"

if [[ "${1:-}" == "--print" ]]; then
  sam3_setting "$2"
  exit 0
fi

python="$workshop/.venv-sam3/bin/python"
if [[ ! -x "$python" ]]; then
  echo "SAM 3 environment is missing: $python (run scripts/setup_sam3_env.sh)" >&2
  exit 1
fi

checkout="$(sam3_setting checkout)"
commit="$(sam3_setting commit)"
if [[ ! -d "$checkout/.git" ]]; then
  echo "SAM 3 reference checkout is missing: $checkout" >&2
  exit 1
fi
if [[ "$(git -C "$checkout" rev-parse HEAD)" != "$commit" ]]; then
  echo "SAM 3 reference checkout is not at commit $commit: $checkout" >&2
  exit 1
fi

# What the reference's docker/rocm101/run.sh passes into its container (lines 133-140).
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
SAM3_ONNX_DIR="$(sam3_setting compiled_dir)"
export SAM3_ONNX_DIR
# Not in the reference's container, which has one model folder; here the
# worker (src/so101_smolvla/sam3_worker.py) reads it from this variable.
SAM3_MODEL_DIR="$(sam3_setting model_dir)"
export SAM3_MODEL_DIR
export SAM3_SOURCE_COMMIT="$commit"
# As run.sh:17-21 decides it: any tracked change or untracked file makes the source dirty.
export SAM3_SOURCE_DIRTY=0
if [[ -n "$(git -C "$checkout" status --porcelain --untracked-files=normal)" ]]; then
  export SAM3_SOURCE_DIRTY=1
fi
export PYTHONPATH="$checkout"

# The reference records which container image built its model files. There is no
# image here; the environment's package list stands in for it.
export SAM3_DOCKER_IMAGE_REF="venv:.venv-sam3 (no Docker; ROCm 10.1.0 from Python packages)"
packages="$(uv pip freeze --python "$python" | sha256sum)"
export SAM3_DOCKER_IMAGE_ID="venv-sha256:${packages%% *}"

# The graphics settings of the reference's container (Dockerfile:73-74) are the
# defaults of its tracker/rocm_env.py; they are read from there and set outright,
# as in the container, so that a value of the calling shell cannot replace them.
eval "$("$python" - "$checkout/tracker/rocm_env.py" <<'PY'
import runpy
import shlex
import sys

for name, value in runpy.run_path(sys.argv[1])["_DEFAULTS"].items():
    print(f"export {name}={shlex.quote(value)}")
PY
)"

# MIOpen (the convolution library PyTorch uses on AMD) times every candidate
# kernel the first time it meets a convolution shape. On this machine, with the
# ROCm 10.1 Python packages, that timer reads zero, every candidate is rejected
# ("Invalid elapsed time detected in EvaluateInvokers") and the convolution fails
# with "No suitable algorithm was found" (ROCm/MIOpen issue 3964). FAST picks a
# kernel from MIOpen's database without timing. The parts of the reference's live
# pipeline that still run in PyTorch (mask decoder, memory encoder) need it.
# Measured 2026-11-22: a plain fp16 conv2d fails without it and works with it.
export MIOPEN_FIND_MODE=FAST

# The reference tunes its kernels by timing and asks for its board's performance
# power mode first (SAM3_EC_POWER_MODE, export/build_text_prompt_mig.py:64-114).
# This machine has no such board interface; its counterpart is the ACPI platform
# profile. Its value is passed on as it is, so a build in another profile is refused.
profile="$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || true)"
case "$profile" in
  performance) export SAM3_EC_POWER_MODE=performance ;;
  balanced)    export SAM3_EC_POWER_MODE=balanced ;;
  low-power)   export SAM3_EC_POWER_MODE=quiet ;;
esac

cd "$checkout"
exec "$python" "$@"
