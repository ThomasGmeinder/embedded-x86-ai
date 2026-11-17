#!/usr/bin/env bash
# Fine-tune the action expert. The system ROCm packages are not touched.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export MUJOCO_GL=egl
MICRO="${MICRO_BATCH:-8}"
ACCUM="${ACCUM_STEPS:-8}"
exec .venv/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --policy.n_action_steps=1 \
  --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --dataset.repo_id=local/so101_box \
  --dataset.root=scripted_testing/dataset \
  --dataset.video_backend=pyav \
  --output_dir=scripted_testing/train \
  --job_name=so101_smolvla \
  --steps=20000 \
  --policy.scheduler_decay_steps=20000 \
  --batch_size="$MICRO" \
  --accelerator.gradient_accumulation.steps="$ACCUM" \
  --save_checkpoint=true \
  --save_freq=20000 \
  --log_freq=50 \
  --num_workers=2
