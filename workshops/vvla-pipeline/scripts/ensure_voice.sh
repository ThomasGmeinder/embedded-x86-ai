#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Everything the voice front ends in config/pipeline.yaml need: each voice
# model's GGUF and its audio projector, the Silero VAD model, and a
# llama-server built from a llama.cpp that knows the models.
#   d1-omni-2b     Liquid AI's d1-Omni-2B (a decision model) and its projector
#   gemma4-e4b-it  Gemma 4 E4B and its audio projector
# Both are put in place, whichever voice_intent names: the pipeline and the
# twin notebook use that one, the student lesson's intent parser and the tests
# use Gemma 4. A machine set up before the models were chosen, or with
# bootstrap.sh --skip-models, has none of them, and the microphone stays off
# without them. The pipeline and notebook launchers call this, like
# ensure_ros2_control.sh. It downloads what is missing, does nothing when
# everything is present, and fails with the reason when something cannot be put
# in place.
#
#   scripts/ensure_voice.sh                # the models, then the llama-server check
#   scripts/ensure_voice.sh --models-only  # bootstrap.sh: it builds llama.cpp itself
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv}"
PY="${VENV_DIR}/bin/python"
HF="${VENV_DIR}/bin/hf"
LLAMA_DIR="${ROOT}/third_party/llama.cpp"
SILERO_URL="https://github.com/snakers4/silero-vad/raw/v6.2.3/src/silero_vad/data/silero_vad.onnx"
SILERO_SHA256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

fail() { echo "ensure_voice: $*" >&2; exit 1; }

MODELS_ONLY=0
case "${1:-}" in
  "") ;;
  --models-only) MODELS_ONLY=1 ;;
  *) echo "usage: $0 [--models-only]" >&2; exit 2 ;;
esac

[[ -x "$PY" ]] || fail "no venv at ${VENV_DIR} - run ./bootstrap.sh first."

# config/pipeline.yaml is the authority for the voice front end and its paths:
# the model's own section is named by voice_intent, the Silero model is in the
# listening section both models share.
# Run from ROOT: python - puts the working directory first on sys.path, so a
# caller standing in another clone would otherwise read that clone's config.
# Every voice model the pipeline can be switched to (vla_pipeline.llm.voice.VOICE_INTENTS),
# the one voice_intent names first. One line per model: its name, then its three paths.
VOICE_ENV="$(cd "$ROOT" && "$PY" - <<'PY'
import shlex
from vla_pipeline.llm.voice import VOICE_INTENTS
from vla_pipeline.utils.config import load_config, resolve
cfg = load_config()
intent = str(cfg.get("voice_intent"))
print("VOICE_INTENT=%s" % shlex.quote(intent))
listening = cfg.get("listening") or {}
print("SILERO_OUT=%s" % shlex.quote(str(resolve(listening["vad_model"])) if listening.get("vad_model") else ""))
names = [intent] + [name for name in VOICE_INTENTS if name != intent]
print("VOICE_MODELS=%s" % shlex.quote(" ".join(names)))
for name in names:
    model = cfg.get(name) or {}
    for key, var in (("model_gguf", "GGUF"), ("mmproj", "MMPROJ"), ("server_bin", "SERVER")):
        print("%s_%s=%s" % (var, name.replace("-", "_"), shlex.quote(str(resolve(model[key])) if model.get(key) else "")))
PY
)" || fail "could not read config/pipeline.yaml"
eval "$VOICE_ENV"

# Per model: where its files come from, and the llama.cpp commit that added it.
model_source() {
  case "$1" in
    gemma4-e4b-it)
      MODEL_REPO="ggml-org/gemma-4-E4B-it-GGUF"
      MODEL_NAME="Gemma 4 audio"
      LLAMA_NEEDS="547765a93"   # the Gemma 4 audio encoder (mtmd, PR #21421)
      ;;
    d1-omni-2b)
      MODEL_REPO="LiquidAI/d1-Omni-2B-GGUF"
      MODEL_NAME="d1-Omni-2B"
      LLAMA_NEEDS="d05706b4f"   # model : add LiquidAI/d1-2B and LiquidAI/d1-omni-2B (PR #31275)
      ;;
    *) return 1 ;;
  esac
}

# The paths of model $1, from its own section of the config.
model_paths() {
  local gguf="GGUF_${1//-/_}" mmproj="MMPROJ_${1//-/_}" server="SERVER_${1//-/_}"
  MODEL_GGUF="${!gguf:-}"; MODEL_MMPROJ="${!mmproj:-}"; MODEL_SERVER="${!server:-}"
  [[ -n "$MODEL_GGUF" ]] || fail "config/pipeline.yaml ${1} has no model_gguf path."
  [[ -n "$MODEL_MMPROJ" ]] || fail "config/pipeline.yaml ${1} has no mmproj path."
  [[ -n "$MODEL_SERVER" ]] || fail "config/pipeline.yaml ${1} has no server_bin path."
}

model_source "$VOICE_INTENT" \
  || fail "voice_intent in config/pipeline.yaml is '${VOICE_INTENT}'; it must be gemma4-e4b-it or d1-omni-2b."
[[ -n "$SILERO_OUT" ]] || fail "config/pipeline.yaml listening has no vad_model path."

# The files of every model: the pipeline and the twin notebook use the one voice_intent
# names, the student lesson's intent parser and the tests use Gemma 4, and a change of
# voice_intent then needs no download.
for model in $VOICE_MODELS; do
  model_source "$model" || fail "no download source is known for the voice model '${model}'."
  model_paths "$model"
  for file in "$MODEL_GGUF" "$MODEL_MMPROJ"; do
    [[ -f "$file" ]] && continue
    [[ -x "$HF" ]] || fail "${file} is missing and there is no hf CLI in ${VENV_DIR} - run ./bootstrap.sh."
    echo "ensure_voice: downloading $(basename "$file") from ${MODEL_REPO}"
    mkdir -p "$(dirname "$file")"
    "$HF" download "$MODEL_REPO" "$(basename "$file")" --local-dir "$(dirname "$file")" \
      || fail "could not download $(basename "$file") from ${MODEL_REPO}."
    [[ -f "$file" ]] || fail "hf download finished but ${file} is not there."
  done
done

if ! { [[ -f "$SILERO_OUT" ]] && echo "${SILERO_SHA256}  ${SILERO_OUT}" | sha256sum -c --status; }; then
  echo "ensure_voice: downloading the Silero VAD model (v6.2.3)"
  mkdir -p "$(dirname "$SILERO_OUT")"
  curl --fail --location --show-error -o "${SILERO_OUT}.part" "$SILERO_URL" \
    || fail "could not download ${SILERO_URL}."
  echo "${SILERO_SHA256}  ${SILERO_OUT}.part" | sha256sum -c --status \
    || fail "the Silero VAD download does not match sha256 ${SILERO_SHA256}."
  mv "${SILERO_OUT}.part" "$SILERO_OUT"
fi

[[ "$MODELS_ONLY" -eq 1 ]] && exit 0

REBUILD="./bootstrap.sh --skip-apt --skip-models --skip-compile"
# Each model's llama-server (the same binary unless the config says otherwise) must be built
# from a llama.cpp that knows that model.
for model in $VOICE_MODELS; do
  model_source "$model"
  model_paths "$model"
  [[ -x "$MODEL_SERVER" ]] || fail "${MODEL_SERVER} is missing - run ${REBUILD} (it builds llama.cpp)."
  # The binary reports the commit it was built from, e.g.
  # "version: 0.6.0-dev (build 11548, commit 781dbc5ac)".
  BUILT="$("$MODEL_SERVER" --version 2>&1 | sed -n 's/.*commit \([0-9a-f]\{7,\}\).*/\1/p' | head -n1)"
  [[ -n "$BUILT" ]] || fail "${MODEL_SERVER} --version does not name its commit."
  [[ -d "${LLAMA_DIR}/.git" ]] || fail "${LLAMA_DIR} is not a git clone, so llama-server ${BUILT} cannot be checked for ${MODEL_NAME}."
  if ! git -C "$LLAMA_DIR" merge-base --is-ancestor "$LLAMA_NEEDS" "$BUILT" 2>/dev/null; then
    fail "llama-server was built from llama.cpp ${BUILT}, which predates ${MODEL_NAME} (${LLAMA_NEEDS}).
       Update and rebuild it: git -C third_party/llama.cpp pull --ff-only && ${REBUILD}"
  fi
done
