#!/usr/bin/env bash
# Sets up the Gemma 4 voice path of this workshop, in the workshop's own .venv, for the
# backend of configs/models.toml (or the one given with --backend):
#
#   1. the Python packages of so101_smolvla.voice (requests, scipy, sounddevice,
#      onnxruntime, silero-vad, huggingface_hub) and of the notebooks (ipykernel,
#      jupyterlab, nbconvert, ipywidgets), held to constraints.txt;
#   backend igpu (llama-server on the integrated graphics):
#   2. llama.cpp at the pinned commit, built with HIP for the graphics target of
#      configs/models.toml, into third_party/llama.cpp (not in git);
#   3. the two Gemma model files, into the Hugging Face cache (~/.cache/huggingface),
#      never into the workshop.
#   backend npu (FastFlowLM on the NPU):
#   2. FastFlowLM (flm) at or above the minimum version of configs/models.toml, and the
#      NPU's device node; both are system installs this script only checks and names;
#   3. the model of configs/models.toml in FastFlowLM's own store (~/.config/flm/models),
#      pulled by flm, never into the workshop.
#
# Every setting comes from configs/models.toml. Runs again without doing work that
# is done; ends with a message naming what it could not put in place.
#
#   scripts/setup_voice.sh                   # everything, for the backend of the settings
#   scripts/setup_voice.sh --check           # only say what is missing, change nothing
#   scripts/setup_voice.sh --backend igpu    # the other backend (igpu or npu); combines with --check
set -euo pipefail

cd "$(dirname "$0")/.."
workshop="$PWD"
settings="$workshop/configs/models.toml"
python="$workshop/.venv/bin/python"

check_only=0
backend=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) check_only=1; shift ;;
    --backend) [[ $# -ge 2 ]] || { echo "usage: $0 [--check] [--backend igpu|npu]" >&2; exit 2; }
               backend="$2"; shift 2 ;;
    *) echo "usage: $0 [--check] [--backend igpu|npu]" >&2; exit 2 ;;
  esac
done

die() { echo "setup_voice: $*" >&2; exit 1; }
setting() { # setting <dotted.key>: a value of configs/models.toml, e.g. setting command.npu.model
  /usr/bin/python3 -c '
import sys, tomllib
with open(sys.argv[1], "rb") as handle:
    value = tomllib.load(handle)
for part in sys.argv[2].split("."):
    value = value[part]
print(value)' "$settings" "$1"
}

[[ -x "$python" ]] || die "no Python at $python; create .venv as README.md, Environment, says"
[[ -n "$backend" ]] || backend="$(setting command.backend)"
case "$backend" in igpu|npu) ;; *) die "unknown backend '$backend'; igpu or npu" ;; esac
echo "backend: $backend"

missing=()

echo "== 1 of 3: the voice packages in .venv, and the notebooks' kernel and JupyterLab"
packages=(requests scipy sounddevice onnxruntime "silero-vad==6.2.3" huggingface_hub ipykernel jupyterlab nbconvert ipywidgets)
if "$python" -c "import requests, scipy, sounddevice, onnxruntime, silero_vad, huggingface_hub, ipykernel, jupyterlab, nbconvert, ipywidgets" 2>/dev/null; then
  echo "present"
elif [[ $check_only -eq 1 ]]; then
  missing+=("the voice packages in .venv (${packages[*]})")
else
  # constraints.txt holds the ROCm Torch wheels of README.md, Environment, in place.
  if [[ -f constraints.txt ]]; then export UV_OVERRIDE="$workshop/constraints.txt"; fi
  uv pip install --system-certs --python "$python" "${packages[@]}"
fi

if [[ "$backend" == igpu ]]; then
  [[ -d /opt/rocm ]] || die "no ROCm at /opt/rocm; llama-server is built with HIP against it"
  repository="$(setting command.igpu.build.repository)"
  commit="$(setting command.igpu.build.commit)"
  checkout="$workshop/$(setting command.igpu.build.checkout)"
  server_bin="$workshop/$(setting command.igpu.build.server_bin)"
  gfx_target="$(setting command.igpu.build.gfx_target)"
  gemma_repo="$(setting command.igpu.repo)"
  model_file="$(setting command.igpu.model)"
  mmproj_file="$(setting command.igpu.mmproj)"

  echo "== 2 of 3: llama-server, llama.cpp commit ${commit:0:9} for $gfx_target"
  built_commit() { # the commit a llama-server reports, or nothing
    [[ -x "$server_bin" ]] || return 0
    "$server_bin" --version 2>&1 | sed -n 's/.*commit \([0-9a-f]\{7,\}\).*/\1/p' | head -n1
  }
  built="$(built_commit)"
  if [[ -n "$built" && "$commit" == "$built"* ]]; then
    echo "present: $server_bin (commit $built)"
  elif [[ $check_only -eq 1 ]]; then
    missing+=("llama-server at $server_bin built from commit ${commit:0:9}${built:+ (found commit $built)}")
  else
    if [[ ! -d "$checkout/.git" ]]; then
      git clone --no-checkout "$repository" "$checkout"
    fi
    git -C "$checkout" fetch --quiet origin "$commit" 2>/dev/null || git -C "$checkout" fetch --quiet origin
    git -C "$checkout" checkout --quiet --detach "$commit"
    [[ "$(git -C "$checkout" rev-parse HEAD)" == "$commit" ]] || die "$checkout is not at commit $commit"
    # The flags the proven build used (HIP, one graphics target, release build), and
    # no web page in the server: at this commit the server embeds a web page that a
    # build downloads ready-made from a Hugging Face bucket; the copy for this commit
    # is gone and a newer one fails the commit's own check. The page is not used
    # here; without it the server embeds an empty table and serves its API as before.
    rm -rf "$checkout/build/tools/ui"  # a page downloaded by an earlier build attempt
    ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIPCXX=/opt/rocm/llvm/bin/clang++ \
      cmake -S "$checkout" -B "$checkout/build" \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS="$gfx_target" \
        -DGPU_TARGETS="$gfx_target" \
        -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
        -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_BUILD_UI=OFF \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$checkout/build" --config Release -j"$(nproc)" --target llama-server
    built="$(built_commit)"
    [[ -n "$built" && "$commit" == "$built"* ]] || die "$server_bin was built but reports commit '${built:-none}', not ${commit:0:9}"
    echo "built: $server_bin (commit $built)"
  fi

  echo "== 3 of 3: the Gemma files in the Hugging Face cache ($gemma_repo)"
  for file in "$model_file" "$mmproj_file"; do
    if path="$("$python" - "$gemma_repo" "$file" <<'PY' 2>/dev/null
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(sys.argv[1], sys.argv[2], local_files_only=True))
PY
    )"; then
      echo "present: $path"
    elif [[ $check_only -eq 1 ]]; then
      missing+=("$file of $gemma_repo in the Hugging Face cache")
    else
      echo "downloading $file"
      "$python" - "$gemma_repo" "$file" <<'PY'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(sys.argv[1], sys.argv[2]))
PY
    fi
  done

else
  flm_bin="$(setting command.npu.binary)"
  min_version="$(setting command.npu.min_version)"
  model_tag="$(setting command.npu.model)"
  releases="https://github.com/ROCm/FastFlowLM/releases"

  echo "== 2 of 3: FastFlowLM ($flm_bin) $min_version or later, and the NPU"
  if ! command -v "$flm_bin" >/dev/null 2>&1; then
    missing+=("FastFlowLM: no $flm_bin on the PATH; install the Ubuntu 24.04 .deb of $releases ($min_version or later)")
  else
    have="$("$flm_bin" version -j | /usr/bin/python3 -c 'import json, sys; print(json.load(sys.stdin)["version"])')"
    if [[ "$(printf '%s\n%s\n' "$min_version" "$have" | sort -V | head -n1)" == "$min_version" ]]; then
      echo "present: $flm_bin $have"
    else
      missing+=("FastFlowLM $have is installed; $min_version or later is needed ($releases)")
    fi
  fi
  if [[ -e /dev/accel/accel0 ]]; then
    echo "present: /dev/accel/accel0 (the amdxdna driver)"
  else
    missing+=("the NPU device /dev/accel/accel0: the amdxdna driver and XRT, as workshops/ryzen-ai-always-on/README.md installs them")
  fi

  echo "== 3 of 3: the FastFlowLM model $model_tag in FastFlowLM's store"
  installed() { # whether flm list shows the tag downloaded
    command -v "$flm_bin" >/dev/null 2>&1 || return 1
    "$flm_bin" list -j | /usr/bin/python3 -c '
import json, sys
tag = sys.argv[1]
models = json.load(sys.stdin)["models"]
sys.exit(0 if any(m.get("model") == tag and m.get("installed") for m in models) else 1)' "$model_tag"
  }
  if installed; then
    echo "present: $model_tag"
  elif [[ $check_only -eq 1 || ${#missing[@]} -gt 0 ]]; then
    missing+=("the FastFlowLM model $model_tag (flm pull $model_tag)")
  else
    "$flm_bin" pull "$model_tag"
    installed || die "flm pull $model_tag ended but flm list does not show it installed"
    echo "pulled: $model_tag"
  fi
fi

if [[ ${#missing[@]} -gt 0 ]]; then
  printf 'setup_voice: missing %s\n' "${missing[@]}" >&2
  exit 1
fi
echo "setup_voice: done"
