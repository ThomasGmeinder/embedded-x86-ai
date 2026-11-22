#!/usr/bin/env bash
# Sets up the Gemma 4 voice path of this workshop, in the workshop's own .venv:
#
#   1. the Python packages of so101_smolvla.voice (requests, scipy, sounddevice,
#      onnxruntime, silero-vad, huggingface_hub) and of the notebooks (ipykernel,
#      jupyterlab, nbconvert), held to constraints.txt;
#   2. llama.cpp at the pinned commit, built with HIP for the graphics target of
#      configs/voice.json, into third_party/llama.cpp (not in git);
#   3. the two Gemma model files, into the Hugging Face cache (~/.cache/huggingface),
#      never into the workshop.
#
# Every setting comes from configs/voice.json. Runs again without doing work that
# is done; ends with a message naming what it could not put in place.
#
#   scripts/setup_voice.sh            # everything
#   scripts/setup_voice.sh --check    # only say what is missing, change nothing
set -euo pipefail

cd "$(dirname "$0")/.."
workshop="$PWD"
settings="$workshop/configs/voice.json"
python="$workshop/.venv/bin/python"

check_only=0
case "${1:-}" in
  "") ;;
  --check) check_only=1 ;;
  *) echo "usage: $0 [--check]" >&2; exit 2 ;;
esac

die() { echo "setup_voice: $*" >&2; exit 1; }
setting() { # setting <section> <key>: a value of configs/voice.json
  /usr/bin/python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])' \
    "$settings" "$1" "$2"
}

[[ -x "$python" ]] || die "no Python at $python; create .venv as README.md, Environment, says"
[[ -d /opt/rocm ]] || die "no ROCm at /opt/rocm; llama-server is built with HIP against it"

repository="$(setting llama_cpp repository)"
commit="$(setting llama_cpp commit)"
checkout="$workshop/$(setting llama_cpp checkout)"
server_bin="$workshop/$(setting llama_cpp server_bin)"
gfx_target="$(setting llama_cpp gfx_target)"
gemma_repo="$(setting gemma repo)"
model_file="$(setting gemma model_file)"
mmproj_file="$(setting gemma mmproj_file)"

missing=()

echo "== 1 of 3: the voice packages in .venv, and the notebooks' kernel and JupyterLab"
packages=(requests scipy sounddevice onnxruntime "silero-vad==6.2.3" huggingface_hub ipykernel jupyterlab nbconvert)
if "$python" -c "import requests, scipy, sounddevice, onnxruntime, silero_vad, huggingface_hub, ipykernel, jupyterlab, nbconvert" 2>/dev/null; then
  echo "present"
elif [[ $check_only -eq 1 ]]; then
  missing+=("the voice packages in .venv (${packages[*]})")
else
  # constraints.txt holds the ROCm Torch wheels of README.md, Environment, in place.
  if [[ -f constraints.txt ]]; then export UV_OVERRIDE="$workshop/constraints.txt"; fi
  uv pip install --system-certs --python "$python" "${packages[@]}"
fi

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

if [[ ${#missing[@]} -gt 0 ]]; then
  printf 'setup_voice: missing %s\n' "${missing[@]}" >&2
  exit 1
fi
echo "setup_voice: done"
