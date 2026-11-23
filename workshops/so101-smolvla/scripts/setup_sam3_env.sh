#!/usr/bin/env bash
# Builds .venv-sam3, the Python environment of the fast SAM 3 reference
# (sam3-tracker-rocm: MIGraphX, AMD's graph compiler, behind ONNX Runtime).
# No Docker. The reference's container takes ROCm 10.1 (AMD's compute stack)
# from system packages; here it comes from Python packages, as in .venv.
#
# Sources, none of them copied into this script:
#   configs/models.toml, section [outlines]         where the checkout is, and its commit
#   <checkout>/docker/rocm101/build.sh              wheel addresses and SHA256 checksums
#   <checkout>/docker/rocm101/runtime-constraints.txt   versions of the other packages
#   <checkout>/requirements.txt                     the reference's package list
#   README.md, "Environment"                        the ROCm and Torch install command
set -euo pipefail

cd "$(dirname "$0")/.."
workshop="$PWD"
# shellcheck source=scripts/sam3_reference_settings.sh
source "$workshop/scripts/sam3_reference_settings.sh"

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
if [[ -n "$(git -C "$checkout" status --short)" ]]; then
  echo "SAM 3 reference checkout has local changes: $checkout" >&2
  exit 1
fi

if [[ ! -x .venv-sam3/bin/python ]]; then
  uv venv --python 3.12 .venv-sam3
fi
python=.venv-sam3/bin/python

echo "== 1 of 3: ROCm 10.1.0 and Torch as Python packages (README.md, Environment)"
# torchvision's device package is named here as well: its version 0.29.0a0 counts
# as a pre-release, and uv accepts a pre-release only for a package named directly.
uv pip install --system-certs --python "$python" \
  --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "rocm[libraries,device-gfx1151]==10.1.0" \
  "torch[device-gfx1151]==2.14.0+rocm10.1.0" \
  "torchvision[device-gfx1151]==0.29.0a0+rocm10.1.0" \
  "amd-torchvision-device-gfx1151==0.29.0a0+rocm10.1.0"

echo "== 2 of 3: the reference's MIGraphX and ONNX Runtime wheels, checksums checked"
# The reference's own download cache (build.sh, variable CACHE).
cache="${SAM3_BINARY_CACHE:-$HOME/.cache/sam3-runtime-binaries/$(< "$checkout/VERSION")}"
mkdir -p "$cache"
# Defines the address variables and the array WHEELS exactly as build.sh does.
eval "$(sed -n '/^TORCH_BASE=/,/^)$/p' "$checkout/docker/rocm101/build.sh")"
wheels=()
for row in "${WHEELS[@]}"; do
  read -r expected url name <<<"$row"
  case "$name" in
    migraphx*|onnxruntime*) ;;
    *) continue ;;   # Torch and its device packages came with step 1
  esac
  file="$cache/$name"
  if [[ ! -f "$file" ]]; then
    curl --fail --location --output "$file.part" -- "$url"
    mv -- "$file.part" "$file"
  fi
  actual="$(sha256sum -- "$file")"
  actual="${actual%% *}"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch: $file" >&2
    exit 1
  fi
  echo "checksum ok: $name"
  wheels+=("$file")
done
if [[ "${#wheels[@]}" -ne 4 ]]; then
  echo "expected 4 MIGraphX and ONNX Runtime wheels in build.sh, found ${#wheels[@]}" >&2
  exit 1
fi
uv pip install --python "$python" --no-deps "${wheels[@]}"

echo "== 3 of 3: the reference's other packages, held to its constraints"
# Package names are those of the reference's Dockerfile (docker/rocm101/Dockerfile:63-65);
# every version comes from runtime-constraints.txt, except setuptools, which the
# Dockerfile pins itself.
uv pip install --system-certs --python "$python" --only-binary :all: \
  --constraints "$checkout/docker/rocm101/runtime-constraints.txt" \
  transformers safetensors opencv-python networkx jinja2 filelock fsspec sympy \
  "setuptools==80.9.0" \
  -r "$checkout/requirements.txt"

echo "== installed versions"
"$python" - <<'PY'
import importlib.metadata as m

for name in (
    "rocm", "torch", "amd-torch-device-gfx1151", "amd-torch-device-gfx115x",
    "torchvision", "triton", "migraphx", "migraphx-libs",
    "onnxruntime", "onnxruntime-ep-migraphx",
    "numpy", "transformers", "opencv-python", "onnx",
):
    print(f"{name}=={m.version(name)}")
PY
echo "setup_sam3_env: done"
