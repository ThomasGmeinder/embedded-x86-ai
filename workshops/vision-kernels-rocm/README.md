# Build Physical AI Kernels for Vision Applications — with AMD ROCm

Write custom HIP/ROCm perception kernels and watch a natural-language-driven
robot arm run *your* code in real time. The notebook is a guided tour of the
agent in `demo_v3_rocm.py`: the perception kernels are pulled out into editable
cells, you implement them, and the live agent runs your version on the AMD iGPU.

## What you build

Eight GPU perception kernels feeding a Genesis-simulated Franka arm:

- Depth normalize, normal colorize, segmentation colorize
- Color-lock target refinement
- Tactile reduction (closed-loop grasp gating)
- RGB→gray, Gaussian blur, Sobel edges

A Llama-3.2-3B planner turns commands like *"stack red on green"* into a plan;
your kernels drive perception; the arm executes in simulation.

## Tested environment

| Component | Version |
|---|---|
| Hardware | AMD Radeon™ 8060S (Strix Halo iGPU, `gfx1151`, RDNA 3.5) |
| OS | Ubuntu 24.04 |
| ROCm | 7.2 |
| PyTorch | 2.12.1+rocm7.2 |

## Prerequisites

- **ROCm 7.2** and **Python 3.10** (the ROCm torch wheels are cp310 builds).
- If ROCm isn't installed yet, `scripts/setup_rocm.sh` installs it on
  **Ubuntu 24.04** (ROCm 7.2 via `amdgpu-install`, then a reboot):

  ```bash
  bash scripts/setup_rocm.sh   # sudo + reboot required; Ubuntu 24.04 only
  ```

  On another distro, follow AMD's official ROCm install guide instead.

## Setup

**Quick path — run the installer** (venv + torch-rocm + Genesis, plus the
ROCm-matched `ld.lld` linker + ffmpeg that Genesis GPU JIT needs):

```bash
bash install.sh
```

<details>
<summary>Manual path (what the installer does)</summary>

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 1. PyTorch for ROCm — install FIRST so the GPU build is in place.
#    The notebook was captured on torch 2.12.1+rocm7.2.
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2

# 2. Everything else (torch is intentionally NOT in requirements.txt so this
#    step can't overwrite the GPU build with a CPU wheel).
pip install -r requirements.txt
```

Genesis JITs its physics kernels and links them with `ld.lld`; that linker must
match your ROCm LLVM or `gs.init()` fails with `unknown abi version`. Point
`/usr/local/bin/ld.lld` at ROCm's bundled copy and install ffmpeg:

```bash
sudo ln -sf "$(ls -d /opt/rocm*/lib/llvm/bin/ld.lld | sort -Vr | head -1)" /usr/local/bin/ld.lld
sudo apt-get install -y ffmpeg
```

</details>

Verify ROCm sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## The LLM (optional but recommended)

Llama-3.2-3B runs locally on the AMD iGPU via llama.cpp/Vulkan to turn
plain-English commands into plans. Without it, the planner falls back to regex
(basic phrasings only), so the LLM is recommended. Put the binary and model at
these paths and the notebook finds them — no env vars needed:

```bash
# Build deps + Vulkan toolchain
sudo apt install -y git cmake build-essential libvulkan-dev vulkan-tools glslc \
    spirv-headers glslang-tools

# Build llama-server (Vulkan)
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_VULKAN=ON
cmake --build ~/llama.cpp/build --config Release -j"$(nproc)"
mkdir -p ~/llama-vulkan/bin && cp ~/llama.cpp/build/bin/llama-server ~/llama-vulkan/bin/

# Model (hf CLI comes with huggingface_hub: pip install -U huggingface_hub)
hf download bartowski/Llama-3.2-3B-Instruct-GGUF \
    Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir ~/models
```

Other locations: set `LLAMA_BIN` and `LLAMA_MODEL`.

## Run

```bash
source .venv/bin/activate
HSA_OVERRIDE_GFX_VERSION=11.0.0 jupyter lab ROCm_Physical_AI_Agent_Workshop.ipynb
```

Work top to bottom: import the module, confirm ROCm is live, then fill in each
kernel cell. The final cells run the full agent and render a video.

To run faster (no live 3D window; the notebook still renders the video), add
`VK_HEADLESS=1`:

```bash
VK_HEADLESS=1 HSA_OVERRIDE_GFX_VERSION=11.0.0 jupyter lab ROCm_Physical_AI_Agent_Workshop.ipynb
```

## Files

| File | What it is |
|------|-----------|
| `ROCm_Physical_AI_Agent_Workshop.ipynb` | The lab — edit the kernel cells here |
| `demo_v3_rocm.py` | The agent engine (planner, executor, scene, kernels) |
| `install.sh` | Sets up the venv, deps, and Genesis GPU JIT linker |
| `scripts/setup_rocm.sh` | Installs ROCm 7.2 (Ubuntu 24.04) |
| `scripts/setup_env.sh` | Builds the venv + installs torch-rocm, Genesis, deps |
| `requirements.txt` | Python deps (torch-rocm installed separately, see above) |
