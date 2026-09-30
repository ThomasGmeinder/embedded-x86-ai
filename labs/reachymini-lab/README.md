# Reachy Mini Lab — Local AI Robot

![Reachy Mini](reachy_mini_hello.gif)

A **~10 minute hands-on lab** that takes a Reachy Mini desktop robot from a
cloud voice → 100% offline on local hardware → one that **sees and answers
questions** through its own camera. The LLM, vision model, and TTS all run
on-device via Ollama, GPU-accelerated by **ROCm on AMD hardware**.

**What you'll build across three tasks:**

1. **`lab/emo_v1.py`** — expressive robot with a **cloud** voice (Edge-TTS)
2. **`lab/emo_v2.py`** — same robot, **100% offline** (Piper-TTS + local LLM), snappier
3. **`lab/emo_v3.py`** — Reachy gets **eyes**: a local vision model answers your questions about what it sees

The step-by-step guide is **[`lab/LAB.md`](lab/LAB.md)**.

---

## Hardware

- **[Reachy Mini](https://huggingface.co/blog/reachy-mini)** robot over USB (`/dev/ttyACM0`), with speaker and camera built-in. Pick one up at the [Pollen Robotics store](https://store.pollen-robotics.com).
- A Linux machine running Ubuntu 24.04 (AMD Strix Halo recommended)
- Network for initial setup; Tasks 2 & 3 run fully offline after that

**No physical robot?** The lab also runs with the built-in MuJoCo simulator. Pass `--sim` to the daemon (see the [Run the Lab](#run-the-lab) section).

---

## Setup

Two ordered steps: install ROCm **first** (if you have an AMD GPU), then run
`./setup.sh`. Order matters: `setup.sh` starts Ollama, and Ollama only offloads
to the GPU if ROCm is already present.

### Step 0 — Install ROCm (AMD GPU acceleration)

Follow this if you have an AMD GPU (which you should on the target hardware).
ROCm is what lets Ollama offload the LLM and vision model to the AMD iGPU.

> No AMD GPU? Skip this. Ollama falls back to CPU automatically.

Follow **[`install-rocm.md`](install-rocm.md)** for the full instructions. Short
version for AMD Strix Halo / Ryzen AI machines (Ubuntu 24.04):

```bash
# 1) OEM kernel, then REBOOT
sudo apt update && sudo apt install -y linux-oem-24.04c
sudo reboot

# 2) AMD driver + ROCm, then REBOOT
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt install -y ./amdgpu-install_7.2.70200-1_all.deb
amdgpu-install -y --usecase=rocm --no-dkms
sudo usermod -a -G render,video $LOGNAME
sudo reboot

# 3) Verify
rocminfo | grep gfx          # expect gfx1151 on Strix Halo
```

### Step 1 — Run `./setup.sh`

```bash
git clone https://github.com/amd/embedded-x86-ai.git
cd embedded-x86-ai/labs/reachymini-lab
./setup.sh
```

`setup.sh` is idempotent (safe to re-run). Use `--help` for options, or
`--skip-models` to skip the slow Ollama model pulls on re-runs.

**What it does:**
1. `apt` installs system packages (Python, audio, camera/GStreamer stack)
2. Creates `venv/` and installs `requirements.txt` + `reachy-mini[mujoco]`
3. Installs Ollama and pulls the chat LLM (`qwen3.5:0.8b`) and vision model (`qwen2.5vl:3b`)
4. Verifies the committed Piper voice in `models/`
5. Pre-caches the Task 1 moves library (set `HF_TOKEN` first, see below)
6. Checks ROCm is detected (does **not** install it)
7. Snapshots lab files into `.lab-baseline/` for `./reset.sh`

### Notes

- **Piper voice is committed** in [`models/`](models/), no download needed on a normal clone. Add voices by dropping `.onnx` + matching `.onnx.json` files from [Piper Voices](https://huggingface.co/rhasspy/piper-voices) into `models/`.
- **Moves library** (Task 1 recorded animations) needs a Hugging Face token:
  ```bash
  export HF_TOKEN=<your token>
  ./setup.sh
  ```
- **Vision task** reads the camera directly via V4L2/`ffmpeg`, bypassing the daemon's media server. Start the daemon with **`--no-media`** or Task 3 will fail with `Device or resource busy`. The Arducam is auto-detected by name; override with `--camera-device /dev/videoN` (`v4l2-ctl --list-devices` lists them). Live preview is browser-based (`--preview-web`, `http://localhost:8080`).
- **GPU check:** once ROCm is installed, verify Ollama is using it:
  ```bash
  ollama ps    # after a query: PROCESSOR should read "100% GPU"
  ```

---

## Run the Lab

Use **two terminals**.

**Terminal A — start the daemon (leave running):**

No physical robot:
```bash
source venv/bin/activate
export PYGLFW_LIBRARY_VARIANT=x11   # required on X11 / Xwayland
reachy-mini-daemon --sim            # opens a MuJoCo 3D window
```

With a physical robot:
```bash
# One-time: grant serial port access
sudo usermod -aG dialout $USER && newgrp dialout

source venv/bin/activate
reachy-mini-daemon --no-media       # --no-media frees the camera for Task 3
```

**Terminal B — open the notebook:**

```bash
source venv/bin/activate && jupyter lab
```

Open **[`lab/lab.ipynb`](lab/lab.ipynb)**, pick the **venv kernel**, and run the
**Setup** cell. Follow **[`lab/LAB.md`](lab/LAB.md)** from there.

### Fallback: terminal scripts

If Jupyter has trouble, the same three tasks run as scripts:

```bash
python lab/emo_v1.py --chat          # Task 1 (cloud voice)
python lab/emo_v2.py --chat          # Task 2 (100% offline)
python lab/emo_v3.py --preview-web   # Task 3 (vision; http://localhost:8080)

# Task 3 simulation mode - default to built-in camera on laptop/computer 
# Run `v4l2-ctl --list-devices` to find your device
python lab/emo_v3.py --preview-web --camera-device /dev/video0
```

### Audio / volume

Open **Settings → Sound**, pick the Reachy Mini speaker as Output Device and set
the volume. Or use `alsamixer` in a terminal (F6 to select the device).

---

## Reset

To restore a clean slate (clears notebook outputs, resets edited lab files):

```bash
./reset.sh
```

Then in JupyterLab: **File → Reload Notebook from Disk** and re-run the Setup
cell. Use `./reset.sh --recapture` to make the current state the new baseline.

---

## Repo layout

```
lab/
  lab.ipynb        Primary notebook (all 3 tasks)
  _labkit.py       Notebook plumbing (chat bar, connect/shutdown, camera)
  emo_v1/2/3.py    Fallback terminal scripts
  LAB.md           Step-by-step guide + troubleshooting
models/            Piper voice models (committed)
utils/             ASR, Ollama, action/emotion test helpers
```

---

## Credits

Forked from [alexhegit/ReachyMiniChat](https://github.com/alexhegit/ReachyMiniChat)
(Apache-2.0). See [`LICENSE`](LICENSE).
