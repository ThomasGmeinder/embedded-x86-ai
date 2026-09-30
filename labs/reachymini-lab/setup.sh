#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

#
# setup.sh — one-time setup for a Reachy Mini mini-lab station.
#
# Run this ONCE on each fresh booth machine (physical station) after cloning the
# repo. It consolidates the "One-time setup (per station)" steps documented in
# README.md so all three identical stations can be provisioned the same way.
#
# The three lab steps are referred to as "Task 1..3"; "station" means one physical
# booth machine (AMD Strix Halo PC + one Reachy Mini).
#
# The script is idempotent: re-running it is safe and mostly fast (it checks
# before doing). It does NOT require sudo overall — sudo is used only for the
# apt step so the venv/pip artifacts stay owned by the normal user.
#
set -euo pipefail

# --- locate the repo root (this script's own directory) so CWD doesn't matter ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- options ---------------------------------------------------------------
SKIP_MODELS=0

usage() {
    cat <<'EOF'
Usage: ./setup.sh [OPTIONS]

One-time setup for a Reachy Mini mini-lab station. Run from the cloned repo
(the script cd's to its own directory, so it works from any CWD).

Options:
  --skip-models   Skip the slow Ollama model pulls (for quick re-runs).
  -h, --help      Show this help and exit.

Sections performed (mirrors README's "One-time setup (per station)"):
  1. System packages (apt)          5. Recorded-moves library (Task 1, needs HF_TOKEN)
  2. Python environment (venv)      6. GPU acceleration (ROCm) check
  3. Ollama + models                7. Reset baseline snapshot (.lab-baseline/)
  4. Piper voice models             Final: next-steps summary

After setup, the lab is primarily run from the notebook: from the repo root,
`source venv/bin/activate && jupyter lab`, open lab/lab.ipynb (venv kernel), and
run the Setup cell first. Terminal scripts (lab/emo_v*.py) are the fallback.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-models) SKIP_MODELS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "❌ Unknown option: $1" >&2; echo; usage; exit 1 ;;
    esac
done

# --- pretty output helpers -------------------------------------------------
section() {
    echo
    echo "=================================================================="
    echo "🤖 $1"
    echo "=================================================================="
}
ok()   { echo "✅ $*"; }
warn() { echo "⚠️  $*"; }
err()  { echo "❌ $*" >&2; }

have() { command -v "$1" >/dev/null 2>&1; }

# --- lab baseline (for ./reset.sh) -----------------------------------------
# Pristine copies of every attendee-editable lab file are snapshotted here so
# ./reset.sh can restore a clean slate between attendees: the primary attendee
# notebook, the three fallback task scripts, plus the LAB.md guide they have
# open. Keep this list in sync with LAB_FILES in reset.sh.
# Snapshots key off each file's basename, so these names must be unique.
LAB_BASELINE_DIR=".lab-baseline"
LAB_FILES=(lab/lab.ipynb lab/emo_v1.py lab/emo_v2.py lab/emo_v3.py lab/LAB.md)

# --- sudo handling ---------------------------------------------------------
# We only need sudo for the apt step. Resolve a runner up front so the rest of
# the script runs as the normal user (venv/pip owned by them, not root).
SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
    if have sudo; then
        SUDO="sudo"
    else
        SUDO="__NOSUDO__"
    fi
fi

# ==========================================================================
# 1. System packages (incl. camera/GStreamer for Task 3)
# ==========================================================================
install_system_packages() {
    section "1/7 System packages (apt)"

    if ! have apt-get; then
        warn "apt-get not found — this doesn't look like a Debian/Ubuntu system."
        warn "Install the equivalent packages manually (see README section 1) and re-run."
        return 0
    fi

    if [[ "$SUDO" == "__NOSUDO__" ]]; then
        err "sudo is not available and you are not root; cannot install apt packages."
        err "Run the apt install line from README section 1 manually, then re-run this script."
        return 0
    fi

    # Package list mirrors README section 1, plus v4l-utils.
    # Judgment call: README's Task 3 troubleshooting uses `v4l2-ctl` (from
    # v4l-utils) to list cameras, so we add it here for camera debugging.
    local pkgs=(
        python3 python3-venv python3-pip curl espeak ffmpeg
        libsndfile1 portaudio19-dev libcairo2-dev libgirepository1.0-dev
        python3-gi gir1.2-gst-plugins-base-1.0 libgstreamer1.0-0 gstreamer1.0-tools
        gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
        gstreamer1.0-libav
        v4l-utils
    )

    echo "Updating apt package index..."
    $SUDO apt-get update -y
    echo "Installing packages (already-present ones are skipped by apt)..."
    $SUDO apt-get install -y "${pkgs[@]}"
    ok "System packages installed."
}

# ==========================================================================
# 2. Python environment
# ==========================================================================
setup_python_env() {
    section "2/7 Python environment (venv)"

    if [[ ! -d venv ]]; then
        echo "Creating virtual environment in ./venv ..."
        python3 -m venv venv
        ok "Created venv/."
    else
        ok "venv/ already exists — reusing it."
    fi

    # Use the venv's interpreter directly; no need to 'source activate'.
    local py="$SCRIPT_DIR/venv/bin/python"

    echo "Upgrading pip..."
    "$py" -m pip install --upgrade pip

    # requirements.txt includes the notebook deps (jupyterlab + ipywidgets) used
    # by lab/lab.ipynb, alongside the core script-lab deps.
    echo "Installing requirements.txt (script lab + notebook deps) ..."
    "$py" -m pip install -r requirements.txt

    echo "Installing reachy-mini[mujoco] ..."
    "$py" -m pip install "reachy-mini[mujoco]"

    ok "Python environment ready."
}

# ==========================================================================
# 3. Ollama + models (LLM and vision model)
# ==========================================================================
OLLAMA_MODELS=(qwen3.5:0.8b qwen2.5vl:3b)

ensure_ollama_running() {
    # Prefer the systemd service if it exists; otherwise fall back to a
    # background `ollama serve`. Then wait for the API to answer.
    if have systemctl && systemctl list-unit-files 2>/dev/null | grep -q '^ollama\.service'; then
        $SUDO systemctl enable --now ollama >/dev/null 2>&1 || systemctl --user start ollama >/dev/null 2>&1 || true
    fi

    if curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
        ok "Ollama service is responding."
        return 0
    fi

    echo "Starting 'ollama serve' in the background..."
    nohup ollama serve >/tmp/ollama-setup.log 2>&1 &

    local i
    for i in $(seq 1 30); do
        if curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
            ok "Ollama is up."
            return 0
        fi
        sleep 1
    done

    warn "Ollama did not become ready within 30s (see /tmp/ollama-setup.log)."
    return 1
}

setup_ollama() {
    section "3/7 Ollama + models"

    if have ollama; then
        ok "Ollama already installed ($(ollama --version 2>/dev/null | head -n1))."
    else
        echo "Installing Ollama via the official script..."
        curl -fsSL https://ollama.com/install.sh | sh
        if have ollama; then
            ok "Ollama installed."
        else
            err "Ollama install did not produce an 'ollama' binary. Check network and re-run."
            return 1
        fi
    fi

    if [[ "$SKIP_MODELS" -eq 1 ]]; then
        warn "--skip-models set: skipping model pulls."
        warn "Run without --skip-models to fetch: ${OLLAMA_MODELS[*]}"
        return 0
    fi

    if ! ensure_ollama_running; then
        warn "Skipping model pulls because Ollama isn't reachable."
        warn "Start it manually ('ollama serve') and re-run this script."
        return 0
    fi

    local installed model
    installed="$(ollama list 2>/dev/null || true)"
    for model in "${OLLAMA_MODELS[@]}"; do
        if echo "$installed" | grep -q "^${model}[[:space:]]"; then
            ok "Model already present: $model"
        else
            echo "Pulling model: $model ..."
            if ollama pull "$model"; then
                ok "Pulled $model"
            else
                warn "Failed to pull $model — check network and re-run (safe to retry)."
            fi
        fi
    done
}

# ==========================================================================
# 4. Piper voice models
# ==========================================================================
# The default English voice is committed in models/. Verify it exists; only try
# to download if missing (never clobber existing files). Extra swap voices for
# Task 2 (amy, ryan) are fetched best-effort by setup_extra_piper_voices below.
PIPER_MODEL_NAME="en-us-blizzard_lessac-medium.onnx"
PIPER_CONFIG_NAME="en-us-blizzard_lessac-medium.onnx.json"
# Best-effort fallback source (Piper Voices on Hugging Face). The committed file
# uses a custom name, so this URL may not resolve; we fail gracefully if so.
PIPER_HF_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

setup_piper_voices() {
    section "4/7 Piper voice models"

    mkdir -p models
    local onnx="models/${PIPER_MODEL_NAME}"
    local cfg="models/${PIPER_CONFIG_NAME}"

    if [[ -s "$onnx" && -s "$cfg" ]]; then
        ok "Piper voice present: ${PIPER_MODEL_NAME} (+ .json). Not re-downloading."
        return 0
    fi

    warn "Default Piper voice missing (${PIPER_MODEL_NAME}). Attempting download..."
    if ! have curl; then
        err "curl not available; cannot download Piper voice."
        err "Manually place ${PIPER_MODEL_NAME} + .json into models/ (see README section 4)."
        return 0
    fi

    # Try the upstream lessac-medium voice as a fallback, saving under the name
    # the lab scripts expect. This is best-effort.
    local dl_ok=1
    if [[ ! -s "$onnx" ]]; then
        curl -fSL "${PIPER_HF_BASE}/en_US-lessac-medium.onnx" -o "$onnx" || dl_ok=0
    fi
    if [[ ! -s "$cfg" ]]; then
        curl -fSL "${PIPER_HF_BASE}/en_US-lessac-medium.onnx.json" -o "$cfg" || dl_ok=0
    fi

    if [[ "$dl_ok" -eq 1 && -s "$onnx" && -s "$cfg" ]]; then
        ok "Downloaded a Piper voice into models/."
    else
        rm -f "$onnx" "$cfg" 2>/dev/null || true
        err "Could not fetch a Piper voice automatically."
        err "Download ${PIPER_MODEL_NAME} + .onnx.json from Piper Voices"
        err "  (https://huggingface.co/rhasspy/piper-voices) into models/ and re-run."
    fi

    setup_extra_piper_voices
}

# Optional extra English voices the Task 2 "swap PIPER_MODEL" step offers. These
# are NOT committed (too large), so a fresh station downloads them here. Purely
# best-effort: if a download fails the lab still works with the default voice.
setup_extra_piper_voices() {
    if ! have curl; then
        return 0
    fi
    local base="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US"
    # name  ->  relative path under $base (dir/file stem)
    local names=("en_US-amy-medium" "en_US-ryan-medium")
    local paths=("amy/medium/en_US-amy-medium" "ryan/medium/en_US-ryan-medium")
    local i
    for i in "${!names[@]}"; do
        local name="${names[$i]}" rel="${paths[$i]}"
        local onnx="models/${name}.onnx" cfg="models/${name}.onnx.json"
        if [[ -s "$onnx" && -s "$cfg" ]]; then
            ok "Extra Piper voice present: ${name}."
            continue
        fi
        echo "    Fetching optional voice: ${name}..."
        local ok_dl=1
        curl -fsSL "${base}/${rel}.onnx"      -o "$onnx" || ok_dl=0
        curl -fsSL "${base}/${rel}.onnx.json" -o "$cfg"  || ok_dl=0
        if [[ "$ok_dl" -eq 1 && -s "$onnx" && -s "$cfg" ]]; then
            ok "Downloaded extra voice: ${name}."
        else
            rm -f "$onnx" "$cfg" 2>/dev/null || true
            warn "Skipped optional voice ${name} (download failed) — default voice still works."
        fi
    done
}

# ==========================================================================
# 5. Recorded-moves library (Task 1) — needs Hugging Face auth
# ==========================================================================
setup_moves_library() {
    section "5/7 Recorded-moves library (Task 1)"

    if [[ -z "${HF_TOKEN:-}" ]]; then
        warn "HF_TOKEN not set — skipping the Task 1 moves-library warm-up."
        warn "To cache the dances/emotions moves, set a token and re-run:"
        warn "    export HF_TOKEN=<your token>"
        warn "    export HF_HOME=\${HOME}/huggingface_cache"
        warn "    ./setup.sh"
        return 0
    fi

    export HF_HOME="${HF_HOME:-${HOME}/huggingface_cache}"
    mkdir -p "$HF_HOME"
    ok "HF_TOKEN detected; using HF_HOME=$HF_HOME"

    warn "Warming the moves cache requires the reachy-mini-daemon to be running."
    echo "Running utils/test_actions.py to download + cache the recorded moves..."
    local py="$SCRIPT_DIR/venv/bin/python"
    if [[ ! -x "$py" ]]; then
        warn "venv python not found; skipping (run section 2 first)."
        return 0
    fi
    if "$py" utils/test_actions.py; then
        ok "Recorded-moves library cached."
    else
        warn "test_actions.py did not complete (is reachy-mini-daemon running?)."
        warn "This is non-fatal — the library also downloads on first run of emo_v1.py."
    fi
}

# ==========================================================================
# 6. GPU acceleration (ROCm) — detect only, never install inline
# ==========================================================================
check_rocm() {
    section "6/7 GPU acceleration (ROCm)"

    local rocm_present=0
    if have rocminfo || compgen -G "/opt/rocm*" >/dev/null 2>&1; then
        rocm_present=1
    fi

    if [[ "$rocm_present" -eq 0 ]]; then
        warn "ROCm not detected (no /opt/rocm* and no rocminfo)."
        warn "The LLM/vision model will fall back to CPU (slower but functional)."
        warn "To enable GPU acceleration, follow install-rocm.md and re-run."
        return 0
    fi

    ok "ROCm appears to be installed."
    if have rocminfo; then
        if rocminfo 2>/dev/null | grep -q gfx; then
            echo "GPU(s) visible to ROCm:"
            rocminfo 2>/dev/null | grep -i gfx | sed 's/^/    /' | head -n 8 || true
        else
            warn "rocminfo ran but reported no gfx target — GPU may not be usable."
        fi
    fi

    echo
    echo "To confirm Ollama actually uses the GPU:"
    echo "    ollama ps            # after a query, PROCESSOR should read '100% GPU'"
    echo "    journalctl -u ollama | grep -i rocm"
    echo
    echo "If 'ollama ps' shows '100% CPU', set HSA_OVERRIDE_GFX_VERSION=11.0.0 in"
    echo "the ollama service environment and restart it (fallback; not needed on the"
    echo "verified Strix Halo setup)."
}

# ==========================================================================
# 7. Reset baseline snapshot (.lab-baseline/) — for ./reset.sh
# ==========================================================================
# Snapshot the pristine lab files so ./reset.sh can restore a clean slate
# between attendees. Idempotent AND safe: we only create the baseline if it does
# not already exist, so re-running setup.sh never overwrites a known-good
# baseline with a possibly-edited file. To intentionally refresh the golden copy
# after editing a lab file, use `./reset.sh --recapture` (see reset.sh).
# NOTE: LAB_FILES here MUST stay identical to LAB_FILES in reset.sh.
snapshot_lab_baseline() {
    section "7/7 Reset baseline snapshot (.lab-baseline/)"

    if [[ -d "$LAB_BASELINE_DIR" ]]; then
        ok "Baseline already exists at ${LAB_BASELINE_DIR}/ — leaving it untouched."
        warn "(Delete ${LAB_BASELINE_DIR}/ and re-run setup.sh to re-capture from the current files.)"
        return 0
    fi

    local missing=0 f
    for f in "${LAB_FILES[@]}"; do
        if [[ ! -f "$f" ]]; then
            warn "Expected lab file not found: $f"
            missing=1
        fi
    done
    if [[ "$missing" -eq 1 ]]; then
        err "Cannot capture a complete baseline — some lab files are missing (see above)."
        err "Skipping baseline creation; fix the checkout and re-run setup.sh."
        return 0
    fi

    echo "Creating ${LAB_BASELINE_DIR}/ from the current pristine lab files..."
    mkdir -p "$LAB_BASELINE_DIR"
    for f in "${LAB_FILES[@]}"; do
        cp "$f" "$LAB_BASELINE_DIR/"
        echo "    snapshot: $f"
    done
    ok "Baseline captured — run ./reset.sh between attendees to restore these files."
}

# ==========================================================================
# Final: next steps
# ==========================================================================
print_next_steps() {
    section "Setup complete — next steps"
    cat <<EOF
Prerequisites (both must be running before the lab):
    - Robot daemon (Terminal A, below), started with --no-media.
    - Ollama serving with both models pulled (qwen3.5:0.8b, qwen2.5vl:3b).

Start the robot daemon (Terminal A — leave running all day):
    source venv/bin/activate && reachy-mini-daemon --no-media
    (one-time serial access: sudo usermod -aG dialout \$USER, then re-login)
    (--no-media frees the camera for Task 3's vision task)

Run the lab — NOTEBOOK (primary), from the repo root (Terminal B):
    source venv/bin/activate && jupyter lab
    Then open lab/lab.ipynb, pick the venv kernel, and run the Setup cell first.
    Between attendees: run ./reset.sh, then File > Reload Notebook from Disk
    (the notebook's "Want a fresh start?" section walks attendees through this).
    (Optional companion: lab/explainer.ipynb — peek under the hood.)

Fallback — terminal scripts (if Jupyter has trouble):
    source venv/bin/activate
    python lab/emo_v1.py --chat          # Task 1 (cloud voice)
    python lab/emo_v2.py --chat          # Task 2 (100% offline)
    python lab/emo_v3.py --preview-web   # Task 3 (vision; http://localhost:8080)
    Then follow lab/LAB.md at the table.

Between attendees (revert their TRY-ME edits to the notebook + scripts):
    ./reset.sh          # restores the lab files from ${LAB_BASELINE_DIR}/

EOF
    if [[ -z "${HF_TOKEN:-}" ]]; then
        warn "Task 1 moves library was SKIPPED (no HF_TOKEN)."
        warn "Set HF_TOKEN and re-run ./setup.sh to cache it (or it downloads on first emo_v1.py run)."
    fi
    ok "Station provisioning finished."
}

# --- main ------------------------------------------------------------------
main() {
    section "Reachy Mini mini-lab — station setup"
    echo "Repo root: $SCRIPT_DIR"
    [[ "$SKIP_MODELS" -eq 1 ]] && warn "--skip-models is set (Ollama model pulls will be skipped)."

    install_system_packages
    setup_python_env
    setup_ollama
    setup_piper_voices
    setup_moves_library
    check_rocm
    snapshot_lab_baseline
    print_next_steps
}

main "$@"
