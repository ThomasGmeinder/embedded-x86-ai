#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# reset.sh — restore the lab to a clean slate between attendees (or re-capture the
# pristine baseline after an intentional lab edit).
#
# WHY THIS EXISTS: attendees edit the `# >>> TRY ME <<<` cells (notebook) and the
# fallback scripts. A button/cell running INSIDE the notebook can only reset the
# Python *variables* (see the "Reset to defaults" cell) — it CANNOT rewrite the
# visible edited cell text or clear outputs, because the kernel doesn't own the
# document. So the only true "clean slate" is restoring the files themselves,
# which is what this operator-run script does. (lab/lab.ipynb is NOT tracked by
# git, so there's no `git checkout` to fall back on — this snapshot IS the golden
# copy.)
#
# WHAT IT DOES:
#   Default (RESTORE): copies every attendee-editable lab file back from the
#     pristine snapshot in .lab-baseline/ (captured by setup.sh step 7 or by
#     `./reset.sh --recapture`), re-normalizes lab.ipynb (clear outputs +
#     validate), and clears stray Jupyter checkpoints.
#   --recapture (CAPTURE): re-snapshots the CURRENT lab files INTO .lab-baseline/
#     (overwriting). Use this once, after an intentional edit to a lab file, to
#     make the edited version the new golden copy.
#
# Keep LAB_FILES below in sync with snapshot_lab_baseline() in setup.sh.
#
# USAGE (from anywhere):
#   ./reset.sh              # restore the lab files to the pristine baseline
#   ./reset.sh --recapture  # overwrite the baseline with the CURRENT lab files
#   ./reset.sh --help       # show help
#
# After a RESTORE, in JupyterLab: File → Reload Notebook from Disk (or reopen the
# tab), and re-run the Setup cell.

set -euo pipefail

# Resolve the repo root as the directory this script lives in, so it works no
# matter what CWD the operator runs it from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

LAB_BASELINE_DIR=".lab-baseline"
# Attendee-editable files restored on reset. MUST match LAB_FILES in setup.sh
# (see snapshot_lab_baseline there). Snapshots key off each file's basename, so
# these names must be unique.
LAB_FILES=(lab/lab.ipynb lab/emo_v1.py lab/emo_v2.py lab/emo_v3.py lab/LAB.md)

VENV_PY="$REPO_ROOT/venv/bin/python"

usage() {
  cat <<'EOF'
Usage: ./reset.sh [OPTIONS]

Restore the mini-lab to a clean slate between attendees, or re-capture the
pristine baseline after an intentional lab edit. Runs from any CWD (it cd's to
its own directory).

Options:
  (no flag)          RESTORE: copy each lab file back from .lab-baseline/,
                     re-normalize lab.ipynb (clear outputs + validate), and clear
                     Jupyter checkpoints. Use this between attendees.
  --recapture        CAPTURE: re-snapshot the CURRENT lab files INTO
                     .lab-baseline/ (overwriting). Run this once after an
                     intentional edit so the edited version becomes the new
                     golden copy. (Alias: --capture.)
  -h, --help         Show this help and exit.
EOF
}

MODE="restore"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --recapture|--capture) MODE="recapture"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "❌ Unknown option: $1" >&2; echo; usage; exit 1 ;;
  esac
done

# Run lab/lab.ipynb through nbformat to guarantee it's pristine + valid: clear
# every code cell's outputs/execution_count, normalize, and validate. Best-effort
# — if the venv python isn't present we warn (not crash) so a restore still
# completes. Used by both RESTORE (on the restored copy) and CAPTURE (before
# snapshotting) so stale outputs never enter or leave the baseline.
normalize_notebook() {
  local nb="$1"
  if [[ ! -f "$nb" ]]; then
    return 0
  fi
  if [[ ! -x "$VENV_PY" ]]; then
    echo "⚠️ venv python not found ($VENV_PY) — skipping notebook normalize for $nb." >&2
    echo "   (Run ./setup.sh to create the venv.)" >&2
    return 0
  fi
  "$VENV_PY" - "$nb" <<'PY'
import sys
import nbformat

path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
for cell in nb.cells:
    if cell.cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
nbformat.validator.normalize(nb)
nbformat.validate(nb)
nbformat.write(nb, path)
print(f"✅ Normalized + validated {path}")
PY
}

# ---------------------------------------------------------------------------
# CAPTURE mode: overwrite the baseline with the CURRENT lab files.
# ---------------------------------------------------------------------------
if [[ "$MODE" == "recapture" ]]; then
  echo "Re-capturing the lab baseline from the CURRENT files into $LAB_BASELINE_DIR/ ..."
  missing=0
  for f in "${LAB_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "⚠️ Expected lab file not found: $f" >&2
      missing=1
    fi
  done
  if [[ "$missing" -eq 1 ]]; then
    echo "❌ Cannot capture a complete baseline — some lab files are missing (see above)." >&2
    exit 1
  fi

  # Normalize the live notebook first so the snapshot is guaranteed pristine.
  normalize_notebook "lab/lab.ipynb"

  mkdir -p "$LAB_BASELINE_DIR"
  for f in "${LAB_FILES[@]}"; do
    cp -f "$f" "$LAB_BASELINE_DIR/"
    echo "    snapshot: $f"
  done
  echo
  echo "✅ Baseline re-captured. ./reset.sh (no flag) now restores THIS version. 🤖"
  exit 0
fi

# ---------------------------------------------------------------------------
# RESTORE mode (default): restore lab files from the pristine baseline.
# ---------------------------------------------------------------------------
if [[ ! -d "$LAB_BASELINE_DIR" ]]; then
  echo "❌ Baseline not found: $LAB_BASELINE_DIR/" >&2
  echo "   Run ./setup.sh first (its step 7 snapshots the pristine lab files)," >&2
  echo "   or capture one now with: ./reset.sh --recapture" >&2
  exit 1
fi

# Guard: the baseline dir must contain at least one snapshot file.
if [[ -z "$(ls -A "$LAB_BASELINE_DIR" 2>/dev/null)" ]]; then
  echo "❌ Baseline is empty: $LAB_BASELINE_DIR/" >&2
  echo "   Capture one with: ./reset.sh --recapture   (or re-run ./setup.sh)" >&2
  exit 1
fi

# 1) Restore each lab file from its pristine snapshot (keyed by basename). Warn
#    (don't crash) on any missing snapshot so a partial baseline still restores
#    what it can.
restored=0
for f in "${LAB_FILES[@]}"; do
  snap="$LAB_BASELINE_DIR/$(basename "$f")"
  if [[ -f "$snap" ]]; then
    cp -f "$snap" "$f"
    echo "✅ Restored $f"
    restored=$((restored + 1))
  else
    echo "⚠️ No baseline snapshot for $f (skipping) — expected $snap" >&2
  fi
done

if [[ "$restored" -eq 0 ]]; then
  echo "❌ Nothing restored — the baseline appears empty. Re-create it with:" >&2
  echo "     ./reset.sh --recapture   (or: rm -rf $LAB_BASELINE_DIR && ./setup.sh)" >&2
  exit 1
fi

# 2) Re-normalize the restored notebook so it's guaranteed pristine + valid, even
#    if the snapshot happened to carry stale outputs.
normalize_notebook "lab/lab.ipynb"

# 3) Clear Jupyter checkpoints so a reopened tab doesn't resurrect old content.
if [[ -d "lab/.ipynb_checkpoints" ]]; then
  rm -rf "lab/.ipynb_checkpoints"
  echo "✅ Cleared Jupyter checkpoints (lab/.ipynb_checkpoints)."
fi

echo
echo "Next: in JupyterLab do File → Reload Notebook from Disk (or reopen the tab),"
echo "then run the Setup cell. Ready for the next attendee. 🤖"
