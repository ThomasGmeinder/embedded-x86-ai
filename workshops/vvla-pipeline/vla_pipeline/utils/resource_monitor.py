#!/usr/bin/env python3
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
GUI Resource Monitor - full edition
-----------------------------------
Sources:
  top -bn1                  -> CPU breakdown (us/sy/ni/id/wa), load avg, tasks,
                               RAM (used/free/buff/avail), swap, process list
  radeontop -d - -l 1       -> GPU pipeline stats (gpu/ee/vgt/ta/tc/sx/sh/spi/
                               smx/sc/pa/db/cb/cr), VRAM, GTT, mclk, sclk
  xrt-smi examine           -> NPU (power mode, columns, firmware, AIE contexts);
                               aie-partitions JSON -> command_completions counter
                               for inferences/s + duty-cycle % utilisation
  /sys/class/drm/card*      -> iGPU overall busy % (amdgpu gpu_busy_percent) —
                               the real engine-busy gauge (compute + graphics)
  /sys/class/hwmon          -> CPU temp (k10temp), GPU temp (amdgpu)

Run:  python -m vla_pipeline.utils.resource_monitor
      python -m vla_pipeline.utils.resource_monitor --headless   # no-display / SSH
      python -m vla_pipeline.utils.resource_monitor --once       # print one snapshot

From a Jupyter notebook, launch it in its own window from a cell at the top of
the notebook (re-running the cell reuses the existing process instead of
spawning a second window):

    import subprocess, sys, pathlib
    _here = pathlib.Path.cwd()
    REPO_ROOT = next(p for p in [_here, *_here.parents]
                     if (p / "vla_pipeline" / "utils" / "resource_monitor.py").exists())
    if globals().get("_MON_PROC") and _MON_PROC.poll() is None:
        print(f"monitor already running (PID {_MON_PROC.pid}) — run _MON_PROC.terminate() to close it")
    else:
        _MON_PROC = subprocess.Popen(
            [sys.executable, "-m", "vla_pipeline.utils.resource_monitor"],
            cwd=str(REPO_ROOT),
        )
        print(f"resource monitor opened in a separate window (PID {_MON_PROC.pid})")
        print("To close it:  _MON_PROC.terminate()")

Without a display (e.g. over SSH) the Tk GUI can't open, so the module falls
back to a terminal readout driven by the same collectors. Force it with
--headless, or capture a single snapshot with --once.

Note: radeontop may need root / `video` group. xrt-smi must be in PATH.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # headless box with no Tk at all
    tk = None
    ttk = None

UPDATE_INTERVAL = 1.0  # seconds between poll cycles
NPU_UTIL_INTERVAL = 1.0  # fast JSON completions poll (utilisation)
NPU_META_INTERVAL = 5.0  # slow text report poll (power mode, firmware, ...)
NPU_INFER_MS = 20.0  # assumed NPU time per inference for the duty estimate
HISTORY_LEN = 90
NUM_PROCS = 10  # rows in the process table

# ================================================================ collectors


def run(cmd, timeout):
    """Run cmd with a timeout, returning its stdout (empty string on any failure)."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        ).stdout
    except Exception:
        return ""


def read_top():
    """Parse `top -bn1` into a dict."""
    out = run(["top", "-bn1", "-w", "200"], 6)
    d = {}
    m = re.search(r"load average:\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)", out)
    if m:
        d["load"] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    m = re.search(r"up\s+(.+?),\s+\d+\s+user", out)
    if m:
        d["uptime"] = m.group(1).strip()
    m = re.search(
        r"Tasks:\s*(\d+)\s+total,\s*(\d+)\s+running,\s*(\d+)\s+sleeping,"
        r"\s*(\d+)\s+stopped,\s*(\d+)\s+zombie",
        out,
    )
    if m:
        d["tasks"] = tuple(int(x) for x in m.groups())
    m = re.search(
        r"%Cpu\(s\):\s*([\d.]+)\s*us,\s*([\d.]+)\s*sy,\s*([\d.]+)\s*ni,"
        r"\s*([\d.]+)\s*id,\s*([\d.]+)\s*wa",
        out,
    )
    if m:
        us, sy, ni, idle, wa = (float(x) for x in m.groups())
        d["cpu"] = {
            "us": us,
            "sy": sy,
            "ni": ni,
            "id": idle,
            "wa": wa,
            "total": round(100.0 - idle, 1),
        }
    m = re.search(
        r"MiB Mem\s*:\s*([\d.]+)\s*total,\s*([\d.]+)\s*free,"
        r"\s*([\d.]+)\s*used,\s*([\d.]+)\s*buff/cache",
        out,
    )
    if m:
        t, f, u, b = (float(x) / 1024.0 for x in m.groups())
        d["mem"] = {"total": t, "free": f, "used": u, "buff": b}
    m = re.search(
        r"MiB Swap\s*:\s*([\d.]+)\s*total,\s*([\d.]+)\s*free,"
        r"\s*([\d.]+)\s*used\.\s*([\d.]+)\s*avail Mem",
        out,
    )
    if m:
        st, sf, su, av = (float(x) / 1024.0 for x in m.groups())
        d["swap"] = {"total": st, "free": sf, "used": su}
        d["avail"] = av

    # process rows
    procs = []
    in_table = False
    for line in out.splitlines():
        if re.match(r"\s*PID\s+USER", line):
            in_table = True
            continue
        if in_table:
            parts = line.split(None, 11)
            if len(parts) == 12:
                pid, user, _, _, _, res, _, s, pcpu, pmem, t, cmd = parts
                try:
                    procs.append((pid, user, float(pcpu), float(pmem), res, t, cmd))
                except ValueError:
                    continue
    procs.sort(key=lambda p: p[2], reverse=True)
    d["procs"] = procs[:NUM_PROCS]
    return d


# radeontop dump fields -> display labels
GPU_PIPES = [
    ("gpu", "Graphics pipe"),
    ("ee", "Event Engine"),
    ("vgt", "Vertex Grp + Tess"),
    ("ta", "Texture Addresser"),
    ("tc", "Texture Cache"),
    ("sx", "Shader Export"),
    ("sh", "Seq. Instr Cache"),
    ("spi", "Shader Interpolator"),
    ("smx", "Shader Mem Exch"),
    ("sc", "Scan Converter"),
    ("pa", "Primitive Assembly"),
    ("db", "Depth Block"),
    ("cb", "Color Block"),
    ("cr", "Clip Rectangle"),
]


def read_radeontop():
    """Parse one radeontop dump line into pipeline %, memory, and clocks."""
    out = run(["radeontop", "-d", "-", "-l", "1"], 10)
    line = ""
    for ln in out.splitlines():
        if "gpu" in ln and "%" in ln:
            line = ln
    if not line:
        return None
    d = {"pipes": {}}
    for key, _ in GPU_PIPES:
        m = re.search(r"\b%s\s+([\d.]+)%%" % re.escape(key), line)
        if m:
            d["pipes"][key] = float(m.group(1))
    m = re.search(r"vram\s+([\d.]+)%\s+([\d.]+)mb", line)
    if m:
        d["vram"] = (float(m.group(1)), float(m.group(2)))
    m = re.search(r"gtt\s+([\d.]+)%\s+([\d.]+)mb", line)
    if m:
        d["gtt"] = (float(m.group(1)), float(m.group(2)))
    m = re.search(r"mclk\s+([\d.]+)%\s+([\d.]+)ghz", line)
    if m:
        d["mclk"] = (float(m.group(1)), float(m.group(2)))
    m = re.search(r"sclk\s+([\d.]+)%\s+([\d.]+)ghz", line)
    if m:
        d["sclk"] = (float(m.group(1)), float(m.group(2)))
    return d


def _find_xrt_smi():
    """Locate the xrt-smi binary on PATH or in its usual install locations."""
    for cand in ("xrt-smi", "/opt/xilinx/xrt/bin/xrt-smi", "/usr/bin/xrt-smi"):
        p = shutil.which(cand)
        if p:
            return p
    return None


_XRT_SMI = _find_xrt_smi()


def _parse_hw_contexts(out):
    """
    Fallback text parser for the HW Contexts / AIE Partitions table (older
    xrt-smi, or when the JSON report is unavailable). Each context spans a
    group of pipe-delimited lines between |---- separators:

        |690273   |1       |0 |0 |0 |Normal |
        |N/A      |Idle    |0 |0 |  |26     |
        |N/A      |2528 KB |  |  |  |1      |
        |         |        |  |  |  |2000   |

    Row 0: PID | Ctx ID | Submissions | Migrations | Err | Priority
    Row 1: Process Name | Status | Completions | Suspensions | - | GOPS
    Row 2: Memory Usage | Instr BO | - | - | - | FPS
    Row 3: - | - | - | - | - | Latency
    """
    lines = out.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if "HW Contexts" in ln or "AIE Partitions" in ln:
            start = i
            break
    if start is None:
        return []

    groups, current = [], []
    for ln in lines[start + 1 :]:
        s = ln.strip()
        if not s.startswith("|"):
            if current:
                groups.append(current)
                current = []
            if s and not s.startswith("|"):
                break  # left the table
            continue
        if re.match(r"\|[=\-]", s):  # |====| or |----| separator
            if current:
                groups.append(current)
                current = []
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        current.append(cells)
    if current:
        groups.append(current)

    ctxs = []
    for g in groups:

        def cell(r, c):
            """Return group cell (r, c), or empty string if out of range."""
            try:
                return g[r][c]
            except IndexError:
                return ""

        if not re.match(r"\d+$", cell(0, 0)):
            continue  # header group or junk
        ctxs.append(
            {
                "pid": cell(0, 0),
                "ctx_id": cell(0, 1),
                "priority": cell(0, 5),
                "pname": cell(1, 0),
                "status": cell(1, 1),
                "gops": cell(1, 5),
                "mem": cell(2, 0),
                "instr_bo": cell(2, 1),
                "fps": cell(2, 5),
                "latency": cell(3, 5),
            }
        )
    return ctxs


def _read_npu_json():
    """Run the structured aie-partitions JSON report; return the parsed dict.

    Unlike the free-text ``examine --report all`` table (whose column layout
    differs between xrt-smi versions and NPU generations), the JSON report is
    stable and carries the ``command_completions`` counter we use to derive
    utilisation. Returns None if xrt-smi is missing or the call fails.
    """
    if not _XRT_SMI:
        return None
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="npu_")
    os.close(fd)
    try:
        run(
            [
                _XRT_SMI,
                "examine",
                "-r",
                "aie-partitions",
                "-f",
                "JSON",
                "-o",
                tmp,
                "--force",
            ],
            20,
        )
        with open(tmp) as f:
            return json.load(f)
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _ctx_get(ctx, *names):
    """First non-empty value among ctx[name] for the given candidate keys."""
    for n in names:
        v = ctx.get(n)
        if v not in (None, ""):
            return str(v)
    return ""


def _parse_npu_json(data):
    """Extract (total_completions, ctxs) from the aie-partitions JSON.

    Sums ``command_completions`` across every hardware context and returns a
    per-context list. ``total`` is None when the expected structure isn't
    present (e.g. an older xrt-smi that doesn't emit this counter).
    """
    if not data:
        return None, []
    total = 0
    found = False
    ctxs = []
    for dev in data.get("devices", []):
        ap = dev.get("aie_partitions") or {}
        for part in ap.get("partitions", []):
            for ctx in part.get("hw_contexts", []):
                comp = ctx.get("command_completions")
                if isinstance(comp, bool):
                    comp = None
                elif isinstance(comp, (int, float)):
                    total += int(comp)
                    found = True
                elif isinstance(comp, str) and comp.strip().isdigit():
                    comp = int(comp)
                    total += comp
                    found = True
                else:
                    comp = None
                ctxs.append(
                    {
                        "pid": _ctx_get(ctx, "pid", "process_id"),
                        "ctx_id": _ctx_get(ctx, "id", "context_id", "ctx_id"),
                        "status": _ctx_get(ctx, "status"),
                        "priority": _ctx_get(ctx, "priority", "qos", "priority_level"),
                        "pname": _ctx_get(ctx, "process_name", "name"),
                        "gops": _ctx_get(ctx, "gops", "gflops"),
                        "mem": _ctx_get(ctx, "memory", "mem", "memory_usage"),
                        "instr_bo": _ctx_get(ctx, "instruction_buffer_size"),
                        "completions": str(comp) if comp is not None else "",
                        "fps": "",
                        "latency": "",
                    }
                )
    return (total if found else None), ctxs


class _NpuUtilTracker:
    """Derive inferences/s and a duty-cycle % from the completions counter.

    The NPU exposes no hardware utilisation gauge, but xrt-smi reports a
    monotonically increasing ``command_completions`` counter per context.
    Sampling it over time gives throughput (Δcount / Δt), and multiplying by
    the assumed per-inference time yields an occupancy estimate:

        inferences/s = Δcompletions / Δt
        duty%        = inferences/s * infer_ms/1000 * 100   (clamped to 100)
    """

    def __init__(self, infer_ms=NPU_INFER_MS):
        self._infer_ms = infer_ms
        self._last_total = None
        self._last_t = None

    def update(self, total):
        """Feed the latest completions total; return (ips, duty_pct)."""
        now = time.monotonic()
        if total is None:
            self._last_total = self._last_t = None
            return None, None
        ips = duty = None
        if self._last_total is not None and self._last_t is not None:
            dt = now - self._last_t
            dcount = total - self._last_total
            if dt > 0 and dcount >= 0:
                ips = dcount / dt
                duty = min(ips * self._infer_ms / 1000.0 * 100.0, 100.0)
        self._last_total = total
        self._last_t = now
        return ips, duty


_NPU_TRACKER = _NpuUtilTracker()


def npu_util_pct(fps, latency_us):
    """Legacy fallback estimate of NPU busy % (fps * latency / wall-clock).

    Only used when the JSON completions counter is unavailable and the text
    table still exposes fps/latency. Returns None if either input is
    missing/zero; clamps to 100%.
    """
    if not fps or not latency_us:
        return None
    return min(fps * latency_us / 1e6 * 100.0, 100.0)


def read_xrt_util():
    """Fast poll: JSON completions -> inferences/s + duty%, plus contexts."""
    total, ctxs = _parse_npu_json(_read_npu_json())
    ips, duty = _NPU_TRACKER.update(total)
    return {"ctxs": ctxs, "ips": ips, "util_pct": duty, "completions": total}


def read_xrt_meta():
    """Slow poll: power mode / columns / firmware / power / memory (text report)."""
    out = run([_XRT_SMI, "examine", "--report", "all"], 20)
    if not out:
        out = run([_XRT_SMI, "examine"], 15)
    d = {}
    if out:
        for key, pat in (
            ("power_mode", r"Power Mode\s*:\s*(\S+)"),
            ("columns", r"Total Columns\s*:\s*(\d+)"),
            ("firmware", r"NPU Firmware Version\s*:\s*(\S+)"),
            ("power", r"Estimated Power\s*:\s*(\S+)"),
            ("total_mem", r"Total Memory Usage\s*:\s*(\S+)"),
        ):
            m = re.search(pat, out)
            if m:
                d[key] = m.group(1).strip()
    d["_no_ctx_text"] = bool(out and "No hardware contexts" in out)
    d["_text_ctxs"] = _parse_hw_contexts(out) if out else []
    return d


def _npu_merge(meta, util):
    """Merge the slow metadata and fast utilisation polls into one dict.

    Contexts prefer the JSON report (util); the text-table parse is a
    fallback for older xrt-smi. State and a legacy util fallback are derived
    from whichever context list we end up with.
    """
    d = dict(meta or {})
    d.update(util or {})
    ctxs = (util or {}).get("ctxs") or (meta or {}).get("_text_ctxs") or []
    d["ctxs"] = ctxs

    if any(c.get("status", "").lower() == "active" for c in ctxs):
        d["state"] = "active"
    elif ctxs:
        d["state"] = "loaded"
    else:
        d["state"] = "idle"

    # If JSON completions weren't available, fall back to the old fps*latency
    # estimate from any text-table cells so we still show *something*.
    if d.get("util_pct") is None:
        best = None
        for c in ctxs:
            try:
                u = npu_util_pct(float(c.get("fps") or 0), float(c.get("latency") or 0))
            except (ValueError, TypeError):
                u = None
            if u is not None:
                best = u if best is None else max(best, u)
        d["util_pct"] = best

    d.pop("_no_ctx_text", None)
    d.pop("_text_ctxs", None)
    return d


def read_xrt():
    """Single-shot NPU snapshot (headless / --once): metadata + utilisation."""
    if not _XRT_SMI:
        return None
    return _npu_merge(read_xrt_meta(), read_xrt_util())


def _find_hwmon(names):
    """Find the hwmon directory whose 'name' file matches one of names."""
    for path in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            with open(path + "/name") as f:
                if f.read().strip() in names:
                    return path
        except OSError:
            continue
    return None


_CPU_HWMON = _find_hwmon({"k10temp", "coretemp", "zenpower"})
_GPU_HWMON = _find_hwmon({"amdgpu"})


def _read_temp(hwmon_dir):
    """Read the first temp*_input sensor under hwmon_dir, in degrees C."""
    if not hwmon_dir:
        return None
    for f in sorted(glob.glob(hwmon_dir + "/temp*_input")):
        try:
            with open(f) as fh:
                return round(int(fh.read().strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return None


def _find_gpu_busy():
    """Locate the amdgpu overall-busy sysfs node (authoritative iGPU gauge)."""
    for p in sorted(glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")):
        if os.access(p, os.R_OK):
            return p
    return None


_GPU_BUSY = _find_gpu_busy()


def read_gpu_busy():
    """Overall iGPU busy % from amdgpu ``gpu_busy_percent``.

    radeontop's ``gpu`` counter tracks only the graphics/render pipe and misses
    ROCm/HIP compute load — exactly what the VLA pipeline drives the iGPU with —
    so it reads near-idle under compute. This DRM sysfs node is the driver's true
    engine-busy gauge (compute + graphics). Returns None if the node is missing
    or unreadable.
    """
    if not _GPU_BUSY:
        return None
    try:
        with open(_GPU_BUSY) as fh:
            return min(float(fh.read().strip()), 100.0)
    except (OSError, ValueError):
        return None


# ====================================================================== GUI

BG, CARD, FG, DIM = "#16161f", "#222230", "#e6e6f2", "#8a8aa3"
C_CPU, C_RAM, C_GPU, C_NPU = "#4fc3f7", "#81c784", "#ff8a65", "#ce93d8"
MONO = ("Monospace", 9)

# Fall back to plain ``object`` as the base when Tk isn't importable, so the
# module still loads (and --headless / --once still work) on a box with no Tk.
_Canvas = tk.Canvas if tk is not None else object
_Frame = tk.Frame if tk is not None else object
_Tk = tk.Tk if tk is not None else object


class Spark(_Canvas):
    """Scrolling sparkline canvas for a 0-100 percentage history."""

    def __init__(self, parent, color, height=30, **kw):
        super().__init__(parent, bg=CARD, highlightthickness=0, height=height, **kw)
        self.color = color
        self.data = deque(maxlen=HISTORY_LEN)

    def push(self, v):
        """Append v to the history and redraw the sparkline."""
        self.data.append(0.0 if v is None else v)
        self.delete("all")
        w = self.winfo_width() or 220
        h = self.winfo_height() or 30
        if len(self.data) < 2:
            return
        step = w / (HISTORY_LEN - 1)
        pts = []
        for i, val in enumerate(self.data):
            x = w - (len(self.data) - 1 - i) * step
            y = h - (min(max(val, 0), 100) / 100.0) * (h - 4) - 2
            pts.extend((x, y))
        self.create_line(*pts, fill=self.color, width=2)


class MiniBar(_Frame):
    """label | bar | value — used for GPU pipeline rows and NPU inf/s."""

    def __init__(self, parent, label, color):
        super().__init__(parent, bg=CARD)
        self.color = color
        tk.Label(
            self, text=label, bg=CARD, fg=DIM, font=MONO, width=19, anchor="w"
        ).pack(side="left")
        self.canvas = tk.Canvas(
            self, width=110, height=10, bg="#101018", highlightthickness=0
        )
        self.canvas.pack(side="left", padx=4)
        self.val = tk.Label(
            self, text="--", bg=CARD, fg=FG, font=MONO, width=8, anchor="e"
        )
        self.val.pack(side="left")

    def set(self, pct, text=None):
        """Update the bar width and value label. pct is 0-100 (or None to clear)."""
        self.canvas.delete("all")
        if pct is not None:
            w = min(max(pct, 0), 100) / 100.0 * 110
            self.canvas.create_rectangle(0, 0, w, 10, fill=self.color, outline="")
            self.val.config(text=text or f"{pct:.2f}%")
        else:
            self.val.config(text=text or "N/A")


class UtilBar(_Frame):
    """Wide bar + bold value label — the primary utilisation row on each card.

    All cards use the same BAR_W so their percentage bars align in one column.
    """

    BAR_W = 210  # fixed pixel width shared across all cards

    def __init__(self, parent, color):
        super().__init__(parent, bg=CARD)
        self._color = color
        self.canvas = tk.Canvas(
            self, width=self.BAR_W, height=14, bg="#101018", highlightthickness=0
        )
        self.canvas.pack(side="left", padx=(0, 10))
        self.val = tk.Label(
            self,
            text="--",
            bg=CARD,
            fg=color,
            font=("TkDefaultFont", 14, "bold"),
            anchor="w",
        )
        self.val.pack(side="left")

    def set(self, pct, text):
        """Draw bar to pct (0-100, or None for empty) and display text."""
        self.canvas.delete("all")
        if pct is not None:
            w = min(max(pct, 0.0), 100.0) / 100.0 * self.BAR_W
            self.canvas.create_rectangle(0, 0, w, 14, fill=self._color, outline="")
        self.val.config(text=text)


def card(parent, title, color):
    """Build a titled card frame used to group related widgets."""
    f = tk.Frame(parent, bg=CARD, padx=12, pady=8)
    tk.Label(f, text=title, bg=CARD, fg=color, font=("TkDefaultFont", 10, "bold")).pack(
        anchor="w"
    )
    return f


class MonitorApp(_Tk):
    """Tkinter window that polls top/radeontop/xrt-smi and renders live CPU/RAM/GPU/NPU cards."""

    def __init__(self):
        super().__init__()
        self.title("Resource Monitor - Ryzen AI 9 HX 375 / Radeon 890M / NPU Strix")
        self.configure(bg=BG)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#1b1b26",
            fieldbackground="#1b1b26",
            foreground=FG,
            rowheight=20,
            font=MONO,
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background="#2c2c3c",
            foreground=DIM,
            font=("TkDefaultFont", 9, "bold"),
            relief="flat",
        )
        style.map("Treeview", background=[("selected", "#3a3a55")])

        # ---------- header
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=12, pady=(10, 4))
        self.clock = tk.Label(
            head, text="", bg=BG, fg=FG, font=("TkDefaultFont", 12, "bold")
        )
        self.clock.pack(side="left")
        self.head_info = tk.Label(head, text="", bg=BG, fg=DIM, font=MONO)
        self.head_info.pack(side="right")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=4)
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="y", anchor="n")
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0), anchor="n")

        # ---------- CPU card
        self.cpu_card = card(left, "CPU", C_CPU)
        self.cpu_card.pack(fill="x", pady=(0, 8))
        self.cpu_util = UtilBar(self.cpu_card, C_CPU)
        self.cpu_util.pack(anchor="w", pady=(2, 0))
        self.cpu_spark = Spark(self.cpu_card, C_CPU, width=230)
        self.cpu_spark.pack(fill="x", pady=3)
        self.cpu_detail = tk.Label(
            self.cpu_card, text="", bg=CARD, fg=DIM, font=MONO, justify="left"
        )
        self.cpu_detail.pack(anchor="w")

        # ---------- RAM card
        self.ram_card = card(left, "MEMORY", C_RAM)
        self.ram_card.pack(fill="x", pady=(0, 8))
        self.ram_util = UtilBar(self.ram_card, C_RAM)
        self.ram_util.pack(anchor="w", pady=(2, 0))
        self.ram_spark = Spark(self.ram_card, C_RAM, width=230)
        self.ram_spark.pack(fill="x", pady=3)
        self.ram_detail = tk.Label(
            self.ram_card, text="", bg=CARD, fg=DIM, font=MONO, justify="left"
        )
        self.ram_detail.pack(anchor="w")

        # ---------- NPU card
        self.npu_card = card(left, "NPU  (xrt-smi)", C_NPU)
        self.npu_card.pack(fill="x")
        # primary util bar — state text + activity bar
        self.npu_util = UtilBar(self.npu_card, C_NPU)
        self.npu_util.pack(anchor="w", pady=(2, 0))
        # inf/s mini-bar — same class as GPU pipeline bars so widths align
        self.npu_fps_bar = MiniBar(self.npu_card, "Inferences/s", C_NPU)
        self.npu_fps_bar.pack(anchor="w", pady=(4, 0))
        # Advanced toggle button
        self._npu_adv_open = False
        self.npu_adv_btn = tk.Button(
            self.npu_card,
            text="Advanced ▶",
            bg=CARD,
            fg=DIM,
            font=MONO,
            relief="flat",
            cursor="hand2",
            activebackground=CARD,
            activeforeground=FG,
            command=self._toggle_npu_adv,
        )
        self.npu_adv_btn.pack(anchor="w", pady=(6, 0))
        # Collapsible frame — not packed initially (hidden by default)
        self.npu_adv_frame = tk.Frame(self.npu_card, bg=CARD)
        self.npu_detail = tk.Label(
            self.npu_adv_frame, text="", bg=CARD, fg=DIM, font=MONO, justify="left"
        )
        self.npu_detail.pack(anchor="w")

        # ---------- GPU card (all radeontop pipes)
        self.gpu_card = card(right, "GPU  (amdgpu)", C_GPU)
        self.gpu_card.pack(fill="x")
        # primary iGPU utilisation — amdgpu gpu_busy_percent (compute + graphics),
        # matching the CPU/RAM/NPU cards. The radeontop pipe bars below are the
        # per-engine breakdown, not the headline number.
        self.gpu_util = UtilBar(self.gpu_card, C_GPU)
        self.gpu_util.pack(anchor="w", pady=(2, 0))
        self.gpu_spark = Spark(self.gpu_card, C_GPU, width=230)
        self.gpu_spark.pack(fill="x", pady=3)
        grid = tk.Frame(self.gpu_card, bg=CARD)
        grid.pack(fill="x", pady=4)
        self.pipe_bars = {}
        for i, (key, label) in enumerate(GPU_PIPES):
            bar = MiniBar(grid, label, C_GPU)
            bar.grid(row=i % 7, column=i // 7, sticky="w", padx=(0, 14), pady=1)
            self.pipe_bars[key] = bar
        mem = tk.Frame(self.gpu_card, bg=CARD)
        mem.pack(fill="x", pady=(6, 0))
        self.vram_bar = MiniBar(mem, "VRAM", "#ef5350")
        self.gtt_bar = MiniBar(mem, "GTT", "#ef5350")
        self.mclk_bar = MiniBar(mem, "Memory Clock", "#26c6da")
        self.sclk_bar = MiniBar(mem, "Shader Clock", "#26c6da")
        self.vram_bar.grid(row=0, column=0, sticky="w", padx=(0, 14), pady=1)
        self.gtt_bar.grid(row=1, column=0, sticky="w", padx=(0, 14), pady=1)
        self.mclk_bar.grid(row=0, column=1, sticky="w", pady=1)
        self.sclk_bar.grid(row=1, column=1, sticky="w", pady=1)
        self.gpu_foot = tk.Label(self.gpu_card, text="", bg=CARD, fg=DIM, font=MONO)
        self.gpu_foot.pack(anchor="w", pady=(4, 0))

        # ---------- process table
        proc_card = card(right, "TOP PROCESSES  (top)", "#ffd54f")
        proc_card.pack(fill="both", expand=True, pady=(8, 0))
        cols = ("pid", "user", "cpu", "mem", "res", "time", "cmd")
        self.tree = ttk.Treeview(
            proc_card, columns=cols, show="headings", height=NUM_PROCS
        )
        widths = {
            "pid": 70,
            "user": 70,
            "cpu": 60,
            "mem": 60,
            "res": 80,
            "time": 80,
            "cmd": 220,
        }
        heads = {
            "pid": "PID",
            "user": "USER",
            "cpu": "%CPU",
            "mem": "%MEM",
            "res": "RES",
            "time": "TIME+",
            "cmd": "COMMAND",
        }
        for c in cols:
            self.tree.heading(c, text=heads[c])
            anchor = "w" if c in ("user", "cmd") else "e"
            self.tree.column(c, width=widths[c], anchor=anchor, stretch=(c == "cmd"))
        self.tree.pack(fill="both", expand=True, pady=4)

        self.status = tk.Label(
            self, text="starting...", bg=BG, fg=DIM, font=("TkDefaultFont", 9)
        )
        self.status.pack(pady=(0, 8))

        # ---------- polling
        self._top = self._gpu = self._npu = None
        self._gpu_busy = None
        self._stop = threading.Event()
        threading.Thread(target=self._poll_fast, daemon=True).start()
        threading.Thread(target=self._poll_npu, daemon=True).start()
        self.after(300, self._refresh)
        self.protocol("WM_DELETE_WINDOW", self._close)

    # ---------------------------------------------------------------- toggle
    def _toggle_npu_adv(self):
        """Show/hide the NPU advanced detail frame."""
        if self._npu_adv_open:
            self.npu_adv_frame.pack_forget()
            self._npu_adv_open = False
            self.npu_adv_btn.config(text="Advanced ▶")
        else:
            self.npu_adv_frame.pack(fill="x", pady=(4, 0))
            self._npu_adv_open = True
            self.npu_adv_btn.config(text="Advanced ▼")

    def _poll_fast(self):
        """Background loop: refresh CPU/RAM/GPU stats every UPDATE_INTERVAL seconds."""
        while not self._stop.is_set():
            self._top = read_top()
            self._gpu = read_radeontop()
            self._gpu_busy = read_gpu_busy()
            time.sleep(UPDATE_INTERVAL)

    def _poll_npu(self):
        """Background loop: poll the completions counter fast (utilisation) and
        the slow text report (power mode / firmware / ...) less often."""
        meta = None
        last_meta = 0.0
        while not self._stop.is_set():
            if _XRT_SMI is None:
                self._npu = None
                time.sleep(NPU_META_INTERVAL)
                continue
            now = time.monotonic()
            if meta is None or (now - last_meta) >= NPU_META_INTERVAL:
                meta = read_xrt_meta()
                last_meta = now
            self._npu = _npu_merge(meta, read_xrt_util())
            time.sleep(NPU_UTIL_INTERVAL)

    # ---------------------------------------------------------------- UI tick
    def _refresh(self):
        """Redraw every card from the latest polled data; reschedules itself."""
        self.clock.config(text=time.strftime("%H:%M:%S"))
        t, g, n = self._top, self._gpu, self._npu

        if t:
            load = t.get("load")
            tasks = t.get("tasks")
            head = []
            if "uptime" in t:
                head.append("up " + t["uptime"])
            if load:
                head.append("load %.2f %.2f %.2f" % load)
            if tasks:
                head.append(f"{tasks[0]} tasks ({tasks[1]} run, " f"{tasks[4]} zombie)")
            self.head_info.config(text="  ·  ".join(head))

            cpu = t.get("cpu")
            if cpu:
                temp = _read_temp(_CPU_HWMON)
                label = f"{cpu['total']:.1f}%" + (
                    f"   {temp}°C" if temp is not None else ""
                )
                self.cpu_util.set(cpu["total"], label)
                self.cpu_spark.push(cpu["total"])
                self.cpu_detail.config(
                    text=f"us {cpu['us']:.1f}  sy {cpu['sy']:.1f}  "
                    f"ni {cpu['ni']:.1f}\nid {cpu['id']:.1f}  "
                    f"wa {cpu['wa']:.1f}"
                )

            mem, swap = t.get("mem"), t.get("swap")
            if mem:
                pct = mem["used"] / mem["total"] * 100.0
                self.ram_util.set(
                    pct, f"{mem['used']:.1f} / {mem['total']:.0f} GB  ({pct:.0f}%)"
                )
                self.ram_spark.push(pct)
                lines = [f"free {mem['free']:.1f}G  buff/cache {mem['buff']:.1f}G"]
                if "avail" in t:
                    lines[0] += f"  avail {t['avail']:.1f}G"
                if swap:
                    lines.append(
                        f"swap {swap['used']:.2f} / " f"{swap['total']:.1f}G used"
                    )
                self.ram_detail.config(text="\n".join(lines))

            self.tree.delete(*self.tree.get_children())
            for p in t.get("procs", []):
                pid, user, pcpu, pmem, res, tm, cmd = p
                self.tree.insert(
                    "",
                    "end",
                    values=(pid, user, f"{pcpu:.1f}", f"{pmem:.1f}", res, tm, cmd),
                )

        # Primary iGPU utilisation from amdgpu sysfs (true compute+graphics busy);
        # fall back to the radeontop graphics pipe only if sysfs is unavailable.
        busy = self._gpu_busy
        if busy is None and g:
            busy = g["pipes"].get("gpu")
        gtemp = _read_temp(_GPU_HWMON)
        if busy is not None:
            label = f"{busy:.1f}%" + (f"   {gtemp}°C" if gtemp is not None else "")
            self.gpu_util.set(busy, label)
        else:
            self.gpu_util.set(None, "N/A")
        self.gpu_spark.push(busy)

        if g:
            for key, _ in GPU_PIPES:
                self.pipe_bars[key].set(g["pipes"].get(key))
            if "vram" in g:
                pct, mb = g["vram"]
                self.vram_bar.set(pct, f"{mb:.0f}M {pct:.0f}%")
            if "gtt" in g:
                pct, mb = g["gtt"]
                self.gtt_bar.set(pct, f"{mb:.0f}M {pct:.0f}%")
            if "mclk" in g:
                pct, ghz = g["mclk"]
                self.mclk_bar.set(pct, f"{ghz:.2f}G {pct:.0f}%")
            if "sclk" in g:
                pct, ghz = g["sclk"]
                self.sclk_bar.set(pct, f"{ghz:.2f}G {pct:.0f}%")
            self.gpu_foot.config(text="pipe breakdown via radeontop")
        elif g is None:
            self.gpu_foot.config(
                text="radeontop returned no data (needs root / video group?)"
            )

        if n:
            state = n.get("state", "?")
            ctxs = n.get("ctxs", [])
            util_pct = n.get("util_pct")  # duty-cycle % from completions
            ips = n.get("ips")  # inferences/s from counter delta

            state_label = state.upper() + (f"  ·  {len(ctxs)} ctx" if ctxs else "")
            if util_pct is not None:
                state_label = f"{util_pct:.0f}%  ·  " + state_label
            self.npu_util.set(util_pct, state_label)
            self.npu_fps_bar.set(
                min(ips, 100.0) if ips is not None else None,
                f"{ips:.1f}/s" if ips is not None else "N/A",
            )

            # advanced detail (only rendered when the section is expanded)
            lines = []
            if util_pct is not None:
                lines.append(
                    f"utilisation: {util_pct:.1f}%  " f"(~{NPU_INFER_MS:.0f}ms/infer)"
                )
            if ips is not None:
                lines.append(f"inferences : {ips:.1f}/s")
            if n.get("completions") is not None:
                lines.append(f"completions: {n['completions']}")
            if "power_mode" in n:
                lines.append(f"power mode : {n['power_mode']}")
            if "columns" in n:
                lines.append(f"columns    : {n['columns']}")
            if "firmware" in n:
                lines.append(f"firmware   : {n['firmware']}")
            if n.get("total_mem") and n["total_mem"] != "N/A":
                lines.append(f"total mem  : {n['total_mem']}")
            if n.get("power") and n["power"] != "N/A":
                lines.append(f"est. power : {n['power']}")
            for c in ctxs:
                lines.append("─" * 26)
                lines.append(
                    f"pid {c.get('pid', '')}  ctx {c.get('ctx_id', '')}  "
                    f"{c.get('status', '').lower()}  "
                    f"{c.get('priority', '').lower()}".rstrip()
                )
                extra = []
                if c.get("mem") and c["mem"] != "N/A":
                    extra.append(f"mem {c['mem']}")
                if c.get("gops"):
                    extra.append(f"{c['gops']} GOPS")
                if c.get("completions"):
                    extra.append(f"{c['completions']} done")
                if c.get("fps"):
                    extra.append(f"{c['fps']} fps")
                if c.get("latency"):
                    extra.append(f"{c['latency']}us lat")
                if extra:
                    lines.append("  ".join(extra))
            self.npu_detail.config(text="\n".join(lines))

        elif n is None:
            self.npu_util.set(None, "N/A")
            self.npu_fps_bar.set(None, "N/A")
            self.npu_detail.config(text="xrt-smi not found / no output")

        self.status.config(
            text=f"top + radeontop every {UPDATE_INTERVAL:.0f}s  ·  "
            f"npu util every {NPU_UTIL_INTERVAL:.0f}s  ·  temps via hwmon"
        )
        self.after(500, self._refresh)

    def _close(self):
        """Stop the poller threads and destroy the window."""
        self._stop.set()
        self.destroy()


# ================================================================== headless


def _bar(pct, width=24):
    """ASCII gauge for a 0-100 percentage (None -> empty)."""
    pct = 0.0 if pct is None else max(0.0, min(100.0, pct))
    fill = int(round(pct / 100.0 * width))
    return "[" + "#" * fill + "-" * (width - fill) + f"] {pct:5.1f}%"


def render_console(top, gpu, npu, cpu_temp, gpu_temp, gpu_busy=None):
    """Format one poll cycle as a terminal readout, using the GUI's collectors."""
    lines = [f"Resource Monitor  ·  {time.strftime('%H:%M:%S')}", "=" * 52]

    cpu = (top or {}).get("cpu")
    if cpu:
        temp = f"  {cpu_temp}°C" if cpu_temp is not None else ""
        lines.append(f"CPU  {_bar(cpu['total'])}{temp}")
        lines.append(
            f"     us {cpu['us']:.0f}  sy {cpu['sy']:.0f}  "
            f"ni {cpu['ni']:.0f}  id {cpu['id']:.0f}  wa {cpu['wa']:.0f}"
        )
    load = (top or {}).get("load")
    if load:
        lines.append(
            f"     load {load[0]:.2f} {load[1]:.2f} {load[2]:.2f}"
            f"   up {(top or {}).get('uptime', '?')}"
        )

    mem = (top or {}).get("mem")
    if mem:
        used_pct = 100.0 * mem["used"] / mem["total"] if mem["total"] else 0.0
        lines.append(
            f"RAM  {_bar(used_pct)}  " f"{mem['used']:.1f}/{mem['total']:.1f} GiB"
        )

    # amdgpu gpu_busy_percent is the true busy gauge; radeontop pipe is fallback.
    busy = gpu_busy
    if busy is None and gpu:
        busy = gpu["pipes"].get("gpu")
    if busy is not None:
        temp = f"  {gpu_temp}°C" if gpu_temp is not None else ""
        lines.append(f"iGPU {_bar(busy)}{temp}")
        if gpu and "vram" in gpu:
            gtt = gpu.get("gtt", (0.0, 0.0))
            lines.append(f"     vram {gpu['vram'][1]:.0f} MB   gtt {gtt[1]:.0f} MB")
    else:
        lines.append("iGPU [utilisation unavailable]")

    if npu:
        # duty-cycle % derived from the command_completions counter delta
        util = npu.get("util_pct")
        ips = npu.get("ips")
        if util is not None:
            lines.append(f"NPU  {_bar(util)}")
        else:
            lines.append("NPU  [util N/A]")
        head = []
        if ips is not None:
            head.append(f"{ips:.1f} inf/s")
        if npu.get("completions") is not None:
            head.append(f"{npu['completions']} done")
        tail = ("  " + "  ".join(head)) if head else ""
        lines.append(
            f"     state={npu.get('state', '?')}  "
            f"power_mode={npu.get('power_mode', '?')}  "
            f"est_power={npu.get('power', 'N/A')}{tail}"
        )
        for c in npu.get("ctxs", []):
            extra = []
            if c.get("completions"):
                extra.append(f"{c['completions']} done")
            if c.get("gops"):
                extra.append(f"{c['gops']} GOPS")
            if c.get("mem") and c["mem"] != "N/A":
                extra.append(f"mem {c['mem']}")
            if c.get("fps"):
                extra.append(f"{c['fps']} fps")
            tail = ("  " + "  ".join(extra)) if extra else ""
            lines.append(
                f"     pid {c.get('pid', '')} ctx {c.get('ctx_id', '')} "
                f"{c.get('status', '').lower()}{tail}"
            )
    else:
        lines.append("NPU  [xrt-smi not found]")

    return "\n".join(lines)


def run_headless(interval=UPDATE_INTERVAL, once=False):
    """Terminal readout loop sharing the GUI's collectors. Ctrl-C to quit."""
    try:
        while True:
            out = render_console(
                read_top(),
                read_radeontop(),
                read_xrt(),
                _read_temp(_CPU_HWMON),
                _read_temp(_GPU_HWMON),
                read_gpu_busy(),
            )
            if once:
                print(out)
                return
            sys.stdout.write("\033[2J\033[H" + out + "\n")  # clear + home
            sys.stdout.flush()
            time.sleep(max(0.2, interval))
    except KeyboardInterrupt:
        pass


def main(argv=None):
    """CLI entry point: launch the Tk GUI, or the terminal readout with --headless/--once."""
    parser = argparse.ArgumentParser(
        description="AMD APU resource monitor (CPU / RAM / iGPU / NPU)."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="force the terminal readout instead of the Tk GUI",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="print a single snapshot and exit (implies --headless)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=UPDATE_INTERVAL,
        help="seconds between refreshes in headless mode",
    )
    args = parser.parse_args(argv)

    want_headless = args.headless or args.once or tk is None
    # No X display (typical over SSH) -> the Tk GUI can't open, so don't try.
    if (
        not want_headless
        and sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
    ):
        want_headless = True

    if want_headless:
        run_headless(interval=args.interval, once=args.once)
        return

    try:
        MonitorApp().mainloop()
    except tk.TclError as exc:  # display vanished / Tk failed to init
        print(
            f"GUI unavailable ({exc}); falling back to terminal readout.",
            file=sys.stderr,
        )
        run_headless(interval=args.interval, once=args.once)


if __name__ == "__main__":
    main()
