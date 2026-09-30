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

"""In-process resource sampling (Linux / Strix Halo).

Metrics:
  - CPU utilization (% across all cores)         via psutil
  - RAM utilization (% of total)                 via psutil
  - iGPU compute utilization (%)                 amdgpu gpu_busy_percent
  - NPU utilization (%)                          pipeline-fed duty cycle (NPU_ACTIVITY),
                                                 xrt-smi completions fallback
  - APU package power (W)                         amdgpu PPT sensor (Package Power Tracking), EMA-smoothed
  - CPU / iGPU temperature (deg C)               k10temp Tctl / amdgpu edge

Linux notes (Strix Halo / Ryzen AI Max, amdxdna + amdgpu):
  - iGPU utilization is a REAL hardware counter: /sys/class/drm/card*/device/gpu_busy_percent.
  - APU package power comes from the amdgpu hwmon "PPT" (Package Power Tracking) sensor
    (power1_average). This is the whole-package power budget and is readable without root.
    (RAPL /sys/class/powercap/.../energy_uj is root-only here due to the Platypus CVE
    mitigation, so we deliberately do NOT use it.)
  - NPU has no hardware utilization %. Primary path: the inference pipeline feeds measured
    per-frame latencies to NPU_ACTIVITY, and the duty cycle is the summed busy time over a
    trailing window (no subprocess). Fallback when no pipeline is feeding it (e.g. a
    standalone dashboard): derive a duty cycle from the amdxdna driver's per-HW-context
    `command_completions` counter, exposed via `xrt-smi examine -r aie-partitions -f JSON`,
    as duty% = inferences/sec * npu_infer_ms. `npu_infer_ms` defaults to a turbo-mode
    estimate and can be set from the pipeline's measured latency for accuracy.
  - Requires the XRT env (xrt-smi on PATH + its libs on LD_LIBRARY_PATH). If xrt-smi is not
    callable, NPU falls back to 0.0 gracefully.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import psutil


class _NpuActivityTracker:
    """Frame-rate NPU activity fed by the inference pipeline (Option B).

    The inference loop calls `record(infer_ms)` once per frame. The monitor then
    derives a duty cycle from the REAL measured latencies over a trailing window:

        duty% = sum(infer_ms in window) / window_span

    This makes the NPU gauge frame-rate granular and accurate (it uses the actual
    per-frame latency, not a fixed estimate) with no xrt-smi subprocess. When no
    pipeline is feeding it, the monitor falls back to the xrt-smi completions path.
    """

    def __init__(self, window_secs: float = 2.0, active_gap_secs: float = 0.75):
        self._window = window_secs
        self._active_gap = active_gap_secs
        self._lock = threading.Lock()
        self._events: deque[tuple[float, float]] = deque()  # (monotonic_t, infer_ms)

    def record(self, infer_ms: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._events.append((now, infer_ms))
            self._trim(now)

    def _trim(self, now: float) -> None:
        ev = self._events
        while ev and now - ev[0][0] > self._window:
            ev.popleft()

    def stats(self) -> tuple[bool, float, float]:
        """Return (active, infers_per_s, duty_pct) over the trailing window."""
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            if not self._events:
                return False, 0.0, 0.0
            n = len(self._events)
            busy_ms = sum(ms for _, ms in self._events)
            first_t = self._events[0][0]
            last_t = self._events[-1][0]
        span = max(now - first_t, 1e-6)
        infers_per_s = n / span
        duty = min(busy_ms / 1000.0 / span * 100.0, 100.0)
        active = (now - last_t) < self._active_gap
        return active, infers_per_s, duty


# Module-level singleton the inference pipeline records into.
NPU_ACTIVITY = _NpuActivityTracker()


@dataclass
class Sample:
    """One reading of system state."""

    t: float
    cpu_pct: float
    ram_pct: float
    igpu_pct: float
    npu_pct: float
    apu_power_w: Optional[float] = None
    apu_power_w_ema: Optional[float] = None
    cpu_temp: Optional[float] = None
    igpu_temp: Optional[float] = None
    npu_active: bool = False
    npu_infers_per_s: float = 0.0


class ResourceMonitor:
    """Light-weight sampler. Call `.sample()` at whatever cadence you want.

    Args:
        npu_infer_ms: assumed NPU busy time per inference, used to convert the
            driver's completions/sec into a duty-cycle %. Set this to your measured
            per-frame inference latency for an accurate gauge (turbo BF16 yolo26s ≈ 20 ms).
        npu_poll_every: minimum interval (s) between (subprocess) xrt-smi polls.
        ema_window_secs / sample_rate_hz: EMA smoothing for APU power.
    """

    def __init__(
        self,
        npu_infer_ms: float = 20.0,
        npu_poll_every: float = 1.0,
        ema_window_secs: float = 3.0,
        sample_rate_hz: float = 1.0,
    ):
        # psutil cpu_percent needs a priming call
        psutil.cpu_percent(interval=None)

        # --- NPU duty-cycle tracking ---
        self.npu_infer_ms = npu_infer_ms
        self._npu_poll_every = npu_poll_every
        self._xrt_smi: Optional[str] = shutil.which("xrt-smi")
        self._npu_last_completions: Optional[int] = None
        self._npu_last_poll_t: float = 0.0
        self._npu_infers_per_s: float = 0.0
        self._npu_active: bool = False
        self._npu_tmp = os.path.join(
            tempfile.gettempdir(), f"_npu_aie_{os.getpid()}.json"
        )

        # --- APU power EMA ---
        n = max(ema_window_secs * sample_rate_hz, 1.0)
        self._ema_alpha = 2.0 / (n + 1.0)
        self._apu_power_ema: Optional[float] = None

        # --- Linux sysfs paths (discovered once) ---
        self._igpu_busy_path: Optional[str] = None
        self._igpu_power_path: Optional[str] = None
        self._igpu_temp_path: Optional[str] = None
        self._cpu_temp_path: Optional[str] = None
        self._discover_linux_paths()

    # ------------------------------------------------------------------
    # Linux sysfs discovery
    # ------------------------------------------------------------------
    def _discover_linux_paths(self) -> None:
        # iGPU busy% — amdgpu exposes this per DRM card.
        for p in glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"):
            if os.access(p, os.R_OK):
                self._igpu_busy_path = p
                break

        for hwmon in glob.glob("/sys/class/hwmon/hwmon*"):
            name = self._read_text(os.path.join(hwmon, "name"))
            if name == "amdgpu":
                # PPT (Package Power Tracking) — whole-package power, microwatts.
                cand = os.path.join(hwmon, "power1_average")
                if os.access(cand, os.R_OK):
                    self._igpu_power_path = cand
                # edge temperature, millidegrees
                cand_t = os.path.join(hwmon, "temp1_input")
                if os.access(cand_t, os.R_OK):
                    self._igpu_temp_path = cand_t
            elif name == "k10temp":
                # Prefer the Tctl label; fall back to temp1_input.
                chosen = None
                for lbl in glob.glob(os.path.join(hwmon, "temp*_label")):
                    if self._read_text(lbl) == "Tctl":
                        chosen = lbl.replace("_label", "_input")
                        break
                if chosen is None:
                    chosen = os.path.join(hwmon, "temp1_input")
                if os.access(chosen, os.R_OK):
                    self._cpu_temp_path = chosen

    @staticmethod
    def _read_text(path: str) -> Optional[str]:
        try:
            with open(path) as f:
                return f.read().strip()
        except Exception:
            return None

    @classmethod
    def _read_float_file(
        cls, path: Optional[str], scale: float = 1.0
    ) -> Optional[float]:
        if not path:
            return None
        raw = cls._read_text(path)
        if raw is None:
            return None
        try:
            return float(raw) * scale
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # GPU plumbing (amdgpu sysfs)
    # ------------------------------------------------------------------
    def _read_gpu_utils(self) -> tuple[float, float]:
        """Returns (igpu_pct, npu_pct), each capped at 100 %."""
        igpu = self._read_float_file(self._igpu_busy_path) or 0.0
        npu = self._read_npu_util_linux()
        return min(igpu, 100.0), min(npu, 100.0)

    # ------------------------------------------------------------------
    # NPU duty-cycle from amdxdna driver counters
    # ------------------------------------------------------------------
    def _read_npu_completions(self) -> Optional[int]:
        """Sum command_completions across all AIE HW contexts via xrt-smi JSON."""
        if not self._xrt_smi:
            return None
        try:
            subprocess.run(
                [
                    self._xrt_smi,
                    "examine",
                    "-r",
                    "aie-partitions",
                    "-f",
                    "JSON",
                    "-o",
                    self._npu_tmp,
                    "--force",
                ],
                capture_output=True,
                timeout=4.0,
                check=False,
            )
            with open(self._npu_tmp) as f:
                data = json.load(f)
        except Exception:
            return None

        total = 0
        found = False
        active = False
        for dev in data.get("devices", []):
            parts = dev.get("aie_partitions", {})
            for part in parts.get("partitions", []):
                for ctx in part.get("hw_contexts", []):
                    if str(ctx.get("status", "")).lower() == "active":
                        active = True
                    c = ctx.get("command_completions")
                    try:
                        total += int(c)
                        found = True
                    except (TypeError, ValueError):
                        pass
        self._npu_active = active
        return total if found else None

    def _read_npu_util_linux(self) -> float:
        """NPU duty-cycle %.

        Prefers Option B (pipeline-fed real per-frame timing via NPU_ACTIVITY) for a
        frame-rate-granular, accurate gauge. Falls back to the xrt-smi completions
        counter when no pipeline is feeding the tracker (e.g. standalone dashboard).
        """
        active, ips, duty = NPU_ACTIVITY.stats()
        if active:
            self._npu_active = True
            self._npu_infers_per_s = ips
            return min(duty, 100.0)
        return self._read_npu_util_xrtsmi()

    def _read_npu_util_xrtsmi(self) -> float:
        """Convert completions/sec into a duty-cycle % (throttled to npu_poll_every)."""
        now = time.monotonic()
        if now - self._npu_last_poll_t >= self._npu_poll_every:
            completions = self._read_npu_completions()
            if completions is not None:
                if self._npu_last_completions is not None:
                    dt = now - self._npu_last_poll_t
                    dc = completions - self._npu_last_completions
                    if dt > 0 and dc >= 0:
                        self._npu_infers_per_s = dc / dt
                self._npu_last_completions = completions
            else:
                self._npu_infers_per_s = 0.0
            self._npu_last_poll_t = now
        return min(self._npu_infers_per_s * self.npu_infer_ms / 1000.0 * 100.0, 100.0)

    # ------------------------------------------------------------------
    # Power / temperature
    # ------------------------------------------------------------------
    def _read_apu_power(self) -> Optional[float]:
        # amdgpu PPT sensor reports microwatts.
        w = self._read_float_file(self._igpu_power_path, scale=1e-6)
        if w is None:
            return None
        if self._apu_power_ema is None:
            self._apu_power_ema = w
        else:
            self._apu_power_ema = (
                self._ema_alpha * w + (1.0 - self._ema_alpha) * self._apu_power_ema
            )
        return w

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample(self) -> Sample:
        """Take one snapshot. Cheap (~5 ms) except on the ~1 Hz NPU poll tick."""
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        igpu, npu = self._read_gpu_utils()

        apu_power = self._read_apu_power()
        cpu_temp = self._read_float_file(self._cpu_temp_path, scale=1e-3)
        igpu_temp = self._read_float_file(self._igpu_temp_path, scale=1e-3)

        return Sample(
            t=time.time(),
            cpu_pct=float(cpu),
            ram_pct=float(ram),
            igpu_pct=float(igpu),
            npu_pct=float(npu),
            apu_power_w=apu_power,
            apu_power_w_ema=self._apu_power_ema,
            cpu_temp=cpu_temp,
            igpu_temp=igpu_temp,
            npu_active=self._npu_active,
            npu_infers_per_s=self._npu_infers_per_s,
        )

    def close(self) -> None:
        try:
            if os.path.exists(self._npu_tmp):
                os.unlink(self._npu_tmp)
        except Exception:
            pass

    def __enter__(self) -> "ResourceMonitor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def avg_load_during(run_fn):
    """Run ``run_fn()`` while sampling CPU load and APU power on a side thread.

    Returns ``(result, avg_cpu_pct, avg_power_w)``. Used by the CPU-vs-NPU
    benchmark so it can report the real wins (a freed CPU and lower power), not
    just fps. ``avg_power_w`` is None if the power sensor isn't readable.
    """
    mon = ResourceMonitor()
    samples: list[tuple[float, Optional[float]]] = []
    stop = threading.Event()

    def _poll() -> None:
        while not stop.is_set():
            stop.wait(0.5)  # let CPU% accumulate over the interval
            if stop.is_set():
                break
            s = mon.sample()
            samples.append((s.cpu_pct, s.apu_power_w))

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    try:
        result = run_fn()
    finally:
        stop.set()
        t.join(timeout=1.0)
        mon.close()

    cpus = [c for c, _ in samples if c is not None]
    pws = [p for _, p in samples if p is not None]
    avg_cpu = sum(cpus) / len(cpus) if cpus else 0.0
    avg_pw = (sum(pws) / len(pws)) if pws else None
    return result, avg_cpu, avg_pw
