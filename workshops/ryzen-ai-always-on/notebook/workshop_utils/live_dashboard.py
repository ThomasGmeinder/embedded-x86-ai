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

"""Live in-notebook resource dashboard — Task Manager dark style."""

from __future__ import annotations

import threading
import time
from collections import deque
from io import BytesIO

import ipywidgets as widgets
import matplotlib.figure
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from IPython.display import display

from .resource_monitor import ResourceMonitor, Sample


_METRICS = [
    ("RAM", "ram_pct", "#7bba5a"),
    ("CPU", "cpu_pct", "#5299d3"),
    ("iGPU", "igpu_pct", "#c97bd4"),
    ("NPU", "npu_pct", "#d4885a"),
]

_BG = "#0a0a0a"
_GRID = "#1e1e1e"
_FG = "#e0e0e0"
_DIM = "#555555"


class LiveDashboard:
    def __init__(
        self,
        window_seconds: float = 60.0,
        sample_period: float = 1.0,
        figsize: tuple[float, float] = (13.0, 3.2),
        dpi: int = 110,
        npu_infer_ms: float = 20.0,
    ):
        self.window_seconds = window_seconds
        self.sample_period = sample_period
        self.npu_infer_ms = npu_infer_ms
        self._dpi = dpi
        self._figsize = figsize
        self._latest: Sample | None = None

        self._maxlen = max(int(window_seconds / sample_period) + 1, 10)
        self._t: deque[float] = deque(maxlen=self._maxlen)
        self._series: dict[str, deque[float]] = {
            field: deque(maxlen=self._maxlen) for _, field, _ in _METRICS
        }

        # Fixed width (not window-scaling, which got huge). The dashboard is a
        # wide 4-panel strip, so it needs more width than the video feed to stay
        # readable; 1000px keeps the numbers legible without filling the window.
        self._image = widgets.Image(format="png", width=1000)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._monitor: ResourceMonitor | None = None
        self._lock = threading.Lock()

        # Render a blank frame immediately so the widget has valid content
        self._image.value = self._render_frame()

    def _render_frame(self) -> bytes:
        """Build a fresh figure, draw current data, return PNG bytes."""
        fig = matplotlib.figure.Figure(figsize=self._figsize, facecolor=_BG)
        gs = gridspec.GridSpec(
            1,
            len(_METRICS),
            figure=fig,
            left=0.04,
            right=0.98,
            top=0.88,
            bottom=0.18,
            wspace=0.06,
        )

        with self._lock:
            now = self._t[-1] if self._t else time.time()
            xs = (
                np.array([t - now for t in self._t])
                if self._t
                else np.array([-self.window_seconds, 0.0])
            )
            s_latest = self._latest

            for i, (label, field, color) in enumerate(_METRICS):
                ax = fig.add_subplot(gs[0, i])
                ax.set_facecolor(_BG)
                ax.set_ylim(0, 100)
                ax.set_xlim(-self.window_seconds, 0)

                xticks = np.linspace(-self.window_seconds, 0, 5)
                ax.set_xticks(xticks)
                ax.set_xticklabels(
                    [f"{int(x)}s" if x != 0 else "now" for x in xticks],
                    color=_DIM,
                    fontsize=7,
                )
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.set_yticklabels(
                    ["0", "25", "50", "75", "100%"], color=_DIM, fontsize=7
                )
                ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
                for spine in ax.spines.values():
                    spine.set_edgecolor(_GRID)

                ys = (
                    np.array(list(self._series[field]))
                    if self._series[field]
                    else np.zeros(2)
                )
                plot_xs = (
                    xs
                    if len(xs) == len(ys)
                    else np.linspace(-self.window_seconds, 0, len(ys))
                )

                ax.fill_between(plot_xs, ys, alpha=0.20, color=color, zorder=2)
                ax.plot(plot_xs, ys, color=color, lw=1.8, zorder=3)

                latest = float(ys[-1]) if len(ys) else 0.0
                if field == "npu_pct":
                    # Lead with the HARD number — inferences/sec, a real driver
                    # counter — and show the duty-cycle % underneath as an estimate
                    # (there is no true hardware utilization % for the NPU).
                    ips = (
                        getattr(s_latest, "npu_infers_per_s", 0.0) if s_latest else 0.0
                    )
                    ax.text(
                        0.04,
                        0.90,
                        f"{ips:.0f} inf/s",
                        transform=ax.transAxes,
                        color=color,
                        fontsize=16,
                        fontweight="bold",
                        va="top",
                        ha="left",
                        zorder=4,
                    )
                    ax.text(
                        0.04,
                        0.66,
                        f"~{latest:.0f}% busy (est.)",
                        transform=ax.transAxes,
                        color=color,
                        fontsize=8,
                        va="top",
                        ha="left",
                        zorder=4,
                    )
                else:
                    ax.text(
                        0.04,
                        0.90,
                        f"{latest:.0f}%",
                        transform=ax.transAxes,
                        color=color,
                        fontsize=18,
                        fontweight="bold",
                        va="top",
                        ha="left",
                        zorder=4,
                    )
                ax.text(
                    0.04,
                    0.04,
                    label,
                    transform=ax.transAxes,
                    color=_FG,
                    fontsize=9,
                    fontweight="bold",
                    va="bottom",
                    ha="left",
                    zorder=4,
                )

        fig.text(
            0.04,
            0.965,
            "System Utilization",
            ha="left",
            va="top",
            color=_FG,
            fontsize=10,
            fontweight="bold",
        )

        # Power + temperature readout (shown when sysfs sensors provide values; else hidden).
        # APU package power is the efficiency headline, so it is rendered large and
        # in a warm colour; the temperatures ride underneath as supporting context.
        with self._lock:
            s = self._latest
        if s is not None:
            power = (
                s.apu_power_w_ema if s.apu_power_w_ema is not None else s.apu_power_w
            )
            if power is not None:
                # White (not orange) so it does not clash with the NPU colour.
                fig.text(
                    0.985,
                    0.985,
                    f"APU {power:.0f} W",
                    ha="right",
                    va="top",
                    color="#ffffff",
                    fontsize=20,
                    fontweight="bold",
                    fontfamily="monospace",
                )
            temps = []
            if s.cpu_temp is not None:
                temps.append(f"CPU {s.cpu_temp:.0f}\u00b0C")
            if s.igpu_temp is not None:
                temps.append(f"iGPU {s.igpu_temp:.0f}\u00b0C")
            if temps:
                fig.text(
                    0.985,
                    0.885,
                    "   ".join(temps),
                    ha="right",
                    va="top",
                    color="#c8c8c8",
                    fontsize=12,
                    fontfamily="monospace",
                )

        canvas = FigureCanvasAgg(fig)
        buf = BytesIO()
        canvas.print_figure(buf, format="png", dpi=self._dpi, facecolor=_BG)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def display(self) -> None:
        display(self._image)

    @property
    def widget(self) -> widgets.Image:
        """The image widget, so callers can embed the dashboard in their own
        layout (e.g. directly under a live demo) instead of auto-displaying it."""
        return self._image

    def start(self, auto_display: bool = True) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if auto_display:
            self.display()
        self._monitor = ResourceMonitor(npu_infer_ms=self.npu_infer_ms)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._monitor is not None:
            self._monitor.close()
            self._monitor = None

    def __enter__(self) -> "LiveDashboard":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        assert self._monitor is not None
        next_tick = time.monotonic()
        while not self._stop.is_set():
            sample = self._monitor.sample()
            with self._lock:
                self._latest = sample
                self._t.append(sample.t)
                for _, field, _ in _METRICS:
                    self._series[field].append(getattr(sample, field))
            self._image.value = self._render_frame()
            next_tick += self.sample_period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                self._stop.wait(timeout=sleep)
            else:
                next_tick = time.monotonic()
