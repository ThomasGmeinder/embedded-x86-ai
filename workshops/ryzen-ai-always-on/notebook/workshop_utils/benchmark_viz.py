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

"""CPU-versus-NPU benchmark comparison charts for the notebook."""

import numpy as np
import matplotlib.pyplot as plt

CPU_COLOR = "#e07"
NPU_COLOR = "#07e"


def plot_cpu_vs_npu(results, cpu_key="CPU FP32", npu_key="NPU BF16", npu_warmup=20):
    """Render the CPU-versus-NPU comparison charts and return the metrics.

    ``results`` is a dict keyed by engine name; each value is
    ``(times_ms_list, avg_cpu_load, avg_watts)`` as recorded by the Section 5
    TRY ME. Three branches: no runs -> print a prompt; one engine -> that
    engine's latency histogram + throughput bar + a nudge to run the other;
    both engines -> latency histogram, throughput bars, and a
    utilization/power/efficiency table.

    Returns a dict of the computed metrics (``p50_cpu``, ``p50_npu``,
    ``cpu_fps``, ``npu_fps`` when available; a partial dict for the one-engine
    case; an empty dict when nothing was recorded) so the caller can recover
    ``p50_npu`` for downstream use.
    """
    # These static charts render inline, while Section 7's live dashboard draws
    # on its own Agg canvas. Re-select an inline/interactive backend here so this
    # renders no matter what order Sections 6 and 7 were run in.
    try:
        plt.switch_backend("module://matplotlib_inline.backend_inline")
    except Exception:
        pass  # already inline (or a non-Jupyter frontend) - leave the backend as-is

    # Reads whatever Section 5 recorded. Both engines -> full comparison; one
    # engine -> that engine's charts plus a nudge to run the other; none -> a
    # short prompt.
    have = {k: results[k] for k in (cpu_key, npu_key) if k in results}

    if not have:
        print(
            "Run the Section 5 TRY ME first (CPU, then swap to NPU) to record both engines."
        )
        return {}
    elif len(have) < 2:
        missing = npu_key if cpu_key in have else cpu_key
        label = next(iter(have))
        times, load, watts = have[label]
        lat = (
            times[npu_warmup:] if label == npu_key else times
        )  # skip AIE warm-up for NPU only
        p = np.percentile(lat, 50)
        fps = 1000 / p
        color = CPU_COLOR if label == cpu_key else NPU_COLOR

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
        ax1.hist(lat, bins=30, alpha=0.6, label=f"{label}  p50={p:.0f} ms", color=color)
        ax1.set_xlabel("Latency per frame (ms)")
        ax1.set_ylabel("Frames")
        ax1.set_title("Latency distribution")
        ax1.legend()

        b = ax2.bar([0], [fps], 0.4, label=label, color=color, alpha=0.6)
        ax2.bar_label(b, fmt="%.0f", padding=2)
        ax2.set_ylim(0, fps * 1.3)
        ax2.set_xlim(-0.7, 0.7)
        ax2.set_xticks([0])
        ax2.set_xticklabels([label])
        ax2.set_ylabel("Frames per second (fps)")
        ax2.set_title("Throughput")
        ax2.legend()
        plt.tight_layout()
        plt.show()

        _w = f", ~{watts:.0f} W" if watts else ""
        print(f"Only {label} recorded ({fps:.1f} fps, CPU ~{load:.0f}% busy{_w}).")
        print(
            f"Run the Section 5 TRY ME again on {missing} to see the full side-by-side."
        )

        metric = "p50_cpu" if label == cpu_key else "p50_npu"
        fps_metric = "cpu_fps" if label == cpu_key else "npu_fps"
        return {metric: p, fps_metric: fps}
    else:
        cpu_times, cpu_load, cpu_watts = have[cpu_key]
        npu_times, npu_load, npu_watts = have[npu_key]
        cpu_lat = cpu_times  # CPU has no warm-up ramp
        npu_lat = npu_times[npu_warmup:]  # skip AIE warm-up frames
        p50, p50_npu = np.percentile(cpu_lat, 50), np.percentile(npu_lat, 50)
        cpu_fps, npu_fps = 1000 / p50, 1000 / p50_npu

        # One figure, whole story: per-frame latency distribution on top, throughput
        # in the middle, the system-cost scorecard below.
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(9, 9), gridspec_kw={"height_ratios": [1, 1, 0.5]}
        )
        w = 0.38

        all_times = list(cpu_lat) + list(npu_lat)
        bins = np.linspace(min(all_times), max(all_times), 30)
        ax1.hist(
            cpu_lat,
            bins=bins,
            alpha=0.6,
            label=f"CPU FP32  p50={p50:.0f} ms",
            color=CPU_COLOR,
        )
        ax1.hist(
            npu_lat,
            bins=bins,
            alpha=0.6,
            label=f"NPU BF16  p50={p50_npu:.0f} ms",
            color=NPU_COLOR,
        )
        ax1.set_xlabel("Latency per frame (ms)")
        ax1.set_ylabel("Frames")
        ax1.set_title("Latency distribution")
        ax1.legend()

        b1 = ax2.bar(-w / 2, cpu_fps, w, label="CPU FP32", color=CPU_COLOR, alpha=0.6)
        b2 = ax2.bar(w / 2, npu_fps, w, label="NPU BF16", color=NPU_COLOR, alpha=0.6)
        ax2.bar_label(b1, fmt="%.0f", padding=2)
        ax2.bar_label(b2, fmt="%.0f", padding=2)
        ax2.set_ylim(
            0, max(cpu_fps, npu_fps) * 1.2
        )  # headroom so labels clear the spine
        ax2.set_xlim(-0.7, 0.7)
        ax2.set_xticks([-w / 2, w / 2])
        ax2.set_xticklabels(["CPU FP32", "NPU BF16"])
        ax2.set_ylabel("Frames per second (fps)")
        ax2.set_title("Throughput")
        ax2.legend()

        ax3.axis("off")
        ax3.set_title("Utilization and power")
        rows = [["CPU utilization", f"{cpu_load:.0f}%", f"{npu_load:.0f}%"]]
        if cpu_watts and npu_watts:
            rows.append(["APU power", f"{cpu_watts:.0f} W", f"{npu_watts:.0f} W"])
            eff = (npu_fps / npu_watts) / (cpu_fps / cpu_watts)
            rows.append(["Efficiency", "1x", f"{eff:.1f}x frames/W"])
        table = ax3.table(
            cellText=rows,
            colLabels=["Metric", "CPU FP32", "NPU BF16"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)
        table[(0, 1)].get_text().set_color(CPU_COLOR)
        table[(0, 2)].get_text().set_color(NPU_COLOR)

        plt.tight_layout()
        plt.show()

        return {
            "p50_cpu": p50,
            "p50_npu": p50_npu,
            "cpu_fps": cpu_fps,
            "npu_fps": npu_fps,
        }
