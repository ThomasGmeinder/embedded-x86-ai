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

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

NPU_COLOR = "#6c8ebf"
CPU_COLOR = "#d6a04a"


def plot_partition(cache_dir, cache_key):
    cache_dir = Path(cache_dir)
    trace = pd.read_csv(cache_dir / cache_key / "graph_partition_trace.csv", header=0)
    gops = pd.read_csv(cache_dir / cache_key / "gops.csv", header=0)
    trace.columns = ["Node", "Type", "Subgraph", "Status"]
    gops.columns = ["Node", "OPs", "Note"]

    df = trace.merge(gops, on="Node", how="left").fillna({"OPs": 0})
    # An empty Subgraph cell (parsed by pandas as NaN) means the op is not in an
    # offloaded partition and runs on CPU. Guard against NaN: str(NaN) == "nan",
    # which is truthy and would misclassify every CPU op as NPU.
    df["Device"] = df["Subgraph"].apply(
        lambda s: "NPU" if (isinstance(s, str) and s.strip()) else "CPU"
    )
    df["GOPs"] = df["OPs"] / 1e9
    npu_gops = df[df["Device"] == "NPU"]["GOPs"].sum()
    cpu_gops = df[df["Device"] == "CPU"]["GOPs"].sum()
    total = npu_gops + cpu_gops

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(14, 5))

    pivot = (
        df.groupby(["Type", "Device"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["NPU", "CPU"], fill_value=0)
    )
    pivot = pivot.iloc[pivot["NPU"].argsort()]
    pivot["NPU"].plot(kind="barh", ax=ax_bar, color=NPU_COLOR, label="NPU")
    pivot["CPU"].plot(
        kind="barh", ax=ax_bar, color=CPU_COLOR, label="CPU", left=pivot["NPU"]
    )
    ax_bar.set(xlabel="Node count", title="Operator types — NPU vs CPU (node count)")
    ax_bar.legend(loc="lower right")
    ax_bar.spines[["top", "right"]].set_visible(False)

    wedges, _ = ax_pie.pie(
        [npu_gops, cpu_gops or 1e-9],
        colors=[NPU_COLOR, CPU_COLOR],
        startangle=90,
        wedgeprops={"width": 0.5, "edgecolor": "white"},
    )
    ax_pie.set_title("Compute split (GOPs)")
    ax_pie.legend(
        wedges,
        [
            f"NPU  {npu_gops:.2f} GFLOPs ({100*npu_gops/total:.1f}%)",
            f"CPU  {cpu_gops:.4f} GFLOPs ({100*cpu_gops/total:.1f}%)",
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
    )
    plt.tight_layout()
    plt.show()

    cpu_ops = df[df["Device"] == "CPU"][["Node", "Type"]].reset_index(drop=True)
    reasons_path = (
        cache_dir
        / cache_key
        / "vaiml_partition_fe.flexml"
        / "aie_unsupported_original_ops_with_reasons.json"
    )
    if reasons_path.exists():
        reason_map = {
            r.get("node_name", r.get("name", "")): r.get("reason", "")
            for r in json.loads(reasons_path.read_text())
        }
        cpu_ops["Reason"] = cpu_ops["Node"].map(reason_map).fillna("")
    print(f"\nCPU-fallback operators ({len(cpu_ops)} of {len(df)} total):")
    print(cpu_ops.to_string(index=False))
