# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""One control process, one perception process, one console process.

``/control/state`` has a single publisher. A second ``run_pipeline.sh``
must exit before it starts another.
"""

from __future__ import annotations

import os

_MARKERS = (
    "vla_pipeline.main",
    "vla_pipeline.perception",
    "vla_pipeline.console_main",
)


def running_pipeline() -> list[tuple[int, str]]:
    """``(pid, command line)`` for every pipeline process except this one."""
    me = os.getpid()
    found: list[tuple[int, str]] = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == me:
            continue
        try:
            raw = open(f"/proc/{pid}/cmdline", "rb").read()
        except OSError:
            continue
        text = raw.replace(b"\0", b" ").decode(errors="replace").strip()
        if any(marker in text for marker in _MARKERS):
            found.append((pid, text))
    return sorted(found)


def format_running() -> str:
    """One ``pid command`` line per running pipeline process, or an empty string."""
    return "\n".join(f"{pid} {cmd}" for pid, cmd in running_pipeline())
