# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Static TODO scanner (PROVIDED).

Answers one question - *which TODOs are still stubs?* - WITHOUT importing any
component code. It reads the source files, so it works on any machine: no NPU,
no ROS 2, no GPU, no cameras, not even numpy or onnxruntime need to import.
Both the live app's preflight (``app.py``) and the top-level end-to-end runner
(``selftest.py``) use it so a half-finished project gets a tidy checklist
instead of a stack trace.

A TODO is a marked block::

    # >>> TODO 1.1: npu_available - notebooks/...
    ...your code...
    # <<< TODO 1.1

"implemented" just means the block no longer contains ``NotImplementedError`` -
the exact signal the selftest harnesses key off at runtime, computed here
statically instead.
"""

from __future__ import annotations

import re
from pathlib import Path

from common.config import PROJECT_ROOT
from common.feedback import TODO_INDEX

# A TODO block: '# >>> TODO <id> ...' up to the matching '# <<< TODO <id>'.
_BLOCK = re.compile(
    r"#\s*>>>\s*TODO\s+(?P<id>\d+\.\d+)\b.*?#\s*<<<\s*TODO\s+(?P=id)\b",
    re.DOTALL,
)


def _key(todo_id: str):
    """Numeric sort key so 2.2 sorts before 2.10 (not lexically after)."""
    try:
        return [int(x) for x in todo_id.split(".")]
    except ValueError:
        return [99]


def _iter_sources(root: Path):
    """Yield every project ``.py`` file that might carry a TODO block.

    Stays inside the active tree (``project/`` or ``solution/``); the shared
    ``common/`` package lives elsewhere and carries no TODOs.
    """
    for f in sorted(root.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        yield f


def scan(root: Path | str | None = None) -> dict[str, bool]:
    """Map every TODO id found under ``root`` to whether it's implemented.

    Implemented == its marker block no longer raises ``NotImplementedError``.
    An id with more than one block counts as done only when every block is.
    """
    root = Path(root) if root else PROJECT_ROOT
    status: dict[str, bool] = {}
    for f in _iter_sources(root):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if ">>> TODO" not in text:  # cheap skip for the common case
            continue
        for m in _BLOCK.finditer(text):
            tid = m.group("id")
            done = "NotImplementedError" not in m.group(0)
            status[tid] = done and status.get(tid, True)
    return status


def remaining(root: Path | str | None = None) -> list[str]:
    """TODO ids still unimplemented, in workshop order."""
    return sorted([t for t, done in scan(root).items() if not done], key=_key)


def implemented(root: Path | str | None = None) -> list[str]:
    """TODO ids already implemented, in workshop order."""
    return sorted([t for t, done in scan(root).items() if done], key=_key)


def all_ids(root: Path | str | None = None) -> list[str]:
    """Every TODO id discovered under ``root``, in workshop order."""
    return sorted(scan(root).keys(), key=_key)


def describe(todo_id: str) -> tuple[str, str, str]:
    """``(where, what, notebook)`` for an id, from the shared TODO index."""
    return TODO_INDEX.get(todo_id, ("?", todo_id, "?"))


def format_list(ids, *, indent="    ") -> str:
    """Render ``ids`` as ``TODO x.y: what  ->  file   (read notebook)`` lines."""
    lines = []
    for t in sorted(ids, key=_key):
        where, what, nb = describe(t)
        lines.append(f"{indent}TODO {t}: {what}  ->  {where}   (read {nb})")
    return "\n".join(lines)


def compact(ids) -> str:
    """A short comma-joined id list (for the 'also still to do' footnote)."""
    return ", ".join(sorted({str(t) for t in ids}, key=_key))


def progress(root: Path | str | None = None) -> tuple[int, int]:
    """``(implemented_count, total_count)`` of TODOs under ``root``."""
    st = scan(root)
    return sum(1 for v in st.values() if v), len(st)


def progress_line(root: Path | str | None = None) -> str:
    """One-line tally: ``TODOs: 9/21 implemented (12 remaining)``."""
    done, total = progress(root)
    return f"TODOs: {done}/{total} implemented ({total - done} remaining)"
