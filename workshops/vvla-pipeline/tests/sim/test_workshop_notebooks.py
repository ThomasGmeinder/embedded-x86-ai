# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The workshop README's happy path, headless: every notebook, in the README's order.

``workshop/README.md`` has the student run ``./run_notebooks.sh``, then the
notebooks in ``01_ai_models``, ``02_ros2`` and ``03_integration``. Here each
notebook runs top to bottom, as stored, in a fresh ``vvla-workshop`` kernel.
A live cell gets the clicks its own buttons ask for. The notebooks that
print their ONNX provider must have printed the NPU, and the intent
notebook must show Gemma's parse of a typed command.

``cd workshop && ./run_notebooks.sh --test`` runs this after the README's setup.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from notebook_harness import CPU_FALLBACK, NOTEBOOKS, PIPE_ROOT, run_notebook
from twin_ros_rig import join_test_domain

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.docs, pytest.mark.slow, pytest.mark.timeout(900)]
pytest.importorskip("rclpy")
pytest.importorskip("nbformat")
pytest.importorskip("nbclient")

README = PIPE_ROOT / "workshop" / "README.md"


def _readme_order() -> list[str]:
    m = re.search(r"Read and run them in order:(.*?)\.\s", README.read_text(encoding="utf-8"), re.S)
    return re.findall(r"`(\w+)`", m.group(1)) if m else []


ORDER = _readme_order()
PATHS = [
    path
    for folder in ORDER
    for path in sorted((NOTEBOOKS / folder).glob("*.ipynb"))
]

# The student's actions in each live cell (by cell id); any other live cell is
# watched until it draws three frames (notebook_harness.DEFAULT_ACTIONS).
SCRIPTS = {
    "02_robot": {
        "cell-10": [("click", "rest pose")],
    },
    "03_intent_node": {
        "578bb604bee3": [
            ("type", "stop robot"), ("wait_for", "stop fast path"),
            ("type", "pick up the ball"), ("wait_for", "[LLM parse]"),
        ],
    },
    "01_behavior": {
        "cell-14": [("type", "wave at me"), ("wait_for", "[toy_wave] waving!")],
        # The dance prints its tempo, then one line per beat, before it parks.
        "cell-16": [("click", "dance"), ("wait_for", "parked:", r"^\d+\.\d+ s/beat, \d+ beats$|^  beat \d+: ")],
    },
    # Typed into the console's text box, then the reset button, then another
    # pickup. The cell ends with end session, which the notebook says to click
    # before the clean-up cell.
    "02_digital_twin": {
        "c0000016": [
            ("wait_for", "Ready."),
            ("type", "pick up the toothbrush"), ("wait_for", "Got the toothbrush"),
            ("type", "set it down"), ("wait_for", "Done."),
            ("type", "pick up the bottle"), ("wait_for", "Got the bottle"),
            ("type", "set it down"), ("wait_for", "Done."),
            ("click", "reset"), ("wait_for", "Reset. Arm parked, objects back on the table."),
            ("type", "pick up the toothbrush"), ("wait_for", "Got the toothbrush"),
        ],
    },
}

# What the notebook's own output must show. The YOLO notebooks print the
# provider the session got.
EXPECT = {
    "03_intent_node": ["[LLM parse]", "'pick_place'"],
    "01_npu_yolo": ["VitisAIExecutionProvider"],
    "01_behavior": ["VitisAIExecutionProvider"],
}
# cell-14 of the cameras notebook subscribes RELIABLE on purpose, so the student
# sees the best-effort publisher refuse the match. That warning is the lesson.
TOLERATE = {
    "01_cameras": {"cell-14": re.compile(r"incompatible QoS")},
}



def test_readme_lists_every_notebook_folder():
    folders = sorted(
        p.name for p in NOTEBOOKS.iterdir()
        if p.is_dir() and not p.name.startswith(".") and any(p.glob("*.ipynb"))
    )
    assert ORDER == folders, f"workshop/README.md says run {ORDER}; the notebooks are in {folders}"


@pytest.mark.parametrize("path", PATHS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_notebook_runs(path: Path, tmp_path):
    join_test_domain()
    twin = None
    if path.stem == "01_behavior":
        from vla_pipeline.utils.config import load_config
        from vvla_workshop.twin_ros import TwinSession

        twin = TwinSession(load_config())
    try:
        # 02_digital_twin starts its own launches through so101_sim.stack and
        # stops them when this kernel exits.
        run = run_notebook(path, SCRIPTS.get(path.stem), TOLERATE.get(path.stem))
    finally:
        if twin is not None:
            twin.close()
    fallback = [line.strip() for line in run.output.splitlines() if CPU_FALLBACK.search(line)]
    assert not fallback, f"{path.name} fell back to the CPU: {fallback}"
    for phrase in EXPECT.get(path.stem, ()):
        assert phrase in run.output, f"{path.name} never showed {phrase!r}"
