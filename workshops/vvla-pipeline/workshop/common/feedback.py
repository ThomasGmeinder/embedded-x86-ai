# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Self-test reporting (PROVIDED).

The component harnesses (``selftest_models``, ``selftest_ros`` and
``selftest_integration``) funnel their checks through this module so every
run prints the same per-TODO status table with targeted hints, and exits
nonzero only on real failures.

Statuses:
- ``PASS``  - your code ran and produced the expected result.
- ``FAIL``  - your code ran but did the wrong thing; the hint names the
  integration point to look at.
- ``TODO``  - the stub still raises NotImplementedError; go to its notebook.
- ``SKIP``  - this machine can't run the check (missing hardware/deps); the
  detail says what would unlock it.
"""

from __future__ import annotations

import argparse
import importlib
import traceback
from dataclasses import dataclass
from pathlib import Path

# The workshop's TODO map: id -> (your file in project/, what, its finished
# reference in the pipeline, the notebook that teaches it). The reference has
# the same name and parameters (tests/sim/test_copies.py checks it).
TODO_INDEX = {
    "1.1": ("models/npu.py", "npu_available", "vla_pipeline/utils/npu_session.py",
            "notebooks/01_ai_models/01_npu_yolo.ipynb"),
    "1.2": ("models/npu.py", "build_npu_session", "vla_pipeline/utils/npu_session.py",
            "notebooks/01_ai_models/01_npu_yolo.ipynb"),
    "1.3": ("models/npu.py", "active_provider", "vla_pipeline/utils/npu_session.py",
            "notebooks/01_ai_models/01_npu_yolo.ipynb"),
    "1.4": ("models/hands.py", "create_hand_tracker", "vla_pipeline/vision/mediapipe_hands.py",
            "notebooks/01_ai_models/02_cpu_mediapipe.ipynb"),
    "1.5": ("models/llm.py", "llama_server_command", "vla_pipeline/llm/llama_intent.py",
            "notebooks/02_ros2/03_intent_node.ipynb"),
    "1.6": ("models/llm.py", "parse_intent", "vla_pipeline/llm/llama_intent.py",
            "notebooks/02_ros2/03_intent_node.ipynb"),
    "2.3": ("ros/camera_client.py", "Ros2Camera._setup_ros", "vla_pipeline/vision/camera.py",
            "notebooks/02_ros2/01_cameras.ipynb"),
    "2.7": ("ros/moveit_arm.py", "go_to_named", "vla_pipeline/robot/moveit_arm.py",
            "notebooks/02_ros2/02_robot.ipynb"),
    "2.8": ("ros/moveit_arm.py", "go_to_joints", "vla_pipeline/robot/moveit_arm.py",
            "notebooks/02_ros2/02_robot.ipynb"),
    "2.9": ("ros/moveit_arm.py", "grip", "vla_pipeline/robot/moveit_arm.py",
            "notebooks/02_ros2/02_robot.ipynb"),
    "2.10": ("ros/intent_node.py", "IntentNode._setup_ros", "vla_pipeline/llm/intent_node.py",
             "notebooks/02_ros2/03_intent_node.ipynb"),
    "2.11": ("ros/intent_node.py", "IntentNode._on_transcript", "vla_pipeline/llm/intent_node.py",
             "notebooks/02_ros2/03_intent_node.ipynb"),
    "3.1": ("integration/dispatch.py", "build_registry", "vla_pipeline/user_robot_control.py",
            "notebooks/03_integration/01_behavior.ipynb"),
    "3.2": ("integration/dispatch.py", "handle_transcript", "vla_pipeline/user_robot_control.py",
            "notebooks/03_integration/01_behavior.ipynb"),
    "3.3": ("integration/mimic.py", "mimic_tick", "vla_pipeline/behaviors/gesture_mimic.py",
            "notebooks/03_integration/01_behavior.ipynb"),
    "3.4": ("integration/behaviors.py", "run_dance", "vla_pipeline/behaviors/dance.py",
            "notebooks/03_integration/01_behavior.ipynb"),
}


ARTIFACTS = Path(__file__).resolve().parents[1] / "project" / "_artifacts"  # what the self-tests write


def implementation(todo_id: str, reference: bool = False):
    """The module holding ``todo_id``: your file in ``project/``, or with ``reference`` the pipeline's."""
    where, _what, ref, _nb = TODO_INDEX[todo_id]
    path = ref if reference else f"vvla_workshop/project/{where}"
    return importlib.import_module(path[: -len(".py")].replace("/", "."))


def reference_flag(description: str) -> bool:
    """A self-test's command line: True with ``--reference`` (check the pipeline's finished code)."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--reference", action="store_true", help="check the pipeline's finished code")
    return ap.parse_args().reference


@dataclass
class CheckResult:
    """One check's outcome: which TODO it covers, its name, status, and detail."""

    todo: str  # TODO id ("1.2") or "-" for infrastructure checks
    name: str
    status: str  # PASS / FAIL / TODO / SKIP
    detail: str = ""


class Reporter:
    """Prints and accumulates PASS/FAIL/TODO/SKIP results for one self-test run."""

    def __init__(self, title: str):
        self.title = title
        self.results: list = []
        self.artifacts: list = []
        print(f"\n=== {title} ===")

    # ------------------------------------------------------------------

    def add(self, todo: str, name: str, status: str, detail: str = "") -> None:
        """Record one result and print its status line."""
        self.results.append(CheckResult(todo, name, status, detail))
        mark = {"PASS": "[ok]", "FAIL": "[x]", "TODO": "…", "SKIP": "-"}[status]
        line = f"  [{status:4s}] {mark} {name}"
        if todo != "-" and todo in TODO_INDEX:
            where, what, _ref, _nb = TODO_INDEX[todo]
            line += f"   (TODO {todo}: {what} in project/{where})"
        print(line)
        if detail:
            for row in detail.splitlines():
                print(f"          {row}")

    def artifact(self, path) -> None:
        """Record and print the path to a generated artifact (e.g. a saved image)."""
        self.artifacts.append(str(path))
        print(f"          artifact: {path}")

    # ------------------------------------------------------------------

    def run(self, todo: str, name: str, fn, hint: str = ""):
        """Run one check. Returns ``(ok, value)``.

        - NotImplementedError  → TODO (with the notebook to read)
        - AssertionError       → FAIL, the assertion message IS the feedback
        - other exceptions     → FAIL with hint + traceback tail
        """
        try:
            value = fn()
        except NotImplementedError as e:
            nb = TODO_INDEX.get(todo, ("", "", "", ""))[3]
            self.add(todo, name, "TODO", f"not implemented yet → open {nb}\n({e})")
            return False, None
        except AssertionError as e:
            self.add(todo, name, "FAIL", str(e) or "assertion failed")
            return False, None
        except Exception as e:
            tail = traceback.format_exc().strip().splitlines()[-1]
            detail = f"{tail}"
            if hint:
                detail += f"\nhint: {hint}"
            self.add(todo, name, "FAIL", detail)
            return False, None
        self.add(todo, name, "PASS")
        return True, value

    # ------------------------------------------------------------------

    def summary(self) -> int:
        """Print the PASS/FAIL/TODO/SKIP tally and the next TODOs; return the exit code."""
        counts = {s: 0 for s in ("PASS", "FAIL", "TODO", "SKIP")}
        for r in self.results:
            counts[r.status] += 1
        print(
            f"\n--- {self.title}: "
            + "  ".join(f"{k} {v}" for k, v in counts.items())
            + " ---"
        )
        todos = sorted(
            {r.todo for r in self.results if r.status == "TODO"},
            key=lambda s: [int(x) for x in s.split(".")] if s[0].isdigit() else [99],
        )
        if todos:
            print("Next up:")
            for t in todos:
                where, what, _ref, nb = TODO_INDEX[t]
                print(f"  TODO {t}: {what}  →  project/{where}   (read {nb})")
        if counts["FAIL"] == 0 and counts["TODO"] == 0 and counts["PASS"] > 0:
            print("ALL CHECKS PASSED [ok]")
        return 1 if counts["FAIL"] else 0
