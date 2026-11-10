# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Digital-twin scenario tests: fixtures, options, video bookkeeping.

    cd workshop && ./run_notebooks.sh --test          # notebooks and the Instructor's corner
    ./run_sim_tests.sh                                # NPU load order and the twin scenarios
    ./run_sim_tests.sh --sim-yolo-device cpu          # YOLO on the CPU
    ./run_sim_tests.sh --sim-video failed             # keep only failing runs' videos
    ./run_sim_tests.sh --sim-repeat 20 -x -k mimic    # hunt a flaky test
    ./run_sim_tests.sh --sim-eyes seg                 # ground-truth boxes instead of YOLO

A failed test leaves ``_artifacts/sim_tests/failures/<test>/``: what each
fixture with a ``failure_report`` saved (joint timeline, twin node log,
notebook kernel log), summarized at the end of its report.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
try:
    from vla_pipeline.utils.config import REPO_ROOT as PIPE_ROOT
except ImportError as exc:
    pytest.exit(f"vla_pipeline is not installed in {sys.prefix} ({exc}): run ./bootstrap.sh", returncode=3)


def pytest_addoption(parser):
    g = parser.getgroup("sim", "SO-101 digital twin")
    g.addoption("--sim-video", choices=("all", "failed", "none"), default="all",
                help="which scenario recordings to keep (default: all)")
    g.addoption("--sim-video-dir", default=str(PIPE_ROOT / "_artifacts" / "sim_tests"),
                help="where recordings + index.html go")
    g.addoption("--sim-eyes", choices=("yolo", "seg"), default="yolo",
                help="arm-camera detector: the pipeline's YOLOv26s on the rendered frames, as in the "
                     "notebook (default), or ground-truth boxes from the twin's segmentation")
    g.addoption("--sim-yolo-device", choices=("cpu", "npu"), default="npu",
                help="where the YOLO eyes run (default npu, as in the notebooks; "
                     "./run_sim_tests.sh sources scripts/ryzen_ai_env.sh)")
    g.addoption("--sim-repeat", type=int, default=1,
                help="run every selected test this many times, to catch a flaky one")


def pytest_generate_tests(metafunc):
    repeat = metafunc.config.getoption("--sim-repeat")
    if repeat > 1:
        metafunc.fixturenames.append("_sim_run")
        metafunc.parametrize("_sim_run", range(1, repeat + 1), ids=lambda i: f"run{i}")


def _require_this_clone() -> None:
    """Exit unless this is the clone's venv and this clone's so101_sim.

    .venv/bin/python is a symlink to the system interpreter, so resolving
    sys.executable makes /usr/bin/pytest look like the venv. sys.prefix is
    the venv root only when the venv interpreter was used.
    """
    venv = (PIPE_ROOT / ".venv").resolve()
    prefix = Path(sys.prefix).resolve()
    commands = "cd workshop && ./run_notebooks.sh --test\n./run_sim_tests.sh"
    if PIPE_ROOT != Path(__file__).resolve().parents[2]:
        pytest.exit(f"vla_pipeline is {PIPE_ROOT}'s, not this clone's\n{commands}", returncode=3)
    if prefix != venv:
        pytest.exit(f"{sys.executable} is not {venv / 'bin' / 'python'}\n{commands}", returncode=3)
    try:
        import so101_sim
    except ImportError as exc:
        pytest.exit(f"so101_sim is not installed in {venv} ({exc})\n{commands}", returncode=3)
    src = Path(so101_sim.__file__).resolve()
    root = (PIPE_ROOT / "so101_sim").resolve()
    if root not in src.parents:
        pytest.exit(f"so101_sim is {src}, not this clone's {root}\n{commands}", returncode=3)


# These tests construct a MoveItPy in this process and need the domain empty.
# Collected with the shared ``twin`` fixture, the second constructor segfaults
# when the first MoveItPy is destroyed (moveit2#3721). ./run_sim_tests.sh
# runs them in their own processes. Fail here, before any bringup.
_OWN_MOVEIT_PY = frozenset({
    "test_the_twin_domain_is_private",
    "test_the_node_dies_with_its_kernel",
    "test_reset_service_restores_the_scene_and_parks_the_arm",
})


def _still_selected(config, items):
    """Items ``--deselect`` keeps. Pytest applies that option in its own hook,
    and the node id it compares depends on the root directory."""
    prefixes = tuple(config.getoption("deselect") or [])
    if not prefixes:
        return items
    kept = []
    for item in items:
        node = item.nodeid
        if any(node.startswith(prefix) or prefix.endswith(node) for prefix in prefixes):
            continue
        kept.append(item)
    return kept


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    items = _still_selected(config, items)
    moveit_files = set()
    shared, own = [], []
    for item in items:
        path = Path(str(item.fspath)).name
        if path in {"test_twin_infra.py", "test_twin_ros.py"}:
            moveit_files.add(path)
        name = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
        if "twin" in item.fixturenames or "rig" in item.fixturenames:
            shared.append(item.nodeid)
        if name in _OWN_MOVEIT_PY:
            own.append(item.nodeid)
    if len(moveit_files) > 1:
        pytest.exit(
            "test_twin_infra.py and test_twin_ros.py each construct a MoveItPy. "
            "A second one in this process segfaults on destruction (moveit2#3721). "
            "Run ./run_sim_tests.sh.",
            returncode=3,
        )
    if shared and own:
        listed = "\n".join(own)
        pytest.exit(
            "These tests construct their own MoveItPy and need a free domain. "
            "Collected with the shared twin, the second constructor segfaults "
            f"(moveit2#3721):\n{listed}\nRun ./run_sim_tests.sh.",
            returncode=3,
        )


def pytest_configure(config):
    _require_this_clone()
    config.addinivalue_line("markers", "sim: digital-twin scenario (tier 1 unless also hw)")
    config.addinivalue_line("markers", "hw: tier 2 - full ROS 2 graph on the iGPU")
    config.addinivalue_line("markers", "slow: long-running scenario")
    config.addinivalue_line("markers", "docs: runs what workshop/README.md documents (notebooks, commands)")
    config._sim_results = {}


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    item._sim_t0 = time.time()


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    rep = yield
    setattr(item, f"rep_{rep.when}", rep)
    if rep.failed and rep.when in ("setup", "call"):
        bundle = _failure_bundle(item)
        if bundle:
            rep.sections.append(("failure bundle", bundle))
    return rep


def _failure_bundle(item) -> str:
    """Evidence from every fixture with a ``failure_report(dest, since)``, saved under failures/<test>/."""
    reporters, seen = [], set()
    for value in getattr(item, "funcargs", {}).values():
        for obj in value if isinstance(value, tuple) else (value,):
            report = getattr(obj, "failure_report", None)
            if callable(report) and id(obj) not in seen:
                seen.add(id(obj))
                reporters.append((type(obj).__name__, report))
    if not reporters:
        return ""
    dest = Path(item.config.getoption("--sim-video-dir")) / "failures" / _slug(item.nodeid)
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    parts = [f"saved in {dest}"]
    for name, report in reporters:
        try:
            parts.append(report(dest, getattr(item, "_sim_t0", None)))
        except Exception as e:
            parts.append(f"{name}.failure_report failed: {e!r}")
    return "\n\n".join(parts)


def _tier(item) -> str:
    return "tier2" if item.get_closest_marker("hw") else "tier1"


def _slug(nodeid: str) -> str:
    name = nodeid.split("::", 1)[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


# ---------------------------------------------------------------------------
# configs / services
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def pipeline_cfg():
    from vla_pipeline.utils.config import load_config

    cfg = load_config()
    cfg["dance"]["duration_s"] = 5.0  # keep the dance scenario short
    cfg.setdefault("sim", {})["require_hw_gl"] = False
    return cfg


@pytest.fixture(scope="session")
def mock_llm():
    """The workshop's llama-server imposter (keyword routing, records payloads)."""
    from vvla_workshop.fixtures import MockLlamaServer

    with MockLlamaServer() as server:
        yield server
