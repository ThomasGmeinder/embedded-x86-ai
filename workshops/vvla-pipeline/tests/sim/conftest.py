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

Videos + timelines land in ``_artifacts/sim_tests/<tier>/`` with an
``index.html`` that plays them all. Each ``.jsonl`` timeline holds the
props' poses, grasps and contacts every 0.2 s, every touch between a prop
and another body, the run's log lines and the full failure message
(:mod:`so101_sim.record`). A failed test also leaves
``_artifacts/sim_tests/failures/<test>/``: what each fixture with a
``failure_report`` saved (joint timeline, twin node log, notebook kernel log),
summarized at the end of its report.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
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


def pytest_sessionfinish(session, exitstatus):
    from so101_sim.record import write_index

    results = getattr(session.config, "_sim_results", {})
    for tier_dir, res in results.items():
        if Path(tier_dir).is_dir():
            write_index(tier_dir, res, title=f"SO-101 digital-twin scenarios ({Path(tier_dir).name})")


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


@pytest.fixture
def world():
    """A twin whose physics does not run (``SimWorld(start=False)``), one per test."""
    from so101_sim.world import SimWorld

    sim = SimWorld(start=False)
    yield sim
    sim.close()


@pytest.fixture(scope="module")
def probe(pipeline_cfg):
    """The test domain's joint states and graph, read by this process (``twin_ros_rig.Probe``)."""
    from twin_ros_rig import Probe, join_test_domain

    join_test_domain()
    p = Probe(pipeline_cfg)
    yield p
    p.close()


@dataclass
class Twin:
    world: object
    renderer: object


@pytest.fixture(scope="session")
def twins():
    """``twins(cfg)``: the session's one SimWorld + RenderService, created on first use."""
    made: list[Twin] = []

    def get(cfg: dict) -> Twin:
        if not made:
            from so101_sim.render import RenderService
            from so101_sim.world import SimWorld
            from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR, REST_POSE

            sim = dict(cfg.get("sim") or {})
            gm = dict(sim.get("gripper_map") or {})
            gm.setdefault("closed_cal", GRIPPER_CLOSED_FLOOR)
            sim["gripper_map"] = gm
            world = SimWorld({"sim": sim}, rest_pose=REST_POSE)
            made.append(Twin(world, RenderService(world, require_hw=False)))
        return made[0]

    yield get
    for t in made:
        t.renderer.close()
        t.world.close()


def _video_path(request, tier: str) -> Path | None:
    mode = request.config.getoption("--sim-video")
    if mode == "none":
        return None
    out = Path(request.config.getoption("--sim-video-dir")) / tier
    return out / f"{_slug(request.node.nodeid)}.mp4"


def _finish(request, rig, tier: str) -> None:
    item = request.node
    call = getattr(item, "rep_call", None)
    setup = getattr(item, "rep_setup", None)
    if call is None:
        status = "failed" if (setup is not None and setup.failed) else "skipped"
        message = ""
    else:
        xfail = hasattr(call, "wasxfail")
        if call.passed and not xfail:
            status = "passed"
        elif call.skipped and xfail:
            status = "xfailed"
        elif call.passed and xfail:
            status = "xpassed"
        else:
            status = "failed" if call.failed else "skipped"
        message = ""
        if call.failed and call.longrepr is not None:
            crash = getattr(call.longrepr, "reprcrash", None)
            message = crash.message if crash is not None else str(call.longrepr)[-300:]
        elif xfail:
            message = str(call.wasxfail)
    mode = request.config.getoption("--sim-video")
    keep = mode == "all" or (mode == "failed" and status in ("failed", "xpassed"))
    path = rig.finish(status, message, keep=keep)
    if path is not None:
        results = request.config._sim_results.setdefault(str(path.parent), {})
        results[path.stem] = status


@pytest.fixture(scope="session")
def eyes(request, pipeline_cfg):
    """``eyes()``: the pipeline's YOLOv26s ``ObjectDetector`` (one session),
    or ``None`` with ``--sim-eyes seg``."""
    made: list = []
    kind = request.config.getoption("--sim-eyes")
    device = request.config.getoption("--sim-yolo-device")

    def get():
        if kind == "seg":
            return None
        if not made:
            from vla_pipeline.vision.yolo_detect_npu import ObjectDetector

            made.append(ObjectDetector(pipeline_cfg, device=device))
        providers = made[0].session.get_providers()
        if device == "npu" and "VitisAIExecutionProvider" not in providers:
            pytest.fail(
                f"--sim-yolo-device npu, but the detector runs on {providers}: "
                "source scripts/ryzen_ai_env.sh with RYZEN_AI_WHEELS set",
                pytrace=False,
            )
        return made[0]

    return get


@pytest.fixture
def rig(request, pipeline_cfg, twins, mock_llm, eyes):
    """The tier-1 rig: the pipeline on the session's twin."""
    from rigs import PipelineRig

    mock_llm.requests.clear()
    made = PipelineRig(
        twins(pipeline_cfg), pipeline_cfg, mock_llm.base_url,
        recorder_path=_video_path(request, "tier1"), title=f"pipeline: {request.node.name}",
        detector=eyes(),
    )
    yield made
    _finish(request, made, "tier1")


@pytest.fixture
def video_path(request):
    """Per-test recording path for tier-2 rigs (None with --sim-video none)."""
    return _video_path(request, "tier2")


@pytest.fixture
def finish_rig(request):
    """Tier-2 tests register their rig here so it is stamped + closed."""
    holder = {}
    yield holder
    if "rig" in holder:
        _finish(request, holder["rig"], "tier2")
