# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The recording's ``.jsonl`` timeline holds what a failure analysis needs.

Every ``state`` sample carries each prop's pose, motion, grasp and contacts
(:meth:`SimWorld.props_report`); a prop starting or stopping to touch
another body is a ``touch`` event; the log records of the run are ``log``
events; ``result`` keeps the whole failure message. The index page lists
the events without the samples and log records.
"""

from __future__ import annotations

import json
import logging
import math
import threading

import numpy as np
import pytest

pytestmark = pytest.mark.sim

_KEYS = {"xyz", "yaw", "tilt", "speed", "spin", "grasped", "touching"}


def test_props_report_gives_pose_motion_grasp_and_contacts(world):
    """A bottle on its side with the ball against it, the toothbrush between
    the rest pose's nearly closed jaws, the book where the scene file has it."""
    from so101_sim.world import TABLE_Z

    side = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)  # 90 deg about x
    world.place("bottle", (0.30, 0.10, TABLE_Z + 0.02), side)
    world.place("sports_ball", (0.30 + 0.039, 0.10, TABLE_Z + 0.02))  # 1 mm into the bottle
    jaw = world.data.geom("moving_jaw_box3").xpos.copy()
    world.place("toothbrush", jaw)
    world.advance(world.dt)

    report = world.props_report()
    assert set(report) == set(world.objects)
    for name, prop in report.items():
        assert set(prop) == _KEYS, name
        assert prop["grasped"] is (name == "toothbrush"), name
        json.dumps(prop)
    bottle, ball, brush, book = (report[n] for n in ("bottle", "sports_ball", "toothbrush", "book"))
    assert bottle["tilt"] == pytest.approx(90.0, abs=2.0), bottle
    assert np.linalg.norm(np.subtract(bottle["xyz"], (0.30, 0.10, TABLE_Z + 0.02))) < 0.002, bottle
    assert "bottle_collision" in ball["touching"].get("bottle", []), ball
    assert "sports_ball" in bottle["touching"], bottle
    assert {"moving_jaw_so101_v1", "gripper"} <= set(brush["touching"]), brush
    assert book["touching"] == {}, f"the table is left out: {book}"
    assert book["tilt"] == pytest.approx(0.0, abs=1.0), book
    fall = 1000.0 * 9.81 * world.dt
    assert book["speed"] == pytest.approx(fall, abs=2.0), f"one step of free fall is {fall:.0f} mm/s: {book}"


def test_the_timeline_records_objects_touches_logs_and_the_whole_result(twins, pipeline_cfg, tmp_path):
    """A real recording: the ball resting on the book, one log line, a long failure message."""
    from so101_sim.record import SimRecorder, _TimelineLog

    twin = twins(pipeline_cfg)
    w = twin.world
    w.reset()
    book = w.body_pos("book")
    w.place("sports_ball", (book[0], book[1], 0.007 + 0.02))  # on the book's top face
    log = logging.getLogger("tests.timeline")
    message = "AssertionError: " + "x" * 900
    rec = SimRecorder(w, tmp_path / "run.mp4", renderer=twin.renderer, title="timeline").start()
    sampled = threading.Event()

    def after_recorder(service, due, t_sim):  # registered after the recorder's hook: it ran first
        if rec._3d_view in due:
            sampled.set()

    try:
        log.info("marker %d", 7)
        twin.renderer.add_hook(after_recorder)
        assert sampled.wait(10.0), "no rendered pass of the recorder's view"
        rec.stamp("failed", message)
    finally:
        twin.renderer.remove_hook(after_recorder)
        rec.stop(keep=True)
        w.reset()
    log.info("after the recording")
    assert not any(isinstance(h, _TimelineLog) for h in logging.getLogger().handlers)

    lines = [json.loads(line) for line in (tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()]
    start = lines[0]
    assert start["event"] == "record_start" and start["objects"] == w.objects, start
    assert start["scene"] == w.scene_path.name, start
    states = [e for e in lines if e["event"] == "state"]
    assert states, "no state sample"
    props = states[0]["props"]
    assert set(props) == set(w.objects) and all(set(p) == _KEYS for p in props.values()), props
    assert "book" in props["sports_ball"]["touching"], props["sports_ball"]
    assert {"behavior", "hand", "tripped", "joints", "goal", "gripper_current"} <= set(states[0]), states[0]
    touches = [e for e in lines if e["event"] == "touch" and e["object"] == "sports_ball"]
    assert touches and "book" in touches[0]["began"], touches
    logs = [e for e in lines if e["event"] == "log"]
    assert any(e["msg"] == "marker 7" and e["logger"] == "tests.timeline" and e["level"] == "INFO" for e in logs), logs
    assert not any(e["msg"] == "after the recording" for e in logs)
    result = next(e for e in lines if e["event"] == "result")
    assert result["status"] == "failed" and result["message"] == message


def test_the_index_lists_events_not_samples_or_log_lines(tmp_path):
    """450 samples and log lines ahead of the result no longer push it off the page."""
    from so101_sim.record import write_index

    (tmp_path / "run.mp4").write_bytes(b"")
    events = [{"t_sim": 0.1 * i, "event": "state", "props": {}} for i in range(450)]
    events += [{"t_sim": 45.0, "event": "log", "msg": "a planning line"}]
    events += [{"t_sim": 45.1, "event": "touch", "object": "bottle", "began": ["gripper"], "ended": []}]
    events += [{"t_sim": 45.2, "event": "result", "status": "failed", "message": "the bottle fell"}]
    (tmp_path / "run.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")

    page = write_index(tmp_path, {"run": "failed"}).read_text(encoding="utf-8")
    assert "<b>touch</b>" in page and "<b>result</b>" in page and "the bottle fell" in page
    assert "<b>state</b>" not in page and "<b>log</b>" not in page
    assert "timeline (2 events)" in page
