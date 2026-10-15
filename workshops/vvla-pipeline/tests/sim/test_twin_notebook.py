# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``02_digital_twin.ipynb`` section 5, driven the way JupyterLab drives it.

The notebook runs top to bottom in the ``vvla-workshop`` kernel. Then the
console cell (``c0000016``) runs for real: its ``ui_events`` loop and its
dispatch thread. The test types into the console's text box and clicks its
buttons with the same widget messages a browser sends. So a typed command
takes the console's own path: ``submit`` -> ``on_transcript`` -> the
``commands`` queue -> ``dispatch_loop`` -> ``run_command`` ->
``handle_transcript``.

What the kernel does is checked three ways: a second client on the
kernel's iopub channel (matched by content, so it does not matter which
cell the dispatch thread's output is attributed to), ROS readings of the
twin taken by this process, and the kernel namespace once the session ends.

Only the in-memory copy of the notebook gets a stub cell (no microphone, no
Llama, no webcam). The file on disk is not changed.

Tier 2 (ROS 2): ``source /opt/ros/jazzy/setup.bash && pytest tests/sim -m hw``.
The notebook's YOLO runs on the NPU (the CPU without the VitisAI provider),
so no other kernel may hold both NPU contexts while this runs.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import pytest

from twin_ros_rig import BODY_JOINTS, Probe, join_test_domain, quat_close, xml_poses

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.slow, pytest.mark.timeout(900)]
pytest.importorskip("rclpy")
nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")

NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "workshop" / "notebooks" / "03_integration" / "02_digital_twin.ipynb"
)
KERNEL = "vvla-workshop"
RESET_LINE = "Reset. Arm parked, objects back on the table."

STUBS = '''\
# notebook test: no microphone, no Llama, no webcam (a scripted person stands in)
import common.speech as _speech
import models.llm as _llm
import ros.camera_client as _camera_client


class _NoMic:
    error = "stubbed by the notebook test"
    transcribing = False

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        return self

    def stop(self):
        pass


class _NoWhisper:
    provider = "stub"

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, *args, **kwargs):
        return ""


class _NoCamera:
    def read(self):
        return False, None

    def close(self):
        pass


def _no_llama(*args, **kwargs):
    raise RuntimeError("stubbed by the notebook test")


_speech.MicListener, _speech.Whisper = _NoMic, _NoWhisper
_llm.LlamaIntentParser = _no_llama
_camera_client.make_camera = lambda *args, **kwargs: _NoCamera()
'''

NAMESPACE = '''\
import json as _json
print("NS " + _json.dumps({
    "trace": pickup_trace,
    "dispatcher_alive": dispatcher.is_alive(),
    "log": [line for line, _color in console._log],
    "pid": twin.pid,
}))
'''


class Watcher:
    """A second client on the kernel's iopub channel: stdout and JSON displays, any parent."""

    def __init__(self, connection_info: dict):
        from jupyter_client import BlockingKernelClient

        self._kc = BlockingKernelClient()
        self._kc.load_connection_info(connection_info)
        self._kc.start_channels(shell=False, stdin=False, hb=False, control=False)
        self._lock = threading.Lock()
        self.stdout: list[str] = []
        self.records: list[dict] = []
        self._running = True
        self._thread = threading.Thread(target=self._pump, daemon=True, name="iopub-watch")
        self._thread.start()

    def _pump(self) -> None:
        while self._running:
            try:
                msg = self._kc.get_iopub_msg(timeout=0.2)
            except Exception:
                continue
            kind, content = msg["msg_type"], msg["content"]
            with self._lock:
                if kind == "stream" and content.get("name") == "stdout":
                    self.stdout.append(content["text"])
                elif kind in ("display_data", "execute_result"):
                    record = content.get("data", {}).get("application/json")
                    if isinstance(record, dict) and "text" in record:
                        self.records.append(record)

    def said(self, phrase: str) -> bool:
        with self._lock:
            return phrase in "".join(self.stdout)

    def record(self, text: str) -> dict | None:
        with self._lock:
            return next((r for r in self.records if r["text"] == text), None)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self._kc.stop_channels()


async def until(predicate, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.1)
    raise AssertionError(f"timed out after {timeout:.0f} s waiting for {what}")


class Console:
    """The running console cell: its widgets, found by label, and the messages a browser sends."""

    def __init__(self, client, watcher: Watcher):
        self._client = client
        self.watcher = watcher
        self.widgets: dict[str, str] = {}

    def find_widgets(self) -> bool:
        found = {}
        for comm_id, state in self._client.widget_state.items():
            model = state.get("_model_name")
            if model == "TextModel" and state.get("placeholder", "").startswith("or type a command"):
                found["typed"] = comm_id
            elif model == "ButtonModel" and state.get("description") in ("stop", "reset", "end session"):
                found[state["description"]] = comm_id
        self.widgets = found
        return {"typed", "stop", "reset", "end session"} <= found.keys()

    def _send(self, comm_id: str, data: dict) -> None:
        kc = self._client.kc
        kc.shell_channel.send(kc.session.msg("comm_msg", {"comm_id": comm_id, "data": data}))

    def type(self, text: str) -> None:
        """Type ``text`` into the console's box and press Enter."""
        self._send(self.widgets["typed"], {"method": "update", "state": {"value": text}, "buffer_paths": []})

    def click(self, label: str) -> None:
        self._send(self.widgets[label], {"method": "custom", "content": {"event": "click"}})

    async def record(self, text: str, timeout: float = 240.0) -> dict:
        return await until(lambda: self.watcher.record(text), timeout, f"the record of {text!r}")


def _cell(nb, cell_id: str) -> int:
    return next(i for i, c in enumerate(nb.cells) if c.get("id") == cell_id)


def _stdout(cell) -> str:
    return "".join(o.get("text", "") for o in cell.get("outputs", []) if o.get("output_type") == "stream")


def run_console(script) -> tuple[dict, int]:
    """Run the notebook through ``c0000014``, then ``script(console)`` against the live console cell.

    Returns the kernel namespace read after "end session" and the twin node's pid.
    The cleanup cell ``c0000018`` runs last.
    """
    from nbclient import NotebookClient

    nb = nbformat.read(NOTEBOOK, as_version=4)
    nb.cells.insert(_cell(nb, "28bc3f9c") + 1, nbformat.v4.new_code_cell(STUBS))
    nb.cells.append(nbformat.v4.new_code_cell(NAMESPACE))
    upto, console_cell, cleanup = _cell(nb, "c0000014"), _cell(nb, "c0000016"), _cell(nb, "c0000018")
    probe_cell = len(nb.cells) - 1
    client = NotebookClient(
        nb, kernel_name=KERNEL, timeout=900, resources={"metadata": {"path": str(NOTEBOOK.parent)}}
    )

    async def main():
        async with client.async_setup_kernel(env=os.environ.copy()):
            watcher = Watcher(client.km.get_connection_info())
            try:
                for i in range(upto + 1):
                    await client.async_execute_cell(nb.cells[i], i)
                task = asyncio.create_task(client.async_execute_cell(nb.cells[console_cell], console_cell))
                console = Console(client, watcher)
                try:
                    await until(console.find_widgets, 60.0, "the console widgets")
                    await script(console)
                finally:
                    if console.widgets.get("end session"):
                        console.click("end session")
                    await asyncio.wait_for(task, timeout=120.0)
                await client.async_execute_cell(nb.cells[probe_cell], probe_cell)
                line = next(t for t in _stdout(nb.cells[probe_cell]).splitlines() if t.startswith("NS "))
                namespace = json.loads(line[3:])
                await client.async_execute_cell(nb.cells[cleanup], cleanup)
                return namespace
            finally:
                watcher.close()

    namespace = asyncio.run(main())
    return namespace, namespace["pid"]


def _gone(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
    except FileNotFoundError:
        return True
    return state == "Z"


@pytest.fixture(scope="module")
def observers(workshop_cfg):
    from common.twin_ros import TwinScene

    join_test_domain()
    probe = Probe(workshop_cfg["robot"]["state_topic"])
    scene = TwinScene()
    yield probe, scene
    scene.close()
    probe.close()


@pytest.fixture
def no_twin(observers, workshop_cfg):
    probe, _scene = observers
    state = workshop_cfg["robot"]["state_topic"]
    assert probe.wait_publishers(state, 0, timeout=15), "a twin already runs on the test domain"
    yield
    assert probe.wait_publishers(state, 0, timeout=20), "the notebook's twin node outlived its cleanup cell"


def _assert_closed(namespace: dict, pid: int) -> None:
    assert not namespace["dispatcher_alive"], "the console's dispatch thread outlived end session"
    deadline = time.time() + 15.0
    while not _gone(pid) and time.time() < deadline:
        time.sleep(0.1)
    assert _gone(pid), "the cleanup cell left the twin node running"


@pytest.mark.parametrize(
    "text, body, spoken",
    [
        ("pick up the ball", "sports_ball", "sports ball"),
        ("pick up the bottle", "bottle", "bottle"),
        ("pick up the toothbrush", "toothbrush", "toothbrush"),
    ],
    ids=["ball", "bottle", "toothbrush"],
)
def test_console_pickup(observers, no_twin, text, body, spoken):
    _probe, scene = observers
    seen = {}

    async def script(console):
        console.type(text)
        record = await console.record(text)
        seen["record"] = record
        seen["reading"] = scene.reading(body, after=time.time(), timeout=5.0)
        await until(lambda: console.watcher.said(f"Got the {spoken}"), 5.0, f"'Got the {spoken}'")

    namespace, pid = run_console(script)
    record, reading = seen["record"], seen["reading"]
    assert record["body"] == body
    assert record["grasped"], record
    assert record["z"] - record["z0"] > 0.03, record
    assert reading["grasped"] and reading["xyz"][2] > 0.08, reading
    assert namespace["trace"] == [record], "the namespace trace and the iopub record differ"
    _assert_closed(namespace, pid)


def test_console_reset(observers, no_twin, workshop_cfg):
    from common.motion import REST_POSE

    probe, scene = observers
    home = xml_poses(workshop_cfg)
    seen = {}

    async def script(console):
        console.type("pick up the toothbrush")
        first = await console.record("pick up the toothbrush")
        assert first["grasped"], first
        console.click("reset")
        await until(lambda: console.watcher.said(RESET_LINE), 15.0, "the reset line")
        await asyncio.sleep(1.5)  # the props settle; the servos reach the parked goal
        seen["scene"] = scene.latest(after=time.time(), timeout=5.0)
        seen["joints"] = probe.joints(after=time.time(), timeout=5.0)
        console.type("pick up the ball")
        seen["ball"] = await console.record("pick up the ball")

    namespace, pid = run_console(script)
    for name, qpos in home.items():
        got = seen["scene"]["bodies"][name]
        assert got["xyz"] == pytest.approx(list(qpos[:3]), abs=0.01), name
        assert quat_close(got["quat"], qpos[3:7]), name
        assert not got["grasped"], name
    joints = seen["joints"]
    for k in BODY_JOINTS:
        assert joints[k] == pytest.approx(REST_POSE[k], abs=3.0), (k, joints)
    assert joints["gripper"] >= 70.0, joints
    ball = seen["ball"]
    assert ball["grasped"] and ball["z"] - ball["z0"] > 0.03, ball
    assert [r["text"] for r in namespace["trace"]] == ["pick up the toothbrush", "pick up the ball"]
    panel = " ".join(line.strip() for line in namespace["log"])  # the panel wraps long lines
    assert f"robot: {RESET_LINE}" in panel, namespace["log"]
    _assert_closed(namespace, pid)


def test_console_stop(observers, no_twin):
    probe, _scene = observers
    seen = {}

    async def script(console):
        start = probe.joints(timeout=10.0)
        console.type("pick up the bottle")

        def moving():
            j = probe.joints()
            return j and max(abs(j[k] - start[k]) for k in BODY_JOINTS) > 3.0

        await until(moving, 30.0, "the arm to start moving")
        console.click("stop")
        await asyncio.sleep(1.0)
        first = probe.joints(after=time.time())
        await asyncio.sleep(1.0)
        second = probe.joints(after=time.time())
        seen["drift"] = {k: round(second[k] - first[k], 2) for k in BODY_JOINTS}
        seen["record"] = await console.record("pick up the bottle", timeout=30.0)

    namespace, pid = run_console(script)
    moved = {k: d for k, d in seen["drift"].items() if abs(d) > 1.0}
    assert not moved, f"the arm kept moving after stop: {moved}"
    assert seen["record"]["grasped"] is False, seen["record"]
    assert [r["text"] for r in namespace["trace"]] == ["pick up the bottle"]
    _assert_closed(namespace, pid)
