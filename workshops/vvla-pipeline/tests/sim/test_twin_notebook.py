# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``02_digital_twin.ipynb`` section 5, driven the way JupyterLab drives it.

The module starts one ``vvla-workshop`` kernel and runs the notebook through
``c0000014`` once. Each test runs only the console cell (``c0000016``): its
``ui_events`` loop and its dispatch thread. The test types into the console's
text box and clicks its buttons with the same widget messages a browser sends.
So a typed command takes the console's own path: ``submit`` ->
``voice.on_transcript`` -> the ``voice.commands`` queue -> ``voice.dispatch_loop``
-> ``voice.run`` -> ``handle_transcript`` (``voice`` is the notebook's
``common.twin_voice.VoiceCommands``).

What the kernel does is checked three ways: a second client on the
kernel's iopub channel (matched by content, so it does not matter which
cell the dispatch thread's output is attributed to), ROS readings of the
twin taken by this process, and the kernel namespace once the session ends.

The in-memory copy of the notebook starts with ``RIG_STANDINS``
(:mod:`notebook_harness`): a photo of people is the webcam and there is no
microphone. Whisper, Llama on the iGPU and YOLO on the NPU are real. The
file on disk is not changed.

``cd workshop && ./run_notebooks.sh --test``
(without that setup YOLO cannot load the NPU and the test fails). The
notebook kernel holds both YOLO NPU contexts, so this runs apart from the
tests whose YOLO runs in the pytest process.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from pathlib import Path

import pytest

from notebook_harness import (
    NOTEBOOKS,
    Console,
    Watcher,
    cell_index,
    cell_stdout,
    client_for,
    kernel_env,
    until,
    with_standins,
)
from twin_ros_rig import BODY_JOINTS, Probe, join_test_domain, quat_close, xml_poses

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.docs, pytest.mark.slow, pytest.mark.timeout(900)]
pytest.importorskip("rclpy")
nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")

NOTEBOOK = NOTEBOOKS / "03_integration" / "02_digital_twin.ipynb"
RESET_LINE = "Reset. Arm parked, objects back on the table."  # common.twin_voice.RESET_LINE

NAMESPACE = '''\
import json as _json
print("NS " + _json.dumps({
    "trace": voice.trace,
    "dispatcher_alive": dispatcher.is_alive(),
    "log": [line for line, _color in console._log],
    "pid": twin.pid,
    "sessions": _SESSIONS,
}))
'''

# voice.reset_scene() returns None. It clears ctx.stop_event, which the console cell's finally leaves set.
SETTLE = '''\
voice.reset_scene()
voice.trace.clear()
console._log.clear()
print("SETTLED")
'''

SESSIONS = '''\
import json as _json
print("SESSIONS " + _json.dumps(_SESSIONS))
'''

# Same line the notebook-run test rejects. Checked once on this kernel's whole transcript.
_FALLBACK = re.compile(r"fall(s|ing)? back to (the )?cpu", re.I)

def _gone(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
    except FileNotFoundError:
        return True
    return state == "Z"


def _assert_dispatcher_dead(namespace: dict) -> None:
    assert not namespace["dispatcher_alive"], "the console's dispatch thread outlived end session"


def _assert_yolo_on_npu(sessions) -> None:
    yolo = [(model, provider) for model, provider in sessions if "yolo" in model.lower()]
    assert yolo, "the notebook made no YOLO session"
    off_npu = [(Path(model).name, provider) for model, provider in yolo if provider != "VitisAIExecutionProvider"]
    assert not off_npu, f"YOLO ran off the NPU: {off_npu}"


def _assert_twin_gone(pid: int) -> None:
    deadline = time.time() + 15.0
    while not _gone(pid) and time.time() < deadline:
        time.sleep(0.1)
    assert _gone(pid), "the cleanup cell left the twin node running"


class NotebookKernel:
    """One kernel for the module. Scenarios share it; the console cell is the part that re-runs.

    ``nbclient`` binds the client to the loop that started the kernel, so that loop
    lives on one thread and every scenario is submitted onto it.
    """

    def __init__(self):
        self.watcher: Watcher | None = None
        self.needs_new_kernel = False
        self.last_pid: int | None = None
        self.sessions = None
        self._loop = asyncio.new_event_loop()
        self._open = threading.Event()
        self._closing = threading.Event()
        self._down = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, name="twin-notebook", daemon=False)
        self._serve_fut = None
        self._load_notebook()

    def _load_notebook(self) -> None:
        self.nb = with_standins(nbformat.read(NOTEBOOK, as_version=4))
        self.client = client_for(self.nb, NOTEBOOK, timeout=900)
        self.console_cell = cell_index(self.nb, "c0000016")
        self.cleanup = cell_index(self.nb, "c0000018")

    def _thread_main(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _call(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def start(self) -> None:
        self._thread.start()
        self._serve_fut = asyncio.run_coroutine_threadsafe(self._serve(), self._loop)
        try:
            while not self._open.wait(0.2):
                if self._serve_fut.done():
                    self._serve_fut.result()
        except Exception:
            self.finish()
            raise

    def finish(self) -> None:
        try:
            # _down is set once the kernel process has exited. Coverage needs a live kernel.
            if self._open.is_set() and not self._closing.is_set() and not self._down.is_set():
                self._call(self._coverage_and_cleanup())
        finally:
            self._closing.set()
            self._down.wait(timeout=180)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=30)

    def run(self, script, end_with: str = "end session", after=None) -> dict:
        """Run ``script`` against a fresh console cell. ``end_with`` is "end session", "interrupt", or "rerun"."""
        return self._call(self._scenario(script, end_with, after))

    async def _serve(self) -> None:
        self._down.clear()
        try:
            async with self.client.async_setup_kernel(env=kernel_env()):
                self.watcher = Watcher(self.client.km.get_connection_info())
                try:
                    await self._through("c0000014")
                    self._open.set()
                    while not self._closing.is_set():
                        await asyncio.sleep(0.1)
                finally:
                    if self.watcher is not None:
                        self.watcher.close()
                        self.watcher = None
        finally:
            self._down.set()

    async def _restart(self) -> None:
        """SIGINT during mimic wedges the NPU session, and the rerun test closes the stack.

        A new kernel process is the recovery. Re-running those cells in the same process
        starts a second twin or dies with the wedged device.
        """
        self._closing.set()
        while not self._down.is_set():
            await asyncio.sleep(0.05)
        self._closing.clear()
        self._open.clear()
        self._load_notebook()
        asyncio.create_task(self._serve())
        await asyncio.sleep(0)
        deadline = time.monotonic() + 900
        while not self._open.is_set():
            if self._down.is_set():
                raise RuntimeError("the notebook kernel did not come back")
            if time.monotonic() > deadline:
                raise TimeoutError("the notebook kernel did not come back")
            await asyncio.sleep(0.1)

    async def _through(self, cell_id: str) -> None:
        for i in range(cell_index(self.nb, cell_id) + 1):
            await self.client.async_execute_cell(self.nb.cells[i], i)

    async def _execute_new(self, source: str) -> str:
        cell = nbformat.v4.new_code_cell(source)
        self.nb.cells.append(cell)
        await self.client.async_execute_cell(cell, len(self.nb.cells) - 1)
        return cell_stdout(cell)

    async def _probe(self) -> dict:
        text = await self._execute_new(NAMESPACE)
        line = next(t for t in reversed(text.splitlines()) if t.startswith("NS "))
        return json.loads(line[3:])

    async def _settle(self) -> None:
        text = await self._execute_new(SETTLE)
        if "reset failed" in text or "SETTLED" not in text:
            raise RuntimeError(f"voice.reset_scene() did not park the arm:\n{text}")

    async def _end_cell(self, task, console: Console, end_with: str) -> float:
        from nbclient.exceptions import CellExecutionError

        ending = time.monotonic()
        if end_with == "interrupt":
            await self.client.km.interrupt_kernel()
        elif end_with == "rerun":
            for i in (self.console_cell, self.cleanup):
                self.client.kc.execute(self.nb.cells[i].source)
        elif not task.done() and console.widgets.get("end session"):
            console.click("end session")
        try:
            await asyncio.wait_for(task, timeout=120.0)
        except asyncio.TimeoutError:
            await self.client.km.interrupt_kernel()
            raise
        except CellExecutionError as e:
            if end_with != "interrupt" or "KeyboardInterrupt" not in str(e):
                raise
        return time.monotonic() - ending

    async def _scenario(self, script, end_with: str, after) -> dict:
        if self.needs_new_kernel:
            await self._restart()
            self.needs_new_kernel = False
        assert self.watcher is not None
        mark = self.watcher.mark()
        task = asyncio.create_task(
            self.client.async_execute_cell(self.nb.cells[self.console_cell], self.console_cell)
        )
        console = Console(self.client, self.watcher, before=set(self.client.widget_state), since=mark)
        error = None
        try:
            await until(console.find_widgets, 60.0, "the console widgets")
            await script(console)
        except Exception as e:
            error = e
        ended_s = await self._end_cell(task, console, end_with)
        if error is None and after is not None:
            try:
                await after()
            except Exception as e:
                error = e
        try:
            namespace = await self._probe()
        except Exception as e:
            self.needs_new_kernel = True
            raise error or e
        namespace.update(
            ended_s=ended_s,
            parked=self.watcher.said("session ended, arm parked", since=mark),
            twin_alive=not _gone(namespace["pid"]),
        )
        self.last_pid = namespace["pid"]
        self.sessions = namespace["sessions"]
        if end_with in ("rerun", "interrupt"):
            # Interrupt wedges the NPU session. Rerun already ran the cleanup cell.
            # Either way the next scenario needs its own kernel. The probe above already
            # recorded whether this scenario's twin was still alive.
            self.needs_new_kernel = True
        elif error is None:
            await self._settle()
        if error is not None:
            raise error
        return namespace

    async def _coverage_and_cleanup(self) -> None:
        sessions = self.sessions
        try:
            text = await self._execute_new(SESSIONS)
            line = next(t for t in reversed(text.splitlines()) if t.startswith("SESSIONS "))
            sessions = json.loads(line[len("SESSIONS "):])
        except Exception:
            if sessions is None:
                raise
        _assert_yolo_on_npu(sessions)
        assert self.watcher is not None
        fallback = [line.strip() for line in self.watcher.text().splitlines() if _FALLBACK.search(line)]
        assert not fallback, f"02_digital_twin fell back to the CPU: {fallback}"
        if self.last_pid is not None and not _gone(self.last_pid):
            # A second hands.close() raises, so skip this when a scenario already ran c0000018.
            await self.client.async_execute_cell(self.nb.cells[self.cleanup], self.cleanup)
            _assert_twin_gone(self.last_pid)


@pytest.fixture(scope="module")
def observers(workshop_cfg):
    from common.twin_ros import TwinScene

    join_test_domain()
    probe = Probe(workshop_cfg["robot"]["state_topic"])
    scene = TwinScene()
    yield probe, scene
    scene.close()
    probe.close()


@pytest.fixture(scope="module")
def kernel(observers, workshop_cfg):
    probe, _scene = observers
    state = workshop_cfg["robot"]["state_topic"]
    assert probe.wait_publishers(state, 0, timeout=15), "a twin already runs on the test domain"
    session = NotebookKernel()
    session.start()
    try:
        yield session
    finally:
        session.finish()
    assert probe.wait_publishers(state, 0, timeout=20), "the notebook's twin node outlived its cleanup cell"


@pytest.mark.parametrize(
    "text, body, spoken",
    [
        ("pick up the ball", "sports_ball", "sports ball"),
        ("pick up the bottle", "bottle", "bottle"),
        ("pick up the toothbrush", "toothbrush", "toothbrush"),
    ],
    ids=["ball", "bottle", "toothbrush"],
)
def test_console_pickup(observers, kernel, text, body, spoken):
    _probe, scene = observers
    seen = {}

    async def script(console):
        console.type(text)
        record = await console.record(text)
        seen["record"] = record
        seen["reading"] = scene.reading(body, after=time.time(), timeout=5.0)
        await until(lambda: console.said(f"Got the {spoken}"), 5.0, f"'Got the {spoken}'")

    namespace = kernel.run(script)
    record, reading = seen["record"], seen["reading"]
    assert record["body"] == body
    assert record["grasped"], record
    assert record["z"] - record["z0"] > 0.03, record
    assert reading["grasped"] and reading["xyz"][2] > 0.08, reading
    assert namespace["trace"] == [record], "the namespace trace and the iopub record differ"
    _assert_dispatcher_dead(namespace)


def test_console_reset(observers, kernel, workshop_cfg):
    from common.motion import REST_POSE

    probe, scene = observers
    home = xml_poses(workshop_cfg)
    seen = {}

    async def script(console):
        console.type("pick up the toothbrush")
        first = await console.record("pick up the toothbrush")
        assert first["grasped"], first
        console.click("reset")
        await until(lambda: console.said(RESET_LINE), 15.0, "the reset line")
        await asyncio.sleep(1.5)  # the props settle; the servos reach the parked goal
        seen["scene"] = scene.latest(after=time.time(), timeout=5.0)
        seen["joints"] = probe.joints(after=time.time(), timeout=5.0)
        console.type("pick up the ball")
        seen["ball"] = await console.record("pick up the ball")

    namespace = kernel.run(script)
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
    _assert_dispatcher_dead(namespace)


def _arm_moved(probe, start) -> bool:
    j = probe.joints()
    return bool(j) and max(abs(j[k] - start[k]) for k in BODY_JOINTS) > 3.0


def test_console_end_session_during_mimic(observers, kernel):
    """"end session" while the arm mirrors the webcam: the console cell ends
    within 20 s and parks the arm. The module teardown closes the twin once."""
    probe, _scene = observers

    async def script(console):
        start = probe.joints(timeout=10.0)
        console.type("copy my movements")
        await until(lambda: _arm_moved(probe, start), 30.0, "the arm to follow the webcam")

    namespace = kernel.run(script)
    assert namespace["ended_s"] < 20.0, f"the console cell took {namespace['ended_s']:.0f} s to end"
    assert namespace["parked"], "the console cell did not park the arm"
    _assert_dispatcher_dead(namespace)


def test_console_interrupted(observers, kernel):
    """The kernel's stop button ends the console like "end session": the arm
    parks at rest, and the twin keeps running until the cleanup cell closes it."""
    from common.motion import REST_POSE

    probe, _scene = observers
    seen = {}

    async def script(console):
        start = probe.joints(timeout=10.0)
        console.type("copy my movements")
        await until(lambda: _arm_moved(probe, start), 30.0, "the arm to follow the webcam")

    async def after():
        seen["joints"] = probe.joints(after=time.time(), timeout=5.0)

    namespace = kernel.run(script, end_with="interrupt", after=after)
    assert namespace["ended_s"] < 20.0, f"the console cell took {namespace['ended_s']:.0f} s to end"
    assert namespace["parked"], "the interrupted console cell did not park the arm"
    assert namespace["twin_alive"], "the kernel's stop button also stopped the twin program"
    joints = seen["joints"]
    assert joints, "no joint state from the twin after the interrupt"
    off = {k: round(joints[k] - REST_POSE[k], 1) for k in BODY_JOINTS if abs(joints[k] - REST_POSE[k]) > 4.0}
    assert not off, f"the arm did not reach rest after the interrupt: {off}"
    _assert_dispatcher_dead(namespace)


def test_console_run_and_advance(observers, kernel):
    """Run and advance on the running console cell, then on the cleanup cell:
    the session ends and parks the arm, the console cell does not start a
    second session, and the cleanup cell closes the twin."""
    probe, _scene = observers
    seen = {}

    async def script(console):
        seen["console"] = console
        start = probe.joints(timeout=10.0)
        console.type("copy my movements")
        await until(lambda: _arm_moved(probe, start), 30.0, "the arm to follow the webcam")

    async def after():
        console = seen["console"]
        await until(lambda: console.said("twin closed"), 60.0, "the cleanup cell's 'twin closed'")
        console._client.kc.execute('print("kernel free")')
        await until(lambda: console.said("kernel free"), 30.0, "a cell run after the cleanup")
        seen["parked"] = console.count("session ended, arm parked")

    namespace = kernel.run(script, end_with="rerun", after=after)
    assert namespace["ended_s"] < 20.0, f"the console cell took {namespace['ended_s']:.0f} s to end"
    assert seen["parked"] == 1, "the console cell ran a second session"
    _assert_dispatcher_dead(namespace)
    _assert_twin_gone(namespace["pid"])


def test_console_stop(observers, kernel):
    """The red stop button ends the console cell: the in-flight pickup is not a grasp
    and the arm parks. Saying "stop" only halts the behavior. The module teardown closes the twin."""
    from common.motion import REST_POSE

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
        await until(lambda: console.said("session ended, arm parked"), 20.0, "the console cell to end")
        seen["record"] = await console.record("pick up the bottle", timeout=30.0)
        seen["joints"] = probe.joints(after=time.time(), timeout=5.0)

    namespace = kernel.run(script)
    assert namespace["ended_s"] < 20.0, f"the console cell took {namespace['ended_s']:.0f} s to end"
    assert namespace["parked"], "the stop button did not end the console cell"
    assert seen["record"]["grasped"] is False, seen["record"]
    assert [r["text"] for r in namespace["trace"]] == ["pick up the bottle"]
    joints = seen["joints"]
    assert joints, "no joint state from the twin after stop"
    off = {k: round(joints[k] - REST_POSE[k], 1) for k in BODY_JOINTS if abs(joints[k] - REST_POSE[k]) > 4.0}
    assert not off, f"the arm did not reach rest after stop: {off}"
    _assert_dispatcher_dead(namespace)
