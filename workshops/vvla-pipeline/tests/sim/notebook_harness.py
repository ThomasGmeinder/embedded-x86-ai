# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Run a workshop notebook the way a student does in JupyterLab.

Each notebook runs top to bottom, as stored, in its own ``vvla-workshop``
kernel, the kernel ``bootstrap.sh`` registers, started as VS Code starts it:
ROS 2 and the NPU runtime come from its ``kernel.json`` alone
(:func:`kernel_env`). A cell that loops on ``ui_events`` (a live camera, a
console, the dance controls) runs while a scripted student types into its
text box and clicks its buttons with the widget messages a browser sends,
then clicks the button that ends it. The test reads each cell's output as
the cell prints it. A finished line, or a line the cell checker treats as
an error, ends a wait for a phrase that has not appeared.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from vla_pipeline.utils.config import REPO_ROOT as PIPE_ROOT

logger = logging.getLogger(__name__)

NOTEBOOKS = PIPE_ROOT / "workshop" / "notebooks"
# A line that says a model left the NPU; the notebook tests reject it.
CPU_FALLBACK = re.compile(r"fall(s|ing)? back to (the )?cpu", re.I)
# Jupyter paints a cell's stderr red and still marks the cell as run. An
# exception already fails the cell. These lines do too: a ROS error or
# warning, a Python traceback, a joint commanded outside its limit, and
# librcutils' error-overwrite banner.
CELL_ERROR = re.compile(
    r"\[ERROR\]"
    r"|\[WARN\]"
    r"|Traceback \(most recent call last\)"
    r"|outside its limit"
    r"|rcutils_set_error_state"
    r"|error state is being overwritten"
)
KERNEL = "vvla-workshop"

# A cell that loops on ui_events ends when one of these is clicked, in this order of preference.
FINISH = ("end session", "done", "stop")


def kernel_env() -> dict:
    """This process's environment without what the kernel's ``kernel.json``
    sets (ROS 2, the NPU runtime), without a display, and without pytest's
    ``PYTEST_CURRENT_TEST``.

    The kernel then gets ROS 2 and the NPU only from its kernelspec, as when
    VS Code starts it, and the notebooks' resource HUD fails quietly instead
    of opening windows. ipykernel does not capture file-descriptor output
    while ``PYTEST_CURRENT_TEST`` is set. Without that capture, what ROS and
    MoveIt print from C++ never reaches a cell, and :func:`cell_errors`
    misses it, though JupyterLab shows it in the cell.
    """
    from jupyter_client.kernelspec import KernelSpecManager

    baked = KernelSpecManager().get_kernel_spec(KERNEL).env
    dropped = ("DISPLAY", "WAYLAND_DISPLAY", "PYTEST_CURRENT_TEST")
    return {k: v for k, v in os.environ.items() if k not in baked and k not in dropped}


class Watcher:
    """A second client of the kernel, as a browser tab is: it reads iopub
    (stdout, stderr, JSON displays and widget state, any parent) and sends
    widget messages and code on its own shell socket.

    Widget state is kept from the same comm messages the browser gets, so a
    waiter woken by a message sees the state that message carried.
    """

    def __init__(self, connection_info: dict):
        import zmq
        from jupyter_client import BlockingKernelClient

        self._kc = BlockingKernelClient()
        self._kc.load_connection_info(connection_info)
        self._kc.start_channels(shell=True, stdin=False, hb=False, control=False)
        self._lock = threading.Lock()
        self.stdout: list[str] = []
        self.stderr: list[str] = []
        self.records: list[dict] = []
        self._widgets: dict[str, dict] = {}
        self._updates = 0  # widget updates the kernel sent (a live view's frames)
        self._listeners: list = []
        wake = f"inproc://iopub-watch-{id(self)}"
        self._wake_rx = self._kc.context.socket(zmq.PAIR)
        self._wake_rx.bind(wake)
        self._wake_tx = self._kc.context.socket(zmq.PAIR)
        self._wake_tx.connect(wake)
        self._thread = threading.Thread(target=self._pump, daemon=True, name="iopub-watch")
        self._thread.start()

    def _pump(self) -> None:
        import zmq

        iopub = self._kc.iopub_channel
        poller = zmq.Poller()
        poller.register(iopub.socket, zmq.POLLIN)
        poller.register(self._wake_rx, zmq.POLLIN)
        while True:
            ready = dict(poller.poll())
            if self._wake_rx in ready:
                return
            self._take(iopub.get_msg(timeout=0))

    def _take(self, msg: dict) -> None:
        kind, content = msg["msg_type"], msg["content"]
        with self._lock:
            if kind == "stream":
                (self.stdout if content.get("name") == "stdout" else self.stderr).append(content["text"])
            elif kind in ("display_data", "execute_result"):
                record = content.get("data", {}).get("application/json")
                if isinstance(record, dict) and "text" in record:
                    self.records.append(record)
            elif kind == "comm_open":
                self._widgets[content["comm_id"]] = dict(content.get("data", {}).get("state", {}))
            elif kind == "comm_msg":
                data = content.get("data", {})
                if data.get("method") in ("update", "echo_update"):
                    self._widgets.setdefault(content["comm_id"], {}).update(data.get("state", {}))
                    self._updates += data["method"] == "update"
            for wake in self._listeners:  # under the lock: none runs after its unsubscribe returns
                wake()

    def send_comm(self, comm_id: str, data: dict) -> None:
        """A widget message from this client, as a browser sends it.

        Never on nbclient's socket: nbclient waits there for a cell's reply
        with a separate poller, and a send on that socket can swallow the
        poller's wake-up.
        """
        kc = self._kc
        kc.shell_channel.send(kc.session.msg("comm_msg", {"comm_id": comm_id, "data": data}))

    def execute(self, code: str) -> None:
        """Run ``code`` from this client, as a second browser tab would. Its reply is not read."""
        self._kc.execute(code)

    def subscribe(self, callback):
        """Call ``callback()`` after every kernel message from now on; returns the unsubscribe."""
        with self._lock:
            self._listeners.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                self._listeners.remove(callback)

        return unsubscribe

    def widgets(self) -> dict[str, dict]:
        """Every widget's state, by comm id, as the kernel last sent it."""
        with self._lock:
            return {cid: dict(state) for cid, state in self._widgets.items()}

    def updates_since(self, since: "Mark") -> int:
        with self._lock:
            return self._updates - since.updates

    def mark(self) -> "Mark":
        """Cursor for output that arrives after this call. Earlier scenarios stay behind it."""
        with self._lock:
            return Mark(len(self.stdout), len(self.records), self._updates, len(self.stderr))

    def output_since(self, since: "Mark") -> tuple[str, str]:
        """``(stdout, stderr)`` the kernel printed after ``since``, from any thread."""
        with self._lock:
            return "".join(self.stdout[since.stdout:]), "".join(self.stderr[since.stderr:])

    def said(self, phrase: str, since: "Mark | int" = 0) -> bool:
        with self._lock:
            return phrase in "".join(self.stdout[_stdout_at(since):])

    def count(self, phrase: str, since: "Mark | int" = 0) -> int:
        with self._lock:
            return "".join(self.stdout[_stdout_at(since):]).count(phrase)

    def text(self) -> str:
        with self._lock:
            return "".join(self.stdout) + "".join(self.stderr)

    def record(self, text: str, since: "Mark | int" = 0) -> dict | None:
        """The latest record of ``text`` after ``since``.

        The latest record in the whole list is still the previous scenario's until
        this command finishes, so a wait has to start behind the mark.
        """
        with self._lock:
            found = [r for r in self.records[_records_at(since):] if r["text"] == text]
            return found[-1] if found else None

    def close(self) -> None:
        self._wake_tx.send(b"")  # the pump returns on it
        self._thread.join()
        self._wake_tx.close()
        self._wake_rx.close()
        self._kc.stop_channels()


@dataclass(frozen=True)
class Mark:
    """How far the watcher had read when a scenario began."""

    stdout: int = 0
    records: int = 0
    updates: int = 0
    stderr: int = 0


def _stdout_at(since: Mark | int) -> int:
    return since.stdout if isinstance(since, Mark) else int(since)


def _records_at(since: Mark | int) -> int:
    return since.records if isinstance(since, Mark) else 0


async def until(predicate, what: str, *sources, task=None):
    """``predicate()`` once it holds, checked again whenever a source reports new data.

    ``sources`` have ``subscribe(callback) -> unsubscribe`` (the
    :class:`Watcher`, a ROS probe); ``task`` ending also wakes the check.
    A wait whose event never comes is the module's hang guard's to fail.
    """
    loop = asyncio.get_running_loop()
    changed = asyncio.Event()

    def wake(*_):
        loop.call_soon_threadsafe(changed.set)

    unsubscribe = [source.subscribe(wake) for source in sources]
    if task is not None:
        task.add_done_callback(wake)
    started = loop.time()
    try:
        while True:
            changed.clear()
            value = predicate()
            if value:
                logger.info("%s after %.2f s", what, loop.time() - started)
                return value
            await changed.wait()
    finally:
        for off in unsubscribe:
            off()
        if task is not None:
            task.remove_done_callback(wake)


class Console:
    """Widgets of a running cell, found by label, and the messages a browser sends them.

    ``before`` is the watcher's widget ids from before the cell ran; the
    cell's own widgets are the ones that appear after.
    """

    def __init__(self, watcher: Watcher, before: set | None = None, since: Mark | int = 0):
        self.watcher = watcher
        self._before = set(before or ())
        self.since = since
        self.widgets: dict[str, str] = {}

    def said(self, phrase: str) -> bool:
        return self.watcher.said(phrase, since=self.since)

    def count(self, phrase: str) -> int:
        return self.watcher.count(phrase, since=self.since)

    def _new(self):
        return {cid: s for cid, s in self.watcher.widgets().items() if cid not in self._before}

    def buttons(self) -> dict[str, str]:
        return {s["description"]: cid for cid, s in self._new().items() if s.get("_model_name") == "ButtonModel"}

    def text_box(self) -> str | None:
        return next((cid for cid, s in self._new().items() if s.get("_model_name") == "TextModel"), None)

    def type(self, text: str, comm_id: str | None = None) -> None:
        """Type ``text`` into a text box and press Enter."""
        target = comm_id or self.widgets.get("typed") or self.text_box()
        self.watcher.send_comm(target, {"method": "update", "state": {"value": text}, "buffer_paths": []})

    def click(self, label: str) -> None:
        comm_id = self.widgets.get(label) or self.buttons()[label]
        self.watcher.send_comm(comm_id, {"method": "custom", "content": {"event": "click"}})


def _install_kernel_parent_death() -> None:
    """The notebook kernel dies with this process, including a SIGKILL of pytest.

    Jupyter only records the parent pid. A killed pytest reparents the kernel to
    systemd, and that kernel keeps the twin's domain lock. ``prctl`` is set in
    the child before exec, so the kernel gets SIGKILL when this process dies.
    """
    import ctypes
    import signal

    from jupyter_client.manager import KernelManager

    if getattr(KernelManager, "_vvla_pdeath", False):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    original = KernelManager._launch_kernel

    def launch(self, kernel_cmd, **kw):
        previous = kw.get("preexec_fn")

        def die_with_parent():
            libc.prctl(1, signal.SIGKILL)
            if previous is not None:
                previous()

        kw["preexec_fn"] = die_with_parent
        return original(self, kernel_cmd, **kw)

    KernelManager._launch_kernel = launch
    KernelManager._vvla_pdeath = True


def client_for(nb, path: Path):
    """An nbclient client with no per-cell timeout: each cell ends on its own reply."""
    from nbclient import NotebookClient

    _install_kernel_parent_death()
    return NotebookClient(nb, kernel_name=KERNEL, timeout=None, resources={"metadata": {"path": str(path.parent)}})


def _stream_text(output) -> str:
    text = output.get("text", "")
    return "".join(text) if isinstance(text, list) else text


def cell_errors(cell, tolerate=None) -> str:
    """Error text Jupyter showed for ``cell`` while still counting the cell as run.

    An exception is a separate failure: nbclient raises before this is asked.
    ``tolerate`` is a pattern for a line the cell is written to produce.
    """
    found = []
    for output in cell.get("outputs") or []:
        kind = output.get("output_type")
        if kind == "error":
            found.append(f"{output.get('ename', 'error')}: {output.get('evalue', '')}")
        elif kind == "stream" and output.get("name") == "stderr":
            found.extend(
                line for line in _stream_text(output).splitlines()
                if CELL_ERROR.search(line) and not (tolerate and tolerate.search(line))
            )
    return "\n".join(found)


def assert_cell_clean(cell, where: str, tolerate=None) -> None:
    """Raise when ``cell`` printed an error. ``where`` names the notebook and the cell."""
    found = cell_errors(cell, tolerate)
    if found:
        raise AssertionError(f"{where} printed an error:\n{found}")


def cell_text(cell) -> str:
    """The stdout, stderr and exception text Jupyter showed for ``cell``."""
    parts = []
    for output in cell.get("outputs") or []:
        kind = output.get("output_type")
        if kind == "stream":
            parts.append(_stream_text(output))
        elif kind == "error":
            parts.append(f"{output.get('ename', 'error')}: {output.get('evalue', '')}\n")
    return "".join(parts)


def cell_answer(stdout: str, stderr: str, progress: str | None = None) -> str | None:
    """What ``stdout`` and ``stderr`` have finished showing, or None while they are still going.

    A complete stdout line is the cell's answer, unless the regex ``progress``
    matches it: a line the cell prints on the way to its answer. A line
    :data:`CELL_ERROR` matches is the cell turning red. A trailing partial
    line is not an answer yet.
    """
    errors = [line for line in stderr.splitlines() if CELL_ERROR.search(line)]
    if errors:
        return "\n".join(errors)
    if "\n" not in stdout:
        return None
    finished = [
        line for line in stdout[: stdout.rfind("\n") + 1].splitlines()
        if line.strip() and not (progress and re.search(progress, line))
    ]
    return "\n".join(finished).strip() or None


@dataclass
class NotebookRun:
    output: str


# A student action: ("type", text), ("click", label), ("wait_for", phrase),
# ("wait_for", phrase, progress): lines matching the regex progress may come
# first, or ("updates", n): the cell sent n widget updates, a live view's frames.
# The kernel handles the clicks in the order they arrive, so the button that ends
# the cell comes after every earlier action.
DEFAULT_ACTIONS = (("updates", 3),)


async def _interact(client, watcher, nb, index: int, actions) -> None:
    """Run the looping cell at ``index``, act on its widgets, then click the button that ends it."""
    console = Console(watcher, before=set(watcher.widgets()))
    task = asyncio.create_task(client.async_execute_cell(nb.cells[index], index))
    cell = nb.cells[index].get("id")
    await until(lambda: task.done() or any(b in console.buttons() for b in FINISH),
                f"the buttons of cell {cell}", watcher, task=task)
    if task.done():
        task.result()
        return
    mark = watcher.mark()  # a wait counts output from the last type/click on
    for action in actions:
        kind = action[0]
        if kind == "type":
            # The cell creates its widgets one message at a time: wait for this one.
            await until(lambda: task.done() or console.text_box(), f"the text box of cell {cell}",
                        watcher, task=task)
            if not task.done():
                mark = watcher.mark()
                console.type(action[1])
        elif kind == "click":
            await until(lambda: task.done() or action[1] in console.buttons(),
                        f"the {action[1]!r} button of cell {cell}", watcher, task=task)
            if not task.done():
                mark = watcher.mark()
                console.click(action[1])
        elif kind == "wait_for":
            phrase = action[1]
            progress = action[2] if len(action) > 2 else None
            shown = mark

            def answer():
                nonlocal shown
                fresh, fresh_err = watcher.output_since(shown)
                if fresh or fresh_err:
                    logger.info("cell %s:\n%s", cell, (fresh + fresh_err).rstrip())
                    shown = watcher.mark()
                if task.done():
                    return True
                out, err = watcher.output_since(mark)
                if phrase in out:
                    return True
                said = cell_answer(out, err, progress)
                if said is None:
                    return False
                raise AssertionError(
                    f"{cell} showed:\n{said}\nwhile the test was waiting for {phrase!r}"
                )

            await until(answer, f"{phrase!r} from cell {cell}", watcher, task=task)
        elif kind == "updates":
            await until(lambda: task.done() or watcher.updates_since(mark) >= action[1],
                        f"{action[1]} widget updates from cell {cell}", watcher, task=task)
        else:
            raise ValueError(f"unknown student action {action!r}")
        if task.done():
            task.result()
            return
    console.click(next(b for b in FINISH if b in console.buttons()))
    # Not wait_for: nbclient reports a cancelled cell as "Kernel died".
    await asyncio.wait({task})
    task.result()


def run_notebook(path: Path, script: dict | None = None, tolerate: dict | None = None) -> NotebookRun:
    """Every cell of ``path`` in order, as stored, in a fresh kernel.

    ``script`` maps a looping cell's id to the student's actions (default:
    watch it draw three frames). ``tolerate`` maps a cell id to a pattern for
    stderr that cell is written to print. Any other error line still fails it.
    """
    import nbformat
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(path, as_version=4)
    client = client_for(nb, path)
    script = script or {}

    async def main() -> NotebookRun:
        async with client.async_setup_kernel(env=kernel_env()):
            watcher = Watcher(client.km.get_connection_info())
            try:
                for i, cell in enumerate(nb.cells):
                    if cell.cell_type != "code":
                        continue
                    try:
                        if "ui_events(" in cell.source:
                            await _interact(client, watcher, nb, i, script.get(cell.get("id"), DEFAULT_ACTIONS))
                        else:
                            await client.async_execute_cell(cell, i)
                    except CellExecutionError as e:
                        raise AssertionError(f"{path.name}, cell {cell.get('id')}:\n{e}") from None
                    shown = cell_text(cell).rstrip()
                    if shown:
                        logger.info("%s cell %s:\n%s", path.name, cell.get("id"), shown)
                    assert_cell_clean(
                        cell, f"{path.name}, cell {cell.get('id')}",
                        (tolerate or {}).get(cell.get("id")),
                    )
                return NotebookRun(output=watcher.text())
            finally:
                watcher.close()

    return asyncio.run(main())
