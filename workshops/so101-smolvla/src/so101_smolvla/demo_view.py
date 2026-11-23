"""The demo in a notebook: one picture widget, a text box, two buttons, one render thread.

The render thread blocks on the session's ``changed`` event, draws the picture
once and sets it on the widget; it never draws when nothing changed. Drawing
is capped at ``fps`` frames a second: after a draw the thread waits out the
rest of the frame time before it blocks on the event again (a rate cap, not a
wait for a condition). A one-shot timer redraws once after the voice meter's
silence time, so "no microphone signal" appears even when nothing else changes.
The cell that shows the view returns at once; the text box and the buttons are
widget callbacks.
"""

from __future__ import annotations

import threading
import time

from so101_smolvla.demo import DemoSession
from so101_smolvla.demo_console import FRAME_SHAPE, SILENT_AFTER_S, compose, jpeg


class DemoView:
    def __init__(self, session: DemoSession, *, fps: float = 10.0, next_seed=None) -> None:
        import ipywidgets as widgets

        self.session = session
        self.fps = fps
        self._next_seed = next_seed or (lambda: self.session.seed + 1)
        self._closing = threading.Event()
        self._watchdog: threading.Timer | None = None
        self.image = widgets.Image(format="jpeg", layout=widgets.Layout(width=f"{FRAME_SHAPE[1]}px"))
        self.text = widgets.Text(placeholder="type a command, then Enter", continuous_update=False,
                                 layout=widgets.Layout(width="420px"))
        self.new_layout = widgets.Button(description="new layout", button_style="warning")
        self.stop = widgets.Button(description="stop", button_style="danger")
        self.box = widgets.VBox([self.image, widgets.HBox([self.text, self.new_layout, self.stop])])
        self.text.observe(self._typed, names="value")
        self.new_layout.on_click(lambda _: self.session.new_layout(self._next_seed()))
        self.stop.on_click(lambda _: self.close())
        self._thread = threading.Thread(target=self._render, daemon=True, name="demo-render")
        self.frames = 0
        self.drawn = threading.Event()  # set after every draw; a test clears and waits on it

    def show(self) -> "DemoView":
        from IPython.display import display

        self.image.value = jpeg(compose(self.session.snapshot, self.session.transcript, self.session.voice, time.monotonic()))
        display(self.box)
        self._thread.start()
        return self

    def _typed(self, change) -> None:
        text = change["new"].strip()
        if text:
            self.text.value = ""
            self.session.submit(text)

    def _render(self) -> None:
        session, changed = self.session, self.session.changed
        while not self._closing.is_set():
            changed.wait()  # blocks until something changed or the view closes
            if self._closing.is_set():
                return
            changed.clear()
            began = time.monotonic()
            self.image.value = jpeg(compose(session.snapshot, session.transcript, session.voice, began))
            self.frames += 1
            self.drawn.set()
            self._arm_watchdog()
            rest = 1.0 / self.fps - (time.monotonic() - began)
            if rest > 0:
                time.sleep(rest)  # the frame-rate cap

    def _arm_watchdog(self) -> None:
        """One redraw after the voice meter's silence time, so a stopped microphone shows as such."""
        if self._watchdog is not None:
            self._watchdog.cancel()
        if self.session.voice.available:
            self._watchdog = threading.Timer(SILENT_AFTER_S + 0.1, self.session.changed.set)
            self._watchdog.daemon = True
            self._watchdog.start()

    def close(self) -> None:
        """End the render thread; the session and the stack live on."""
        self._closing.set()
        if self._watchdog is not None:
            self._watchdog.cancel()
        self.session.changed.set()  # wakes the render thread so it sees the flag
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=5)
        self.text.disabled = self.new_layout.disabled = self.stop.disabled = True
