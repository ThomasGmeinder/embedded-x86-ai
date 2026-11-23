"""The demo in a notebook: one picture widget, a text box, two buttons, one render thread.

The render thread blocks on the session's ``changed`` event, draws the picture
once and sets it on the widget; it never draws when nothing changed. Drawing
is capped at ``fps`` frames a second for the camera pictures and the voice
meter: after a draw the thread waits out the rest of the frame time before it
blocks on the event again (a rate cap, not a wait for a condition). The two
events of a command (the session's ``command received`` and ``transcript
received``) end that pause through the session's ``urgent`` event, so each is
on the picture one draw (about 9 ms, measured) after it happens. A one-shot timer redraws once after the voice meter's
silence time, so "no microphone signal" appears even when nothing else changes.
The cell that shows the view returns at once; the text box and the buttons are
widget callbacks. So does the notebook's stop cell: it shows a button
(``StopControl``) and stops nothing until that is pressed, so running every
cell in a row leaves the demo running.
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
            # Cleared before the state is read: an event that arrives from here on is either in
            # this draw or starts the next one at once.
            session.urgent.clear()
            changed.clear()
            began = time.monotonic()
            self.image.value = jpeg(compose(session.snapshot, session.transcript, session.voice, began))
            self.frames += 1
            self.drawn.set()
            self._arm_watchdog()
            rest = 1.0 / self.fps - (time.monotonic() - began)
            if rest > 0:
                # The frame-rate cap for the camera pictures and the voice meter. A command or
                # its transcript ends it at once: those are drawn when they happen.
                session.urgent.wait(rest)

    def _arm_watchdog(self) -> None:
        """One redraw after the voice meter's silence time, so a stopped microphone shows as such."""
        if self._watchdog is not None:
            self._watchdog.cancel()
        if self.session.voice.available:
            self._watchdog = threading.Timer(SILENT_AFTER_S + 0.1, self.session.changed.set)
            self._watchdog.daemon = True
            self._watchdog.start()

    def close(self) -> None:
        """End the render thread; the session and the stack live on. The picture's last frame says
        that it is stopped: a frame from before (for one, the first, drawn before the microphone
        opened, which reads "no microphone") would otherwise stand there as if it were current."""
        if self._closing.is_set():
            return
        self._closing.set()
        if self._watchdog is not None:
            self._watchdog.cancel()
        self.session.changed.set()  # wakes the render thread so it sees the flag
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=5)
        self.image.value = jpeg(compose(
            self.session.snapshot, self.session.transcript, self.session.voice, time.monotonic(), stopped=True,
        ))
        self.text.disabled = self.new_layout.disabled = self.stop.disabled = True


class StopControl:
    """A button that stops the demo when it is pressed. Showing it stops nothing.

    The notebook's stop cell shows this instead of stopping: "Run All" runs that cell right after
    the live cell, and a cell that stopped everything ended the demo a second after it began
    (2026-11-23: the picture then stood on its first frame, "no microphone"). ``stop`` does the
    stopping and returns the lines to show; it runs once, on the first press.
    """

    def __init__(self, stop) -> None:
        import ipywidgets as widgets

        self._stop = stop
        self.done = False
        self.said = ""  # what the stop reported, once pressed
        self.button = widgets.Button(description="stop everything", button_style="danger")
        self.output = widgets.Output()
        self.box = widgets.VBox([self.button, self.output])
        self.button.on_click(self._pressed)

    def show(self) -> "StopControl":
        from IPython.display import display

        display(self.box)
        return self

    def _pressed(self, _=None) -> None:
        if self.done:
            return
        self.done = True
        self.button.disabled = True
        self.said = "\n".join(self._stop())
        self.output.append_stdout(self.said + "\n")
