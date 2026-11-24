"""The utilization and power table as a window of its own process.

    python -m so101_smolvla.utilization_window [--place X,Y] [--renderer auto|gpu|software] [--report-after N]

Usually started through scripts/run_utilization_meters.sh (so101_smolvla.utilization_meters.open_meters
does that). One process: the sampler (so101_smolvla.utilization) and a Qt Quick scene
(utilization_table.qml), drawn as an X11 window without a frame, outside the window manager, with a
transparent background, that does not take the keyboard. It draws on the display the process inherits
(DISPLAY): the desktop's, or through an SSH connection with X11 forwarding the viewing machine's.

- One sampling clock: a Qt timer takes a reading twice per second (the firmware filters over 1 s).
- "Is it running" has one authority: the lock file of utilization_meters, held with flock while the
  program lives, with its process id, display and window number inside.
- Once the first frame is on the screen the program prints one line, ``shown display=... window=...
  renderer=... host=... log=... lock=...``; open_meters() reads it.
- SIGTERM and SIGINT are events of the event loop (a wake-up socket), as is a click on the "×";
  both end the program with status 0.
- ``--report-after N``: after N readings, print ``report {...}`` with the table's name, the caption
  (the processor's name), the texts of the six cells and each row's bar read back from the scene, then
  end. The tests' way to see what the scene shows.
- The window is named ``WINDOW_NAME`` ("Utilization and power"): its title, and the words in its card.

Renderer rule (plan of 2026-11-22, step 0 item 10, measured on this machine): on a local display the
graphics-processor renderer, whose own graphics time was 0.02 % of a 60 s interval; on a forwarded display
(a host part before the colon, such as ``localhost:10.0``) the software renderer, because the graphics
renderer falls back to software OpenGL on the remote machine and cost seven times the processor time.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import time
from pathlib import Path

from so101_smolvla.utilization import Sampler, SourceError, apu_name
from so101_smolvla.utilization_meters import LOCK_NAME, LOG_NAME, hold_lock, runtime_dir

SCENE = Path(__file__).with_name("utilization_table.qml")
WINDOW_NAME = "Utilization and power"  # the table's name: the window's title and the words in its card
INTERVAL_MS = 500
COLOURS = {"CPU": "#fb7185", "iGPU": "#fb923c", "NPU": "#22d3ee"}
ROWS = ("CPU", "iGPU", "NPU")


def choose_renderer(display: str) -> str:
    """"gpu" on a local display (":0"), "software" on a forwarded one ("localhost:10.0")."""
    return "software" if display.split(":", 1)[0] else "gpu"


def rows_from_reading(reading) -> list[dict]:
    """The scene's rows from one reading; a missing value is None with its reason beside it.

    Every row is one utilization (its bar and its number) and one power. The NPU's is the mean
    of its eight columns, as ``Reading.npu_percent`` holds it: the columns themselves are not
    shown, so that no row looks measured in more detail than the others.
    """
    fields = {
        "CPU": ("cpu_percent", "cpu_power_w"),
        "iGPU": ("gpu_percent", "gpu_power_w"),
        "NPU": ("npu_percent", "npu_power_w"),
    }
    rows = []
    for label in ROWS:
        percent_name, power_name = fields[label]
        rows.append(
            {
                "label": label,
                "colour": COLOURS[label],
                "percent": getattr(reading, percent_name),
                "power": getattr(reading, power_name),
                "reason_util": reading.reasons.get(percent_name, ""),
                "reason_power": reading.reasons.get(power_name, ""),
            }
        )
    return rows


def _items_by_name(item) -> dict:
    """Every item of the scene with an object name, by that name. Walks the visual tree: items a
    Repeater creates are visual children of its parent, not QObject children."""
    found = {}
    for child in item.childItems():
        name = child.objectName()
        if name:
            found[name] = child
        found.update(_items_by_name(child))
    return found


def cell_texts(window) -> dict:
    """What the scene shows: its name, the caption, the six cells' texts and each row's bar.

    ``bars`` holds per row the filled share of its one bar, or None where no bar is drawn (a
    missing value).
    """
    items = _items_by_name(window.contentItem())
    texts = {"name": items["name"].property("text"), "caption": items["caption"].property("text")}
    for label in ROWS:
        for kind in ("util", "power"):
            texts[f"{kind}_{label}"] = items[f"{kind}_{label}"].property("text")
    bars = {}
    for label in ROWS:
        track = items[f"bar_{label}"]
        fill = track.childItems()[0]
        bars[label] = round(fill.property("width") / track.property("width"), 3) if track.property("visible") else None
    texts["bars"] = bars
    return texts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--place", help="X,Y of the window's top left corner (default: the primary screen's top right)")
    parser.add_argument("--renderer", choices=("auto", "gpu", "software"), default="auto")
    parser.add_argument("--report-after", type=int, metavar="N", help="print the scene's texts after N readings and end")
    args = parser.parse_args(argv)

    display = os.environ.get("DISPLAY", "")
    if not display:
        sys.exit("no DISPLAY: the table needs an X11 display (the desktop's, or ssh with X11 forwarding)")
    renderer = choose_renderer(display) if args.renderer == "auto" else args.renderer
    # One place decides the platform and the renderer, before Qt is loaded.
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    if renderer == "software":
        os.environ["QT_QUICK_BACKEND"] = "software"
    else:
        os.environ.pop("QT_QUICK_BACKEND", None)

    lock_path = runtime_dir() / LOCK_NAME
    log_path = runtime_dir() / LOG_NAME
    lock = hold_lock(lock_path)
    if lock is None:
        sys.exit(f"the table is running already (lock {lock_path} is held)")
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("started pid %d display %s renderer %s", os.getpid(), display, renderer)

    from PySide6.QtCore import QObject, QSocketNotifier, QTimer, QUrl
    from PySide6.QtGui import QGuiApplication, QSurfaceFormat
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuick import QQuickWindow

    surface = QSurfaceFormat.defaultFormat()
    surface.setAlphaBufferSize(8)
    QSurfaceFormat.setDefaultFormat(surface)
    app = QGuiApplication(sys.argv[:1])
    engine = QQmlApplicationEngine()
    engine.load(QUrl.fromLocalFile(str(SCENE)))
    if not engine.rootObjects():
        sys.exit(f"the scene {SCENE} did not load")
    window = engine.rootObjects()[0]
    if not isinstance(window, QQuickWindow):
        sys.exit(f"the scene's root is a {type(window).__name__}, not a window")

    host = socket.gethostname()
    sampler = Sampler()
    state = {"shown": False, "readings": 0}
    # The table's name: the window's title for the desktop (window lists, xwininfo) and, from the
    # same property, the words in the card. Beside it the processor the three rows are parts of.
    window.setTitle(WINDOW_NAME)
    try:
        window.setProperty("caption", apu_name())
    except SourceError as error:
        logging.info("processor name missing: %s", error)
        window.setProperty("caption", "processor name not found")
    window.setProperty("rows", rows_from_reading(sampler.sample()))
    if args.place:
        x, y = (int(value) for value in args.place.split(","))
    else:
        screen = app.primaryScreen().geometry()
        x, y = screen.x() + screen.width() - int(window.width()) - 24, screen.y() + 40
    window.setPosition(x, y)

    def shown() -> None:
        if state["shown"]:
            return
        state["shown"] = True
        facts = {
            "pid": os.getpid(), "display": display, "window": int(window.winId()),
            "renderer": window.rendererInterface().graphicsApi().name, "host": host,
            "log": str(log_path), "lock": str(lock_path),
        }
        lock.write(facts)
        print("shown " + " ".join(f"{key}={value}" for key, value in facts.items()), flush=True)

    def tick() -> None:
        try:
            reading = sampler.sample()
        except Exception:  # a source that broke after the start: logged, the cells keep their last values
            logging.exception("reading failed")
            return
        window.setProperty("rows", rows_from_reading(reading))
        for name, reason in reading.reasons.items():
            logging.info("%s missing: %s", name, reason)
        state["readings"] += 1
        if args.report_after is not None and state["readings"] >= args.report_after:
            try:
                print("report " + json.dumps(cell_texts(window)), flush=True)
            except Exception:
                logging.exception("the report failed")
                app.exit(1)
                return
            app.quit()

    # Signals arrive as bytes on a socket pair, which the event loop watches.
    wake_read, wake_write = socket.socketpair()
    wake_write.setblocking(False)
    signal.set_wakeup_fd(wake_write.fileno())
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: None)

    def signalled() -> None:
        wake_read.recv(64)
        logging.info("ending on a signal")
        app.quit()

    notifier = QSocketNotifier(wake_read.fileno(), QSocketNotifier.Type.Read)
    notifier.activated.connect(signalled)

    window.frameSwapped.connect(shown)
    timer = QTimer()
    timer.setInterval(INTERVAL_MS)  # the measurement interval of the table
    timer.timeout.connect(tick)
    timer.start()
    window.show()
    code = app.exec()
    logging.info("ended with status %d", code)
    lock.release()
    sys.exit(code)


if __name__ == "__main__":
    main()
