"""The window program: the mouse ends and moves it, the cells show the sampler's values, a missing value shows n/a."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from so101_smolvla.utilization import Reading, apu_name
from so101_smolvla.utilization_meters import LAUNCHER, close_meters, running_facts
from so101_smolvla.utilization_window import SCENE, WINDOW_NAME, choose_renderer, rows_from_reading
from test_utilization import BusyProcessors
from test_utilization_meters import _xwininfo
from utilization_machines import WORKSHOP, this_machine

MACHINE = this_machine()
live = pytest.mark.skipif(MACHINE is None, reason="this machine is not in tests/utilization_machines.py")
PAD = 20  # the shadow margin of the scene; the card starts at (PAD, PAD) in the window


def _reading(**missing) -> Reading:
    values = dict(host="h", time=0.0, cpu_percent=42.0, cpu_power_w=8.9, gpu_percent=74.0, gpu_power_w=18.1,
                  npu_percent=62.0, npu_power_w=1.2, npu_columns=(60.0, 62.0, 64.0, 60.0, 62.0, 64.0, 60.0, 64.0),
                  socket_power_w=40.0, filter_s=1.0, reasons={})
    for name, reason in missing.items():
        values[name] = None
        values["reasons"][name] = reason
    return Reading(**values)


def test_the_renderer_rule_is_gpu_on_a_local_display_and_software_on_a_forwarded_one():
    assert choose_renderer(":0") == "gpu"
    assert choose_renderer(":1.0") == "gpu"
    assert choose_renderer("localhost:10.0") == "software"
    assert choose_renderer("otherhost:11.0") == "software"


def test_the_scenes_rows_carry_one_utilization_each_and_the_reason_of_a_missing_value():
    rows = rows_from_reading(_reading(gpu_percent="gpu_busy_percent reads 130, above 100"))
    assert [row["label"] for row in rows] == ["CPU", "iGPU", "NPU"]
    assert rows[0]["percent"] == 42.0 and rows[0]["power"] == 8.9
    assert rows[1]["percent"] is None
    assert rows[1]["reason_util"] == "gpu_busy_percent reads 130, above 100" and rows[1]["reason_power"] == ""
    # The NPU is one overall utilization like the other two: its eight columns are not handed to the scene.
    assert rows[2]["percent"] == 62.0
    assert all(set(row) == {"label", "colour", "percent", "power", "reason_util", "reason_power"} for row in rows)


SCENE_CHECK = """
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickWindow  # loaded so that the scene's root arrives as a QQuickWindow
from so101_smolvla.utilization import Reading
from so101_smolvla.utilization_window import SCENE, cell_texts, rows_from_reading
app = QGuiApplication([])
engine = QQmlApplicationEngine()
engine.load(QUrl.fromLocalFile(str(SCENE)))
window = engine.rootObjects()[0]
reading = Reading(**json.loads(sys.argv[1]))
reading = Reading(**{**reading.__dict__, "npu_columns": None if reading.npu_columns is None else tuple(reading.npu_columns)})
window.setTitle("A name")
window.setProperty("caption", "A PROCESSOR 123")
window.setProperty("rows", rows_from_reading(reading))
app.processEvents()
print(json.dumps(cell_texts(window)))
"""


def test_the_scene_shows_n_a_for_a_missing_value_and_the_numbers_for_the_rest():
    reading = _reading(gpu_power_w="metrics file is format 2.4, expected 3.0", cpu_percent="no time passed")
    record = reading.as_dict()
    ran = subprocess.run(
        [str(WORKSHOP / ".venv" / "bin" / "python"), "-c", SCENE_CHECK, json.dumps(record)],
        cwd=WORKSHOP, env={**os.environ, "PYTHONPATH": str(WORKSHOP / "src")},
        capture_output=True, text=True, timeout=120,
    )
    assert ran.returncode == 0, ran.stderr
    texts = json.loads(ran.stdout.strip().splitlines()[-1])
    assert texts["name"] == "A name" and texts["caption"] == "A PROCESSOR 123"
    assert texts["util_CPU"] == "n/a" and texts["power_CPU"] == "8.9 W"
    assert texts["util_iGPU"] == "74 %" and texts["power_iGPU"] == "n/a"
    assert texts["util_NPU"] == "62 %" and texts["power_NPU"] == "1.2 W"
    # One bar per row, filled to the row's number; none for the missing value.
    assert texts["bars"] == {"CPU": None, "iGPU": 0.74, "NPU": 0.62}


def _start(*args: str) -> tuple[subprocess.Popen, dict]:
    process = subprocess.Popen([str(LAUNCHER), *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        if line.startswith("shown "):
            facts = dict(part.split("=", 1) for part in line.split()[1:])
            return process, facts
    raise AssertionError(f"the program ended with status {process.wait()} before it was shown")


def _place(window: str) -> tuple[int, int]:
    info = _xwininfo(int(window))
    return int(info["Absolute upper-left X"]), int(info["Absolute upper-left Y"])


@pytest.fixture(autouse=True)
def nothing_left_running():
    yield
    close_meters()


@live
def test_a_drag_moves_the_window_and_a_click_on_x_ends_the_program_with_status_0():
    assert os.environ.get("DISPLAY") and running_facts() is None
    process, facts = _start("--place", "200,200")
    try:
        x, y = _place(facts["window"])
        assert (x, y) == (200, 200)
        subprocess.run(["xdotool", "mousemove", str(x + PAD + 100), str(y + PAD + 100), "mousedown", "1",
                        "mousemove", str(x + PAD + 160), str(y + PAD + 130), "mouseup", "1"], check=True)
        time.sleep(0.5)  # the X server's report follows the pointer events; one bounded wait, then read
        moved = _place(facts["window"])
        assert moved == (x + 60, y + 30), f"the window is at {moved}, expected {(x + 60, y + 30)}"
        x, y = moved
        # The "×" is the card's top right: card x 288..308, y 9..29, plus the pad.
        subprocess.run(["xdotool", "mousemove", str(x + PAD + 298), str(y + PAD + 19), "click", "1"], check=True)
        out, _ = process.communicate(timeout=15)
        assert process.returncode == 0, out
    finally:
        if process.poll() is None:
            process.kill()
    assert running_facts() is None


@live
def test_under_a_processor_load_the_cpu_cell_is_at_or_above_the_floor_and_the_caption_names_the_processor():
    load = BusyProcessors(16)
    try:
        process, facts = _start("--report-after", "5")
        out, _ = process.communicate(timeout=30)
    finally:
        load.stop()
    assert process.returncode == 0, out
    report = json.loads(next(line for line in out.splitlines() if line.startswith("report ")).split(" ", 1)[1])
    print(report)
    floor = 100.0 * 16 / MACHINE["logical_processors"]
    cpu = int(report["util_CPU"].split()[0])
    assert cpu >= floor, report
    # The table's name and, beside it, this machine's processor as the kernel names it.
    assert report["name"] == WINDOW_NAME == "Utilization and power"
    assert report["caption"] == MACHINE["apu_name"] == apu_name()
    # One bar per row, each filled to the row's number.
    assert set(report["bars"]) == {"CPU", "iGPU", "NPU"}
    for label in report["bars"]:
        assert abs(report["bars"][label] - int(report[f"util_{label}"].split()[0]) / 100) <= 0.006, report
    assert report["power_CPU"].endswith(" W") and report["util_NPU"].endswith(" %")
    assert facts["renderer"] == "OpenGL"  # the rule's choice on this machine's local display
