"""so101_smolvla.utilization on the real sources of this machine.

Every live test brackets its reading inside a load: the load reports "running", a first reading is taken,
the measurement interval passes (2.5 s: the firmware filters over 1 s), the reading under test is taken,
and only then is the load told to stop. The thresholds are in tests/utilization_machines.py.
"""

from __future__ import annotations

import ctypes
import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from so101_smolvla import utilization
from so101_smolvla.utilization import (
    EXPECTED_SIZE,
    NPU_COLUMNS,
    Reading,
    Sampler,
    SourceError,
    decode_metrics,
    find_amdgpu_device,
    read_cpu_times,
)
from utilization_machines import VOICE_WORKSHOP, WORKSHOP, this_machine

DATA = Path(__file__).with_name("data")
RECORDED = DATA / "gpu_metrics_v3_0_ryzen_ai_max_395.bin"
INTERVAL_S = 2.5  # the measurement interval inside a load, not a wait for a condition
MACHINE = this_machine()
live = pytest.mark.skipif(
    MACHINE is None, reason="this machine is not in tests/utilization_machines.py; the live tests need its thresholds"
)


def _spin(report, stop):
    """A busy loop on one processor; it reports "running" and spins until the stop pipe has a message."""
    report.send("running")
    while not stop.poll():
        pass


class BusyProcessors:
    """``count`` processes, each spinning on one processor, from "running" until ``stop()``."""

    def __init__(self, count: int) -> None:
        self.processes, self._stops = [], []
        # Fresh interpreters: a fork of the test process, which has torch's threads, is unsafe.
        context = multiprocessing.get_context("spawn")
        for _ in range(count):
            report_end, report = context.Pipe(duplex=False)
            stop, stop_end = context.Pipe(duplex=False)
            process = context.Process(target=_spin, args=(report, stop), daemon=True)
            process.start()
            report.close()
            assert report_end.recv() == "running"
            self.processes.append(process)
            self._stops.append(stop_end)

    def stop(self) -> None:
        for stop in self._stops:
            stop.send("stop")
        for process in self.processes:
            process.join(timeout=10)
            assert process.exitcode == 0


class LoadProcess:
    """A load program that prints "running" and stops when its standard input closes."""

    def __init__(self, command: list[str], cwd: Path | None = None) -> None:
        self.process = subprocess.Popen(
            command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        line = self.process.stdout.readline()
        if line.strip() != "running":
            # communicate() closes the load's standard input, which tells it to stop.
            out, err = self.process.communicate(timeout=60)
            raise AssertionError(f"the load did not report running: {line!r}\n{out}\n{err}")

    def stop(self) -> str:
        out, err = self.process.communicate(timeout=120)  # closes its standard input: the stop signal
        assert self.process.returncode == 0, err
        return out


def _two_readings(sampler: Sampler) -> tuple[Reading, Reading]:
    """A reading at the start of a load and the reading INTERVAL_S later, both inside the load."""
    first = sampler.sample()
    time.sleep(INTERVAL_S)
    return first, sampler.sample()


def _no_power_sums_above_the_chip(reading: Reading) -> None:
    assert None not in (reading.cpu_power_w, reading.gpu_power_w, reading.npu_power_w, reading.socket_power_w)
    assert reading.cpu_power_w + reading.gpu_power_w + reading.npu_power_w <= reading.socket_power_w, reading


def test_the_recorded_metrics_file_of_this_machine_decodes_to_its_recorded_values():
    raw = RECORDED.read_bytes()
    assert len(raw) == EXPECTED_SIZE == 264
    metrics = decode_metrics(raw)
    assert (metrics.header.format_revision, metrics.header.content_revision) == (3, 0)
    # Recorded on 2026-11-22 at 20:30 with the graphics and the NPU idle.
    assert metrics.average_gfx_activity == 1
    assert list(metrics.average_ipu_activity) == [0] * NPU_COLUMNS
    assert metrics.average_socket_power == 19792
    assert metrics.average_ipu_power == 0
    assert metrics.average_gfx_power == 3269
    assert metrics.average_all_core_power == 6228
    assert metrics.time_filter_alphavalue == 1_000_000
    assert metrics.average_vcn_activity == 0
    assert ctypes.sizeof(utilization.GpuMetricsV3) == 264


def test_a_metrics_file_of_another_size_or_format_is_refused_with_the_reason():
    raw = bytearray(RECORDED.read_bytes())
    with pytest.raises(SourceError, match="metrics file is 200 bytes, expected 264"):
        decode_metrics(bytes(raw[:200]))
    with pytest.raises(SourceError, match="metrics file is 300 bytes, expected 264"):
        decode_metrics(bytes(raw) + b"\0" * 36)
    other = bytearray(raw)
    other[2], other[3] = 2, 4
    with pytest.raises(SourceError, match=r"metrics file is format 2\.4, expected 3\.0"):
        decode_metrics(bytes(other))
    wrong_size = bytearray(raw)
    wrong_size[0], wrong_size[1] = 0x10, 0x01
    with pytest.raises(SourceError, match="metrics header says 272 bytes, expected 264"):
        decode_metrics(bytes(wrong_size))


def test_a_reading_is_a_record_with_a_reason_for_every_missing_value():
    reading = Reading(
        host="x", time=1.0, cpu_percent=5.0, cpu_power_w=None, gpu_percent=None, gpu_power_w=None,
        npu_percent=None, npu_power_w=None, npu_columns=None, socket_power_w=None, filter_s=None,
        reasons={"cpu_power_w": "no metrics file"},
    )
    record = reading.as_dict()
    assert record["cpu_percent"] == 5.0 and record["npu_columns"] is None
    assert record["reasons"] == {"cpu_power_w": "no metrics file"}
    assert json.loads(json.dumps(record)) == record


def test_the_cpu_busy_definition_leaves_out_idle_and_iowait():
    fields = Path("/proc/stat").read_text().splitlines()[0].split()
    user, nice, system, idle, iowait, irq, softirq, steal = (int(v) for v in fields[1:9])
    busy, total = read_cpu_times()
    # Read moments apart: the counters only grow, so the test's own values are at most the sampler's.
    assert total >= user + nice + system + idle + iowait + irq + softirq + steal
    assert busy >= user + nice + system + irq + softirq + steal
    assert total - busy >= idle + iowait


@live
def test_this_machine_has_every_source_and_a_reading_holds_every_value():
    device = find_amdgpu_device()
    assert (device / "gpu_metrics").is_file() and (device / "gpu_busy_percent").is_file()
    assert Path(MACHINE["npu_device"]).exists(), f"{MACHINE['npu_device']} is missing on a machine that has it"
    assert os.cpu_count() == MACHINE["logical_processors"]
    sampler = Sampler()
    time.sleep(0.5)  # a measurement interval for the CPU row
    reading = sampler.sample()
    assert reading.reasons == {}, reading.reasons
    assert reading.host == os.uname().nodename
    assert 0.0 <= reading.cpu_percent <= 100.0 and 0.0 <= reading.gpu_percent <= 100.0
    assert len(reading.npu_columns) == NPU_COLUMNS and reading.npu_percent == sum(reading.npu_columns) / NPU_COLUMNS
    assert reading.filter_s == 1.0
    _no_power_sums_above_the_chip(reading)
    print(json.dumps(reading.as_dict()))


@live
def test_cpu_utilization_inside_a_load_of_k_processors_is_at_least_their_share():
    processors = MACHINE["logical_processors"]
    for count in (4, 16):
        sampler = Sampler()
        load = BusyProcessors(count)
        try:
            _, inside = _two_readings(sampler)
        finally:
            load.stop()
        floor = 100.0 * count / processors
        print(f"{count} busy processors: cpu {inside.cpu_percent:.1f} % (floor {floor:.1f}), cores {inside.cpu_power_w} W")
        assert inside.cpu_percent >= floor
        _no_power_sums_above_the_chip(inside)


@live
def test_cpu_power_with_every_processor_busy_is_above_the_full_load_threshold():
    sampler = Sampler()
    load = BusyProcessors(MACHINE["logical_processors"])
    try:
        before, inside = _two_readings(sampler)
    finally:
        load.stop()
    print(f"all processors busy: cpu {inside.cpu_percent:.1f} %, cores {inside.cpu_power_w} W (at the start {before.cpu_power_w} W)")
    assert inside.cpu_percent >= 95.0
    assert inside.cpu_power_w >= MACHINE["cpu_power_full_load_w"]
    _no_power_sums_above_the_chip(inside)


@live
def test_graphics_utilization_and_power_rise_inside_a_rocm_load():
    sampler = Sampler()
    time.sleep(0.5)
    idle = sampler.sample()
    assert idle.gpu_percent is not None and idle.gpu_percent < 10.0, (
        f"the graphics are busy ({idle.gpu_percent} %) before the test's own load; another job holds them"
    )
    load = LoadProcess([str(WORKSHOP / ".venv" / "bin" / "python"), str(Path(__file__).with_name("gpu_load.py"))])
    try:
        _, inside = _two_readings(sampler)
    finally:
        out = load.stop()
    print(f"idle: {idle.gpu_percent} %, {idle.gpu_power_w} W; inside the load: {inside.gpu_percent} %, {inside.gpu_power_w} W; {out.strip()}")
    assert inside.gpu_percent >= MACHINE["gpu_busy_full_load_percent"]
    assert inside.gpu_power_w > idle.gpu_power_w
    _no_power_sums_above_the_chip(inside)


@live
def test_npu_columns_and_power_rise_inside_an_npu_load_and_are_zero_before_it():
    activate = VOICE_WORKSHOP / "activate_venv.sh"
    assert activate.is_file(), f"the voice workshop's {activate} is missing on a machine that has the NPU"
    sampler = Sampler()
    time.sleep(0.5)
    idle = sampler.sample()
    assert idle.npu_columns == (0.0,) * NPU_COLUMNS and idle.npu_power_w == 0.0, "the NPU is busy before the test's own load"
    command = [
        "bash", "-c",
        f"cd {VOICE_WORKSHOP} && source ./activate_venv.sh >/dev/null && exec python -u {Path(__file__).with_name('npu_load.py')}",
    ]
    load = LoadProcess(command)
    try:
        _, inside = _two_readings(sampler)
    finally:
        out = load.stop()
    print(f"inside the NPU load: columns {inside.npu_columns}, mean {inside.npu_percent:.1f} %, {inside.npu_power_w} W; {out.strip()}")
    assert all(0.0 <= value <= 100.0 for value in inside.npu_columns)
    assert inside.npu_percent >= MACHINE["npu_columns_mean_under_load_percent"]
    assert inside.npu_power_w >= MACHINE["npu_power_under_load_w"]
    _no_power_sums_above_the_chip(inside)


def test_the_command_line_prints_one_reading_as_json():
    ran = subprocess.run(
        [sys.executable, "-m", "so101_smolvla.utilization", "--once", "--json", "--interval", "0.3"],
        cwd=WORKSHOP, capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": str(WORKSHOP / "src")},
    )
    assert ran.returncode == 0, ran.stderr
    record = json.loads(ran.stdout)
    assert set(record) >= {"host", "cpu_percent", "gpu_percent", "npu_percent", "npu_columns", "reasons"}
