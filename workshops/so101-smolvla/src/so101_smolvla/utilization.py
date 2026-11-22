"""Utilization and power of the processor, the integrated graphics and the NPU, read from the kernel's files.

Standard library only, so it runs in any Python 3.10 on the measured machine.

Where the numbers come from (SPEC of the meters in the plan of 2026-11-22; verified in its step 0):

- CPU utilization: ``/proc/stat``, the difference of two readings. Busy is all time minus ``idle`` minus
  ``iowait`` (``guest`` and ``guest_nice`` are inside ``user`` and ``nice``). This is psutil's definition.
- iGPU utilization: ``gpu_busy_percent`` of the amdgpu driver, the firmware's graphics-engine busy
  share (graphics and compute; the video engine is not in it).
- Power of all three rows and the NPU's eight columns: the firmware's metrics table, format 3.0, that the
  amdgpu driver hands out as ``gpu_metrics`` (``struct gpu_metrics_v3_0`` in
  ``drivers/gpu/drm/amd/include/kgd_pp_interface.h``, filled by ``smu_v14_0_0_get_gpu_metrics``). The
  firmware filters every value over ``time_filter_alphavalue`` microseconds (1 s on this machine).
  ``average_all_core_power`` is the cores only, ``average_gfx_power`` the graphics, ``average_ipu_power``
  the NPU ("IPU" is the firmware's name for it); they do not add up to ``average_socket_power``, the
  whole chip.

A source that is missing or in an unknown form gives no value and a reason in words. It never gives 0,
and no second source stands behind it.

    python -m so101_smolvla.utilization --once --json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DRM = Path("/sys/class/drm")
DRIVER = "amdgpu"
EXPECTED_FORMAT = (3, 0)
NPU_COLUMNS = 8
CORES = 16
# The structure the metrics file holds. One place; the header and name beside it.
STRUCTURE = "struct gpu_metrics_v3_0, drivers/gpu/drm/amd/include/kgd_pp_interface.h"
_u8, _u16, _u32, _u64 = ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32, ctypes.c_uint64


class MetricsHeader(ctypes.Structure):
    _fields_ = [("structure_size", _u16), ("format_revision", _u8), ("content_revision", _u8)]


class GpuMetricsV3(ctypes.Structure):
    """``struct gpu_metrics_v3_0`` with the C compiler's natural alignment (264 bytes)."""

    _fields_ = [
        ("header", MetricsHeader),
        ("temperature_gfx", _u16), ("temperature_soc", _u16), ("temperature_core", _u16 * CORES),
        ("temperature_skin", _u16),
        ("average_gfx_activity", _u16), ("average_vcn_activity", _u16),
        ("average_ipu_activity", _u16 * NPU_COLUMNS), ("average_core_c0_activity", _u16 * CORES),
        ("average_dram_reads", _u16), ("average_dram_writes", _u16),
        ("average_ipu_reads", _u16), ("average_ipu_writes", _u16),
        ("system_clock_counter", _u64),
        ("average_socket_power", _u32), ("average_ipu_power", _u16), ("average_apu_power", _u32),
        ("average_gfx_power", _u32), ("average_dgpu_power", _u32), ("average_all_core_power", _u32),
        ("average_core_power", _u16 * CORES), ("average_sys_power", _u16),
        ("stapm_power_limit", _u16), ("current_stapm_power_limit", _u16),
        ("clocks", _u16 * 8), ("current_coreclk", _u16 * CORES),
        ("current_core_maxfreq", _u16), ("current_gfx_maxfreq", _u16),
        ("throttle", _u32 * 7), ("time_filter_alphavalue", _u32),
    ]


EXPECTED_SIZE = ctypes.sizeof(GpuMetricsV3)


class SourceError(RuntimeError):
    """A source is missing or not in the form this module knows; the message says which and why."""


def find_amdgpu_device() -> Path:
    """The device folder of the amdgpu graphics device that hands out a metrics file."""
    for device in sorted(DRM.glob("card*/device")):
        driver = device / "driver"
        if driver.is_symlink() and os.path.basename(os.readlink(driver)) == DRIVER and (device / "gpu_metrics").is_file():
            return device
    raise SourceError(f"no {DRIVER} device with a gpu_metrics file under {DRM}")


def decode_metrics(raw: bytes) -> GpuMetricsV3:
    """The metrics file's bytes as the structure, after the size and format checks."""
    if len(raw) != EXPECTED_SIZE:
        raise SourceError(f"metrics file is {len(raw)} bytes, expected {EXPECTED_SIZE} ({STRUCTURE})")
    metrics = GpuMetricsV3.from_buffer_copy(raw)
    fmt = (metrics.header.format_revision, metrics.header.content_revision)
    if fmt != EXPECTED_FORMAT:
        raise SourceError(f"metrics file is format {fmt[0]}.{fmt[1]}, expected {EXPECTED_FORMAT[0]}.{EXPECTED_FORMAT[1]}")
    if metrics.header.structure_size != EXPECTED_SIZE:
        raise SourceError(f"metrics header says {metrics.header.structure_size} bytes, expected {EXPECTED_SIZE}")
    return metrics


def read_metrics(device: Path) -> GpuMetricsV3:
    return decode_metrics((device / "gpu_metrics").read_bytes())


def read_cpu_times() -> tuple[int, int]:
    """(busy, total) jiffies over all processors from /proc/stat; busy leaves out idle and iowait."""
    fields = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()
    if fields[0] != "cpu" or len(fields) < 9:
        raise SourceError(f"/proc/stat does not start with the cpu line this module knows: {fields[:9]}")
    user, nice, system, idle, iowait, irq, softirq, steal = (int(value) for value in fields[1:9])
    total = user + nice + system + idle + iowait + irq + softirq + steal
    return total - idle - iowait, total


def _percent(value: int, what: str) -> tuple[float | None, str | None]:
    if value > 100:
        return None, f"{what} reads {value}, above 100"
    return float(value), None


@dataclass(frozen=True)
class Reading:
    """One reading. A missing value is None, with the reason under its name in ``reasons``."""

    host: str
    time: float
    cpu_percent: float | None
    cpu_power_w: float | None
    gpu_percent: float | None
    gpu_power_w: float | None
    npu_percent: float | None
    npu_power_w: float | None
    npu_columns: tuple[float, ...] | None
    socket_power_w: float | None
    filter_s: float | None
    reasons: dict[str, str]

    def as_dict(self) -> dict:
        record = asdict(self)
        record["npu_columns"] = None if self.npu_columns is None else list(self.npu_columns)
        return record


class Sampler:
    """Reads the three rows. Its only state is the previous /proc/stat reading."""

    def __init__(self) -> None:
        self.host = socket.gethostname()
        self._device: Path | None
        self._device_reason: str | None
        try:
            self._device = find_amdgpu_device()
            self._device_reason = None
        except SourceError as error:
            self._device, self._device_reason = None, str(error)
        self._previous = read_cpu_times()

    def sample(self) -> Reading:
        reasons: dict[str, str] = {}
        now = time.time()

        busy, total = read_cpu_times()
        busy_before, total_before = self._previous
        self._previous = (busy, total)
        if total == total_before:
            cpu_percent, reasons["cpu_percent"] = None, "no time passed since the previous reading"
        else:
            cpu_percent = 100.0 * (busy - busy_before) / (total - total_before)

        gpu_percent = cpu_power = gpu_power = npu_power = npu_percent = socket_power = filter_s = None
        npu_columns = None
        if self._device is None:
            for name in ("gpu_percent", "cpu_power_w", "gpu_power_w", "npu_percent", "npu_power_w",
                         "npu_columns", "socket_power_w"):
                reasons[name] = self._device_reason or "no device"
        else:
            try:
                gpu_percent, reason = _percent(
                    int((self._device / "gpu_busy_percent").read_text(encoding="ascii")), "gpu_busy_percent"
                )
                if reason:
                    reasons["gpu_percent"] = reason
            except (OSError, ValueError) as error:
                reasons["gpu_percent"] = f"gpu_busy_percent: {error}"
            try:
                metrics = read_metrics(self._device)
            except (OSError, SourceError) as error:
                for name in ("cpu_power_w", "gpu_power_w", "npu_percent", "npu_power_w", "npu_columns",
                             "socket_power_w"):
                    reasons[name] = str(error)
            else:
                cpu_power = metrics.average_all_core_power / 1000.0
                gpu_power = metrics.average_gfx_power / 1000.0
                npu_power = metrics.average_ipu_power / 1000.0
                socket_power = metrics.average_socket_power / 1000.0
                filter_s = metrics.time_filter_alphavalue / 1e6
                columns = list(metrics.average_ipu_activity)
                over = [value for value in columns if value > 100]
                if over:
                    reasons["npu_columns"] = reasons["npu_percent"] = f"an NPU column reads {over[0]}, above 100"
                else:
                    npu_columns = tuple(float(value) for value in columns)
                    npu_percent = sum(npu_columns) / len(npu_columns)
        return Reading(
            host=self.host, time=now,
            cpu_percent=cpu_percent, cpu_power_w=cpu_power,
            gpu_percent=gpu_percent, gpu_power_w=gpu_power,
            npu_percent=npu_percent, npu_power_w=npu_power, npu_columns=npu_columns,
            socket_power_w=socket_power, filter_s=filter_s, reasons=reasons,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="one reading of the three rows")
    parser.add_argument("--once", action="store_true", required=True, help="take one reading and exit")
    parser.add_argument("--json", action="store_true", help="print the reading as JSON")
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="the measurement interval of the CPU row in seconds (default 1.0, the firmware's filter time)",
    )
    args = parser.parse_args(argv)
    sampler = Sampler()
    time.sleep(args.interval)  # the measurement interval, not a wait for a condition
    reading = sampler.sample()
    if args.json:
        print(json.dumps(reading.as_dict(), indent=2))
    else:
        for row, percent, power in (
            ("CPU", reading.cpu_percent, reading.cpu_power_w),
            ("iGPU", reading.gpu_percent, reading.gpu_power_w),
            ("NPU", reading.npu_percent, reading.npu_power_w),
        ):
            util = "n/a" if percent is None else f"{percent:3.0f} %"
            watts = "n/a" if power is None else f"{power:5.1f} W"
            print(f"{row:5} {util:>6} {watts:>8}")
        if reading.npu_columns is not None:
            print("NPU columns", [int(value) for value in reading.npu_columns])
        for name, reason in reading.reasons.items():
            print(f"{name}: {reason}", file=sys.stderr)


if __name__ == "__main__":
    main()
