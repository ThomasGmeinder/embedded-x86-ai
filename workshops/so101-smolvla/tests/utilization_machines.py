"""The machines whose sources the utilization tests expect to be complete, in one place.

On a machine named here a missing source fails a test with its reason; it is not skipped. The thresholds
are what step 0 of the plan of 2026-11-22 measured on that machine (TESTING_LEDGER.md of that day).
"""

from __future__ import annotations

import socket
from pathlib import Path

WORKSHOP = Path(__file__).resolve().parents[1]

MACHINES = {
    "xdetgmeinde-lx01": {
        # AMD Ryzen AI Max+ PRO 395, 32 logical processors, GNOME 46 on Wayland, X11 display ":0".
        "logical_processors": 32,
        "metrics_format": (3, 0),
        "npu_device": "/dev/accel/accel0",
        "has_display": True,
        # Firmware all-core power at 99.5 to 99.8 % utilization: 23.4 to 24.8 W (step 0, item 3); at
        # idle 5 to 10 W. A reading 2.5 s into the load, with the 1 s filter, must be above this.
        "cpu_power_full_load_w": 18.0,
        # gpu_busy_percent follows radeontop within 5 points at 30 and 70 % duty (step 0, item 2); a
        # matrix-multiplication loop without pause must read at least this.
        "gpu_busy_full_load_percent": 50.0,
        # YOLO26s on the NPU: columns 32, 58, 77 to 78 % while the load ran, power 1.24 W mean
        # (step 0, item 1). Measured once by hand. The automated NPU load test was removed on
        # 2026-11-22 with the NPU runtime it borrowed from outside this workshop; no test reads
        # these two until this workshop has an NPU load of its own.
        "npu_columns_mean_under_load_percent": 30.0,
        "npu_power_under_load_w": 0.5,
    },
}


def this_machine() -> dict | None:
    """The table row of the machine the tests run on, or None."""
    return MACHINES.get(socket.gethostname())
