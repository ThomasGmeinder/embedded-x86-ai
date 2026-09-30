# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

#!/usr/bin/env python3
"""One-step SO-ARM101 calibration.

Connect the arm, run this, put the arm at its home pose, and press Enter.
It auto-detects the arm, transfers the reference calibration, writes it to the
servos, and verifies.

    sudo python3 calibrate.py

Everything else (port, reference file, output path) is handled automatically.
"""

import glob
import os
import pwd
import shutil
import subprocess
import sys

import serial

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "golden_reference.json")
WORKER = os.path.join(HERE, "calibrate_from_reference.py")
BY_ID_GLOB = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_*-if00"

# Also install the calibration here so LeRobot picks it up directly.
LEROBOT_NAME = "my_awesome_follower_arm.json"
LEROBOT_SUBDIR = ".cache/huggingface/lerobot/calibration/robots/so_follower"


def real_user():
    """The invoking user (not root) when run under sudo: (home, uid, gid)."""
    name = os.environ.get("SUDO_USER") or os.environ.get("USER") or "root"
    info = pwd.getpwnam(name)
    return info.pw_dir, info.pw_uid, info.pw_gid


def _chown(path, uid, gid):
    try:
        os.chown(path, uid, gid)
    except (PermissionError, OSError):
        pass


def bus_alive(port, baud=1_000_000):
    """Ping servo id 1; return True if the motor bus answers.

    A silent bus (with the USB adapter still present) almost always means the
    arm's motor power supply is off.
    """
    try:
        s = serial.Serial(port, baud, timeout=0.15)
    except serial.SerialException:
        return False
    with s:
        for servo_id in (1, 2, 3, 4, 5, 6):
            body = [servo_id, 4, 2, 3, 2]
            s.reset_input_buffer()
            s.write(bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF]))
            s.flush()
            if len(s.read(8)) >= 8:
                return True
    return False


def find_arm():
    """Return the by-id path of the connected arm, prompting if there are several."""
    ports = sorted(glob.glob(BY_ID_GLOB))
    if not ports:
        print("No SO-ARM101 arm found. Plug it in via USB and try again.")
        sys.exit(1)
    if len(ports) == 1:
        return ports[0]
    print("Multiple arms detected:")
    for i, p in enumerate(ports, 1):
        print(f"  {i}) {os.path.basename(p)}")
    while True:
        choice = input(f"Which one? [1-{len(ports)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(ports):
            return ports[int(choice) - 1]


def main():
    port = find_arm()
    serial = os.path.basename(port).split("_Serial_")[-1].replace("-if00", "")
    out = os.path.join(HERE, f"calibrated_{serial}.json")

    print(f"\nArm detected: {serial}")
    print("\n>>> Move the arm to its HOME / middle pose (all joints centered),")
    print(">>> hold it there, then press Enter to calibrate.")
    input()

    if not bus_alive(port):
        print("\nERROR: the arm's motors are not responding.")
        print("The USB adapter is detected, but no servo answered on the bus.")
        print("This almost always means the arm's POWER SUPPLY is off.")
        print("Check that the power brick/switch is on and the barrel jack is")
        print("connected, then run this again.")
        sys.exit(1)

    result = subprocess.run(
        [
            sys.executable,
            WORKER,
            "--port",
            port,
            "--ref",
            REFERENCE,
            "--out",
            out,
            "--commit",
        ]
    )
    if result.returncode == 0:
        home, uid, gid = real_user()
        _chown(out, uid, gid)

        # Install a copy where LeRobot loads it from.
        lerobot_path = os.path.join(home, LEROBOT_SUBDIR, LEROBOT_NAME)
        os.makedirs(os.path.dirname(lerobot_path), exist_ok=True)
        shutil.copyfile(out, lerobot_path)
        _chown(lerobot_path, uid, gid)

        print(f"\nDone. Calibration saved to:\n  {out}\n  {lerobot_path}")
    else:
        print(
            "\nCalibration did not complete (see messages above). "
            "Re-check the home pose and try again."
        )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
