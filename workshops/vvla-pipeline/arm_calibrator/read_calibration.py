# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

#!/usr/bin/env python3
"""Read calibration data directly from an SO-ARM101 arm's servos.

Reads the calibration-relevant registers (homing offset, position limits,
present position) straight from each Feetech STS3215 servo, without relying on
any JSON file. Useful to inspect an arm or verify a calibration after writing.

Usage
-----
    sudo python3 read_calibration.py \\
        --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_<ID>-if00

    # Optionally dump to a LeRobot-format JSON:
    sudo python3 read_calibration.py --port <port> --out arm.json
"""

import argparse
import json
import os
import sys

import serial

from calibrate_from_reference import JOINT_ORDER, REG, decode_sign_magnitude, read_register


def main():
    parser = argparse.ArgumentParser(description="Read SO-ARM101 calibration.")
    parser.add_argument(
        "--port", required=True, help="Serial port or /dev/serial/by-id/... path"
    )
    parser.add_argument(
        "--out", default=None, help="Optional: write a LeRobot-format calibration JSON"
    )
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--timeout", type=float, default=0.1)
    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
    except serial.SerialException as exc:
        print(f"ERROR: could not open {args.port}: {exc}", file=sys.stderr)
        return 1

    print(f"Reading calibration from {args.port} @ {args.baud} baud\n")
    hdr = (
        f"{'ID':>3}  {'joint':<14}  {'model':>6}  {'homing':>7}  "
        f"{'min':>6}  {'max':>6}  {'present':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    calibration = {}
    with ser:
        for servo_id, name in JOINT_ORDER:
            model = read_register(ser, servo_id, *REG["Model_Number"])
            if model is None:
                print(f"{servo_id:>3}  {name:<14}  (no response)")
                continue
            homing = decode_sign_magnitude(
                read_register(ser, servo_id, *REG["Homing_Offset"])
            )
            mn = read_register(ser, servo_id, *REG["Min_Position_Limit"])
            mx = read_register(ser, servo_id, *REG["Max_Position_Limit"])
            present = read_register(ser, servo_id, *REG["Present_Position"])
            print(
                f"{servo_id:>3}  {name:<14}  {model:>6}  {str(homing):>7}  "
                f"{str(mn):>6}  {str(mx):>6}  {str(present):>7}"
            )
            calibration[name] = {
                "id": servo_id,
                "drive_mode": 0,
                "homing_offset": homing,
                "range_min": mn,
                "range_max": mx,
            }

    if args.out and calibration:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(calibration, f, indent=4)
            f.write("\n")
        print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
