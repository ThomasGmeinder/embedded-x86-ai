# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The pipeline's arm layer keeps main's names: ``DirectArm`` owns the Feetech
bus, and it is the same class the workshop calls ``LeRobotArm``."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.sim


def test_direct_arm_is_the_workshop_bus_arm():
    from common.motion import LeRobotArm
    from vla_pipeline.robot import arm_interface

    assert arm_interface.DirectArm is LeRobotArm
    assert "DirectArm" in arm_interface.__all__


def test_the_real_backend_builds_a_direct_arm_from_the_robot_config(monkeypatch):
    from vla_pipeline.robot import arm_interface

    built = []
    monkeypatch.setattr(arm_interface, "DirectArm", lambda *args: built.append(args) or "bus arm")
    cfg = {"robot": {"motor_port": "/dev/ttyACM7", "robot_id": "rig_arm", "max_relative_target": 5.0}}

    assert arm_interface.make_arm(cfg) == "bus arm"
    assert built == [("/dev/ttyACM7", "rig_arm", 5.0)]
