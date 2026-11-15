# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The scripted person stands in for YOLO-pose wherever the console draws a pose.

``open_webcam`` returns it when the webcam gives no frames, and the console's
pose tile calls ``ctx.pose.draw`` on what ``infer`` returned.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from vla_pipeline.vision.yolo_pose_npu import PoseEstimator
from vvla_workshop.fixtures import ScriptedPose

pytestmark = pytest.mark.sim


def test_scripted_pose_draws_what_it_inferred():
    pose = ScriptedPose()
    frame = np.zeros((480, 640, 3), np.uint8)
    drawn = pose.draw(frame, *pose.infer(frame))
    assert drawn.shape == frame.shape
    assert drawn.any(), "the scripted person was not drawn"
    assert not frame.any(), "draw changed the camera frame in place"


def test_scripted_pose_draw_takes_what_pose_estimator_draw_takes():
    assert inspect.signature(ScriptedPose.draw).parameters.keys() == (
        inspect.signature(PoseEstimator.draw).parameters.keys()
    )
