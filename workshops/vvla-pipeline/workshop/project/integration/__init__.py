# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Component 3 - integration: composing models + transport into behavior.

Your TODOs live in ``dispatch.py`` (registry + transcript routing),
``mimic.py`` (the perception → mapping → actuation tick), and
``behaviors.py`` (the dance loop).

Check yourself offline (scripted pose/hands + dry-run arm, no NPU/ROS 2/LLM):

    python -m integration.selftest

The full harness is the live UI:

    python app.py --dry-run --synthetic     # anywhere
    python app.py                           # on the rig
"""
