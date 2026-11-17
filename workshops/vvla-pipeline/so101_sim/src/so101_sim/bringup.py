# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""DDS participant before any Vitis AI session.

``run_notebooks.sh`` starts the notebook kernel with this module, the same
role ``so101_sim.bringup`` has on the gemma4 branch. ``FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA``
does not discover peers when a Vitis session is opened first, and ``rclpy.init``
on its own does not create a participant.

The participant is on a private context. ``MoveItPy`` has to be the first init
of the default context, so this module does not touch that context, the ROS
domain, or the discovery profile. Those come from the process environment,
which is the one ``run_notebooks.sh`` gave the twin launches.
"""

from __future__ import annotations

import sys

_context = None
_participant = None


def ensure_dds_participant() -> None:
    """Create this process's DDS participant on a private context."""
    global _context, _participant
    import rclpy

    if _participant is not None:
        return
    _context = rclpy.Context()
    rclpy.init(context=_context)
    _participant = rclpy.create_node("dds_participant", context=_context)


def main() -> None:
    """``kernel`` creates the participant, then runs the notebook kernel."""
    if len(sys.argv) < 2 or sys.argv[1] != "kernel":
        raise SystemExit("usage: python -m so101_sim.bringup kernel [args...]")
    ensure_dds_participant()
    sys.argv = ["ipykernel_launcher", *sys.argv[2:]]
    from ipykernel.kernelapp import IPKernelApp

    IPKernelApp.launch_instance()


if __name__ == "__main__":
    main()
