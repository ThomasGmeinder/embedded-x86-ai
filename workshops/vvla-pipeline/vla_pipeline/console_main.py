# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Twin console process. It draws overlays and the latched control state.

Space publishes a stop. The reset button publishes a reset. Neither call
runs a model or moves the arm; the control process does that.
"""

from __future__ import annotations

import logging
import signal
import threading

from vla_pipeline.main import apply_sim_overrides, build_parser
from vla_pipeline.user_robot_control import PipelineContext, build_registry


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    from vla_pipeline.utils.config import load_config

    cfg = load_config(args.config)
    if args.digital_twin:
        apply_sim_overrides(cfg)

    from vla_pipeline.console import TwinConsole
    from vla_pipeline.robot.ros2_control_arm import Spinner
    from vla_pipeline.wire import ConsoleLink

    ctx = PipelineContext(cfg=cfg, arm=None, registry=build_registry())
    console = TwinConsole(ctx, voice="typed" if args.text_commands else "mic")
    console.use_tiles()
    spin = Spinner("twin_console")
    link = ConsoleLink(spin.node, console)
    shutdown = threading.Event()

    def _stop(_signum, _frame) -> None:
        shutdown.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        console.run(shutdown, on_stop=link.publish_stop, on_reset=link.publish_reset)
    finally:
        shutdown.set()
        spin.close()


if __name__ == "__main__":
    main()
