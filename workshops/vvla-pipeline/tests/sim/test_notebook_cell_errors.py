# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""A notebook cell that prints an error is a failure, even when it did not raise.

Jupyter shows stderr in red and still marks the cell as run. nbclient raises
only for an exception. The lines here are the ones a manual run of
``02_digital_twin.ipynb`` left in its cells.
"""

from __future__ import annotations

import pytest

from notebook_harness import assert_cell_clean, cell_answer, cell_errors

pytestmark = pytest.mark.sim


def _stderr(text: str) -> dict:
    return {"outputs": [{"output_type": "stream", "name": "stderr", "text": text}]}


def test_a_warning_on_stderr_fails_the_cell():
    cell = _stderr("[WARN] [moveit]: Joint jerk limits are not defined.\n")
    with pytest.raises(AssertionError, match="printed an error"):
        assert_cell_clean(cell, "02_digital_twin.ipynb, cell cb36b108")


def test_a_ros_error_log_fails_the_cell():
    cell = _stderr(
        "[ERROR] [moveit.ros.occupancy_map_monitor]: "
        "No 3D sensor plugin(s) defined for octomap updates\n"
    )
    with pytest.raises(AssertionError, match="printed an error"):
        assert_cell_clean(cell, "02_digital_twin.ipynb, cell cb36b108")


def test_the_planner_typesupport_probe_is_cleared_before_the_next_node(monkeypatch):
    """The OMPL planner's failed typesupport probe must not still be set for the next node.

    Left set, the next ``rcl`` call stores "Expecting token or wildcard" on top
    of it and librcutils prints the overwrite. Jupyter shows that line in red.
    """
    import ctypes

    monkeypatch.setenv("ROS_DOMAIN_ID", "19")
    import moveit.core  # noqa: F401
    from moveit.planning import MoveItPy
    from moveit.utils import create_params_file_from_dict

    from vla_pipeline.robot.moveit_arm import _pipeline, _sensors_3d, moveit_configs

    config = moveit_configs().to_dict() | {"use_sim_time": False} | _sensors_3d()
    robot = MoveItPy(
        node_name="rcutils_probe_moveit",
        launch_params_filepaths=[create_params_file_from_dict(config, "/**")],
    )
    try:
        import rclcpp

        try:
            node = rclcpp.Node("rcutils_probe_mtc")
        except Exception:
            rclcpp.init()
            node = rclcpp.Node("rcutils_probe_mtc")
        _pipeline(node)
        lib = ctypes.CDLL("librcutils.so")
        lib.rcutils_error_is_set.restype = ctypes.c_bool
        assert not lib.rcutils_error_is_set()
    finally:
        robot.shutdown()


def test_the_rcutils_typesupport_banner_fails_the_cell():
    """The banner is an error. A following ROS error is still reported too."""
    banner = (
        ">>> [rcutils|error_handling.c:108] rcutils_set_error_state()\n"
        "This error state is being overwritten:\n"
        "  'Handle's typesupport identifier (rosidl_typesupport_cpp) is not supported by this library, "
        "at ./src/type_support_dispatch.hpp:114'\n"
        "with this new error message:\n"
        "  'Handle's typesupport identifier (rosidl_typesupport_c) is not supported by this library, "
        "at ./src/type_support_dispatch.hpp:116'\n"
        "rcutils_reset_error() should be called after error handling to avoid this.\n"
        "<<<\n"
    )
    with pytest.raises(AssertionError, match="rcutils_set_error_state"):
        assert_cell_clean(_stderr(banner), "02_digital_twin.ipynb, cell b0e4be19")
    mixed = _stderr(banner + "[ERROR] [moveit]: real\n")
    assert "[ERROR] [moveit]: real" in cell_errors(mixed)


def test_a_joint_outside_its_limit_fails_the_cell():
    cell = _stderr("gripper_joint at 1.703379 rad is outside its limit; planning from 1.700000\n")
    with pytest.raises(AssertionError, match="outside its limit"):
        assert_cell_clean(cell, "02_digital_twin.ipynb, cell 365f827e")


def test_an_error_output_fails_the_cell():
    cell = {"outputs": [{"output_type": "error", "ename": "RuntimeError", "evalue": "boom"}]}
    with pytest.raises(AssertionError, match="RuntimeError: boom"):
        assert_cell_clean(cell, "02_digital_twin.ipynb, cell c0000016")


def test_an_error_written_to_the_stderr_file_descriptor_reaches_the_cell(tmp_path):
    """ROS and MoveIt print from C++ to fd 2. The harness's kernel shows that in the cell, as JupyterLab does."""
    import asyncio

    import nbformat

    from notebook_harness import client_for, kernel_env

    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell('import os\nos.write(2, b"[ERROR] [moveit]: from C++\\n")'))
    client = client_for(nb, tmp_path / "fd.ipynb")

    async def run():
        async with client.async_setup_kernel(env=kernel_env()):
            await client.async_execute_cell(nb.cells[0], 0)

    asyncio.run(run())
    assert "[ERROR] [moveit]: from C++" in cell_errors(nb.cells[0])


def test_a_finished_line_is_the_cells_answer():
    assert cell_answer("I can't see a toothbrush in the workspace.\n", "") == (
        "I can't see a toothbrush in the workspace."
    )


def test_a_partial_line_is_not_an_answer_yet():
    assert cell_answer("Got the tooth", "") is None


DANCE_PROGRESS = r"^\d+\.\d+ s/beat, \d+ beats$|^  beat \d+: "


def test_a_progress_line_is_not_the_cells_answer():
    assert cell_answer("0.60 s/beat, 8 beats\n  beat 1: pan= +12.0  roll= -3.0\n", "", DANCE_PROGRESS) is None


def test_a_line_that_is_not_progress_is_still_the_answer():
    assert cell_answer("0.60 s/beat, 8 beats\n  stop! interrupting mid-dance\n", "", DANCE_PROGRESS) == (
        "stop! interrupting mid-dance"
    )


def test_an_error_line_is_the_cells_answer():
    assert cell_answer("", "[WARN] [moveit]: Joint jerk limits are not defined.\n") == (
        "[WARN] [moveit]: Joint jerk limits are not defined."
    )
