# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Tier-2 helpers: the notebook twin (``launch/so101_control.launch.py``) over ROS 2.

Every tier-2 test runs on one private ROS domain, ``TWIN_TEST_DOMAIN_ID``
(default 93), with discovery kept to this machine. It is not the notebook's
default domain (87) and not the default domain the real arm uses, so a test
never reaches either. rclpy reads the domain once per process, so every
tier-2 module in a pytest session shares it.
"""

from __future__ import annotations

import copy
import os
import select
import shutil
import threading
from pathlib import Path

import numpy as np

TEST_DOMAIN_ID = int(os.environ.get("TWIN_TEST_DOMAIN_ID", "93"))



def raise_crashes(crashes: list[str]) -> None:
    """A behavior that raised fails the scenario."""
    if crashes:
        raise AssertionError("a behavior crashed: " + "; ".join(crashes))


def join_test_domain() -> None:
    """Put this process, and every twin or kernel it starts, on the test domain."""
    from vvla_workshop.twin_ros import use_private_domain

    os.environ["TWIN_ROS_DOMAIN_ID"] = str(TEST_DOMAIN_ID)
    use_private_domain(TEST_DOMAIN_ID)


def process_gone(pid: int) -> bool:
    """``pid`` has exited (a zombie counts)."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
    except FileNotFoundError:
        return True
    return state == "Z"


def wait_exit(pid: int) -> bool:
    """True once ``pid`` has exited: the kernel's exit event on a pidfd."""
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return True
    try:
        readable, _, _ = select.select([fd], [], [])
    finally:
        os.close(fd)
    return bool(readable) or process_gone(pid)


def assert_domain_free(message: str, wait: bool = False) -> None:
    """No twin holds the test domain's lock; with ``wait``, a holder's exit is awaited first."""
    from so101_sim.domain_lock import owner

    pid = owner(TEST_DOMAIN_ID)
    if pid and wait:
        wait_exit(pid)
        pid = owner(TEST_DOMAIN_ID)
    assert pid is None, f"{message} (pid {pid})"


def twin_log_report(path, dest: Path, lines: int = 60) -> str:
    """Copy the twin node's log into ``dest``; return its last ``lines`` lines."""
    path = Path(path)
    if not path.exists():
        return f"twin node log: {path} is gone"
    shutil.copyfile(path, dest / "twin_node.log")
    tail = path.read_text(errors="replace").splitlines()[-lines:]
    return f"twin node log -> twin_node.log, last {len(tail)} lines:\n" + "\n".join(tail)


def _scene_model(cfg: dict):
    """The MuJoCo model ``launch/so101_control.launch.py`` loads."""
    import mujoco

    from so101_sim.assets import require
    from so101_sim.config import load_sim_config, scene_path

    require()
    return mujoco.MjModel.from_xml_path(str(scene_path(load_sim_config(cfg))))


def _free_bodies(model) -> list[tuple[str, int]]:
    """Each free-joint body and its ``qpos`` address, in model order."""
    import mujoco

    found = []
    for joint in range(model.njnt):
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[joint])
        found.append((name, int(model.jnt_qposadr[joint])))
    return found


def default_scene_snapshot(cfg: dict) -> dict:
    """Every prop at its XML body position, turned straight, plus the pedestals."""
    import mujoco

    model = _scene_model(cfg)
    data = mujoco.MjData(model)
    bodies = {}
    for name, _adr in _free_bodies(model):
        pos = model.body(name).pos
        bodies[name] = [float(pos[0]), float(pos[1]), float(pos[2]), 1.0, 0.0, 0.0, 0.0]
    mocap = {}
    for stand in ("stand_a", "stand_b"):
        try:
            mid = int(model.body(stand).mocapid[0])
        except KeyError:
            continue
        if mid >= 0:
            mocap[stand] = [float(v) for v in data.mocap_pos[mid]]
    return {"bodies": bodies, "mocap": mocap}


def xml_poses(cfg: dict) -> dict[str, np.ndarray]:
    """Each prop's compiled ``qpos0`` (what reset restores)."""
    model = _scene_model(cfg)
    return {name: np.array(model.qpos0[adr : adr + 7], float) for name, adr in _free_bodies(model)}


def quat_close(a, b, tol: float = 1e-3) -> bool:
    return abs(abs(float(np.dot(np.asarray(a, float), np.asarray(b, float)))) - 1.0) < tol


class TwinRig:
    """The section 5 console's context, on a :class:`~vvla_workshop.twin_ros.TwinSession`.

    The arm camera and the mount camera are wrapped in ``FanoutCamera``
    as the console wraps them, and the operator is the notebook's scripted
    stand-in (``ScriptedPose`` / ``ScriptedHands`` / ``SyntheticCamera``).
    """

    def __init__(self, cfg: dict, detector, scene: dict | None = None):
        from vla_pipeline.console import FanoutCamera
        from vla_pipeline.dispatch import PipelineContext, build_registry
        from vla_pipeline.utils.gesture_map import mapper_from_config
        from vvla_workshop.fixtures import ScriptedHands, ScriptedPose, SyntheticCamera
        from vvla_workshop.twin_ros import TwinSession

        self.cfg = copy.deepcopy(cfg)
        self.cfg["robot"]["backend"] = "sim"
        self.twin = TwinSession(self.cfg, scene=scene)
        self.said: list[str] = []
        self.armcam = FanoutCamera(self.twin.armcam)
        self.mount = FanoutCamera(SyntheticCamera())
        self.ctx = PipelineContext(
            cfg=self.cfg,
            arm=self.twin.arm,
            pose=ScriptedPose(),
            hands=ScriptedHands(),
            mount_camera=self.mount,
            detector=detector,
            arm_camera=self.armcam,
            mapper=mapper_from_config(self.cfg),
            registry=build_registry(),
            headless=True,
        )
        self.ctx.say = self.said.append
        self._crashes: list[str] = []

    def said_any(self, phrase: str) -> bool:
        return any(phrase.lower() in m.lower() for m in self.said)

    def failure_report(self, dest: Path, since: float | None) -> str:
        """What the robot said, and the twin node's log."""
        said = "\n".join(f"  {m}" for m in self.said[-15:]) or "  (nothing)"
        return f"robot said:\n{said}\n\n" + twin_log_report(self.twin.log_path, dest)

    def start(self, text: str) -> threading.Thread:
        """Run ``text`` through ``handle_transcript`` on a thread, as the console's dispatcher does.

        A crash is recorded; :meth:`run` and :meth:`close` raise it, so it fails the test.
        """
        from vla_pipeline.dispatch import handle_transcript
        from vla_pipeline.llm.intents import keyword_parse

        def behave() -> None:
            try:
                handle_transcript(self.ctx, text, keyword_parse)
            except Exception as e:
                self._crashes.append(f"{text!r}: {e!r}")
                raise

        worker = threading.Thread(target=behave, daemon=True, name=text)
        worker.start()
        return worker

    def run(self, text: str) -> None:
        """Run ``text`` to the end of its behavior."""
        self.start(text).join()
        raise_crashes(self._crashes)

    def stop(self) -> None:
        """The console's stop button: ``handle_transcript`` with "stop robot", from another thread."""
        from vla_pipeline.dispatch import handle_transcript
        from vla_pipeline.llm.intents import keyword_parse

        handle_transcript(self.ctx, "stop robot", keyword_parse)

    def close(self) -> None:
        self.ctx.stop_event.set()
        for cam in (self.armcam, self.mount):
            try:
                cam.close()
            except Exception:
                pass
        self.twin.close()
        raise_crashes(self._crashes)
