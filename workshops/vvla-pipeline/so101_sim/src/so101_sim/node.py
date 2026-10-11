# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ROS 2 process hosting the digital twin in place of the arm + arm camera.

It runs the host stack's OWN server nodes - ``So101ServerNode`` with the
twin injected as its bus transport, and ``CameraNode`` with a
:class:`~so101_sim.camera.SimCamera` as its frame source - so the topics,
QoS, rates, de-duplication and health flags are byte-for-byte the real
ones. Clients (the orchestrator, the workshop app, ``ros2 topic echo``)
cannot tell the difference.

    python -m so101_sim.node --config config/pipeline.yaml
    python -m so101_sim.node --config workshop/solution/config/workshop.yaml
    python -m so101_sim.node --config config/pipeline.yaml --device-cameras --viewer

Arm and arm camera must live in ONE process (they share the physics), which
is why this replaces ``robot_node --server`` + ``camera_node --role arm``.
Extra topic: ``/sim/view/image_raw`` (the free 3-D view, bgr8).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

VIEW_TOPIC = "/sim/view/image_raw"


def detect_stack(config_path: str | None, cfg: dict) -> str:
    """``workshop`` for workshop.yaml-style configs, else ``pipeline``."""
    if config_path and Path(config_path).name.startswith("workshop"):
        return "workshop"
    return "pipeline"


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _stack_parts(stack: str, cfg: dict, config_path: str | None):
    """(world, arm, ServerNodeClass, CameraNodeClass) for the host stack."""
    if stack == "workshop":
        # workshop/<solution|project>/config/workshop.yaml -> host that variant's ros/
        if config_path:
            pkg = str(Path(config_path).resolve().parents[1])
            if pkg not in sys.path:
                sys.path.insert(0, pkg)
        from common.motion import SimArmClient, sim_world
        from ros.arm_node import So101ServerNode
        from ros.camera_node import CameraNode

        world = sim_world(cfg)
        return world, SimArmClient(cfg, world), So101ServerNode, CameraNode
    from vla_pipeline.robot.robot_node import So101ServerNode
    from vla_pipeline.robot.sim_arm import SimArmClient, sim_world
    from vla_pipeline.vision.camera_node import CameraNode

    world = sim_world(cfg)
    return world, SimArmClient(cfg, world), So101ServerNode, CameraNode


class ViewPublisher:
    """Publishes a twin camera (default: the free 3-D view) as sensor_msgs/Image."""

    def __init__(self, world, role: str = "view", topic: str = VIEW_TOPIC, fps: float = 15.0):
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        from so101_sim.camera import SimCamera

        self._Image = Image
        self.node = Node("so101_sim_view")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1
        )
        self._pub = self.node.create_publisher(Image, topic, qos)
        self._cam = SimCamera(world, role, fps=fps, blocking=False)
        self.node.create_timer(1.0 / fps, self._tick)

    def _tick(self) -> None:
        ok, frame = self._cam.read()
        if not ok:
            return
        msg = self._Image()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "sim_view"
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        msg.data = frame.tobytes()
        self._pub.publish(msg)

    def destroy(self) -> None:
        self._cam.release()
        self.node.destroy_node()


def build_nodes(
    cfg: dict,
    *,
    stack: str = "pipeline",
    config_path: str | None = None,
    device_cameras: bool = False,
    view: bool = True,
):
    """Create the twin's ROS nodes (rclpy must be initialised).

    Returns ``(world, nodes, closers)``: add ``nodes`` to an executor, call
    each closer on shutdown. Cameras with ``source: sim`` are rendered on the
    twin; ``source: device`` cameras are opened here only if
    ``device_cameras`` (otherwise their usual camera node serves them).
    """
    from so101_sim.camera import SimCamera

    world, arm, ServerNode, CameraNode = _stack_parts(stack, cfg, config_path)
    nodes, closers = [], []
    server = ServerNode(cfg, arm=arm)
    nodes.append(server)
    closers.append(server.destroy_node)
    for role, spec in (cfg.get("cameras") or {}).items():
        if role not in ("arm", "mount") or not isinstance(spec, dict):
            continue
        source = spec.get("source", "sim" if role == "arm" else "device")
        if source == "sim":
            cam = CameraNode(role, spec, frame_source=SimCamera(world, role, blocking=False))
        elif device_cameras:
            cam = CameraNode(role, spec)
        else:
            continue
        nodes.append(cam)
        closers.append(cam.destroy_node)
    if view:
        vp = ViewPublisher(world)
        nodes.append(vp.node)
        closers.append(vp.destroy)
    return world, nodes, closers


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="SO-101 digital twin ROS 2 node")
    ap.add_argument("--config", required=True, help="pipeline.yaml or workshop.yaml")
    ap.add_argument("--stack", choices=["auto", "pipeline", "workshop"], default="auto")
    ap.add_argument("--device-cameras", action="store_true",
                    help="also publish source:device cameras from this process")
    ap.add_argument("--sim-cameras", default=None,
                    help="comma list of camera roles to render on the twin "
                    "(overrides cameras.<role>.source; default: as configured)")
    ap.add_argument("--no-view", action="store_true", help=f"don't publish {VIEW_TOPIC}")
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer window")
    args, ros_args = ap.parse_known_args()

    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
    except ImportError:
        raise SystemExit("rclpy not importable - source /opt/ros/jazzy/setup.bash first") from None

    cfg = _load_yaml(args.config)
    cfg.setdefault("robot", {})["backend"] = "sim"
    if args.viewer:
        cfg.setdefault("sim", {})["viewer"] = True
    for role in filter(None, (args.sim_cameras or "").split(",")):
        cfg.setdefault("cameras", {}).setdefault(role.strip(), {})["source"] = "sim"
    stack = detect_stack(args.config, cfg) if args.stack == "auto" else args.stack

    rclpy.init(args=ros_args)
    world, nodes, closers = build_nodes(
        cfg,
        stack=stack,
        config_path=args.config,
        device_cameras=args.device_cameras,
        view=not args.no_view,
    )
    executor = MultiThreadedExecutor(num_threads=4)
    for n in nodes:
        executor.add_node(n)
    logger.info("digital twin up (%s stack): %d nodes", stack, len(nodes))
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        for close in reversed(closers):
            try:
                close()
            except Exception as e:
                logger.debug("shutdown: %s", e)
        from so101_sim.world import close_shared_world

        close_shared_world()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
