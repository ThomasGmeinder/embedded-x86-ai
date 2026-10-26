# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Component 2 self-test (PROVIDED) - run: ``python -m vvla_workshop.selftest_ros``

Exercises YOUR node wiring (TODOs 2.3, 2.10, 2.11 in ``workshop/project/ros/``)
end to end over real rclpy/DDS in this process: the pipeline's camera node
publishes a synthetic camera for your camera client, and a keyword brain
stands in for llama-server behind your intent node. ``--reference`` runs the
same checks on the pipeline's finished classes. ROS 2 is required
(``source /opt/ros/jazzy/setup.bash``).

Every wait is on the event it expects - a publisher matched, a message
arrived - bounded by a budget that fails the check when it passes.
"""

from __future__ import annotations

import json
import sys
import threading

from vla_pipeline.utils.config import load_config
from vvla_workshop.feedback import ARTIFACTS, Reporter, implementation, reference_flag

DISCOVERY_S = 5.0  # budget for two endpoints in one process to match
REPLY_S = 3.0  # budget for one message through a node


def _matched(event: threading.Event):
    """Event callback that sets ``event`` once the endpoint has a peer."""
    return lambda info: event.set() if info.current_count > 0 else None


def check_camera(rep: Reporter, cfg: dict, reference: bool) -> None:
    """TODO 2.3 - your camera client receives the camera node's frames."""
    import cv2

    from vla_pipeline.robot.ros2_control_arm import Spinner
    from vla_pipeline.vision.camera_node import CameraNode
    from vvla_workshop.fixtures import SyntheticCamera

    Ros2Camera = implementation("2.3", reference).Ros2Camera
    spec = dict(cfg["cameras"]["mount"])
    topic = spec["topic"]
    host = Spinner("selftest_camera_host")
    node = CameraNode("mount", spec, frame_source=SyntheticCamera())
    host.add_node(node)
    client = None
    try:
        ok_client, client = rep.run(
            "2.3", "camera client constructs (subscription + executor)",
            lambda: Ros2Camera("mount", topic, stale_after_s=spec.get("stale_after_s", 1.0)),
            hint="subscribe with BEST-EFFORT QoS - the node publishes best-effort, and a RELIABLE "
            "subscriber never matches it",
        )
        if not ok_client:
            rep.add("2.3", "frame round-trip over the image topic", "SKIP", "needs the client first")
            return

        def roundtrip():
            """Verify a frame published by the node arrives correctly decoded at the client."""
            ok, frame, _arrived = client.next_frame(0.0, DISCOVERY_S + REPLY_S)
            assert ok, (
                f"nothing arrived on {topic} - check the topic name and QoS on both ends "
                "(ros2 topic list / ros2 topic info -v)")
            assert frame.shape == (spec["height"], spec["width"], 3), (
                f"frame shape {frame.shape} != ({spec['height']}, {spec['width']}, 3) - encoding or step "
                "mishandled in the Image message")
            assert int(frame[0, -1, 0]) > 180, (
                "pixels don't match the synthetic pattern - the frame was re-encoded wrong "
                "(bgr8 bytes should pass through unchanged)")
            out = frame.copy()
            cv2.putText(out, f"round-trip OK via {topic}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            path = ARTIFACTS / "camera_roundtrip.png"
            cv2.imwrite(str(path), out)
            rep.artifact(path)

        rep.run("2.3", "frame round-trip over the image topic", roundtrip)
    finally:
        if client is not None:
            client.close()
        node.destroy_node()
        host.close()


def check_intent(rep: Reporter, cfg: dict, reference: bool) -> None:
    """TODOs 2.10-2.11 - the intent node's wiring, routing, stop path and noise gate."""
    from rclpy.event_handler import PublisherEventCallbacks, SubscriptionEventCallbacks
    from std_msgs.msg import String

    from vla_pipeline.llm.intents import keyword_parse
    from vla_pipeline.robot.ros2_control_arm import Spinner

    IntentNode = implementation("2.10", reference).IntentNode
    calls: list[str] = []

    def counting_parser(text: str):
        """Keyword-route ``text`` like the mock brain, recording every call."""
        calls.append(text)
        return keyword_parse(text)

    n = cfg.get("intent_node", {}) or {}
    intent_topic = n.get("intent_topic", "/vla/intent")
    transcript_topic = n.get("transcript_topic", "/vla/transcript")
    host = Spinner("selftest_intent_host")
    ok_node, node = rep.run(
        "2.10", "intent node constructs (String sub + pub, RELIABLE)",
        lambda: IntentNode(cfg, counting_parser),
        hint="std_msgs/String both ways; default reliable QoS (a plain depth like 10) - intents are "
        "commands, not a sensor stream",
    )
    if not ok_node:
        rep.add("2.11", "intent routing checks", "SKIP", "needs TODO 2.10")
        host.close()
        return
    host.add_node(node)

    received: list[dict] = []
    arrived = threading.Condition()

    def on_intent(msg) -> None:
        with arrived:
            received.append(json.loads(msg.data))
            arrived.notify_all()

    sub_matched, pub_matched = threading.Event(), threading.Event()
    host.node.create_subscription(String, intent_topic, on_intent, 10,
                                  event_callbacks=SubscriptionEventCallbacks(matched=_matched(sub_matched)))
    pub = host.node.create_publisher(String, transcript_topic, 10,
                                     event_callbacks=PublisherEventCallbacks(matched=_matched(pub_matched)))

    def say(text: str) -> dict:
        """Publish one transcript; return the next intent the node publishes."""
        with arrived:
            before = len(received)
        pub.publish(String(data=text))
        with arrived:
            assert arrived.wait_for(lambda: len(received) > before, REPLY_S), (
                f"no intent on {intent_topic} within {REPLY_S:.0f} s of {text!r} - is the transcript "
                "subscription wired to _on_transcript, and the result published?")
            return received[before]

    try:
        def discovery():
            """Both topics of the node have a peer."""
            assert pub_matched.wait(DISCOVERY_S), f"nothing subscribes to {transcript_topic}"
            assert sub_matched.wait(DISCOVERY_S), f"nothing publishes on {intent_topic}"

        ok_graph, _ = rep.run("2.10", "the node's subscription and publisher are on the graph", discovery)
        if not ok_graph:
            rep.add("2.11", "intent routing checks", "SKIP", "needs TODO 2.10")
            return

        def normal_routing():
            """Verify a plain transcript round-trips through the parser into published intent JSON."""
            intent = say("pick up the ball")
            assert intent.get("command") == "pick_place", (
                f"published intent {intent} for 'pick up the ball' - expected pick_place (is the parser "
                "being called and its result serialized?)")
            assert intent.get("object") == "ball", f"object slot lost: {intent} - carry cmd.object_name through"

        rep.run("2.11", "transcript → intent JSON round-trip", normal_routing)

        def stop_fast_path():
            """Verify a stop transcript publishes at once without reaching the parser."""
            before = len(calls)
            intent = say("whoa STOP right now!")
            assert intent.get("command") == "stop", f"stop transcript produced {intent}"
            assert len(calls) == before, (
                "the LLM parser was invoked for a STOP transcript - the safety stop must bypass the model "
                "entirely (keyword match on the raw text FIRST)")

        rep.run("2.11", "stop fast path bypasses the LLM", stop_fast_path)

        def noise_gate():
            """A noise transcript, then a command: only the command's intent arrives.

            One reliable publisher to one node keeps their order, so the first
            intent after the noise is the command's when the noise published none.
            """
            before = len(calls)
            pub.publish(String(data="Thank you."))
            intent = say("wave hello")
            assert intent.get("command") == "wave", (
                f"the noise transcript published an intent ({intent}) - Whisper's silence hallucinations "
                "must be dropped (is_noise)")
            assert calls[before:] == ["wave hello"], "the parser was invoked for a noise transcript - gate first"

        rep.run("2.11", "noise transcripts are dropped before the LLM", noise_gate)
    finally:
        node.destroy_node()
        host.close()


def run(reference: bool = False) -> int:
    """Run every component 2 check on your classes (or the reference) over ROS 2; return the exit code."""
    import rclpy

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    rep = Reporter("Component 2 - ROS 2 nodes & topics" + (" (reference)" if reference else ""))
    if not rclpy.ok():
        rclpy.init()
    cfg = load_config()
    check_camera(rep, cfg, reference)
    check_intent(rep, cfg, reference)
    code = rep.summary()
    print("\nOn-hardware next steps (each in its own terminal):")
    print("  python -m vla_pipeline.vision.camera_node --role mount")
    print("  python -m vla_pipeline.llm.intent_node --mock")
    print("  ros2 topic hz /cameras/mount/image_raw && ros2 topic echo /vla/intent")
    return code


def main() -> int:
    """CLI: ``--reference`` checks the pipeline's finished classes."""
    reference = reference_flag("Component 2 self-test (requires ROS 2)")
    try:
        import rclpy
    except ImportError as e:
        raise SystemExit(f"ROS 2 is required: source /opt/ros/jazzy/setup.bash first ({e})") from e
    try:
        return run(reference)
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
