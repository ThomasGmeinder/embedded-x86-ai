# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The Llama intent node - language as a ROS 2 citizen.

Transcripts arrive as ``std_msgs/String`` on ``/vla/transcript`` (from the
speech stack, ``ros2 topic pub``, or a notebook), and parsed intents leave as
JSON on ``/vla/intent`` - so ANY node can react to language without linking
against the LLM. ``_setup_ros`` and ``_on_transcript`` are the workshop's
TODOs 2.10 and 2.11.

- **The stop fast path.** "stop"/"halt"/"freeze" is keyword-matched on the
  raw transcript and published IMMEDIATELY - the safety stop must never wait
  in line behind a language model.
- **QoS asymmetry.** Camera frames are best-effort (a lost frame is free);
  intents are commands, so this node uses default RELIABLE QoS - a lost
  "stop" is not free.

Run it (llama-server running, or ``--mock`` for the keyword brain):
    python -m vla_pipeline.llm.intent_node [--mock]
Then, in another terminal:
    ros2 topic pub --once /vla/transcript std_msgs/String "{data: pick up the ball}"
    ros2 topic echo /vla/intent
"""

from __future__ import annotations

import argparse
import json
import logging

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from vla_pipeline.llm.intents import Intent, ParsedCommand, is_noise, is_stop, keyword_parse
from vla_pipeline.utils.config import load_config

logger = logging.getLogger(__name__)


class IntentNode(Node):
    """Subscribes to transcripts, publishes grammar-shaped intent JSON.

    ``parser`` is injected: any callable ``str -> ParsedCommand`` - on the rig
    :meth:`~vla_pipeline.llm.llama_intent.LlamaIntentParser.parse` (Llama on
    the iGPU), in tests the keyword stand-in.
    """

    def __init__(self, cfg: dict, parser):
        super().__init__("llama_intent_node")
        n = cfg.get("intent_node", {}) or {}
        self.transcript_topic = n.get("transcript_topic", "/vla/transcript")
        self.intent_topic = n.get("intent_topic", "/vla/intent")
        self._parse = parser
        self._pub = None
        self._setup_ros()
        self.get_logger().info(f"intent node: {self.transcript_topic} → {self.intent_topic}")

    def _setup_ros(self) -> None:
        """Wire the transcript subscription and the intent publisher.

        Both are ``std_msgs/String``, with RELIABLE delivery (the rclpy
        default, given a queue depth like ``10``): intents are commands, not a
        sensor stream, and must not be dropped.
        """
        self.create_subscription(String, self.transcript_topic, self._on_transcript, 10)
        self._pub = self.create_publisher(String, self.intent_topic, 10)

    def _on_transcript(self, msg) -> None:
        """Route one transcript: stop fast path → noise gate → LLM → publish.

        1. :func:`~vla_pipeline.llm.intents.is_stop` on the RAW text →
           publish ``{"command": "stop", "object": ""}`` immediately - no LLM
           round trip on the safety path.
        2. :func:`~vla_pipeline.llm.intents.is_noise` → drop silently
           (a noise phrase must not become a robot command).
        3. Otherwise ``self._parse(text)`` and publish the result via
           :meth:`publish_intent`.
        """
        text = msg.data or ""
        if is_stop(text):
            self.publish_intent(ParsedCommand(Intent.STOP, transcript=text))
            return
        if is_noise(text):
            self.get_logger().info(f"ignoring noise transcript: {text!r}")
            return
        self.publish_intent(self._parse(text))

    def publish_intent(self, cmd: ParsedCommand) -> None:
        """Serialize a ParsedCommand as the grammar-shaped JSON."""
        payload = json.dumps({"command": cmd.intent.value, "object": cmd.object_name or ""})
        self._pub.publish(String(data=payload))
        self.get_logger().info(f"intent: {payload}")


def main() -> None:
    """CLI: bring up the intent node (mock or Llama-backed) and spin until interrupted."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Llama intent node")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mock", action="store_true", help="keyword stand-in brain (no llama-server needed)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mock:
        parse = keyword_parse
        print("(mock brain - keyword routing, no LLM)")
    else:
        from vla_pipeline.llm.llama_intent import LlamaIntentParser

        parse = LlamaIntentParser(cfg).parse

    rclpy.init()
    node = IntentNode(cfg, parse)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
