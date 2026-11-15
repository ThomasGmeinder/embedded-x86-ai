# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""TODOs 2.10-2.11: the Gemma intent node - language as a ROS 2 citizen.

Transcripts arrive as ``std_msgs/String`` on ``/vla/transcript``; parsed
intents leave as JSON on ``/vla/intent``, so ANY node can react to language
without linking against the LLM. This subclass is the pipeline's
:class:`~vla_pipeline.llm.intent_node.IntentNode` with YOUR wiring and routing.

Reference: ``vla_pipeline/llm/intent_node.py`` (same method names and parameters).
Notebook: ``notebooks/02_ros2/03_intent_node.ipynb``
Self-test: ``python -m vvla_workshop.selftest_ros``
"""

from __future__ import annotations

from std_msgs.msg import String

from vla_pipeline.llm import intent_node
from vla_pipeline.llm.intents import Intent, ParsedCommand, is_noise, is_stop


class IntentNode(intent_node.IntentNode):
    """The pipeline's intent node, wired and routed by your TODOs."""

    def _setup_ros(self) -> None:
        """Wire the transcript subscription and the intent publisher.

        Both are ``std_msgs/String`` on ``self.transcript_topic`` /
        ``self.intent_topic``. Use RELIABLE delivery (the rclpy default - just
        pass a queue depth like ``10``): intents are commands, not a sensor
        stream, and must not be dropped. Keep the publisher in ``self._pub``.
        """
        # >>> TODO 2.10: IntentNode._setup_ros - notebooks/02_ros2/03_intent_node.ipynb
        raise NotImplementedError(
            "TODO 2.10 (IntentNode._setup_ros): implement me - see notebooks/02_ros2/03_intent_node.ipynb")
        # <<< TODO 2.10

    def _on_transcript(self, msg) -> None:
        """Route one transcript: stop fast path → noise gate → LLM → publish.

        1. :func:`is_stop` on the RAW text → publish
           ``ParsedCommand(Intent.STOP)`` immediately and return - no LLM round
           trip on the safety path.
        2. :func:`is_noise` → drop silently (stock phrases heard in silence
           must not become robot commands).
        3. Otherwise ``self._parse(text)`` and publish the result via
           :meth:`publish_intent`.
        """
        text = msg.data or ""
        # >>> TODO 2.11: IntentNode._on_transcript - notebooks/02_ros2/03_intent_node.ipynb
        raise NotImplementedError(
            "TODO 2.11 (IntentNode._on_transcript): implement me - see notebooks/02_ros2/03_intent_node.ipynb")
        # <<< TODO 2.11
