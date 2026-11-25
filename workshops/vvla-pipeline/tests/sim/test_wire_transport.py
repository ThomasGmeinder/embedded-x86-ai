# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""A spoken command and its words, from perception's link to control's, over ROS 2.

``PerceptionLink`` and ``ControlLink`` on two nodes of one private ROS context
and domain: the real topics, the real encoders, the real ledger and the real
dispatcher's entry points. The command is offered the moment the model has
written it and its words when the answer is complete; control must act on the
command when it arrives and show the words after it. Both are sent on one
reliable topic, so the words cannot arrive first.

No model, no microphone and no arm. ``tests/sim/test_ipc.py`` holds the
orderings of the outbox and the ledger; this is the transport between them.
"""

from __future__ import annotations

import threading

import pytest

pytestmark = [pytest.mark.sim, pytest.mark.timeout(60)]

# Discovery of two nodes in one process and one delivery took well under a second (2026-11-26).
DELIVERY_S = 20.0


@pytest.fixture
def graph():
    """``(perception node, control node)`` in a ROS context and domain of their own, spun on one thread."""
    import domain_coordinator
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    with domain_coordinator.domain_id() as domain:
        context = rclpy.Context()
        rclpy.init(context=context, domain_id=domain)
        perception = rclpy.create_node("wire_test_perception", context=context)
        control = rclpy.create_node("wire_test_control", context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(perception)
        executor.add_node(control)
        stop = threading.Event()
        wake = control.create_guard_condition(lambda: None)

        def spin() -> None:
            while not stop.is_set():
                executor.spin_once()

        thread = threading.Thread(target=spin, daemon=True, name="wire-test")
        thread.start()
        try:
            yield perception, control
        finally:
            stop.set()
            wake.trigger()  # wakes spin_once so the thread sees the stop
            thread.join()
            executor.shutdown()
            perception.destroy_node()
            control.destroy_node()
            rclpy.shutdown(context=context)


def test_a_spoken_command_reaches_control_before_its_words(graph):
    from vla_pipeline.llm.intents import CommandRegistry, Intent, ParsedCommand
    from vla_pipeline.user_robot_control import PipelineContext, UserRobotControl
    from vla_pipeline.wire import ControlLink, PerceptionLink

    perception_node, control_node = graph
    ctx = PipelineContext(cfg={}, arm=None, registry=CommandRegistry())
    ctx.say = lambda _message: None
    voice = UserRobotControl(ctx, lambda text: ParsedCommand(Intent.UNKNOWN, transcript=text))
    seen: list[tuple[str, str | None]] = []
    words_shown = threading.Event()

    def on_event(event: str, **data) -> None:
        if event in ("queued", "heard"):
            seen.append((event, data.get("text")))
        if event == "heard" and data.get("text") == "dance for me":
            words_shown.set()  # the last message sent, whichever of the two orderings below happened

    voice.listeners.append(on_event)
    control = ControlLink(control_node, ctx, voice, lambda: False)
    perception = PerceptionLink(perception_node)

    # Offered at once, without waiting for control's epoch. Two orderings are right, and which
    # one happens depends on whether the epoch has arrived by now:
    #   it has:     wave goes out, then its words, then dance, then its words;
    #   it has not: dance replaces the unsent wave and goes out when the epoch arrives, with its
    #               words behind it; wave and its words never go out.
    # In both, words for a command that is not the one sent last are not sent.
    first = perception.offer_command("wave", "")
    perception.offer_heard(first, "wave at me")
    second = perception.offer_command("dance", "")
    perception.offer_heard(first, "words for a command that is no longer the last")
    perception.offer_heard(second, "dance for me")

    assert words_shown.wait(DELIVERY_S), f"control showed {seen}"
    wave, dance = '{"command":"wave","object":""}', '{"command":"dance","object":""}'
    assert seen in (
        [("queued", wave), ("heard", "wave at me"), ("queued", dance), ("heard", "dance for me")],
        [("queued", dance), ("heard", "dance for me")],
    ), seen
    assert control.heard == "dance for me"
