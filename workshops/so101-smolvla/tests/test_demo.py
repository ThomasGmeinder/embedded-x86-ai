"""demo.DemoSession and demo_view.DemoView in the stand-in scene: the queue, the state, the picture."""

from __future__ import annotations

import threading

from so101_smolvla.demo import COMMAND_RECEIVED, TRANSCRIPT_RECEIVED, DemoSession
from so101_smolvla.pick_place import PickPlaceScene
from so101_smolvla.voice.intents import Intent, ParsedCommand
from test_three_cube_rollout import StandInPolicy, StandInScene

ANSWERS = {
    "put the green cube in the bin": ParsedCommand(Intent.PICK_PLACE, {"object": "green cube", "place": "bin"}, "put the green cube in the bin"),
    "put the red cube in the bin": ParsedCommand(Intent.PICK_PLACE, {"object": "red cube", "place": "bin"}, "put the red cube in the bin"),
    "grab the blue one": ParsedCommand(Intent.PICK_PLACE, {"object": "blue one"}, "grab the blue one"),
    "learn the stack": ParsedCommand(Intent.LEARN_STACK, {}, "learn the stack"),
}


def act(policy, pre, post, observation, task_text):
    return observation.state


class Gemma:
    """Gemma's answers for the test's sentences; ``hold`` keeps a request waiting until released."""

    def __init__(self) -> None:
        self.hold = threading.Event()
        self.hold.set()
        self.entered = threading.Event()  # set when a request has reached Gemma
        self.asked: list[str] = []

    def parse(self, text: str) -> ParsedCommand:
        self.entered.set()
        self.hold.wait()
        self.asked.append(text)
        return ANSWERS.get(text, ParsedCommand(Intent.UNKNOWN, {}, text))


def _session():
    scene, gemma = StandInScene(), Gemma()
    loop = PickPlaceScene(scene, StandInPolicy(), None, None, pictures="others-painted", steps_per_pick=3, act=act, record=False)
    loop.start_layout(4, 10)
    session = DemoSession(loop, gemma.parse, seed=4).start()
    return scene, gemma, session


def _kinds(session, kind):
    return [text for text, k in session.transcript if k == kind]


def test_a_typed_command_runs_one_pick_and_the_state_follows_it():
    scene, gemma, session = _session()
    try:
        assert session.idle.is_set() and session.snapshot.status.startswith("idle")
        assert set(session.snapshot.pictures) == {"top", "wrist"} and session.snapshot.opening == 0.0
        session.submit("put the green cube in the bin")
        assert session.wait_idle(10)
        assert gemma.asked == ["put the green cube in the bin"] and scene.targets == ["green"]
        assert session.results[-1]["cube"] == "green" and session.results[-1]["in_bin_at_end"]
        assert session.snapshot.cubes_in_bin == ("green",) and session.snapshot.cube is None
        assert set(session.snapshot.plans) == {"top", "wrist"}
        assert _kinds(session, "you") == ["you: put the green cube in the bin"]
        assert _kinds(session, "call")[0].startswith("-> pick_place(object: green cube, place: bin)")
        assert any(text.startswith("green cube: placed, rule held at step 1") for text in _kinds(session, "note"))
    finally:
        session.close()


def test_a_spoken_command_arrives_with_gemmas_fields_and_is_not_asked_again():
    scene, gemma, session = _session()
    try:
        session.command_received(ANSWERS["grab the blue one"], 0.9)
        assert session.wait_idle(10)
        assert gemma.asked == [] and scene.targets == ["blue"]
        assert session.results[-1]["gemma"]["how"] == "synthetic voice"
        assert _kinds(session, "you") == ["you: grab the blue one"]
    finally:
        session.close()


def test_a_spoken_call_starts_the_pick_before_its_words_and_the_words_are_said_when_they_come():
    scene, gemma, session = _session()
    try:
        # The call as the listener hands it over the moment Gemma has written it: no words yet.
        session.command_received(ParsedCommand(Intent.PICK_PLACE, {"object": "blue one"}), 1.4)
        assert session.wait_idle(10)
        assert gemma.asked == [] and scene.targets == ["blue"]
        assert _kinds(session, "you") == []  # nothing is made up for the words
        assert _kinds(session, "call")[0].startswith("-> pick_place(object: blue one)  (1.40 s)")
        # The whole answer follows: its words, and no second pick.
        session.transcript_received(ANSWERS["grab the blue one"])
        assert _kinds(session, "you") == ["you: grab the blue one"]
        assert session.wait_idle(10) and scene.targets == ["blue"]
        session.transcript_received(ParsedCommand(Intent.PICK_PLACE, {"object": "blue one"}))  # words not made out
        assert _kinds(session, "you") == ["you: grab the blue one"]
    finally:
        session.close()


def test_the_picture_is_told_of_the_call_when_it_arrives_and_again_of_the_words_while_the_worker_is_busy():
    scene, gemma, session = _session()
    try:
        gemma.hold.clear()  # the worker is held inside a typed request: it cannot say anything
        session.submit("put the green cube in the bin")
        # The typed sentence is shown before Gemma has answered; its call is not there yet.
        assert gemma.entered.wait(10)
        assert _kinds(session, "you") == ["you: put the green cube in the bin"] and _kinds(session, "call") == []

        # A spoken call arrives: the picture has it when command_received returns, on the caller's thread.
        session.changed.clear()
        session.command_received(ParsedCommand(Intent.PICK_PLACE, {"object": "blue one"}), 1.4)
        assert session.changed.is_set()
        assert _kinds(session, "call") == ["-> pick_place(object: blue one)  (1.40 s)"]
        assert _kinds(session, "you") == ["you: put the green cube in the bin"]  # its words are not there yet
        assert scene.targets == []  # nothing moved: the worker is still held

        # The words follow: a second update of the picture.
        session.changed.clear()
        session.transcript_received(ANSWERS["grab the blue one"])
        assert session.changed.is_set()
        assert _kinds(session, "you") == ["you: put the green cube in the bin", "you: grab the blue one"]

        gemma.hold.set()
        assert session.wait_idle(10)
        assert scene.targets == ["green", "blue"]
        # The spoken call was shown once, when it arrived; the worker did not write it again.
        calls = _kinds(session, "call")
        assert len(calls) == 2 and sum(call.startswith("-> pick_place(object: blue one)") for call in calls) == 1
        # The events, by name, in the order they happened: the typed sentence, the spoken call,
        # its words, and the typed sentence's call when Gemma was let go.
        assert [(name, text) for name, _, text in session.events] == [
            (TRANSCRIPT_RECEIVED, "you: put the green cube in the bin"),
            (COMMAND_RECEIVED, "-> pick_place(object: blue one)"),
            (TRANSCRIPT_RECEIVED, "you: grab the blue one"),
            (COMMAND_RECEIVED, "-> pick_place(object: green cube, place: bin)"),
        ]
        times = [at for _, at, _ in session.events]
        assert times == sorted(times)
    finally:
        gemma.hold.set()
        session.close()


def test_a_command_and_its_transcript_are_drawn_when_they_happen_not_at_the_next_frame():
    from so101_smolvla.demo_view import DemoView

    scene, _, session = _session()
    # One frame every 30 s: a draw that waits for the frame pause cannot come within the bounds below.
    view = DemoView(session, fps=1 / 30)
    try:
        view.show()
        session.say("a note")  # an ordinary change: drawn, then the frame pause begins
        assert view.drawn.wait(10)
        frames = view.frames
        for event in (
            lambda: session.command_received(ParsedCommand(Intent.LEARN_STACK, {}), 1.4),
            lambda: session.transcript_received(ANSWERS["learn the stack"], 2.1),
        ):
            view.drawn.clear()
            event()
            assert view.drawn.wait(10)  # the draw the event caused, inside the 30 s pause
            assert view.frames > frames
            frames = view.frames
        calls = _kinds(session, "call")
        assert len(calls) == 1 and calls[0].startswith("-> learn_stack") and calls[0].endswith("  (1.40 s)")
        assert _kinds(session, "you") == ["you: learn the stack  (2.10 s)"]
        # Both events ended the pause through the session's own event, which the draw clears.
        assert [name for name, _, _ in session.events] == [COMMAND_RECEIVED, TRANSCRIPT_RECEIVED]
    finally:
        view.close()
        session.close()


def test_a_refused_sentence_moves_nothing_and_says_why():
    scene, _, session = _session()
    try:
        session.submit("learn the stack")
        assert session.wait_idle(10)
        assert scene.targets == [] and session.results[-1]["cube"] is None
        assert _kinds(session, "error") == ["refused: not a pick command but 'learn_stack'; nothing moves"]
        assert session.snapshot.status.startswith("refused")
    finally:
        session.close()


def test_idle_stays_false_while_a_request_waits_and_a_second_waiting_command_replaces_the_first():
    scene, gemma, session = _session()
    try:
        gemma.hold.clear()  # the first request waits inside Gemma
        session.submit("put the green cube in the bin")
        session.submit("grab the blue one")  # waits in the queue
        session.submit("put the red cube in the bin")  # replaces it
        assert not session.wait_idle(0.2)
        assert "replaced the waiting command" in _kinds(session, "note")
        gemma.hold.set()
        assert session.wait_idle(10)
        assert scene.targets == ["green", "red"]
        assert gemma.asked == ["put the green cube in the bin", "put the red cube in the bin"]
    finally:
        session.close()


def test_a_new_layout_resets_the_scene_from_the_worker_and_the_picture_starts_over():
    scene, _, session = _session()
    try:
        session.submit("put the green cube in the bin")
        assert session.wait_idle(10) and session.snapshot.plans
        session.new_layout(7)
        assert session.wait_idle(10)
        assert scene.seed == 7 and session.snapshot.seed == 7 and session.snapshot.plans == {}
        assert session.snapshot.status.startswith("idle") and session.scene.layout_steps == 0
    finally:
        session.close()
    assert not session._worker.is_alive()


def test_the_view_draws_on_change_only_and_closes_its_thread():
    from so101_smolvla.demo_view import DemoView

    scene, _, session = _session()
    view = DemoView(session, fps=50.0)
    try:
        view.show()
        assert view.image.value[:2] == b"\xff\xd8"
        view.drawn.clear()
        session.submit("put the green cube in the bin")
        assert session.wait_idle(10)
        assert view.drawn.wait(5)
        drawn_once = view.frames
        assert drawn_once >= 1
        # Nothing changes: the pacing lets any pending change through, then the thread blocks.
        view.drawn.clear()
        assert not view.drawn.wait(0.3) or view.frames <= drawn_once + 1
    finally:
        view.close()
        session.close()
    assert not view._thread.is_alive() and view.text.disabled


def test_a_closed_view_shows_the_stopped_picture_not_its_last_frame():
    from so101_smolvla.demo_console import compose, jpeg
    from so101_smolvla.demo_view import DemoView

    _, _, session = _session()
    view = DemoView(session, fps=50.0)
    try:
        view.show()
        first = bytes(view.image.value)  # drawn before any microphone frame: it says "no microphone"
        view.close()
        stopped = jpeg(compose(session.snapshot, session.transcript, session.voice, 0.0, stopped=True))
        assert bytes(view.image.value) == stopped and stopped != first
        view.close()  # a second close changes nothing
        assert bytes(view.image.value) == stopped
    finally:
        session.close()


def test_the_stop_control_stops_nothing_when_shown_and_once_when_pressed():
    from so101_smolvla.demo_view import StopControl

    stops = []

    def stop():
        stops.append(1)
        return ["table closed", "Gemma, SAM 3 and the scene stopped."]

    control = StopControl(stop).show()
    assert stops == [] and not control.done and control.said == "" and not control.button.disabled
    control.button.click()
    assert stops == [1] and control.done and control.button.disabled
    assert control.said == "table closed\nGemma, SAM 3 and the scene stopped."
    control.button.click()  # a second press does nothing
    assert stops == [1]
