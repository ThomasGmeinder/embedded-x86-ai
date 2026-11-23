"""demo.DemoSession and demo_view.DemoView in the stand-in scene: the queue, the state, the picture."""

from __future__ import annotations

import threading

from so101_smolvla.demo import DemoSession
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
        self.asked: list[str] = []

    def parse(self, text: str) -> ParsedCommand:
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
        session.on_spoken(ANSWERS["grab the blue one"], 0.9)
        assert session.wait_idle(10)
        assert gemma.asked == [] and scene.targets == ["blue"]
        assert session.results[-1]["gemma"]["how"] == "synthetic voice"
        assert _kinds(session, "you") == ["you: grab the blue one"]
    finally:
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
