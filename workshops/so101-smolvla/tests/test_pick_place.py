"""pick_place: the one pick-and-place loop, in the stand-in scene of the runner's tests."""

from __future__ import annotations

import pytest

from so101_smolvla.pick_place import (
    PickPlaceScene,
    PlanView,
    StepView,
    command_from_colour,
    command_from_fields,
)
from so101_smolvla.voice.commands import CommandFields
from test_three_cube_rollout import StandInPolicy, StandInScene


def act(policy, pre, post, observation, task_text):
    return observation.state


class Recorder:
    def __init__(self) -> None:
        self.plans: list[PlanView] = []
        self.steps: list[StepView] = []

    def on_plan(self, view: PlanView) -> None:
        self.plans.append(view)

    def on_step(self, view: StepView) -> None:
        self.steps.append(view)


def _scene(record: bool = False):
    scene, policy = StandInScene(), StandInPolicy()
    loop = PickPlaceScene(scene, policy, None, None, pictures="others-painted", steps_per_pick=3, act=act, record=record)
    loop.start_layout(4, 10)
    return scene, policy, loop


def test_a_colour_command_is_written_as_gemma_writes_a_pick():
    assert command_from_colour("blue").command == "pick_place(object: blue cube, place: box)"
    assert command_from_colour("blue").cube == "blue" and command_from_colour("blue").translation is None
    with pytest.raises(ValueError, match="purple"):
        command_from_colour("purple")


def test_gemmas_fields_give_the_cube_or_the_refusal_with_the_sentence_kept():
    fields = CommandFields("grab the green one", "typed", "pick_place", "green one", "", "grab the green one", 0.8)
    pick = command_from_fields(fields, "grab the green one", "green")
    assert pick.cube == "green" and pick.translation["as_expected"] is True
    assert pick.translation["gemma"]["heard"] == "grab the green one"
    refused = command_from_fields(
        CommandFields("learn the stack", "typed", "learn_stack", "", "", "learn the stack", 0.1), "learn the stack"
    )
    assert refused.cube is None and refused.refusal == "not a pick command but 'learn_stack'"
    assert refused.translation["as_expected"] is None


def test_the_scene_runs_commands_one_after_another_and_tells_its_observer_of_every_plan_and_step():
    scene, policy, loop = _scene()
    recorder = Recorder()
    result = loop.run_command(command_from_colour("green"), observer=recorder)
    assert scene.targets == ["green"] and policy.resets == 1
    assert result["cube"] == "green" and result["in_bin_at_end"] and result["first_success_step"] == 1
    assert [view.step for view in recorder.steps] == [0, 1, 2]
    assert all(view.cube == "green" and view.steps_per_pick == 3 for view in recorder.steps)
    # The stand-in scene: the jaw touches and holds the cube after the first step, the cube is in
    # the bin after the second.
    assert recorder.steps[0].grasped and recorder.steps[0].contact and not recorder.steps[0].in_bin
    assert recorder.steps[1].in_bin and recorder.steps[1].success and recorder.steps[2].in_bin
    # The stand-in policy plans every second step: plan steps counted over the layout.
    assert [view.step for view in recorder.plans] == [0, 2]
    assert set(recorder.plans[0].feeds) == {"top", "wrist"}
    assert recorder.plans[0].feeds["top"].legend[1].role == "target"  # green is the second cube
    assert loop.layout_steps == 3 and loop.record.targets == []  # nothing kept without recording
    loop.run_command(command_from_colour("red"), observer=recorder)
    assert scene.targets == ["green", "red"] and policy.resets == 2
    assert [view.step for view in recorder.plans] == [0, 2, 3, 5] and loop.layout_steps == 6
    assert loop.record.sequence == ["green", "red"] and len(loop.record.picks) == 2


def test_recording_keeps_what_a_layouts_episode_folder_needs():
    scene, _, loop = _scene(record=True)
    loop.run_command(command_from_colour("green"))
    record = loop.record
    assert len(record.targets) == len(record.top_frames) == len(record.wrist_frames) == 3
    assert [plan.step for plan in record.plans["top"]] == [0, 2]
    assert record.plans["wrist"][0].command == "pick_place(object: green cube, place: box)"
    assert record.initial_layout == {"bin_xy": [0.3, 0.0], "bin_yaw": 0.0, "cubes": {}}
    assert record.start_state.shape == (6,)


def test_a_refused_command_moves_nothing_and_is_recorded_as_such():
    scene, policy, loop = _scene()
    recorder = Recorder()
    fields = CommandFields("the stack looks nice", "typed", "unknown", "", "", "the stack looks nice", 0.2)
    result = loop.run_command(command_from_fields(fields, "the stack looks nice"), observer=recorder)
    assert scene.targets == [] and policy.resets == 0 and recorder.steps == [] and recorder.plans == []
    assert result["cube"] is None and result["steps"] == 0 and result["refusal"] == "not a pick command but 'unknown'"
    assert loop.record.sequence == [None] and len(loop.record.translations) == 1


def test_a_command_before_a_layout_and_a_bad_mode_are_refused():
    scene, policy = StandInScene(), StandInPolicy()
    with pytest.raises(ValueError, match="picture mode"):
        PickPlaceScene(scene, policy, None, None, pictures="blurred", act=act)
    loop = PickPlaceScene(scene, policy, None, None, pictures="unedited", act=act)
    with pytest.raises(RuntimeError, match="no layout"):
        loop.run_command(command_from_colour("red"))
