"""pick_place: the one pick-and-place loop, in the stand-in scene of the runner's tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from so101_smolvla.pick_place import (
    HOME_ARRIVE_STEPS,
    HOME_DEGREES_PER_STEP,
    HOME_OPEN_JAW,
    HOME_OPEN_STEPS,
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
    with pytest.raises(RuntimeError, match="no layout"):
        loop.warm_up()


class CountingOutlines:
    """Stands in for SAM 3 where only the calls count."""

    def __init__(self) -> None:
        self.layouts = self.outlined = 0

    def start_layout(self) -> None:
        self.layouts += 1

    def outline(self, observation):
        self.outlined += 1
        return observation, {}


class LateScene(StandInScene):
    """A cube that is in the gripper from ``gripped_after`` steps of its command on and is let go
    into the bin after ``placed_after`` steps (None: never). The arm stands where it was sent."""

    def __init__(self, gripped_after: int, placed_after: int | None) -> None:
        super().__init__(places=False)
        self.gripped_after = gripped_after
        self.placed_after = placed_after
        self.actions: list[np.ndarray] = []

    def _facts(self) -> dict[str, object]:
        facts = super()._facts()
        target = facts["target"]
        if target is not None:
            done = self.steps_of_command
            held = done >= self.gripped_after and (self.placed_after is None or done < self.placed_after)
            # A cube is let go when the gripper is opened, by whoever opens it, and lies there.
            held = held and not any(action[5] >= HOME_OPEN_JAW for action in self.actions)
            facts["cubes"][target]["grasped"] = facts["grip"][target]["contact"] = held
        return facts

    def step(self, action):
        self.actions.append(np.array(action, dtype=float))
        if self.placed_after is not None and self.steps_of_command + 1 >= self.placed_after:
            self.in_bin[self.targets[-1]] = True
        outcome = super().step(action)
        return replace(outcome, observation=replace(outcome.observation, state=self.actions[-1].copy()))


def _late(gripped_after: int, placed_after: int | None, give_up_after: int | None = 8, act=act):
    scene = LateScene(gripped_after, placed_after)
    loop = PickPlaceScene(scene, StandInPolicy(), None, None, pictures="others-painted", steps_per_pick=3,
                          give_up_after=give_up_after, act=act, record=True)
    loop.start_layout(4, 10)
    return scene, loop


def test_a_command_whose_steps_end_with_the_cube_in_the_gripper_goes_on_until_the_rule_holds():
    scene, loop = _late(gripped_after=3, placed_after=5)
    recorder = Recorder()
    result = loop.run_command(command_from_colour("blue"), observer=recorder)
    # In the gripper after the third and last step; the rule holds after the fifth: it ends there.
    assert [view.step for view in recorder.steps] == [0, 1, 2, 3, 4]
    assert recorder.steps[2].grasped and recorder.steps[4].success and not recorder.steps[4].going_home
    assert result["steps"] == 5 and result["first_success_step"] == 4 and result["in_bin_at_end"]
    assert not result["gave_up"] and result["home_steps"] == 0
    assert loop.layout_steps == 5 and len(loop.record.targets) == 5


def test_a_command_goes_on_only_with_the_cube_in_the_gripper_and_only_where_it_is_asked_to():
    # Never gripped, never placed: the command is its three steps, and it gave up there. The arm
    # is driven home (it stands at home in this scene: the gripper opens, the pose is held).
    scene, loop = _late(gripped_after=99, placed_after=None)
    recorder = Recorder()
    missed = loop.run_command(command_from_colour("blue"), observer=recorder)
    assert missed["steps"] == 3 and missed["gave_up"] and missed["first_success_step"] is None
    assert missed["home_steps"] == HOME_OPEN_STEPS + HOME_ARRIVE_STEPS == len(scene.actions) - 3
    assert [view.going_home for view in recorder.steps[:4]] == [False, False, False, True]
    # A cube placed within the command's own steps: nothing gives up, SmolVLA brings the arm back.
    scene, loop = _late(gripped_after=1, placed_after=2)
    placed = loop.run_command(command_from_colour("blue"))
    assert placed["steps"] == 3 and not placed["gave_up"] and placed["home_steps"] == 0
    # In the gripper at the end, but nothing says to go on (the batch runner): three steps.
    scene, loop = _late(gripped_after=3, placed_after=5, give_up_after=None)
    result = loop.run_command(command_from_colour("blue"))
    assert result["steps"] == 3 and not result["gave_up"] and result["first_success_step"] is None
    with pytest.raises(ValueError, match="gives up after more steps than its 3"):
        PickPlaceScene(StandInScene(), StandInPolicy(), None, None, pictures="unedited", steps_per_pick=3,
                       give_up_after=3, act=act)


def test_goes_on_step_by_step():
    _, loop = _late(gripped_after=3, placed_after=None)
    assert loop.goes_on(2, in_gripper=False, rule_held=True)  # its own steps, whatever happened
    assert loop.goes_on(3, in_gripper=True, rule_held=False)  # they ended with the cube in the gripper
    assert not loop.goes_on(3, in_gripper=False, rule_held=False)
    assert loop.goes_on(5, in_gripper=False, rule_held=False)  # dropped beside the bin: SmolVLA tries again
    assert not loop.goes_on(5, in_gripper=False, rule_held=True)  # placed
    assert loop.goes_on(7, in_gripper=True, rule_held=False) and not loop.goes_on(8, in_gripper=True, rule_held=False)


def test_a_command_that_gives_up_opens_the_gripper_and_drives_the_arm_home():
    away = np.array([40.0, -10.0, 0.0, 0.0, 0.0, 5.0])  # where the policy holds the arm, jaw nearly shut

    def stay_away(policy, pre, post, observation, task_text):
        return away

    scene, loop = _late(gripped_after=3, placed_after=None, act=stay_away)
    start = loop.record.start_state.copy()
    recorder = Recorder()
    result = loop.run_command(command_from_colour("blue"), observer=recorder)
    # Never placed: all eight steps, then the program: 10 to open, 40 degrees at 2 a step, 10 at the start pose.
    assert result["steps"] == 8 and result["gave_up"]
    assert result["home_steps"] == HOME_OPEN_STEPS + 20 + HOME_ARRIVE_STEPS
    assert result["first_success_step"] is None and not result["in_bin_at_end"]
    home = scene.actions[8:]
    assert len(home) == result["home_steps"] and loop.layout_steps == 8 + len(home) == len(loop.record.targets)
    # The gripper opens where the arm stands and the cube is let go there.
    assert all((target[:5] == away[:5]).all() and target[5] == HOME_OPEN_JAW for target in home[:HOME_OPEN_STEPS])
    views = recorder.steps[8:]
    assert all(view.going_home for view in views) and not any(view.going_home for view in recorder.steps[:8])
    assert recorder.steps[7].grasped and not views[0].grasped
    assert [view.step for view in views] == list(range(8, 8 + len(home)))
    # Then the arm, never more than two degrees a step, the gripper open until the arm is there;
    # the last steps hold the start pose.
    moves = np.abs(np.diff(np.array(home)[:, :5], axis=0))
    assert moves.max() <= HOME_DEGREES_PER_STEP + 1e-9
    assert all(target[5] == HOME_OPEN_JAW for target in home[:-HOME_ARRIVE_STEPS])
    assert all((target == start).all() for target in home[-HOME_ARRIVE_STEPS:])
    assert (loop.observation.state == start).all()
    # The next command starts from the start pose.
    assert loop.run_command(command_from_colour("red"))["arm_from_start_deg"] == 0.0


def test_a_warm_up_runs_each_model_once_and_leaves_the_layout_as_a_fresh_start():
    acts = []

    def counting_act(policy, pre, post, observation, task_text):
        acts.append(task_text)
        return observation.state

    scene, policy, outlines = StandInScene(), StandInPolicy(), CountingOutlines()
    loop = PickPlaceScene(scene, policy, None, None, pictures="others-painted", sam3=outlines, steps_per_pick=3,
                          act=counting_act, record=True)
    first = loop.start_layout(4, 10)
    seconds = loop.warm_up()
    assert set(seconds) == {"sam3", "smolvla"} and all(value >= 0.0 for value in seconds.values())
    # One outline and one plan; the plan's actions are dropped again.
    assert outlines.outlined == 1 and len(acts) == 1 and policy.resets == 1
    # The layout was started again: SAM 3's tracking anew, nothing run in it, the same first pictures.
    assert outlines.layouts == 2 and loop.layout_steps == 0 and loop.record.steps == 0
    assert loop.record.sequence == [] and scene.targets == [] and loop.record.plan_seconds == []
    assert (loop.observation.state == first.state).all() and (loop.observation.top == first.top).all()

    # A command after the warm-up runs as it does without one.
    def targets_of_a_pick(warm: bool):
        scene, policy = StandInScene(), StandInPolicy()
        loop = PickPlaceScene(scene, policy, None, None, pictures="others-painted", steps_per_pick=3, act=act, record=True)
        loop.start_layout(4, 10)
        if warm:
            assert set(loop.warm_up()) == {"smolvla"}  # no SAM 3, no outline
        result = loop.run_command(command_from_colour("green"))
        # A command of three steps with two actions per plan plans at its steps 0 and 2: the
        # seconds of each plan are kept, the warm-up's own plan is not among them.
        assert len(loop.record.plan_seconds) == 2 and all(s >= 0.0 for s in loop.record.plan_seconds)
        return result, [target.tolist() for target in loop.record.targets], scene.targets

    assert targets_of_a_pick(warm=True) == targets_of_a_pick(warm=False)
