"""The one pick-and-place loop: a command names a cube, SmolVLA moves it into the bin.

Used by the batch runner (``three_cube_rollout``), its tests and the demo
notebook alike, so there is one place where a command is turned into a pick:
the cube the command names (``cube_commands.choose_cube``), the outlines the
picture editor paints by (the scene's own or SAM 3's at every plan), the
pictures SmolVLA is given, one action per step for ``steps_per_pick`` steps,
and the record of what happened. A refused command moves nothing and is
recorded as such. With ``give_up_after`` (the demo) a command whose steps end
while the cube is in the gripper goes on until the simulator's rule holds
(``PickPlaceScene.goes_on``). A command that ends without its cube placed,
after its own steps with the gripper empty or at ``give_up_after`` steps, has
given up: a program, not SmolVLA, opens the gripper and drives the arm back
to the pose the layout started in (``PickPlaceScene._go_home``). Without it a command is
its ``steps_per_pick`` steps and no more (the batch runner and its
measurements).

``PickPlaceScene`` owns one scene (``env``), one policy and, with SAM 3, one
outline worker. ``start_layout`` resets the scene once; ``run_command`` runs one
command in it and never resets. An ``Observer`` hears of every plan and every
step; the batch runner prints progress from it, the demo draws from it. With
``record`` the scene keeps what a layout's episode folder needs (every step's
pictures and targets, every plan's tiles); the demo turns that off.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import torch

from so101_smolvla.bin_layouts import BIN_CUBE_COLORS
from so101_smolvla.cube_commands import call_text, choose_cube
from so101_smolvla.feeds_video import Outline, PlanFeeds
from so101_smolvla.grasp_geometry import GripMoment, GripSample
from so101_smolvla.picture_editor import edit_pictures
from so101_smolvla.policy_env import PolicyObservation
from so101_smolvla.sam3_outlines import Sam3Outlines
from so101_smolvla.sapien_config import OUTLINE_CODES
from so101_smolvla.track import JOINT_NAMES
from so101_smolvla.voice.commands import CommandFields

# Steps for one command: the episode length the simulator registers for its
# bin scenes (so101_sim/envs.py:902). A command runs all of them, so the arm
# has the time to come back before the next command starts.
DEFAULT_STEPS_PER_PICK = 400
# The way home of a command that gave up, driven by a program: the gripper is
# opened to this share of its travel (SmolVLA itself opens to 47 % before a
# pick, measured 2026-11-23) over these steps, then the joints go to the pose
# the layout started in, at most this many degrees per step, and that pose is
# held for the last steps while the gripper closes and the arm arrives (with
# one such step the arm stood 3.6 degrees short and the gripper at 36 %,
# measured 2026-11-24).
HOME_OPEN_JAW = 50.0
HOME_OPEN_STEPS = 10
HOME_DEGREES_PER_STEP = 2.0
HOME_ARRIVE_STEPS = 10
# What SmolVLA is shown.
PICTURE_MODES = {
    "unedited": "the pictures as the cameras take them",
    "others-painted": (
        "the two cubes the command does not name are painted over, "
        "on the table and in the bin"
    ),
    "table-others-painted": (
        "the cubes the command does not name are painted over while they "
        "stand on the table; a cube in the bin is left as it is"
    ),
}
# Where the outlines come from that the picture editor paints by, and what
# the video of a layout writes above them. The simulator knows every pixel of
# its scene at every step. SAM 3 (so101_smolvla.sam3_outlines, the reference
# sam3-tracker-rocm) is asked for the pictures SmolVLA plans on and outlines
# cubes only.
OUTLINE_SOURCES = {
    "simulator": "simulator outlines",
    "sam3": "SAM 3 outlines",
}


def pick_place_command(color: str) -> str:
    """The command of a cube given as a colour, as Gemma writes a pick of it."""
    return call_text("pick_place", f"{color} cube", "box")


def cubes_to_paint(
    pictures: str, target: str, facts: dict[str, object]
) -> tuple[str, ...]:
    """The cubes painted over while ``target`` is moved, as the scene stands."""
    if pictures not in PICTURE_MODES:
        raise ValueError(
            f"unknown picture mode {pictures!r}; expected one of {list(PICTURE_MODES)}"
        )
    others = tuple(color for color in BIN_CUBE_COLORS if color != target)
    if pictures == "unedited":
        return ()
    if pictures == "others-painted":
        return others
    return tuple(color for color in others if not facts["cubes"][color]["in_bin"])


def cube_outlines(
    outline_map: np.ndarray | None,
    target: str,
    painted: tuple[str, ...],
    unused: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[Outline, ...]]:
    """The outlines in one picture, numbered, and what is done with each.

    ``outline_map`` is the map of the picture the editor paints by
    (sapien_config.OUTLINE_CODES) and ``painted`` what ``cubes_to_paint``
    returned for the picture. ``unused`` marks the pixels of outlines the
    editor was not given because they are on no cube; where there are any,
    they follow the three cubes as a fourth entry.
    """
    if outline_map is None:
        raise ValueError("the scene hands out no outline maps")
    numbered = np.zeros(outline_map.shape, dtype=np.uint8)
    legend = []
    for number, color in enumerate(BIN_CUBE_COLORS, start=1):
        numbered[outline_map == OUTLINE_CODES[color]] = number
        if color == target:
            role = "target"
        elif color in painted:
            role = "painted over"
        else:
            role = "left as it is"
        legend.append(Outline(label=f"{color} cube", role=role))
    if unused is not None:
        if unused.shape != outline_map.shape or unused.dtype != bool:
            raise ValueError(
                f"unused outlines are {unused.dtype} of shape {unused.shape}, "
                f"the outline map is of shape {outline_map.shape}"
            )
        if unused.any():
            legend.append(Outline(label="no cube colour", role="not used"))
            numbered[unused & (numbered == 0)] = len(legend)
    return numbered, tuple(legend)


@dataclass(frozen=True)
class PickCommand:
    """One command as the scene runs it: its text, the cube it names or none, and where it came from."""

    command: str  # as call_text writes it
    cube: str | None  # None: refused, nothing moves
    # For a sentence through Gemma: sentence, expected_cube, gemma (the
    # fields' record), cube, refusal, as_expected. None for a colour.
    translation: dict | None = None

    @property
    def refusal(self) -> str | None:
        return None if self.translation is None else self.translation["refusal"]


def command_from_colour(color: str) -> PickCommand:
    """A cube given as a colour, written as Gemma writes a pick of it."""
    if color not in BIN_CUBE_COLORS:
        raise ValueError(f"{color!r} is not one of {', '.join(BIN_CUBE_COLORS)}")
    return PickCommand(command=pick_place_command(color), cube=color)


def command_from_fields(
    fields: CommandFields, sentence: str, expected_cube: str | None = None
) -> PickCommand:
    """What Gemma made of a sentence, read for the cube by cube_commands.choose_cube."""
    choice = choose_cube(fields.command, fields.object)
    return PickCommand(
        command=call_text(fields.command, fields.object, fields.place),
        cube=choice.cube,
        translation={
            "sentence": sentence,
            "expected_cube": expected_cube,
            "gemma": fields.as_record(),
            "cube": choice.cube,
            "refusal": choice.refusal,
            "as_expected": None if expected_cube is None else choice.cube == expected_cube,
        },
    )


@dataclass(frozen=True)
class PlanView:
    """What an observer is told at the start of one plan of SmolVLA."""

    step: int  # the plan's first step, counted over the layout
    command: str
    cube: str
    feeds: dict[str, PlanFeeds]  # per camera: taken, numbered outlines, legend, shown
    sam3: dict | None  # with SAM 3: per camera its seconds and outlines


@dataclass(frozen=True)
class StepView:
    """What an observer is told after one step of a command."""

    step: int  # within the command, from 0; ``steps_per_pick`` and above in a command that goes on
    steps_per_pick: int
    command: str
    cube: str
    observation: PolicyObservation  # after the step
    in_bin: bool
    grasped: bool
    contact: bool
    success: bool
    going_home: bool = False  # the command gave up; a program drives the arm home


class Observer(Protocol):
    def on_plan(self, view: PlanView) -> None: ...

    def on_step(self, view: StepView) -> None: ...


@dataclass
class LayoutRecord:
    """Everything a layout's episode folder is written from (``record=True``)."""

    start_state: np.ndarray | None = None
    initial_layout: dict = field(default_factory=dict)
    targets: list[np.ndarray] = field(default_factory=list)
    commanded: list[np.ndarray] = field(default_factory=list)
    achieved: list[np.ndarray] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)
    top_frames: list[np.ndarray] = field(default_factory=list)
    wrist_frames: list[np.ndarray] = field(default_factory=list)
    # Per camera, what it took, what was outlined and what SmolVLA was given
    # at the start of every chunk of actions.
    plans: dict[str, list[PlanFeeds]] = field(
        default_factory=lambda: {"top": [], "wrist": []}
    )
    cube_positions: dict[str, list] = field(
        default_factory=lambda: {color: [] for color in BIN_CUBE_COLORS}
    )
    bin_positions: list = field(default_factory=list)
    # With SAM 3: per plan and camera, what it outlined and how long it took.
    sam3_found: list[dict] = field(default_factory=list)
    # Per plan: the seconds SmolVLA took to compute its actions.
    plan_seconds: list[float] = field(default_factory=list)
    picks: list[dict] = field(default_factory=list)
    sequence: list[str | None] = field(default_factory=list)
    translations: list[dict] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return len(self.targets)


def _refused_record(pick: PickCommand) -> dict:
    translation = pick.translation or {}
    return {
        "command": pick.command,
        "cube": None,
        "refusal": translation.get("refusal"),
        "sentence": translation.get("sentence"),
        "expected_cube": translation.get("expected_cube"),
        "gemma": translation.get("gemma"),
        "success": False,
        "first_success_step": None,
        "in_bin_at_end": False,
        "picked_up": False,
        "held": False,
        "steps": 0,
    }


class PickPlaceScene:
    """One scene, one policy, the outlines' source; a layout of commands run one at a time."""

    def __init__(
        self,
        env,
        policy,
        pre,
        post,
        *,
        pictures: str,
        sam3: Sam3Outlines | None = None,
        steps_per_pick: int = DEFAULT_STEPS_PER_PICK,
        give_up_after: int | None = None,
        act: Callable[..., np.ndarray],
        record: bool = True,
    ) -> None:
        if pictures not in PICTURE_MODES:
            raise ValueError(
                f"unknown picture mode {pictures!r}; expected one of {list(PICTURE_MODES)}"
            )
        if steps_per_pick < 1:
            raise ValueError(f"steps per command must be positive, got {steps_per_pick}")
        if give_up_after is not None and give_up_after <= steps_per_pick:
            raise ValueError(
                f"a command gives up after more steps than its {steps_per_pick}, got {give_up_after}"
            )
        self.give_up_after = give_up_after  # None: a command never goes on; see goes_on
        self.env = env
        self.policy = policy
        self.pre = pre
        self.post = post
        self.pictures = pictures
        self.sam3 = sam3
        self.steps_per_pick = steps_per_pick
        self.act = act  # (policy, pre, post, observation, task_text) -> the joint targets
        self.recording = record
        self.outline_source = OUTLINE_SOURCES["simulator" if sam3 is None else "sam3"]
        self.record = LayoutRecord()
        self.observation: PolicyObservation | None = None
        self.table_height = 0.0
        self.layout_steps = 0  # steps run in this layout, recorded or not
        self._layout_seeds: tuple[int, int] | None = None  # of the layout in force

    def start_layout(self, environment_seed: int, policy_seed: int) -> PolicyObservation:
        """Reset the scene once for a layout of commands; the first observation."""
        torch.manual_seed(policy_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(policy_seed)
        observation = self.env.reset(environment_seed)
        if self.sam3 is not None:
            # SAM 3 follows the cubes through the plans of one layout.
            self.sam3.start_layout()
        self.record = LayoutRecord(
            start_state=observation.state.copy(),
            initial_layout=dict(self.env.facts["initial_layout"]),
        )
        self.table_height = float(self.env.facts["table_height"])
        self.observation = observation
        self.layout_steps = 0
        self._layout_seeds = (environment_seed, policy_seed)
        return observation

    def warm_up(self) -> dict[str, float]:
        """Run SAM 3 and SmolVLA once on the layout's first pictures; the seconds each took.

        The first run of a model after it is loaded is slow (measured 2026-11-23 in the demo:
        SAM 3's first outline of both cameras 6.6 s against 0.5 s, SmolVLA's first plan 1.3 s
        against 0.3 s), and the first command of a session otherwise waits for both. The plan
        draws from the policy's random numbers and the outline moves SAM 3's tracking on, so
        the layout is started again with its own seeds afterwards: the scene, the random
        numbers and the tracking are then as ``start_layout`` leaves them, and the first
        command runs as it does without a warm-up.
        """
        if self.observation is None:
            raise RuntimeError("no layout was started")
        seconds: dict[str, float] = {}
        if self.sam3 is not None:
            began = time.perf_counter()
            self.sam3.outline(self.observation)
            seconds["sam3"] = time.perf_counter() - began
        began = time.perf_counter()
        self.act(self.policy, self.pre, self.post, self.observation, self.env.task_text)
        self.policy.reset()
        seconds["smolvla"] = time.perf_counter() - began
        self.start_layout(*self._layout_seeds)
        return seconds

    def goes_on(self, step: int, in_gripper: bool, rule_held: bool) -> bool:
        """Whether a command runs its step ``step``; the two flags are of the step before.

        A command runs its ``steps_per_pick`` steps. With ``give_up_after`` it goes on past
        them when they end with the cube in the gripper: a pick that came late is finished by
        SmolVLA instead of being left standing with the cube in the air (2026-11-23, the demo:
        a cube gripped at step 340 of 400 was over the bin at step 383 and still held at 399).
        Once it goes on, it ends at the first step at which the simulator's rule holds (the
        cube in the bin, let go, the robot still and off the bin) and not at the end of the
        plan: measured in that session, SmolVLA let the cube go at step 406, gripped it again
        and had it out of the bin by step 444. It ends at the latest at ``give_up_after``
        steps; ``run_command`` then sends the arm home.
        """
        if step < self.steps_per_pick:
            return True
        if self.give_up_after is None or step >= self.give_up_after:
            return False
        if step == self.steps_per_pick:
            return in_gripper
        return not rule_held

    def _keep(self, target: np.ndarray, outcome) -> None:
        """One step of the layout: counted and, with ``record``, kept for the episode folder."""
        self.layout_steps += 1
        if not self.recording:
            return
        record, facts = self.record, outcome.facts
        record.targets.append(target)
        record.commanded.append(outcome.commanded)
        record.achieved.append(outcome.achieved)
        record.successes.append(outcome.success)
        record.top_frames.append(outcome.observation.top)
        record.wrist_frames.append(outcome.observation.wrist)
        for other in BIN_CUBE_COLORS:
            record.cube_positions[other].append(facts["cubes"][other]["position"])
        record.bin_positions.append(facts["bin"]["position"])

    def _go_home(self, pick: PickCommand, step: int, observer: Observer | None) -> int:
        """The way home of a command that gave up, driven by this program and not by SmolVLA.

        The gripper is opened where the arm stands, so a cube in it falls there; then the
        joints go in a straight line to the pose the layout started in. The steps it took;
        ``self.observation`` is the last one's.
        """
        start = self.record.start_state
        stand = self.observation.state
        opened = stand.copy()
        opened[5] = max(float(stand[5]), HOME_OPEN_JAW)
        way = int(np.ceil(np.max(np.abs(start[:5] - stand[:5])) / HOME_DEGREES_PER_STEP))
        targets = [opened] * HOME_OPEN_STEPS
        # The arm first, the gripper still open; the gripper closes to its start in the last steps.
        for share in np.linspace(0.0, 1.0, way + 1)[1:]:
            target = opened + share * (start - opened)
            target[5] = opened[5]
            targets.append(target)
        targets.extend([start.copy()] * HOME_ARRIVE_STEPS)
        color = pick.cube
        for count, target in enumerate(targets):
            outcome = self.env.step(target)
            self._keep(target, outcome)
            self.observation = outcome.observation
            if observer is not None:
                cube = outcome.facts["cubes"][color]
                observer.on_step(
                    StepView(
                        step=step + count,
                        steps_per_pick=self.steps_per_pick,
                        command=pick.command,
                        cube=color,
                        observation=outcome.observation,
                        in_bin=bool(cube["in_bin"]),
                        grasped=bool(cube["grasped"]),
                        contact=bool(outcome.facts["grip"][color]["contact"]),
                        success=bool(outcome.success),
                        going_home=True,
                    )
                )
        return len(targets)

    def run_command(self, pick: PickCommand, observer: Observer | None = None) -> dict:
        """One command in the scene as it stands: the pick's record (``steps`` 0 when refused)."""
        if self.observation is None:
            raise RuntimeError("no layout was started")
        record = self.record
        record.sequence.append(pick.cube)
        if pick.translation is not None:
            record.translations.append(pick.translation)
        if pick.cube is None:
            refused = _refused_record(pick)
            record.picks.append(refused)
            return refused
        color = pick.cube
        env, policy = self.env, self.policy
        action_steps = int(policy.config.n_action_steps)
        facts = env.set_target(color)
        # Table and bin as the top camera sees them now, without cubes and arm.
        scene_without_cubes, _ = env.scene_without_cubes()
        policy.reset()
        stood = {
            other: np.asarray(facts["cubes"][other]["position"]) for other in BIN_CUBE_COLORS
        }
        observation = self.observation
        arm_from_start = float(
            np.max(np.abs(observation.state[:5] - record.start_state[:5]))
        )
        first_success = None
        picked_up = False
        held = False
        grip_moment = GripMoment()
        painted_top, painted_wrist, wrist_kept = [], [], []
        step = 0
        in_gripper = rule_held = False
        while self.goes_on(step, in_gripper, rule_held):
            facts = env.facts
            painted = cubes_to_paint(self.pictures, color, facts)
            # SmolVLA plans when its queue of actions is empty: at the first
            # step of a command and every ``action_steps`` after it.
            planning = step % action_steps == 0
            unused: dict[str, np.ndarray | None] = {"top": None, "wrist": None}
            found_record = None
            if self.sam3 is None:
                outlined = observation
            elif planning:
                outlined, found = self.sam3.outline(observation)
                unused = {camera: found[camera].unused for camera in found}
                found_record = {
                    camera: {
                        "seconds": found[camera].seconds,
                        "outlines": [
                            {
                                "number": outline.number,
                                "pixels": outline.pixels,
                                "cube": outline.cube,
                            }
                            for outline in found[camera].found
                        ],
                    }
                    for camera in found
                }
                record.sam3_found.append(found_record)
            else:
                outlined = None
            if outlined is None:
                # Between two plans SmolVLA takes the next action of the plan
                # it has. It computes from pictures only when it plans, and
                # its queues hold actions only (lerobot's SmolVLAPolicy.reset
                # and select_action), so the pictures of these steps are not
                # used: SAM 3 is not asked and they go on as they are.
                shown = observation
            else:
                shown, count = edit_pictures(
                    outlined,
                    painted,
                    scene_without_cubes,
                    np.asarray(facts["cameras"]["top"]),
                    np.asarray(facts["cameras"]["wrist"]),
                    self.table_height,
                )
                painted_top.append(count.top)
                painted_wrist.append(count.wrist)
                wrist_kept.append(count.wrist_kept)
            if planning:
                feeds = {}
                for camera, taken, outline_map, given in (
                    ("top", observation.top, outlined.top_outline, shown.top),
                    ("wrist", observation.wrist, outlined.wrist_outline, shown.wrist),
                ):
                    numbered, legend = cube_outlines(outline_map, color, painted, unused[camera])
                    feeds[camera] = PlanFeeds(
                        step=self.layout_steps,
                        command=pick.command,
                        taken=taken.copy(),
                        outlines=numbered,
                        legend=legend,
                        shown=given.copy(),
                    )
                if self.recording:
                    for camera in feeds:
                        record.plans[camera].append(feeds[camera])
                if observer is not None:
                    observer.on_plan(
                        PlanView(
                            step=self.layout_steps,
                            command=pick.command,
                            cube=color,
                            feeds=feeds,
                            sam3=found_record,
                        )
                    )
            began = time.perf_counter()
            target = self.act(policy, self.pre, self.post, shown, env.task_text)
            if planning:
                # The step at which SmolVLA computes its next actions from the pictures.
                record.plan_seconds.append(time.perf_counter() - began)
            outcome = env.step(target)
            observation = outcome.observation
            facts = outcome.facts
            self._keep(target, outcome)
            cube = facts["cubes"][color]
            picked_up |= bool(cube["above_table"])
            held |= bool(cube["grasped"])
            grip = facts["grip"][color]
            grip_moment.observe(
                GripSample(
                    step=step,
                    distance=float(grip["distance"]),
                    finger_face_deg=float(grip["finger_face_deg"]),
                    roll_deg=float(outcome.achieved[JOINT_NAMES.index("wrist_roll")]),
                    contact=bool(grip["contact"]),
                )
            )
            if outcome.success and first_success is None:
                first_success = step
            if observer is not None:
                observer.on_step(
                    StepView(
                        step=step,
                        steps_per_pick=self.steps_per_pick,
                        command=pick.command,
                        cube=color,
                        observation=observation,
                        in_bin=bool(cube["in_bin"]),
                        grasped=bool(cube["grasped"]),
                        contact=bool(grip["contact"]),
                        success=bool(outcome.success),
                    )
                )
            in_gripper = bool(cube["grasped"])
            rule_held = bool(outcome.success)
            step += 1
        self.observation = observation
        # Where a command may give up (``give_up_after``), it does when it ends without its cube
        # placed: the rule never held (a pick that missed for all its steps and ended with the
        # gripper empty), or it went on to its last allowed step without the rule holding there.
        # The arm then does not stay where the miss left it (2026-11-23: 144 degrees from its
        # start pose after a missed green cube, where the next command began).
        gave_up = self.give_up_after is not None and (
            first_success is None or (step == self.give_up_after and not rule_held)
        )
        home_steps = self._go_home(pick, step, observer) if gave_up else 0
        ended = env.facts["cubes"]
        translation = pick.translation
        result = {
            "command": pick.command,
            "cube": color,
            **(
                {}
                if translation is None
                else {
                    key: translation[key]
                    for key in ("refusal", "sentence", "expected_cube", "gemma")
                }
            ),
            # The simulator's rule held at some step: the cube inside the
            # bin's footprint, let go, the robot still and off the bin.
            "success": first_success is not None,
            "first_success_step": first_success,
            "in_bin_at_end": bool(ended[color]["in_bin"]),
            # SmolVLA's steps: ``steps_per_pick``, or more where the command went on. Where it
            # gave up, the steps of the program that opened the gripper and drove the arm home.
            "steps": step,
            "gave_up": gave_up,
            "home_steps": home_steps,
            "picked_up": picked_up,
            "held": held,
            **grip_moment.chosen().as_result(),
            "arm_from_start_deg": arm_from_start,
            "others_moved_mm": {
                other: 1000.0
                * float(np.linalg.norm(np.asarray(ended[other]["position"]) - stood[other]))
                for other in BIN_CUBE_COLORS
                if other != color
            },
            # Means over the pictures that were edited: every step with
            # the simulator's outlines, the plans with SAM 3's.
            "painted_pixels": {
                "top_mean": float(np.mean(painted_top)),
                "wrist_mean": float(np.mean(painted_wrist)),
                "wrist_kept_mean": float(np.mean(wrist_kept)),
            },
        }
        record.picks.append(result)
        return result
