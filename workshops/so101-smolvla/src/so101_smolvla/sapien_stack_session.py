"""Persistent command session for color-directed SAPIEN stacking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from so101_smolvla.grasp_geometry import GripMoment, GripSample
from so101_smolvla.policy_env import PolicyObservation
from so101_smolvla.stack_recording_rig import ARM_JOINTS


COLORS = frozenset({"red", "green", "blue"})
COMMAND_PATTERN = re.compile(
    r"^\s*(?:put|stack)\s+([a-z]+)(?:\s+cube)?\s+on\s+"
    r"([a-z]+)(?:\s+cube)?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StackCommand:
    original: str
    canonical: str
    source: str
    target: str


@dataclass
class CommandTrace:
    command: StackCommand
    success: bool
    success_step: int | None
    failure_stage: str | None
    selected: bool
    lifted: bool
    released: bool
    # The gripper against the cube it was told to move, at the grip.
    grip: GripSample
    target: np.ndarray
    commanded: np.ndarray
    achieved: np.ndarray
    cube_positions: dict[str, np.ndarray]
    top: np.ndarray
    wrist: np.ndarray
    facts: dict[str, object]


def parse_stack_command(text: str) -> StackCommand:
    match = COMMAND_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(
            "expected 'put|stack <red|green|blue> [cube] on "
            "<red|green|blue> [cube]'"
        )
    source, target = (value.lower() for value in match.groups())
    unknown = sorted({source, target} - COLORS)
    if unknown:
        raise ValueError(f"unknown cube color: {', '.join(unknown)}")
    if source == target:
        raise ValueError("source and target cube must differ")
    return StackCommand(
        original=text,
        canonical=f"put {source} cube on {target} cube",
        source=source,
        target=target,
    )


def target_can_support(
    command: StackCommand, facts: dict[str, object]
) -> bool:
    table_supported = facts["table_supported"]
    available = set(table_supported)
    if command.source not in available or command.target not in available:
        return False
    if bool(table_supported[command.target]):
        return True
    interfaces = facts["interfaces"]
    for support in available - {command.target, command.source}:
        if (
            bool(interfaces[f"{command.target}_on_{support}"]["supported"])
            and bool(table_supported[support])
        ):
            return True
    return False


class StackCommandSession:
    """Execute commands without resetting the environment between stages."""

    def __init__(
        self,
        *,
        environment,
        policy,
        observation: PolicyObservation,
        action: Callable[[PolicyObservation, str], np.ndarray],
        max_steps_per_command: int,
    ) -> None:
        if max_steps_per_command < 1:
            raise ValueError("max_steps_per_command must be positive")
        self.environment = environment
        self.policy = policy
        self.observation = observation
        self.action = action
        self.max_steps_per_command = max_steps_per_command
        self.completed_interfaces: set[str] = set()

    def execute(self, text: str) -> CommandTrace:
        command = parse_stack_command(text)
        available = set(self.environment.facts["cube_positions"])
        absent = sorted({command.source, command.target} - available)
        if absent:
            raise ValueError(
                f"cube is not present in this task: {', '.join(absent)}"
            )
        if not target_can_support(command, self.environment.facts):
            raise ValueError(
                f"{command.target} cube is not independently supported"
            )

        self.environment.set_command(command.source, command.target)
        self.policy.reset()
        initial_positions = {
            color: np.asarray(position, dtype=np.float64).copy()
            for color, position in self.environment.facts["cube_positions"].items()
        }
        initial_source_z = float(initial_positions[command.source][2])
        source_selected = False
        source_lifted = False
        wrong_cube_lifted = False
        destroyed_completed_stack = False
        targets: list[np.ndarray] = []
        commanded: list[np.ndarray] = []
        achieved: list[np.ndarray] = []
        positions: dict[str, list[list[float]]] = {
            color: [] for color in available
        }
        top: list[np.ndarray] = []
        wrist: list[np.ndarray] = []
        success_step = None
        grip_moment = GripMoment()

        for step in range(self.max_steps_per_command):
            target = self.action(self.observation, command.canonical)
            outcome = self.environment.step(target)
            self.observation = outcome.observation
            targets.append(target)
            commanded.append(outcome.commanded)
            achieved.append(outcome.achieved)
            top.append(self.observation.top)
            wrist.append(self.observation.wrist)
            facts = outcome.facts
            source_selected |= bool(facts["grasped"][command.source])
            grip = facts["grip"][command.source]
            grip_moment.observe(
                GripSample(
                    step=step,
                    distance=float(grip["distance"]),
                    finger_face_deg=float(grip["finger_face_deg"]),
                    roll_deg=float(outcome.achieved[ARM_JOINTS.index("wrist_roll")]),
                    contact=bool(grip["contact"]),
                )
            )
            for color in available:
                positions[color].append(facts["cube_positions"][color])
            source_z = float(facts["cube_positions"][command.source][2])
            source_lifted |= source_z > initial_source_z + 0.01
            for color in available - {command.source}:
                initial_z = float(initial_positions[color][2])
                current_z = float(facts["cube_positions"][color][2])
                wrong_cube_lifted |= current_z > initial_z + 0.01
            destroyed_completed_stack |= any(
                not bool(facts["interfaces"][name]["supported"])
                for name in self.completed_interfaces
            )
            if outcome.success:
                success_step = step
                break

        interface = f"{command.source}_on_{command.target}"
        success = success_step is not None
        if success:
            self.completed_interfaces.add(interface)
            failure_stage = None
        elif destroyed_completed_stack:
            failure_stage = "first stack destroyed during stage two"
        elif (
            not source_selected
            and (
                source_lifted
                or np.linalg.norm(
                    np.asarray(
                        self.environment.facts["cube_positions"][command.source]
                    )
                    - np.asarray(initial_positions[command.source])
                )
                > 0.02
            )
        ):
            failure_stage = "cube displaced without grasp"
        elif wrong_cube_lifted and not source_lifted:
            failure_stage = "wrong cube selected"
        elif not source_lifted:
            failure_stage = "no lift"
        elif float(self.environment.facts["cube_positions"][command.source][2]) <= (
            initial_source_z + 0.01
        ):
            failure_stage = "dropped cube"
        else:
            failure_stage = "final stack unstable"

        return CommandTrace(
            command=command,
            success=success,
            success_step=success_step,
            failure_stage=failure_stage,
            selected=source_selected,
            lifted=source_lifted,
            released=not bool(self.environment.facts["grasped"][command.source]),
            grip=grip_moment.chosen(),
            target=np.stack(targets),
            commanded=np.stack(commanded),
            achieved=np.stack(achieved),
            cube_positions={
                color: np.asarray(values, dtype=np.float32)
                for color, values in positions.items()
            },
            top=np.stack(top),
            wrist=np.stack(wrist),
            facts=dict(self.environment.facts),
        )
