"""Three cubes into one bin, one cube per command, in an order the user gives.

Every command names one cube: ``pick_place(object: green cube, place: box)``.
The commands come either as colours (the user's order, or a draw from the
layout's seed) or as sentences a user types or says, which Gemma 4 turns into
the fields of the voice client (so101_smolvla.voice) and
cube_commands.choose_cube reads for the cube; a sentence that names no one
cube is refused and moves nothing.
SmolVLA is given its trained sentence and, depending on the picture mode, the
pictures as they are or with the cubes the command does not name painted over.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from so101_smolvla.bin_layouts import BIN_CUBE_COLORS
from so101_smolvla.cube_commands import call_text, choose_cube
from so101_smolvla.feeds_video import Outline, PlanFeeds, feed_frames
from so101_smolvla.grasp_geometry import GripMoment, GripSample
from so101_smolvla.picture_editor import PAINT_MARGIN_PX, edit_pictures
from so101_smolvla.policy_env import create_policy_environment
from so101_smolvla.record import accumulate
from so101_smolvla.sam3_outlines import Sam3Outlines
from so101_smolvla.sapien_config import OUTLINE_CODES, THREE_CUBE_TASK
from so101_smolvla.sapien_rollout import (
    _action,
    _atomic_episode,
    _load_policy,
    _package_versions,
)
from so101_smolvla.track import CHECKPOINT, JOINT_NAMES, trajectory_metrics
from so101_smolvla.voice.commands import VoiceCommands

# Steps for one command: the episode length the simulator registers for its
# bin scenes (so101_sim/envs.py:902). A command runs all of them, so the arm
# has the time to come back before the next command starts.
DEFAULT_STEPS_PER_PICK = 400
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
# Where a layout's commands come from (``order_given_by`` of a result).
COMMANDS_FROM = {
    "draw": "seeded draw",
    "order": "user",
    "typed": "typed sentences through Gemma 4",
    "spoken": "sentences in a synthetic voice through Gemma 4",
}


def pick_place_command(color: str) -> str:
    """The command of a cube given as a colour, as Gemma writes a pick of it."""
    return call_text("pick_place", f"{color} cube", "box")


def draw_order(seed: int) -> tuple[str, ...]:
    """The order of an automated run, which stands in for the user's."""
    return tuple(
        str(color) for color in np.random.default_rng(seed).permutation(BIN_CUBE_COLORS)
    )


def parse_order(text: str) -> tuple[str, ...]:
    """The user's order, written as the three colours with commas between them."""
    order = tuple(part.strip().lower() for part in text.split(","))
    if sorted(order) != sorted(BIN_CUBE_COLORS):
        raise ValueError(
            f"order {text!r} must name each of {', '.join(BIN_CUBE_COLORS)} once"
        )
    return order


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


def run_three_cube_episode(
    env,
    policy,
    pre,
    post,
    checkpoint: Path,
    destination: Path,
    *,
    environment_seed: int,
    policy_seed: int,
    steps_per_pick: int,
    pictures: str,
    order: tuple[str, ...] | None = None,
    sam3: Sam3Outlines | None = None,
    sentences: list[dict] | None = None,
    voice: VoiceCommands | None = None,
    spoken: bool = False,
) -> dict:
    """One layout: a command per cube in one persistent scene.

    With ``sam3`` the picture editor paints by SAM 3's outlines of the
    pictures SmolVLA plans on; without it, by the scene's own outline maps.
    With ``sentences`` (``{"say": sentence, "cube": the colour it should
    name, or None}`` each) every command is one sentence, given to Gemma
    through ``voice`` typed or, with ``spoken``, in the synthetic voice, right
    before the pick; the cube is the one its fields name, and a sentence
    whose fields name no one cube is refused and moves nothing.
    """
    if sentences is not None and order is not None:
        raise ValueError("a layout takes its commands as colours or as sentences, not both")
    if sentences is not None and voice is None:
        raise ValueError("sentences need the voice that asks Gemma")
    torch.manual_seed(policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy_seed)
    observation = env.reset(environment_seed)
    outline_source = OUTLINE_SOURCES["simulator" if sam3 is None else "sam3"]
    if sam3 is not None:
        # SAM 3 follows the cubes through the plans of one layout.
        sam3.start_layout()
    # With SAM 3: per plan and camera, what it outlined and how long it took.
    sam3_found: list[dict] = []
    start_state = observation.state.copy()
    initial_layout = dict(env.facts["initial_layout"])
    table_height = float(env.facts["table_height"])
    if sentences is not None:
        commands_from = COMMANDS_FROM["spoken" if spoken else "typed"]
        requests = [dict(entry) for entry in sentences]
    else:
        commands_from = COMMANDS_FROM["draw" if order is None else "order"]
        requests = [
            {"say": None, "cube": color}
            for color in (draw_order(environment_seed) if order is None else order)
        ]
    sequence: list[str | None] = []
    # Per sentence, what Gemma made of it and what was chosen.
    translations: list[dict] = []
    action_steps = int(policy.config.n_action_steps)

    targets, commanded, achieved, successes = [], [], [], []
    top_frames, wrist_frames = [], []
    # Per camera, what it took, what was outlined and what SmolVLA was given
    # at the start of every chunk of actions.
    plans: dict[str, list[PlanFeeds]] = {"top": [], "wrist": []}
    cube_positions = {color: [] for color in BIN_CUBE_COLORS}
    bin_positions = []
    picks = []
    for pick_index, request in enumerate(requests):
        if request["say"] is None:
            color = request["cube"]
            command_text = pick_place_command(color)
            translation = None
        else:
            # The user speaks (or types) now, and the cube is read from what
            # Gemma made of it.
            fields = voice.ask(request["say"], spoken=spoken)
            choice = choose_cube(fields.command, fields.object)
            command_text = call_text(fields.command, fields.object, fields.place)
            translation = {
                "sentence": request["say"],
                "expected_cube": request["cube"],
                "gemma": fields.as_record(),
                "cube": choice.cube,
                "refusal": choice.refusal,
                "as_expected": (
                    None if request["cube"] is None else choice.cube == request["cube"]
                ),
            }
            translations.append(translation)
            color = choice.cube
            print(
                f"{destination.name} command {pick_index + 1}/{len(requests)}: "
                f"{request['say']!r} -> {command_text} "
                f"({fields.seconds:.2f} s) -> "
                f"{'refused: ' + choice.refusal if color is None else color + ' cube'}",
                flush=True,
            )
        sequence.append(color)
        if color is None:
            picks.append(
                {
                    "command": command_text,
                    "cube": None,
                    "refusal": translation["refusal"],
                    "sentence": translation["sentence"],
                    "expected_cube": translation["expected_cube"],
                    "gemma": translation["gemma"],
                    "success": False,
                    "first_success_step": None,
                    "in_bin_at_end": False,
                    "picked_up": False,
                    "held": False,
                    "steps": 0,
                }
            )
            continue
        facts = env.set_target(color)
        # Table and bin as the top camera sees them now, without cubes and arm.
        scene_without_cubes, _ = env.scene_without_cubes()
        policy.reset()
        stood = {
            other: np.asarray(facts["cubes"][other]["position"])
            for other in BIN_CUBE_COLORS
        }
        arm_from_start = float(
            np.max(np.abs(observation.state[:5] - start_state[:5]))
        )
        first_success = None
        picked_up = False
        held = False
        grip_moment = GripMoment()
        painted_top, painted_wrist, wrist_kept = [], [], []
        for step in range(steps_per_pick):
            facts = env.facts
            painted = cubes_to_paint(pictures, color, facts)
            # SmolVLA plans when its queue of actions is empty: at the first
            # step of a command and every ``action_steps`` after it.
            planning = step % action_steps == 0
            unused: dict[str, np.ndarray | None] = {"top": None, "wrist": None}
            if sam3 is None:
                outlined = observation
            elif planning:
                outlined, found = sam3.outline(observation)
                unused = {camera: found[camera].unused for camera in found}
                sam3_found.append(
                    {
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
                )
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
                    table_height,
                )
                painted_top.append(count.top)
                painted_wrist.append(count.wrist)
                wrist_kept.append(count.wrist_kept)
            if planning:
                for camera, taken, outline_map, given in (
                    ("top", observation.top, outlined.top_outline, shown.top),
                    ("wrist", observation.wrist, outlined.wrist_outline, shown.wrist),
                ):
                    numbered, legend = cube_outlines(
                        outline_map, color, painted, unused[camera]
                    )
                    plans[camera].append(
                        PlanFeeds(
                            step=len(targets),
                            command=command_text,
                            taken=taken.copy(),
                            outlines=numbered,
                            legend=legend,
                            shown=given.copy(),
                        )
                    )
            target = _action(policy, pre, post, shown, env.task_text)
            outcome = env.step(target)
            observation = outcome.observation
            facts = outcome.facts
            targets.append(target)
            commanded.append(outcome.commanded)
            achieved.append(outcome.achieved)
            successes.append(outcome.success)
            top_frames.append(observation.top)
            wrist_frames.append(observation.wrist)
            for other in BIN_CUBE_COLORS:
                cube_positions[other].append(facts["cubes"][other]["position"])
            bin_positions.append(facts["bin"]["position"])
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
            if (step + 1) % 50 == 0:
                print(
                    f"{destination.name} command {pick_index + 1}/{len(requests)} "
                    f"({color}) {step + 1}/{steps_per_pick}; "
                    f"in bin={cube['in_bin']} success={outcome.success}",
                    flush=True,
                )
        ended = env.facts["cubes"]
        picks.append(
            {
                "command": command_text,
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
                "picked_up": picked_up,
                "held": held,
                **grip_moment.chosen().as_result(),
                "arm_from_start_deg": arm_from_start,
                "others_moved_mm": {
                    other: 1000.0
                    * float(
                        np.linalg.norm(
                            np.asarray(ended[other]["position"]) - stood[other]
                        )
                    )
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
        )
        print(
            f"{destination.name} command {pick_index + 1}/{len(requests)} "
            f"({color}) in bin at end={picks[-1]['in_bin_at_end']} "
            f"success step={first_success}",
            flush=True,
        )

    final = env.facts
    in_bin = {color: bool(final["cubes"][color]["in_bin"]) for color in BIN_CUBE_COLORS}
    if not targets:
        raise RuntimeError(
            f"{destination.name}: every command was refused, nothing to record"
        )
    target_rows = np.stack(targets)
    achieved_rows = np.stack(achieved)
    result = {
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": checkpoint.name,
        "backend": env.backend,
        "provenance": env.provenance,
        "package_versions": _package_versions(),
        "task_mode": env.task_mode,
        "task": env.task_text,
        "pictures": pictures,
        "paint_margin_px": PAINT_MARGIN_PX,
        "environment_seed": environment_seed,
        "policy_seed": policy_seed,
        "steps_per_pick": steps_per_pick,
        "order": sequence,
        "order_given_by": commands_from,
        **(
            {}
            if sentences is None
            else {"sentences": translations, "voice": voice.provenance}
        ),
        "initial_layout": initial_layout,
        "bin": final["bin"],
        "outline_source": outline_source,
        **({} if sam3 is None else {"sam3": sam3.provenance}),
        # The start of every chunk of actions: its first step, counted over
        # the layout, and what was outlined in each camera's picture. With
        # SAM 3 also everything it outlined there and the seconds it took.
        "plans": [
            {
                "step": top_plan.step,
                "command": top_plan.command,
                "outlines": {
                    camera: [
                        {"label": outline.label, "role": outline.role}
                        for outline in plan.legend
                    ]
                    for camera, plan in (("top", top_plan), ("wrist", wrist_plan))
                },
                **({} if sam3 is None else {"sam3": sam3_found[index]}),
            }
            for index, (top_plan, wrist_plan) in enumerate(
                zip(plans["top"], plans["wrist"], strict=True)
            )
        ],
        "picks": picks,
        "cubes_in_bin": in_bin,
        "cube_heights_mm": {
            color: 1000.0 * float(final["cubes"][color]["position"][2])
            for color in BIN_CUBE_COLORS
        },
        "cubes_in_bin_count": sum(in_bin.values()),
        "success": all(in_bin.values()),
        "raw_model_targets": trajectory_metrics(target_rows),
        "achieved_joints": trajectory_metrics(achieved_rows),
    }
    arrays = {
        "target": target_rows,
        "commanded": np.stack(commanded),
        "achieved": achieved_rows,
        "success": np.asarray(successes, dtype=bool),
        "bin_position": np.asarray(bin_positions, dtype=np.float32),
        "steps_per_pick": np.asarray(steps_per_pick, dtype=np.int64),
    }
    for color in BIN_CUBE_COLORS:
        arrays[f"cube_position_{color}"] = np.asarray(
            cube_positions[color], dtype=np.float32
        )
    _atomic_episode(
        destination,
        result=result,
        arrays=arrays,
        # The start of every chunk of actions: what SmolVLA was shown (``top``,
        # ``wrist``), what the cameras took, the numbered outlines (the
        # numbers are the places, from one, in ``plans`` of result.json) and
        # the chunk's first step.
        policy_observations={
            "step": np.asarray([plan.step for plan in plans["top"]], dtype=np.int64),
            **{
                name: np.stack([getattr(plan, field) for plan in plans[camera]])
                for camera in ("top", "wrist")
                for name, field in (
                    (camera, "shown"),
                    (f"{camera}_taken", "taken"),
                    (f"{camera}_outlines", "outlines"),
                )
            },
        },
        top=np.stack(top_frames),
        wrist=np.stack(wrist_frames),
        feeds=feed_frames(
            {"top": top_frames, "wrist": wrist_frames}, plans, outline_source
        ),
    )
    return result


def write_three_cube_report(rows: list[dict], destination: Path) -> Path:
    layouts = len(rows)
    commands = [pick for row in rows for pick in row["picks"]]
    positions = max(len(row["picks"]) for row in rows)
    translations = [entry for row in rows for entry in row.get("sentences", [])]
    lines = [
        f"# Three-cube bin test {destination.name}",
        "",
        f"Pictures given to SmolVLA: {PICTURE_MODES[rows[0]['pictures']]}.",
        f"Outlines the picture editor paints by: {rows[0]['outline_source']}.",
        f"Sentence given to SmolVLA: \"{rows[0]['task']}\".",
        f"Steps per command: {rows[0]['steps_per_pick']}.",
        f"Commands from: {rows[0]['order_given_by']}.",
    ]
    if translations:
        known = [entry for entry in translations if entry["expected_cube"] is not None]
        lines.append(
            f"Sentences through Gemma 4: {len(translations)}; "
            f"{sum(bool(entry['as_expected']) for entry in known)} of {len(known)} "
            f"with a known cube gave that cube; "
            f"{sum(entry['cube'] is None for entry in translations)} refused "
            f"(nothing moved)."
        )
    lines += [
        "",
        f"All three cubes in the bin at the end: "
        f"{sum(bool(row['success']) for row in rows)} of {layouts} layouts.",
        f"Cubes in the bin at the end: "
        f"{sum(row['cubes_in_bin_count'] for row in rows)} of "
        f"{sum(len(row['cubes_in_bin']) for row in rows)}.",
        f"Commands whose cube was in the bin when the command ended: "
        f"{sum(bool(pick['in_bin_at_end']) for pick in commands)} of {len(commands)}.",
    ]
    for index in range(positions):
        placed = sum(bool(row["picks"][index]["in_bin_at_end"]) for row in rows)
        lines.append(f"Command {index + 1} of a layout: {placed} of {layouts}.")
    for color in BIN_CUBE_COLORS:
        named = [pick for pick in commands if pick["cube"] == color]
        placed = sum(bool(pick["in_bin_at_end"]) for pick in named)
        lines.append(f"Commands for the {color} cube: {placed} of {len(named)}.")
    lines.extend(
        [
            "",
            "Accumulated video: [all feeds](overall-feeds.mp4), "
            "[top](overall-top.mp4), [wrist](overall-wrist.mp4). "
            "The feeds video has a row per camera with three pictures: the scene "
            "as the camera takes it at every step; the outlines the picture "
            f"editor was given ({rows[0]['outline_source']}) when SmolVLA last "
            "planned, with what is done with each; and the picture SmolVLA was "
            "given then. Those pictures and outlines are in each layout's "
            "`policy-observations.npz`.",
            "",
            "A command's entry reads: cube, `in` or `out` of the bin when the "
            "command ended, and the step at which the simulator's rule for a "
            "placed cube first held (`-` if never).",
            "",
            "| Layout seed | Policy seed | Order from | Command 1 | Command 2 | Command 3 | In the bin at the end | All three |",
            "|---:|---:|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        entries = [
            "refused"
            if pick["cube"] is None
            else "{cube} {where} {step}".format(
                cube=pick["cube"],
                where="in" if pick["in_bin_at_end"] else "out",
                step="-" if pick["first_success_step"] is None else pick["first_success_step"],
            )
            for pick in row["picks"]
        ]
        entries += ["-"] * (3 - len(entries))
        in_bin = ", ".join(
            color for color, inside in row["cubes_in_bin"].items() if inside
        )
        lines.append(
            f"| {row['environment_seed']} | {row['policy_seed']} | "
            f"{row['order_given_by']} | {entries[0]} | {entries[1]} | {entries[2]} | "
            f"{in_bin or 'none'} | {'pass' if row['success'] else 'fail'} |"
        )
    if translations:
        lines += [
            "",
            "Every sentence, what Gemma made of it and the cube that was chosen "
            "(`refused: ...` where none was). Seconds are one request to Gemma.",
            "",
            "| Layout seed | Sentence | How | Gemma heard | Call | Cube | Should be | Seconds |",
            "|---:|---|---|---|---|---|---|---:|",
        ]
        for row in rows:
            for entry in row.get("sentences", []):
                gemma = entry["gemma"]
                lines.append(
                    f"| {row['environment_seed']} | {entry['sentence']} | {gemma['how']} | "
                    f"{gemma['heard']} | `{call_text(gemma['command'], gemma['object'], gemma['place'])}` | "
                    f"{entry['cube'] if entry['cube'] else 'refused: ' + entry['refusal']} | "
                    f"{entry['expected_cube'] or '-'} | {gemma['seconds']:.2f} |"
                )
    path = destination / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_three_cube_batch(
    count: int,
    *,
    destination: Path,
    pictures: str,
    steps_per_pick: int = DEFAULT_STEPS_PER_PICK,
    environment_seed_base: int = 0,
    policy_seed_base: int = 0,
    order: tuple[str, ...] | None = None,
    outlines: str = "simulator",
    sentences: list[list[dict]] | None = None,
    spoken: bool = False,
) -> list[dict]:
    """``sentences``: per layout the sentences of its commands (cube_commands);
    then Gemma runs on a server of its own for the batch, with the server's
    log in the batch's folder."""
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if steps_per_pick < 1:
        raise ValueError(f"steps per command must be positive, got {steps_per_pick}")
    if pictures not in PICTURE_MODES:
        raise ValueError(
            f"unknown picture mode {pictures!r}; expected one of {list(PICTURE_MODES)}"
        )
    if outlines not in OUTLINE_SOURCES:
        raise ValueError(
            f"unknown outline source {outlines!r}; expected one of {list(OUTLINE_SOURCES)}"
        )
    if sentences is not None and order is not None:
        raise ValueError("a batch takes its commands as colours or as sentences, not both")
    if sentences is not None and len(sentences) != count:
        raise ValueError(
            f"{len(sentences)} layouts of sentences for a batch of {count} layouts"
        )
    if spoken and sentences is None:
        raise ValueError("spoken commands need sentences")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite batch {destination}")
    destination.mkdir(parents=True)
    voice = None
    sam3 = None
    rows = []
    try:
        if sentences is not None:
            voice = VoiceCommands(log_dir=destination)
        checkpoint, policy, pre, post = _load_policy()
        env = create_policy_environment(
            "sapien",
            horizon=steps_per_pick * len(BIN_CUBE_COLORS),
            task_mode=THREE_CUBE_TASK,
        )
        try:
            if outlines == "sam3":
                sam3 = Sam3Outlines()
            for index in range(count):
                seed = environment_seed_base + index
                rows.append(
                    run_three_cube_episode(
                        env,
                        policy,
                        pre,
                        post,
                        checkpoint,
                        destination / f"seed-{seed:05d}",
                        environment_seed=seed,
                        policy_seed=policy_seed_base + index,
                        steps_per_pick=steps_per_pick,
                        pictures=pictures,
                        order=order,
                        sam3=sam3,
                        sentences=None if sentences is None else sentences[index],
                        voice=voice,
                        spoken=spoken,
                    )
                )
        finally:
            try:
                if sam3 is not None:
                    sam3.close()
            finally:
                env.close()
    finally:
        if voice is not None:
            voice.close()
    (destination / "summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    accumulate(
        [
            destination / f"seed-{environment_seed_base + index:05d}"
            for index in range(count)
        ],
        destination,
        "overall",
        clips=("top", "wrist", "feeds"),
    )
    write_three_cube_report(rows, destination)
    (destination / "COMPLETE.json").write_text(
        json.dumps(
            {
                "complete": True,
                "episodes": count,
                "task_mode": THREE_CUBE_TASK,
                "pictures": pictures,
                "outlines": outlines,
                "commands": rows[0]["order_given_by"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows
