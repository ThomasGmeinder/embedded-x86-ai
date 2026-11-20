"""Where do the model's answers of the reach take the gripper, with real and with scene pictures?

Run from the workshop folder, in this order, for one moment (a number of
frames before the human closes the gripper on the red cube):
    early_reach_check.py render <before>                      (.venv-sapien)
    release_aim_check.py predict early-reach-<before>-pictures.npz early-reach-<before>-predictions.npz   (.venv)
    early_reach_check.py report <before>                      (.venv-sapien)
PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim : ~/.cache/so101-stack-training
: <workshop>/vla_testing/diagnostics.

Why: in the stacking test the model is asked at steps 0, 50, 100, 150 and
plays all 50 targets of each answer. In the episodes that fail, the arm is
already past the red cube at step 100 and the answer given there closes the
gripper high above the table (model_path_against_the_human.py). The other
offline checks ask the model only 25 frames before the human closes, from a
reading the human had already steered onto the cube. This one asks during
the reach, for example 105 and 55 frames before the closing.

"render" poses the stack2-recording-rig scene at the recorded reading of the
moment, cubes standing where the recording had them, and takes both pictures
twice: with the scene's cubes as they are, and with the cubes painted so that
the top camera shows them in the colours measured in the recordings
(analysis/objects/fitted-paint.json).
"report" poses the targets of every answer (real or scene top picture with
real or scene wrist picture, and the human's own next 50 commands) on the
robot model and measures, split by how far the red cube is from the robot:
where the gripper tip is at the last target, whether and where the answer
closes the gripper, and the wrist-flex number at the last target. Nothing is
simulated forward.

Writes analysis/early-reach-<before>-pictures.npz (render),
analysis/early-reach-<before>.json and -per-recording.npz (report).

"report" also measures other pictures of the same moment and recordings
(top_picture_parts.py render <before>), named by their file stem in analysis/:
    early_reach_check.py report <before> <stem>
reads <stem>-pictures.npz and <stem>-predictions.npz, writes <stem>.json and
<stem>-per-recording.npz.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from release_aim_check import ANALYSIS, ANSWER_LENGTH, pairs_in, recorded, tip_at

# (label, top picture, wrist picture); "real_top" and "real_wrist" are the recorded pictures.
PAIRS = (
    ("real top + real wrist", "real_top", "real_wrist"),
    ("scene top + real wrist", "top", "real_wrist"),
    ("real top + scene wrist", "real_top", "wrist"),
    ("scene top + scene wrist", "top", "wrist"),
    ("scene top with recorded cube colours + real wrist", "top_recorded_cubes", "real_wrist"),
    ("real top + scene wrist with recorded cube colours", "real_top", "wrist_recorded_cubes"),
    ("scene top + scene wrist, both with recorded cube colours",
     "top_recorded_cubes", "wrist_recorded_cubes"),
)
# The red cube counts as near the robot below this distance, as in the stacking runs.
NEAR_MM = 265.0
# The gripper counts as closing once its target is this far below the widest
# target of the answer so far (per cent of the gripper's range).
CLOSING_PCT = 10.0
WRIST_FLEX = 3


def pictures_file(before: str) -> str:
    return f"early-reach-{int(before)}-pictures.npz"


def cube_materials(inner) -> list[tuple[str, object]]:
    """(cube colour name, material) of every drawn part of the scene's cubes."""
    from sapien.render import RenderBodyComponent

    found = []
    for colour, cube in inner.cubes.items():
        for obj in cube._objs:
            entity = obj.entity if hasattr(obj, "entity") else obj
            body = entity.find_component_by_type(RenderBodyComponent)
            if body is not None:
                found.extend(
                    (colour, part.material) for shape in body.render_shapes for part in shape.parts)
    if not found:
        raise RuntimeError("the scene's cubes have no drawn part")
    return found


def render(before: str) -> None:
    import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the stacking tasks)
    from grasp_aim_check import scenes_before_closing
    from so101_sim.lerobot_env import So101SimEnv
    from so101_smolvla.sapien_config import environment_arguments
    from so101_smolvla.sapien_env_worker import _configure_robot_appearance

    paint = json.loads((ANALYSIS / "objects" / "fitted-paint.json").read_text())
    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=10)
    rows: dict[str, list] = {
        "episode": [], "frame": [], "state": [], "top": [], "wrist": [],
        "top_recorded_cubes": [], "wrist_recorded_cubes": [],
        "red_cube": [], "table_height": [],
    }
    try:
        _configure_robot_appearance(env)
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        materials = cube_materials(inner)

        def pictures() -> dict[str, np.ndarray]:
            seen = env._format_raw_obs(inner.get_obs())["pixels"]
            return {name: seen[name].copy() for name in ("top", "wrist")}

        for number, frame, reading in scenes_before_closing(env, int(before)):
            seen = pictures()
            as_built = [(m.base_color, m.roughness, m.metallic, m.specular) for _, m in materials]
            for colour, material in materials:
                material.set_base_color([*paint["paint_base_colour"][colour], 1.0])
                material.roughness, material.metallic, material.specular = (
                    paint["roughness"], paint["metallic"], paint["specular"])
            painted = pictures()
            for (_, material), (colour, roughness, metallic, specular) in zip(materials, as_built):
                material.set_base_color(colour)
                material.roughness, material.metallic, material.specular = (
                    roughness, metallic, specular)
            rows["episode"].append(number)
            rows["frame"].append(frame)
            rows["state"].append(reading)
            rows["top"].append(seen["top"])
            rows["wrist"].append(seen["wrist"])
            rows["top_recorded_cubes"].append(painted["top"])
            rows["wrist_recorded_cubes"].append(painted["wrist"])
            rows["red_cube"].append(inner.cubes["red"].pose.p[0].cpu().numpy().copy())
            rows["table_height"].append(float(inner.TABLE_SURFACE_Z))
        if not np.array_equal(pictures()["top"], rows["top"][-1]):
            raise RuntimeError("the cubes were not put back to the scene's own colours")
    finally:
        env.close()
    np.savez_compressed(
        ANALYSIS / pictures_file(before), pairs=np.array(PAIRS),
        **{name: np.array(values) for name, values in rows.items()},
    )
    print(f"pictures of {len(rows['episode'])} recordings, {int(before)} frames before "
          f"the human closes the gripper", flush=True)


def closing_index(targets: np.ndarray) -> int | None:
    """First target of an answer that closes the gripper, or None if none does."""
    gripper = targets[:, 5]
    widest = np.maximum.accumulate(gripper)
    closing = np.flatnonzero(gripper < widest - CLOSING_PCT)
    return int(closing[0]) if len(closing) else None


def report(before: str, stem: str | None = None) -> None:
    """Measure the answers in analysis/<stem>-predictions.npz to the pictures in <stem>-pictures.npz."""
    import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the stacking tasks)
    from so101_sim.lerobot_env import So101SimEnv
    from so101_smolvla.sapien_config import environment_arguments

    stem = stem or f"early-reach-{int(before)}"
    stored = np.load(ANALYSIS / f"{stem}-pictures.npz")
    answers = np.load(ANALYSIS / f"{stem}-predictions.npz")
    if answers["episode"].tolist() != stored["episode"].tolist():
        raise RuntimeError("the answers are not of these pictures")
    red = 1000.0 * stored["red_cube"][:, :2]
    table = 1000.0 * stored["table_height"]
    distance = np.linalg.norm(red, axis=1)
    outward = red / distance[:, None]
    leftward = np.column_stack([-outward[:, 1], outward[:, 0]])
    groups = {"all": np.ones(len(red), dtype=bool), "near": distance < NEAR_MM,
              "far": distance >= NEAR_MM}

    human = []
    for number, frame in zip(stored["episode"].tolist(), stored["frame"].tolist()):
        commands = recorded()["action"][recorded()["episode"] == number]
        human.append(commands[frame:frame + ANSWER_LENGTH])
    sets = {"human": np.array(human), **{name: answers[name] for name in pairs_in(stored)}}

    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=10)
    result: dict[str, dict] = {
        "frames_before_closing": int(before),
        "near_means_red_cube_closer_than_mm": NEAR_MM,
        "recordings": {name: int(chosen.sum()) for name, chosen in groups.items()},
    }
    per_recording: dict[str, np.ndarray] = {"episode": stored["episode"], "red_cube_mm": red}
    try:
        env._env.reset(seed=0)
        inner = env._env.unwrapped

        def place(targets: np.ndarray) -> np.ndarray:
            """(past the red cube, left of it, above the table) in mm, one row per recording."""
            tips = 1000.0 * np.array([tip_at(inner, row) for row in targets])
            off = tips[:, :2] - red
            return np.column_stack(
                [(off * outward).sum(axis=1), (off * leftward).sum(axis=1), tips[:, 2] - table])

        for name, chunks in sets.items():
            last = place(chunks[:, -1])
            when = [closing_index(chunk) for chunk in chunks]
            closes = np.array([index is not None for index in when])
            # An answer that never closes is measured at its last target and not counted.
            at_closing = place(np.array([
                chunk[index if index is not None else -1] for chunk, index in zip(chunks, when)]))
            per_recording[f"{name}: last target, past / left / above table mm"] = last
            per_recording[f"{name}: closing target"] = np.array(
                [index if index is not None else -1 for index in when])
            per_recording[f"{name}: at closing, past / left / above table mm"] = at_closing
            # How far out the last target lies, against how far out the red cube
            # lies: 1 means the reach follows the cube, 0 means it ignores it.
            follows, _ = np.polyfit(distance, last[:, 0] + distance, 1)
            result[name] = {"reach_distance_per_cube_distance": round(float(follows), 2)}
            for group, chosen in groups.items():
                row = {
                    "last_target_past_the_red_cube_mm_median": round(float(np.median(last[chosen, 0])), 1),
                    "last_target_left_of_the_red_cube_mm_median": round(float(np.median(last[chosen, 1])), 1),
                    "last_target_above_table_mm_median": round(float(np.median(last[chosen, 2])), 1),
                    "last_target_wrist_flex_deg_median": round(
                        float(np.median(chunks[chosen, -1, WRIST_FLEX])), 1),
                    "answers_that_close_the_gripper": int((closes & chosen).sum()),
                }
                if (closes & chosen).any():
                    row.update({
                        "closing_target_number_median": float(np.median(
                            [index for index, use in zip(when, chosen) if use and index is not None])),
                        "at_closing_past_the_red_cube_mm_median": round(
                            float(np.median(at_closing[closes & chosen, 0])), 1),
                        "at_closing_above_table_mm_median": round(
                            float(np.median(at_closing[closes & chosen, 2])), 1),
                    })
                result[name][group] = row
    finally:
        env.close()
    print(json.dumps(result, indent=1), flush=True)
    (ANALYSIS / f"{stem}.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    np.savez(ANALYSIS / f"{stem}-per-recording.npz", **per_recording)


if __name__ == "__main__":
    {"render": render, "report": report}[sys.argv[1]](*sys.argv[2:])
