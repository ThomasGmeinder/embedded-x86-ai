"""How much room do the open fingers need around the red cube, and did the recordings give it?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder:
    hand_room_at_the_grasp.py [<run folder of stack2-recording-rig>]

No model, nothing simulated forward. For each real stacking recording whose
grasp and release were read (analysis/cube-placement.npz), the robot of the
stack2-recording-rig scene is posed at the recorded reading of the frame at
which the human starts to close the gripper. The collision shapes of the two
fingers are then measured against the place of the red cube, in millimetres:

  along the pinch line   the horizontal line through the two fingertips
  across the pinch line  horizontal, at a right angle to it; positive is
                         away from the robot base

Only the part of the fingers below a height counts: below the top of the
large cube (40 mm above the table) a finger can run into the large cube.
The red cube's place is the gripper tip 10 frames after the closing starts,
the large cube's place the gripper tip at the release, as in the earlier
measurements of the cube places.

Reported over the recordings:
- how wide the gripper is open when it starts to close, and how far apart
  the fingertips then are;
- how far the fingers reach from the red cube's centre, to the side of each
  finger along the pinch line, across it, and in any direction;
- where the large cube lies against that, and in how many recordings it
  stands inside the fingers' reach.

With a run folder: the same reach for each test of that run, at the step at
which its gripper starts to close, against the test's own cube places.
Writes analysis/hand-room-at-the-grasp.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from random_run_grasp_and_release import CLOSED_PCT, OPEN_PCT, first
from release_aim_check import recorded, recording_rig_scene, tip_at

ANALYSIS = Path.home() / ".cache" / "so101-stack-training" / "analysis"
# The two fingers: the hand with its fixed jaw, and the moving jaw.
FINGER_LINKS = ("gripper_link", "moving_jaw_so101_v1_link")
LARGE_CUBE_MM = 40.0
RED_CUBE_MM = 20.0
HEIGHTS_MM = {"below the large cube's top": LARGE_CUBE_MM, "below the red cube's top": RED_CUBE_MM}
PERCENTILES = (0, 5, 25, 50, 75, 95, 100)
REACH_SIDES = (
    "to the fixed finger's side",
    "to the moving finger's side",
    "towards the robot",
    "away from the robot",
    "in any direction",
)


def spread(values) -> dict[str, float]:
    """How many numbers there are, and their smallest, 5 %, 25 %, median, 75 %, 95 % and largest.

    A grasp at which the hand does not reach a height has no number there
    and is left out of the count.
    """
    numbers = np.asarray(list(values), dtype=np.float64)
    numbers = numbers[np.isfinite(numbers)]
    if not len(numbers):
        return {"count": 0}
    return {
        "count": int(len(numbers)),
        **{
            name: round(float(value), 1)
            for name, value in zip(
                ("min", "p5", "p25", "median", "p75", "p95", "max"),
                np.percentile(numbers, PERCENTILES),
            )
        },
    }


def part_below(corners: np.ndarray, height: float) -> np.ndarray:
    """Corner points of the part of one convex piece that lies below a height."""
    kept = [corners[corners[:, 2] <= height]]
    for triangle in ConvexHull(corners, qhull_options="QJ").simplices:
        for start, end in ((0, 1), (1, 2), (2, 0)):
            low, high = corners[triangle[start]], corners[triangle[end]]
            if (low[2] - height) * (high[2] - height) < 0.0:
                share = (height - low[2]) / (high[2] - low[2])
                kept.append((low + share * (high - low))[None])
    return np.concatenate(kept)


def finger_pieces(inner) -> dict[str, list[np.ndarray]]:
    """Corner points of every convex collision piece of each finger, in the robot base frame, mm."""
    pieces: dict[str, list[np.ndarray]] = {}
    for name in FINGER_LINKS:
        link = inner.agent.robot.links_map[name]._objs[0]
        to_base = link.entity.pose.to_transformation_matrix()
        pieces[name] = []
        for shape in link.get_collision_shapes():
            in_link = shape.local_pose.to_transformation_matrix()
            corners = np.asarray(shape.vertices, dtype=np.float64) * np.asarray(shape.scale)
            moved = (to_base @ in_link @ np.c_[corners, np.ones(len(corners))].T).T[:, :3]
            pieces[name].append(1000.0 * moved)
    return pieces


def fingertips(inner) -> tuple[np.ndarray, np.ndarray]:
    """The two fingertips of the robot as it stands, in the robot base frame, mm."""
    agent = inner.agent
    return tuple(
        1000.0 * finger.pose.p[0].cpu().numpy().astype(np.float64)
        for finger in (agent.finger1_tip, agent.finger2_tip)
    )


def reach_of_the_fingers(inner, red: np.ndarray, table: float) -> dict[str, dict[str, float]]:
    """How far the fingers reach from the red cube's centre, for the robot as it stands."""
    tip_fixed, tip_moving = fingertips(inner)
    along = (tip_moving - tip_fixed)[:2]
    along /= np.linalg.norm(along)
    across = np.array([-along[1], along[0]])
    if across @ red[:2] < 0.0:
        across = -across
    pieces = finger_pieces(inner)
    reach: dict[str, dict[str, float]] = {}
    for label, height in HEIGHTS_MM.items():
        low_parts = [
            part_below(piece, table + height)
            for link_pieces in pieces.values() for piece in link_pieces
            if piece[:, 2].min() < table + height
        ]
        if not low_parts:
            # The hand is still above this height: nothing of it reaches there.
            reach[label] = dict.fromkeys(REACH_SIDES, float("nan"))
            continue
        offset = np.concatenate(low_parts)[:, :2] - red[:2]
        reach[label] = dict(zip(REACH_SIDES, (
            float(-(offset @ along).min()),
            float((offset @ along).max()),
            float(-(offset @ across).min()),
            float((offset @ across).max()),
            float(np.linalg.norm(offset, axis=1).max()),
        )))
    reach["pinch line"] = {"along": along.tolist(), "across": across.tolist()}
    return reach


def large_cube_against_the_fingers(
    reach: dict, red: np.ndarray, large: np.ndarray
) -> dict[str, float]:
    """Where the large cube's centre lies against the fingers' reach below its top."""
    along, across = (np.asarray(reach["pinch line"][name]) for name in ("along", "across"))
    offset = large[:2] - red[:2]
    position = np.array([offset @ along, offset @ across])
    box = reach["below the large cube's top"]
    low = np.array([-box["to the fixed finger's side"], -box["towards the robot"]])
    high = np.array([box["to the moving finger's side"], box["away from the robot"]])
    outside = np.maximum(np.maximum(low - position, position - high), 0.0)
    return {
        "centres apart": float(np.linalg.norm(offset)),
        "along the pinch line": float(abs(position[0])),
        "across the pinch line": float(position[1]),
        # Zero: the cube's centre itself stands inside the fingers' reach.
        "centre outside the fingers' reach": float(np.linalg.norm(outside)),
    }


def summary(rows: list[dict]) -> dict:
    """Spread of every measured number over recordings or tests."""
    result: dict = {"count": len(rows)}
    for name in ("gripper reading when it starts to close, %", "fingertips apart",
                 "gripper tip above the table"):
        result[name] = spread(row[name] for row in rows)
    for label in HEIGHTS_MM:
        result[f"fingers reach from the red cube's centre, {label}"] = {
            side: spread(row["reach"][label][side] for row in rows)
            for side in rows[0]["reach"][label]
        }
    for name in rows[0]["large cube"]:
        result[f"large cube, {name}"] = spread(row["large cube"][name] for row in rows)
    gap = np.array([row["large cube"]["centre outside the fingers' reach"] for row in rows])
    half, corner = LARGE_CUBE_MM / 2.0, LARGE_CUBE_MM / np.sqrt(2.0)
    result["large cube stands inside the fingers' reach"] = {
        "if a face of it looks at the fingers (20 mm from its centre)": int((gap < half).sum()),
        "if a corner of it looks at the fingers (28.3 mm from its centre)": int((gap < corner).sum()),
    }
    return result


def measured(inner, reading: np.ndarray, red: np.ndarray, large: np.ndarray, table: float) -> dict:
    """Everything measured for one grasp: the robot is posed at the reading."""
    tip = 1000.0 * tip_at(inner, reading)
    tip_fixed, tip_moving = fingertips(inner)
    reach = reach_of_the_fingers(inner, red, table)
    return {
        "gripper reading when it starts to close, %": float(reading[5]),
        "fingertips apart": float(np.linalg.norm(tip_moving - tip_fixed)),
        "gripper tip above the table": float(tip[2] - table),
        "reach": reach,
        "large cube": large_cube_against_the_fingers(reach, red, large),
    }


def recordings_measured(inner, table: float) -> list[dict]:
    places = np.load(ANALYSIS / "cube-placement.npz")
    rows = []
    for number, close, release in zip(
        places["training_episode"].tolist(),
        places["training_close_frame"].tolist(),
        places["training_release_frame"].tolist(),
    ):
        state = recorded()["state"][recorded()["episode"] == number].astype(np.float64)
        red = 1000.0 * tip_at(inner, state[min(close + 10, release - 1)])
        large = 1000.0 * tip_at(inner, state[release])
        rows.append({"recording": int(number), **measured(inner, state[close], red, large, table)})
    return rows


def tests_measured(inner, run: Path, table: float) -> list[dict]:
    rows = []
    for folder in sorted(run.glob("seed-*")):
        steps = np.load(folder / "trajectory.npz")
        reading = steps["achieved"].astype(np.float64)
        gripper = reading[:, 5]
        opened = first(gripper > OPEN_PCT)
        closed = first(gripper < CLOSED_PCT, opened) if opened is not None else None
        if closed is None:
            print(f"{folder.name}: the gripper never closes after having been open", flush=True)
            continue
        # The step at which the closing starts: the last step before "closed"
        # at which the gripper was as wide as it had been since it opened.
        widest = float(gripper[opened:closed].max())
        start = opened + int(np.flatnonzero(gripper[opened:closed] >= widest - 1.0)[-1])
        red = 1000.0 * steps["cube_position_red"][start].astype(np.float64)
        large = 1000.0 * steps["cube_position_blue"][start].astype(np.float64)
        rows.append({
            "seed": int(folder.name.split("-")[1]), "step": start,
            **measured(inner, reading[start], red, large, table),
        })
    return rows


def main(run: str | None = None) -> None:
    env = recording_rig_scene()
    try:
        inner = env._env.unwrapped
        table = 1000.0 * float(inner.TABLE_SURFACE_Z)
        result = {"recordings": summary(recordings_measured(inner, table))}
        if run is not None:
            tests = tests_measured(inner, Path(run), table)
            result["tests"] = {
                "run": run,
                **summary(tests),
                "per test": [
                    {
                        "seed": row["seed"], "step": row["step"],
                        "gripper %": round(row["gripper reading when it starts to close, %"], 1),
                        **{name: round(value, 1) for name, value in row["large cube"].items()},
                    }
                    for row in tests
                ],
            }
    finally:
        env.close()
    print(json.dumps(result, indent=1), flush=True)
    (ANALYSIS / "hand-room-at-the-grasp.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
