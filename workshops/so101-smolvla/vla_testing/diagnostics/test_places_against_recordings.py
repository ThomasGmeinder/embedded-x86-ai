"""Do the random stacking tests fail where the recordings never had a cube?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder, after random_run_grasp_and_release.py has been run on every
run named here:
    test_places_against_recordings.py <result .json in analysis/> <run folder>=<its grasp-and-release .json in analysis/> ...

Nothing is simulated. Where the cubes lay in a recording is read the way the
replay of the recordings places them (release_aim_check.replay_commands): the
red cube where the gripper tip is ten frames after the human closed on it, the
large cube where the tip is when the human lets go, with the recorded joint
readings posed on the robot of the stack2-recording-rig scene, so with its
joint zero points. The grasp and release frames are those of
analysis/cube-placement.npz.

Per test episode: where its two cubes started, how many recordings had the
same cube within NEAR_MM of that place, how far the nearest recording's cube
was, and how the episode went (from the grasp-and-release results and the
run's own result files). Over all episodes: whether the miss at the grasp
goes with how unfamiliar the red cube's place is, and the miss at the release
with how unfamiliar the large cube's place is (rank correlation), and the
outcome counts for familiar and unfamiliar places.

Writes analysis/<result>.json and analysis/<result>.png (the places).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from release_aim_check import recorded, recording_rig_scene, tip_at

ANALYSIS = Path.home() / ".cache" / "so101-stack-training" / "analysis"
# A recording counts as having had a cube "there" within this distance.
NEAR_MM = 30.0
# A place counts as familiar from this many recordings on.
FAMILIAR_RECORDINGS = 5
# The red cube rests on the large cube when its underside is this far above the table.
ON_THE_LARGE_CUBE_MM = 30.0
LIFTED_MM = 10.0


def recorded_places(inner) -> tuple[np.ndarray, np.ndarray]:
    """Red and large cube places of every readable recording, millimetres in the robot base frame.

    Also written, with the recordings' numbers, to analysis/recorded-cube-places.npz.
    """
    events = np.load(ANALYSIS / "cube-placement.npz")
    episode, state = recorded()["episode"], recorded()["state"]
    red, large = [], []
    for number, close, release in zip(
            events["training_episode"], events["training_close_frame"],
            events["training_release_frame"]):
        readings = state[episode == number]
        red.append(tip_at(inner, readings[min(close + 10, release - 1)])[:2])
        large.append(tip_at(inner, readings[release])[:2])
    red_mm, large_mm = 1000.0 * np.array(red), 1000.0 * np.array(large)
    np.savez(ANALYSIS / "recorded-cube-places.npz",
             episode=events["training_episode"], red_mm=red_mm, large_mm=large_mm)
    return red_mm, large_mm


def familiarity(place: np.ndarray, places: np.ndarray) -> tuple[int, float]:
    """How many recorded places lie within NEAR_MM of a place, and how far the nearest is."""
    apart = np.linalg.norm(places - place, axis=1)
    return int((apart <= NEAR_MM).sum()), float(apart.min())


def spread(values: np.ndarray) -> list[float]:
    return [round(float(v), 1) for v in np.percentile(values, [5, 50, 95])]


def episode_row(folder: Path, facts: dict, red_places: np.ndarray, large_places: np.ndarray) -> dict:
    result = json.loads((folder / "result.json").read_text())
    layout = result["initial_layout"]
    red = 1000.0 * np.asarray(layout["red"])
    large = 1000.0 * np.asarray(layout["blue"])
    red_count, red_nearest = familiarity(red, red_places)
    large_count, large_nearest = familiarity(large, large_places)
    # The same pair: recordings that had both cubes near these two places.
    both = int(((np.linalg.norm(red_places - red, axis=1) <= NEAR_MM)
                & (np.linalg.norm(large_places - large, axis=1) <= NEAR_MM)).sum())
    grasp, release, end = facts.get("grasp"), facts.get("release"), facts["end"]
    if release is not None:
        release_miss = release["red_cube_from_large_cube_centre_mm"]
    elif end["red_cube_underside_above_table_mm"] > ON_THE_LARGE_CUBE_MM:
        release_miss = end["red_cube_from_large_cube_centre_mm"]
    else:
        release_miss = None
    return {
        "run": folder.parent.name[-15:],
        "seed": facts["seed"],
        "red_cube_place_mm": [round(float(v), 1) for v in red],
        "large_cube_place_mm": [round(float(v), 1) for v in large],
        "recordings_with_the_red_cube_near": red_count,
        "nearest_recorded_red_cube_mm": round(red_nearest, 1),
        "recordings_with_the_large_cube_near": large_count,
        "nearest_recorded_large_cube_mm": round(large_nearest, 1),
        "recordings_with_both_cubes_near": both,
        "grasp_miss_mm": None if grasp is None else round(float(np.hypot(
            grasp["tip_past_red_cube_mm"], grasp["tip_left_of_red_cube_mm"])), 1),
        "grasp_tip_above_table_mm": None if grasp is None else grasp["tip_above_table_mm"],
        "lifted": facts["red_cube_highest_mm"] > LIFTED_MM,
        "release_miss_mm": release_miss,
        "stacked": bool(result["success"]),
        "failure_stage": result["failure_stage"],
    }


def goes_with(rows: list[dict], familiar: str, miss: str) -> dict:
    """Rank correlation between how many recordings had the cube near and the miss."""
    pairs = [(row[familiar], row[miss]) for row in rows if row[miss] is not None]
    if len(pairs) < 3:
        return {"episodes": len(pairs)}
    known, missed = map(np.array, zip(*pairs))
    relation = spearmanr(known, missed)
    often = known >= FAMILIAR_RECORDINGS
    return {
        "episodes": len(pairs),
        "rank_correlation": round(float(relation.statistic), 2),
        "chance_of_this_without_a_relation": round(float(relation.pvalue), 3),
        f"median_miss_mm_with_{FAMILIAR_RECORDINGS}_or_more_recordings_near": (
            round(float(np.median(missed[often])), 1) if often.any() else None),
        "episodes_there": int(often.sum()),
        "median_miss_mm_with_fewer": (
            round(float(np.median(missed[~often])), 1) if (~often).any() else None),
        "episodes_with_fewer": int((~often).sum()),
    }


def counts(rows: list[dict], chosen) -> dict:
    picked = [row for row in rows if chosen(row)]
    return {
        "episodes": len(picked),
        "lifted": sum(row["lifted"] for row in picked),
        "stacked": sum(row["stacked"] for row in picked),
    }


def draw(rows: list[dict], red_places: np.ndarray, large_places: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for axis, places, key, title in (
            (axes[0], red_places, "red_cube_place_mm", "Red cube"),
            (axes[1], large_places, "large_cube_place_mm", "Large cube")):
        axis.scatter(places[:, 1], places[:, 0], s=12, color="0.6", label="recordings")
        for row in rows:
            good = row["lifted"] if key.startswith("red") else row["stacked"]
            axis.scatter(row[key][1], row[key][0], s=70, marker="o" if good else "x",
                         color="tab:green" if good else "tab:red")
            axis.annotate(str(row["seed"]), (row[key][1], row[key][0]),
                          textcoords="offset points", xytext=(5, 5), fontsize=8)
        axis.set_title(f"{title}: tests (green = "
                       f"{'lifted' if key.startswith('red') else 'stacked'}, red cross = not)")
        axis.set_xlabel("to the robot's left, mm")
        axis.invert_xaxis()
        axis.set_aspect("equal")
        axis.legend(loc="lower right")
    axes[0].set_ylabel("in front of the robot base, mm")
    figure.tight_layout()
    figure.savefig(path, dpi=110)


def main(output: str, *runs: str) -> None:
    env = recording_rig_scene()
    try:
        red_places, large_places = recorded_places(env._env.unwrapped)
    finally:
        env.close()

    rows = []
    for run in runs:
        folder, facts_file = run.split("=")
        for facts in json.loads((ANALYSIS / facts_file).read_text())["episodes"]:
            rows.append(episode_row(
                Path(folder) / f"seed-{facts['seed']:05d}", facts, red_places, large_places))
    for row in rows:
        print(json.dumps(row), flush=True)

    result = {
        "recordings": len(red_places),
        "near_mm": NEAR_MM,
        "recorded red cube, in front of the base (5 %, median, 95 %) mm": spread(red_places[:, 0]),
        "recorded red cube, to the left mm": spread(red_places[:, 1]),
        "recorded large cube, in front of the base mm": spread(large_places[:, 0]),
        "recorded large cube, to the left mm": spread(large_places[:, 1]),
        "grasp miss against recordings with the red cube near": goes_with(
            rows, "recordings_with_the_red_cube_near", "grasp_miss_mm"),
        "release miss against recordings with the large cube near": goes_with(
            rows, "recordings_with_the_large_cube_near", "release_miss_mm"),
        "red cube familiar": counts(
            rows, lambda row: row["recordings_with_the_red_cube_near"] >= FAMILIAR_RECORDINGS),
        "red cube not familiar": counts(
            rows, lambda row: row["recordings_with_the_red_cube_near"] < FAMILIAR_RECORDINGS),
        "large cube familiar": counts(
            rows, lambda row: row["recordings_with_the_large_cube_near"] >= FAMILIAR_RECORDINGS),
        "large cube not familiar": counts(
            rows, lambda row: row["recordings_with_the_large_cube_near"] < FAMILIAR_RECORDINGS),
        "both cubes familiar": counts(
            rows, lambda row: row["recordings_with_the_red_cube_near"] >= FAMILIAR_RECORDINGS
            and row["recordings_with_the_large_cube_near"] >= FAMILIAR_RECORDINGS),
        "episodes": rows,
    }
    for name, value in result.items():
        if name != "episodes":
            print(name, json.dumps(value), flush=True)
    (ANALYSIS / output).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    draw(rows, red_places, large_places, (ANALYSIS / output).with_suffix(".png"))


if __name__ == "__main__":
    main(*sys.argv[1:])
