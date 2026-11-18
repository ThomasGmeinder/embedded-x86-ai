"""Run the sim-real checkpoint at each listed cube position.

Videos and results go to ``vla_testing/``. Scripted runs stay in ``scripted_testing/``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from so101_smolvla.layouts import draw_layout
from so101_smolvla.record import _write_clip, accumulate
from so101_smolvla.sim import PHYSICS_HZ, ArmSim
from so101_smolvla.track import (
    CHUNK,
    EVAL_STEPS,
    POWER_ON,
    load_checkpoint,
    predict_chunk,
    rollout,
    trajectory_metrics,
)

ROOT = Path(__file__).resolve().parents[2]
VLA_TESTING = ROOT / "vla_testing"
SIM_YAML = ROOT / "configs" / "sim.yaml"
RANDOM_COUNT = 100


def scenarios(path: Path | None = None) -> list[tuple[str, tuple[float, float] | None]]:
    """Scene default, then each cube position in ``configs/sim.yaml``."""
    loaded = yaml.safe_load((path or SIM_YAML).read_text(encoding="utf-8"))
    rows: list[tuple[str, tuple[float, float] | None]] = [("scene", None)]
    for xy in loaded["distinct_cube_positions"]:
        x, y = float(xy[0]), float(xy[1])
        rows.append((f"cube-{x:g}-{y:g}", (x, y)))
    return rows


def seed_policy(seed: int) -> None:
    """Seed the checkpoint noise the same way every time this scenario is repeated."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare(
    sim: ArmSim,
    cube_xy: tuple[float, float] | None,
    yaw_rad: float | None = None,
) -> list[float]:
    """Power-on pose. ``None`` leaves the cube where the scene XML puts it.

    ``yaw_rad`` is the training yaw: a rotation about vertical, with x and y
    locked, the same lock ``place.py`` uses for the cube.
    """
    sim.reset(cube_xy=cube_xy)
    if yaw_rad is not None:
        half = 0.5 * float(yaw_rad)
        sim.data.qpos[sim.cube_qadr + 3 : sim.cube_qadr + 7] = (
            np.cos(half),
            0.0,
            0.0,
            np.sin(half),
        )
    sim.hold_pose(POWER_ON)
    start = sim.cube_pos()
    return [round(float(start[0]), 4), round(float(start[1]), 4)]


class GripWatch:
    """First jaw contact, or the closest approach when the jaws never touch."""

    def __init__(self, empty_force: float = 0.0) -> None:
        self.empty_force = empty_force
        self.grip: dict | None = None
        self.closest: dict | None = None
        self.picked_up = False
        self.held = False
        self.step = 0

    def observe(self, sim: ArmSim) -> None:
        mid = (sim.data.geom_xpos[sim.tip_a] + sim.data.geom_xpos[sim.tip_b]) / 2
        dist = float(np.linalg.norm(mid - sim.cube_pos()))
        sample = {
            "step": self.step,
            "dist": dist,
            "finger_face_deg": round(sim.finger_face_deg(), 2),
            "roll_deg": round(float(sim.read_state()[4]), 2),
            "contact": sim.jaw_contact(),
        }
        if sample["contact"] and self.grip is None:
            self.grip = sample
        if self.closest is None or dist < self.closest["dist"]:
            self.closest = sample
        if sim.above_table():
            self.picked_up = True
        # A squeeze on the table is not a hold. The cube has to be off the table
        # on this same step, still in the jaws.
        if sim.above_table() and sim.held(self.empty_force):
            self.held = True
        self.step += 1

    def chosen(self) -> dict:
        return self.grip if self.grip is not None else self.closest


def random_directory(root: Path, stamp: str) -> Path:
    return root / f"random-{stamp}"


def empty_close_force(sim: ArmSim) -> float:
    """Gripper force with the jaws shut and the cube out of reach. The hold check uses this."""
    sim.reset((0.5, 0.5))
    state = sim.read_state()
    state[5] = 0.0
    sim.command(state)
    sim.step(PHYSICS_HZ)
    return sim.gripper_force()


def run_scenario(
    sim: ArmSim,
    predict,
    name: str,
    cube_xy: tuple[float, float] | None,
    root: Path,
    steps: int = EVAL_STEPS,
    yaw_rad: float | None = None,
    layout_seed: int | None = None,
    policy_seed: int | None = None,
    empty_force: float = 0.0,
) -> dict:
    if policy_seed is not None:
        seed_policy(policy_seed)
    start = prepare(sim, cube_xy, yaw_rad=yaw_rad)
    watch = GripWatch(empty_force)
    out = rollout(
        sim,
        predict,
        steps=steps,
        chunk_size=CHUNK,
        watch=watch,
    )
    grip = watch.chosen()
    folder = root / name
    _write_clip(out["top"], folder / "top.mp4")
    _write_clip(out["wrist"], folder / "wrist.mp4")
    np.savez_compressed(
        folder / "trajectory.npz",
        target=out["target"],
        commanded=out["commanded"],
        achieved=out["achieved"],
        chunk_index=out["chunk_index"],
    )
    np.savez_compressed(
        folder / "policy-observations.npz",
        top=out["policy_top"],
        wrist=out["policy_wrist"],
    )
    end = sim.cube_pos()
    motion = trajectory_metrics(out["target"])
    row = {
        "scenario": name,
        "cube_start_xy": start,
        "cube_end_xyz": [round(float(v), 4) for v in end],
        "contact": bool(out["contact"]),
        "lifted": bool(out["lifted"]),
        "in_box": bool(sim.in_box()),
        "cube_yaw_deg": None if yaw_rad is None else round(float(np.degrees(yaw_rad)), 2),
        "bin_xy": [round(float(v), 4) for v in sim.data.xpos[sim.box_body][:2]],
        "layout_seed": layout_seed,
        "policy_seed": policy_seed,
        "backend": "mujoco",
        "grip": grip["contact"],
        "grip_step": grip["step"],
        "finger_face_deg": grip["finger_face_deg"],
        "grip_roll_deg": grip["roll_deg"],
        "picked_up": watch.picked_up,
        "held": watch.held,
        "trajectory": motion,
    }
    (folder / "result.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row


def run_all(
    sim: ArmSim,
    predict,
    root: Path | None = None,
    steps: int = EVAL_STEPS,
) -> list[dict]:
    dest = root or VLA_TESTING
    rows = [
        run_scenario(
            sim,
            predict,
            name,
            xy,
            dest,
            steps=steps,
        )
        for name, xy in scenarios()
    ]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


def _count(rows: list[dict], key: str) -> tuple[int, int]:
    passed = sum(1 for row in rows if row[key])
    return passed, len(rows) - passed


def write_report(rows: list[dict], dest: Path) -> Path:
    """One page: links to the accumulated videos, then pass/fail counts and each seed."""
    angles = [float(row["finger_face_deg"]) for row in rows]
    mean = float(np.mean(angles)) if angles else 0.0
    lines = [
        f"# Random regression {dest.name}",
        "",
        "Accumulated video: [top](overall-top.mp4), [wrist](overall-wrist.mp4).",
        "",
        "The finger angle is the angle between the fingertip line and the nearest cube face",
        "at the first jaw contact. Zero is square onto the face. If the jaws never touch,",
        "the angle is taken at the closest approach.",
        "",
        "| Check | Pass | Fail |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("in_box", "Cube in the bin at the end"),
        ("picked_up", "Cube picked up"),
        ("held", "Cube held"),
        ("grip", "Jaws touched the cube"),
    ):
        passed, failed = _count(rows, key)
        lines.append(f"| {label} | {passed} | {failed} |")
    lines.extend(
        [
            "",
            f"Finger-to-face angle at grip: mean {mean:.1f}°, "
            f"median {float(np.median(angles)):.1f}°, max {max(angles):.1f}°. Ideal is 0°.",
            "",
            "| Layout seed | Policy seed | In bin | Picked up | Held | Jaw contact | Finger angle ° | Lift reversals | Elbow reversals |",
            "|---:|---:|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lift = row["trajectory"]["shoulder_lift"]["reversal_rate"]
        elbow = row["trajectory"]["elbow_flex"]["reversal_rate"]
        lines.append(
            "| {layout_seed} | {policy_seed} | {box} | {picked} | {held} | {grip} | {angle:.1f} | {lift:.1%} | {elbow:.1%} |".format(
                layout_seed=row["layout_seed"],
                policy_seed=row["policy_seed"],
                box="pass" if row["in_box"] else "fail",
                picked="pass" if row["picked_up"] else "fail",
                held="pass" if row["held"] else "fail",
                grip="pass" if row["grip"] else "fail",
                angle=row["finger_face_deg"],
                lift=lift,
                elbow=elbow,
            )
        )
    path = dest / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_random(
    sim: ArmSim,
    predict,
    count: int = RANDOM_COUNT,
    root: Path | None = None,
    stamp: str | None = None,
    policy_seed_base: int = 10_000,
) -> list[dict]:
    """Layout seeds ``0`` .. ``count - 1`` with independent policy seeds."""
    dest = random_directory(root or VLA_TESTING, stamp or datetime.now().strftime("%Y%m%d-%H%M%S"))
    baseline = empty_close_force(sim)
    rows = []
    for seed in range(count):
        layout = draw_layout(sim, seed)
        sim.place_bin(layout["bin_xy"], layout["bin_yaw"])
        name = f"seed-{seed:05d}"
        print(
            f"{name} bin={layout['bin_xy'][0]:.3f},{layout['bin_xy'][1]:.3f} "
            f"cube={layout['cube_xy'][0]:.3f},{layout['cube_xy'][1]:.3f}",
            flush=True,
        )
        rows.append(
            run_scenario(
                sim,
                predict,
                name,
                layout["cube_xy"],
                dest,
                yaw_rad=layout["cube_yaw"],
                layout_seed=seed,
                policy_seed=policy_seed_base + seed,
                empty_force=baseline,
            )
        )
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    accumulate([dest / row["scenario"] for row in rows], dest, "overall")
    report = write_report(rows, dest)
    print(f"wrote {report}", flush=True)
    return rows


def main() -> None:
    sim = ArmSim()
    try:
        print("loading checkpoint", flush=True)
        policy, pre, post = load_checkpoint()

        def predict(state, top, wrist):
            print(f"chunk {state.tolist()}", flush=True)
            return predict_chunk(policy, pre, post, state, top, wrist)

        rows = run_all(sim, predict)
    finally:
        sim.close()
    print(json.dumps(rows), flush=True)


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("all", "random"), default="all")
    parser.add_argument("count", nargs="?", type=int, default=RANDOM_COUNT)
    parser.add_argument("stamp", nargs="?")
    parser.add_argument("--policy-seed-base", type=int, default=10_000)
    args = parser.parse_args(argv)

    if args.mode == "random":
        count = args.count
        stamp = args.stamp
        sim = ArmSim()
        try:
            print("loading checkpoint", flush=True)
            policy, pre, post = load_checkpoint()

            def predict(state, top, wrist):
                return predict_chunk(policy, pre, post, state, top, wrist)

            rows = run_random(
                sim,
                predict,
                count=count,
                stamp=stamp,
                policy_seed_base=args.policy_seed_base,
            )
        finally:
            sim.close()
        placed = sum(1 for row in rows if row["in_box"])
        print(f"placed {placed} of {len(rows)}", flush=True)
    else:
        main()


if __name__ == "__main__":
    cli()
