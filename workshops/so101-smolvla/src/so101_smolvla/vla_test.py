"""Run the sim-real checkpoint at each listed cube position.

Videos and results go to ``vla_testing/``. Scripted runs stay in ``scripted_testing/``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from so101_smolvla.record import _write_clip, accumulate
from so101_smolvla.sim import PHYSICS_HZ, ArmSim
from so101_smolvla.track import (
    CHUNK,
    EVAL_STEPS,
    POWER_ON,
    load_checkpoint,
    predict_chunk,
    rollout,
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


def _camera_table_quad(sim: ArmSim) -> np.ndarray:
    """Four corners, xy, where the top camera's field meets the table top.

    The top camera is fixed on the base. Its vertical field is the calibrated
    59.17°, and the 640×480 frame makes the horizontal field wider. That quad is
    the view used to place the tray and the cube. The wrist camera moves with
    the hand, so it does not choose the starting spots.
    """
    cam = sim.cam_id["top"]
    origin = sim.data.cam_xpos[cam]
    rotation = sim.data.cam_xmat[cam].reshape(3, 3)
    half_vert = np.tan(np.deg2rad(float(sim.model.cam_fovy[cam])) / 2.0)
    half_horiz = half_vert * (640.0 / 480.0)
    corners = []
    for side_x, side_y in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
        direction = rotation @ np.array([side_x * half_horiz, side_y * half_vert, -1.0])
        direction = direction / np.linalg.norm(direction)
        if abs(direction[2]) < 1e-8:
            raise RuntimeError("top camera ray misses the table")
        distance = (0.0 - origin[2]) / direction[2]
        point = origin + distance * direction
        corners.append(point[:2])
    return np.asarray(corners, dtype=np.float64)


def _inside(quad: np.ndarray, point: np.ndarray) -> bool:
    signs = []
    for index in range(len(quad)):
        edge = quad[(index + 1) % len(quad)] - quad[index]
        rel = point - quad[index]
        signs.append(edge[0] * rel[1] - edge[1] * rel[0])
    return all(sign >= -1e-8 for sign in signs) or all(sign <= 1e-8 for sign in signs)


def _on_table(sim: ArmSim, point: np.ndarray) -> bool:
    center = sim.model.body_pos[sim.model.body(name="table").id][:2]
    half = sim.model.geom(name="table_top").size[:2]
    return abs(point[0] - center[0]) <= half[0] and abs(point[1] - center[1]) <= half[1]


def _corners(xy: np.ndarray, yaw: float, half: np.ndarray) -> np.ndarray:
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    local = np.array([[half[0], half[1]], [half[0], -half[1]], [-half[0], half[1]], [-half[0], -half[1]]])
    return xy + local @ rotation.T


# Fingertip positions measured on the real robot, 174 cube episodes.
# so101_sim/envs.py: object x p5..p95 is 0.172..0.368, y is -0.099..0.190.
# That file rejects an IK scan of the same arm because the scan produced false negatives.
FINGER_X = (0.172, 0.368)
FINGER_Y = (-0.099, 0.190)


def _in_fingers(point: np.ndarray) -> bool:
    return FINGER_X[0] <= point[0] <= FINGER_X[1] and FINGER_Y[0] <= point[1] <= FINGER_Y[1]


def _clear_of_bin(cube_xy: np.ndarray, bin_xy: np.ndarray, bin_yaw: float, bin_half: np.ndarray, cube_half: float) -> bool:
    """The cube's bounding circle stays outside the tray rim."""
    cosine, sine = np.cos(-bin_yaw), np.sin(-bin_yaw)
    delta = cube_xy - bin_xy
    local = np.array([cosine * delta[0] - sine * delta[1], sine * delta[0] + cosine * delta[1]])
    reach = cube_half * np.sqrt(2.0) + 0.01
    return abs(local[0]) >= bin_half[0] + reach or abs(local[1]) >= bin_half[1] + reach


def _wrist_clips_cube(sim: ArmSim, xy: np.ndarray, yaw: float, half: float) -> bool:
    """The power-on wrist cuts this cube with the picture edge.

    The arm is already at the power-on pose. A cube fully inside or fully
    outside the frame is kept. One that crosses the edge is the corner clip.
    """
    cam = sim.cam_id["wrist"]
    origin = sim.data.cam_xpos[cam]
    rotation = sim.data.cam_xmat[cam].reshape(3, 3)
    half_vert = np.tan(np.deg2rad(float(sim.model.cam_fovy[cam])) / 2.0)
    half_horiz = half_vert * (640.0 / 480.0)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    inside = 0
    outside = False
    for sx in (-half, half):
        for sy in (-half, half):
            for sz in (-half, half):
                world = np.array(
                    [
                        float(xy[0]) + cosine * sx - sine * sy,
                        float(xy[1]) + sine * sx + cosine * sy,
                        0.02 + sz,
                    ]
                )
                local = rotation.T @ (world - origin)
                if local[2] >= -1e-8:
                    outside = True
                    continue
                ndc_x = (local[0] / -local[2]) / half_horiz
                ndc_y = (local[1] / -local[2]) / half_vert
                if abs(ndc_x) <= 1.0 and abs(ndc_y) <= 1.0:
                    inside += 1
                else:
                    outside = True
    # One corner in the frame is a sliver. The corner bug had two or more.
    return inside >= 2 and outside


# Episode 490 frame 80. At this reach pose the training wrist shows the cube and
# the bin here. A pickup that starts elsewhere shows the checkpoint a different picture.
REACH_ARM = np.array([-9.0, 1.6, -8.4, 90.1, 20.6, 43.9], dtype=np.float64)
MATCH_CUBE = np.array([0.308, 0.033])
MATCH_CUBE_YAW = np.deg2rad(-135.0)
MATCH_BIN = np.array([0.326, 0.151])
MATCH_BIN_YAW = np.deg2rad(-45.0)
TRAIN_CUBE_PX = np.array([376.4, 118.3])
TRAIN_TRAY_PX = np.array([132.1, 37.6])
WRIST_PX = 30.0


def _blob_center(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()])


def _reach_wrist_offset(frame: np.ndarray) -> tuple[float, float] | None:
    """How far the cube and the bin sit from the training wrist, in pixels."""
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    cube = (blue > 100) & (green > 100) & (blue > red + 8) & (red < 220)
    arm = (red > 60) & (red > blue + 15) & (green > blue + 8)
    tray = ((red + green + blue) / 3 < 90) & ~arm & ~cube
    cube_at = _blob_center(cube)
    tray_at = _blob_center(tray)
    if cube_at is None or tray_at is None:
        return None
    return (
        float(np.linalg.norm(cube_at - TRAIN_CUBE_PX)),
        float(np.linalg.norm(tray_at - TRAIN_TRAY_PX)),
    )


def draw_layout(sim: ArmSim, seed: int) -> dict:
    """Place the bin and the cube near the training reach picture.

    The bin center stays in the measured finger area. Every bin corner does not,
    because the training bin's corners stick out of that box. The cube stays
    outside the bin. A one-corner sliver in the power-on wrist is kept. Two or
    more corners in that frame is the corner clip and is rejected.
    """
    import mujoco

    sim.hold_pose(POWER_ON)
    quad = _camera_table_quad(sim)
    rng = np.random.Generator(np.random.PCG64(seed))
    cube_half = float(sim.model.geom(name="cube_geom").size[0])
    bin_half = sim.bin_half
    for _ in range(400):
        cube = MATCH_CUBE + rng.normal(0.0, 0.006, 2)
        bin_xy = MATCH_BIN + rng.normal(0.0, 0.006, 2)
        cube_yaw = float(MATCH_CUBE_YAW + rng.normal(0.0, np.deg2rad(8.0)))
        bin_yaw = float(MATCH_BIN_YAW + rng.normal(0.0, np.deg2rad(8.0)))
        cube_corners = _corners(cube, cube_yaw, np.array([cube_half, cube_half]))
        if not all(_inside(quad, corner) and _on_table(sim, corner) and _in_fingers(corner) for corner in cube_corners):
            continue
        if not (_in_fingers(bin_xy) and _inside(quad, bin_xy) and _on_table(sim, bin_xy)):
            continue
        if float(np.hypot(bin_xy[0], bin_xy[1])) > 0.38:
            continue
        if not _clear_of_bin(cube, bin_xy, bin_yaw, bin_half, cube_half):
            continue
        sim.hold_pose(POWER_ON)
        if _wrist_clips_cube(sim, cube, cube_yaw, cube_half):
            continue
        sim.place_bin((float(bin_xy[0]), float(bin_xy[1])), bin_yaw)
        sim.data.qpos[sim.cube_qadr : sim.cube_qadr + 3] = (float(cube[0]), float(cube[1]), 0.02)
        half = 0.5 * cube_yaw
        sim.data.qpos[sim.cube_qadr + 3 : sim.cube_qadr + 7] = (np.cos(half), 0.0, 0.0, np.sin(half))
        mujoco.mj_forward(sim.model, sim.data)
        sim.hold_pose(REACH_ARM)
        if sim.jaw_contains_cube_center() is not None:
            continue
        offset = _reach_wrist_offset(sim.render("wrist"))
        if offset is None or offset[0] > WRIST_PX or offset[1] > WRIST_PX:
            continue
        return {
            "bin_xy": (float(bin_xy[0]), float(bin_xy[1])),
            "bin_yaw": bin_yaw,
            "cube_xy": (float(cube[0]), float(cube[1])),
            "cube_yaw": cube_yaw,
        }
    raise RuntimeError(f"seed {seed} could not match the training reach picture")


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
    seed: int | None = None,
    empty_force: float = 0.0,
) -> dict:
    if seed is not None:
        seed_policy(seed)
    start = prepare(sim, cube_xy, yaw_rad=yaw_rad)
    watch = GripWatch(empty_force)
    out = rollout(sim, predict, steps=steps, chunk_size=CHUNK, watch=watch)
    grip = watch.chosen()
    folder = root / name
    _write_clip(out["top"], folder / "top.mp4")
    _write_clip(out["wrist"], folder / "wrist.mp4")
    end = sim.cube_pos()
    row = {
        "scenario": name,
        "cube_start_xy": start,
        "cube_end_xyz": [round(float(v), 4) for v in end],
        "contact": bool(out["contact"]),
        "lifted": bool(out["lifted"]),
        "in_box": bool(sim.in_box()),
        "cube_yaw_deg": None if yaw_rad is None else round(float(np.degrees(yaw_rad)), 2),
        "bin_xy": [round(float(v), 4) for v in sim.data.xpos[sim.box_body][:2]],
        "seed": seed,
        "grip": grip["contact"],
        "grip_step": grip["step"],
        "finger_face_deg": grip["finger_face_deg"],
        "grip_roll_deg": grip["roll_deg"],
        "picked_up": watch.picked_up,
        "held": watch.held,
    }
    (folder / "result.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row


def run_all(sim: ArmSim, predict, root: Path | None = None, steps: int = EVAL_STEPS) -> list[dict]:
    dest = root or VLA_TESTING
    rows = [run_scenario(sim, predict, name, xy, dest, steps=steps) for name, xy in scenarios()]
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
            "| Seed | In bin | Picked up | Held | Jaw contact | Finger angle ° |",
            "|---:|---|---|---|---|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {seed} | {box} | {picked} | {held} | {grip} | {angle:.1f} |".format(
                seed=row["seed"],
                box="pass" if row["in_box"] else "fail",
                picked="pass" if row["picked_up"] else "fail",
                held="pass" if row["held"] else "fail",
                grip="pass" if row["grip"] else "fail",
                angle=row["finger_face_deg"],
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
) -> list[dict]:
    """Seeds ``0`` .. ``count - 1``. The same seed draws the same cube."""
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
                seed=seed,
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "random":
        count = int(sys.argv[2]) if len(sys.argv) > 2 else RANDOM_COUNT
        sim = ArmSim()
        try:
            print("loading checkpoint", flush=True)
            policy, pre, post = load_checkpoint()

            def predict(state, top, wrist):
                return predict_chunk(policy, pre, post, state, top, wrist)

            rows = run_random(sim, predict, count=count)
        finally:
            sim.close()
        placed = sum(1 for row in rows if row["in_box"])
        print(f"placed {placed} of {len(rows)}", flush=True)
    else:
        main()
