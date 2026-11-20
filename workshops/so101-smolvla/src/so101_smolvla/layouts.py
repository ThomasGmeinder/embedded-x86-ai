"""One deterministic cube/bin layout distribution for recording and evaluation."""

from __future__ import annotations

import numpy as np

from so101_smolvla.reachable_workspace import is_reachable_xy
from so101_smolvla.sim import ArmSim

POWER_ON = np.array([-5.76, -100.0, 92.97, 63.38, -0.53, 1.90], dtype=np.float64)
REACH_ARM = np.array([-9.0, 1.6, -8.4, 90.1, 20.6, 43.9], dtype=np.float64)
MATCH_CUBE = np.array([0.308, 0.033])
MATCH_CUBE_YAW = np.deg2rad(-135.0)
MATCH_BIN = np.array([0.326, 0.151])
MATCH_BIN_YAW = np.deg2rad(-45.0)
TRAIN_CUBE_PX = np.array([376.4, 118.3])
TRAIN_TRAY_PX = np.array([132.1, 37.6])
WRIST_PX = 30.0
def _camera_table_quad(sim: ArmSim) -> np.ndarray:
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
        distance = -origin[2] / direction[2]
        corners.append((origin + distance * direction)[:2])
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
    local = np.array(
        [[half[0], half[1]], [half[0], -half[1]], [-half[0], half[1]], [-half[0], -half[1]]]
    )
    return xy + local @ rotation.T


def _clear_of_bin(
    cube_xy: np.ndarray,
    bin_xy: np.ndarray,
    bin_yaw: float,
    bin_half: np.ndarray,
    cube_half: float,
) -> bool:
    cosine, sine = np.cos(-bin_yaw), np.sin(-bin_yaw)
    delta = cube_xy - bin_xy
    local = np.array([cosine * delta[0] - sine * delta[1], sine * delta[0] + cosine * delta[1]])
    reach = cube_half * np.sqrt(2.0) + 0.01
    return abs(local[0]) >= bin_half[0] + reach or abs(local[1]) >= bin_half[1] + reach


def _wrist_clips_cube(sim: ArmSim, xy: np.ndarray, yaw: float, half: float) -> bool:
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
                    [float(xy[0]) + cosine * sx - sine * sy, float(xy[1]) + sine * sx + cosine * sy, 0.02 + sz]
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
    return inside >= 2 and outside


def _blob_center(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()])


def _reach_wrist_offset(frame: np.ndarray) -> tuple[float, float] | None:
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
    """Draw the established deterministic MuJoCo random-regression layout."""
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
        if not all(
            _inside(quad, corner)
            and _on_table(sim, corner)
            and is_reachable_xy(corner)
            for corner in cube_corners
        ):
            continue
        if not (
            is_reachable_xy(bin_xy)
            and _inside(quad, bin_xy)
            and _on_table(sim, bin_xy)
        ):
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
        sim.data.qpos[sim.cube_qadr + 3 : sim.cube_qadr + 7] = (
            np.cos(half),
            0.0,
            0.0,
            np.sin(half),
        )
        mujoco.mj_forward(sim.model, sim.data)
        sim.hold_pose(REACH_ARM)
        if sim.jaw_contains_cube_center() is not None:
            continue
        offset = _reach_wrist_offset(sim.render("wrist"))
        if offset is None or offset[0] > WRIST_PX or offset[1] > WRIST_PX:
            continue
        return {
            "seed": int(seed),
            "bin_xy": (float(bin_xy[0]), float(bin_xy[1])),
            "bin_yaw": bin_yaw,
            "cube_xy": (float(cube[0]), float(cube[1])),
            "cube_yaw": cube_yaw,
        }
    raise RuntimeError(f"seed {seed} could not produce a valid layout")
