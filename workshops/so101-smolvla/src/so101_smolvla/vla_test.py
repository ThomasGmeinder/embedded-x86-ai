"""Run the sim-real checkpoint at each listed cube position.

Videos and results go to ``vla_testing/``. Scripted runs stay in ``scripted_testing/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from so101_smolvla.record import _write_clip
from so101_smolvla.sim import ArmSim
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
# Published spawn box, robot at the origin. so101_sim/envs.py SPAWN_BOX_POS / SPAWN_BOX_HALF_SIZE.
SPAWN_CENTER = (0.27, 0.045)
SPAWN_HALF = 0.10
RANDOM_COUNT = 100


def scenarios(path: Path | None = None) -> list[tuple[str, tuple[float, float] | None]]:
    """Scene default, then each cube position in ``configs/sim.yaml``."""
    loaded = yaml.safe_load((path or SIM_YAML).read_text(encoding="utf-8"))
    rows: list[tuple[str, tuple[float, float] | None]] = [("scene", None)]
    for xy in loaded["distinct_cube_positions"]:
        x, y = float(xy[0]), float(xy[1])
        rows.append((f"cube-{x:g}-{y:g}", (x, y)))
    return rows


def draw_cube(seed: int) -> tuple[tuple[float, float], float]:
    """Cube xy inside the spawn box, and a yaw about vertical. Both come from ``seed``."""
    rng = np.random.Generator(np.random.PCG64(seed))
    offset = rng.uniform(-SPAWN_HALF, SPAWN_HALF, size=2)
    yaw = float(rng.uniform(-np.pi, np.pi))
    return (float(SPAWN_CENTER[0] + offset[0]), float(SPAWN_CENTER[1] + offset[1])), yaw


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


def run_scenario(
    sim: ArmSim,
    predict,
    name: str,
    cube_xy: tuple[float, float] | None,
    root: Path,
    steps: int = EVAL_STEPS,
    yaw_rad: float | None = None,
    seed: int | None = None,
) -> dict:
    if seed is not None:
        seed_policy(seed)
    start = prepare(sim, cube_xy, yaw_rad=yaw_rad)
    out = rollout(sim, predict, steps=steps, chunk_size=CHUNK)
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
        "seed": seed,
    }
    (folder / "result.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row


def run_all(sim: ArmSim, predict, root: Path | None = None, steps: int = EVAL_STEPS) -> list[dict]:
    dest = root or VLA_TESTING
    rows = [run_scenario(sim, predict, name, xy, dest, steps=steps) for name, xy in scenarios()]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


def run_random(sim: ArmSim, predict, count: int = RANDOM_COUNT, root: Path | None = None) -> list[dict]:
    """Seeds ``0`` .. ``count - 1``. The same seed draws the same cube."""
    dest = (root or VLA_TESTING) / "random"
    rows = []
    for seed in range(count):
        xy, yaw = draw_cube(seed)
        name = f"seed-{seed:05d}"
        print(f"{name} xy={xy[0]:.3f},{xy[1]:.3f} yaw={np.degrees(yaw):.1f}", flush=True)
        rows.append(run_scenario(sim, predict, name, xy, dest, yaw_rad=yaw, seed=seed))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
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
