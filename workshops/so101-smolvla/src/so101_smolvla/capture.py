"""Capture one expert episode to a compressed npz. This module does not import torch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from so101_smolvla.expert import Expert
from so101_smolvla.task import TASK
from so101_smolvla.sim import ArmSim

ROOT = Path(__file__).resolve().parents[2]


def capture(index: int, xy: tuple[float, float], azimuth: float, dest: Path) -> bool:
    sim = ArmSim()
    expert = Expert(sim)
    frames: dict[str, list] = {"state": [], "action": [], "top": [], "wrist": []}
    try:
        def on_tick() -> None:
            frames["state"].append(sim.read_state().astype(np.float32))
            frames["action"].append(sim.commanded().astype(np.float32))
            frames["top"].append(sim.render("top"))
            frames["wrist"].append(sim.render("wrist"))

        ok = expert.run(xy, azimuth=azimuth, on_tick=on_tick)
        if not ok:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest,
            state=np.stack(frames["state"]),
            action=np.stack(frames["action"]),
            top=np.stack(frames["top"]),
            wrist=np.stack(frames["wrist"]),
            task=np.array(TASK),
        )
        return True
    finally:
        sim.close()


if __name__ == "__main__":
    index = int(sys.argv[1])
    x, y = float(sys.argv[2]), float(sys.argv[3])
    azimuth = float(sys.argv[4])
    dest = Path(sys.argv[5])
    raise SystemExit(0 if capture(index, (x, y), azimuth, dest) else 1)
