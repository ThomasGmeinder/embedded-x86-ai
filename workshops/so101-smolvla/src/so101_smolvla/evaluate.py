"""Compare a checkpoint on held-out cube positions."""

from __future__ import annotations

import numpy as np

from so101_smolvla.policy import load_policy, select_action
from so101_smolvla.sim import HOLD_STEPS, TICK_STEPS, ArmSim


def rollout(ticks: int, cube_xy: tuple[float, float]) -> dict:
    sim = ArmSim()
    policy, pre, post, device = load_policy()
    sim.reset(cube_xy)
    times = []
    try:
        for _ in range(ticks):
            action, elapsed = select_action(
                policy, pre, post, sim.read_state(), sim.render("top"), sim.render("wrist"), device
            )
            times.append(elapsed)
            sim.command(action)
            sim.step(TICK_STEPS)
        return {
            "cube": sim.cube_pos().tolist(),
            "in_box": bool(sim.in_box()),
            "lifted": bool(sim.above_table()),
            "mean_inference_s": float(np.mean(times)),
        }
    finally:
        sim.close()


def grasp_attempt(cube_xy: tuple[float, float] | None = None, search_ticks: int = 200) -> dict:
    """Run the base policy until it lifts the cube, then require a 3 s hold.

    Returns the measured facts. The caller treats ``grasped`` false as a failure.
    """
    sim = ArmSim()
    policy, pre, post, device = load_policy()
    sim.reset(cube_xy)
    times = []
    lifted_at = None
    try:
        for tick in range(search_ticks):
            action, elapsed = select_action(
                policy, pre, post, sim.read_state(), sim.render("top"), sim.render("wrist"), device
            )
            times.append(elapsed)
            sim.command(action)
            sim.step(TICK_STEPS)
            if sim.jaw_contact() and sim.above_table():
                lifted_at = tick
                break
        held = False
        if lifted_at is not None:
            for _ in range(HOLD_STEPS // TICK_STEPS):
                action, elapsed = select_action(
                    policy, pre, post, sim.read_state(), sim.render("top"), sim.render("wrist"), device
                )
                times.append(elapsed)
                sim.command(action)
                sim.step(TICK_STEPS)
            held = bool(sim.jaw_contact() and sim.above_table())
        return {
            "cube": [round(float(v), 4) for v in sim.cube_pos()],
            "lifted_at_tick": lifted_at,
            "contact": bool(sim.jaw_contact()),
            "gripper_force": round(sim.gripper_force(), 4),
            "above_table": bool(sim.above_table()),
            "grasped": held,
            "mean_inference_s": round(float(np.mean(times)), 4) if times else None,
        }
    finally:
        sim.close()


def main() -> None:
    result = grasp_attempt()
    print(result, flush=True)
    if not result["grasped"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
