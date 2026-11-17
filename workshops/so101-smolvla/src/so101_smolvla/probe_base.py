"""Run lerobot/smolvla_base on pick and stack. Render before loading PyTorch."""

from __future__ import annotations

import json
import sys

import mujoco
import numpy as np

from so101_smolvla.sim import ROOT, TICK_STEPS, ArmSim
from so101_smolvla.task import TASK

STACK_SCENE = ROOT / "assets" / "robotstudio_so101" / "scene_stack.xml"
STACK_TASK = "Pick up the red cube and put it on top of the blue cube."
TICKS = 60
DECIDE_TICK = 30
CUBE_XY = (0.22, 0.10)
COLORS = {
    "red": (0.85, 0.15, 0.12),
    "blue": (0.20, 0.35, 0.75),
    "green": (0.15, 0.65, 0.20),
}


def _set_color(sim: ArmSim, rgb: tuple[float, float, float]) -> None:
    material = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_MATERIAL, "cube_mat")
    sim.model.mat_rgba[material, :3] = rgb


def _body_pos(sim: ArmSim, joint: str) -> list[float]:
    address = int(sim.model.jnt_qposadr[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint)])
    return [round(float(v), 4) for v in sim.data.qpos[address : address + 3]]


def _run(sim: ArmSim, policy, pre, post, device, select_action, task: str, label: str) -> dict:
    policy.reset()
    start = sim.cube_pos().copy()
    ran = 0
    for tick in range(TICKS):
        action, _elapsed = select_action(
            policy, pre, post, sim.read_state(), sim.render("top"), sim.render("wrist"), device, task=task
        )
        sim.command(action)
        sim.step(TICK_STEPS)
        ran = tick + 1
        pos = sim.cube_pos()
        if tick == 0 or tick + 1 == DECIDE_TICK:
            print(
                f"{label} tick {tick} cube {[round(float(v), 3) for v in pos]} contact {sim.jaw_contact()}",
                flush=True,
            )
        if tick + 1 == DECIDE_TICK and not sim.jaw_contact() and float(np.linalg.norm(pos - start)) < 0.01:
            print(f"{label} stop: no contact and the cube has not moved", flush=True)
            break
    end = sim.cube_pos()
    return {
        "ticks": ran,
        "task": task,
        "start": [round(float(v), 4) for v in start],
        "end": [round(float(v), 4) for v in end],
        "moved_m": round(float(np.linalg.norm(end - start)), 4),
        "contact": bool(sim.jaw_contact()),
        "above_table": bool(sim.above_table()),
        "in_box": bool(sim.in_box()),
        "grasped": bool(sim.jaw_contact() and sim.above_table()),
    }


def main() -> None:
    pick = ArmSim()
    pick.render("top")
    stack = ArmSim(STACK_SCENE)
    stack.render("top")
    from so101_smolvla.policy import load_policy, select_action

    policy, pre, post, device = load_policy()
    print(f"loaded lerobot/smolvla_base on {device}", flush=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    report: dict = {"checkpoint": "lerobot/smolvla_base", "ticks": TICKS, "pick": [], "stack": None}
    try:
        if which in ("all", "pick"):
            for name, rgb in COLORS.items():
                _set_color(pick, rgb)
                pick.reset(CUBE_XY)
                row = _run(pick, policy, pre, post, device, select_action, TASK, f"pick-{name}")
                row["color"] = name
                report["pick"].append(row)
                print(f"pick-{name} {row}", flush=True)
                if not row["grasped"]:
                    print("pick stop: the first cube was not grasped", flush=True)
                    break
        stack.reset((0.18, 0.08))
        row = _run(stack, policy, pre, post, device, select_action, STACK_TASK, "stack")
        red = np.array(_body_pos(stack, "cube_joint"))
        blue = np.array(_body_pos(stack, "cube_blue_joint"))
        row["blue"] = blue.tolist()
        row["stacked"] = bool(
            abs(red[0] - blue[0]) < 0.04 and abs(red[1] - blue[1]) < 0.04 and red[2] > blue[2] + 0.02
        )
        report["stack"] = row
        print(f"stack {row}", flush=True)
    finally:
        dest = ROOT / "scripted_testing" / "base_probe" / "result.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        pick.close()
        stack.close()
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
