"""Closed-loop episode. Control, render, and inference are separate processes."""

from __future__ import annotations

import time
from multiprocessing import Event, Process
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from so101_smolvla.sim import IMAGE_H, IMAGE_W, TICK_STEPS, ArmSim

STATE_N = 10
ACTION_N = 7


def _array(mem: SharedMemory, count: int, dtype) -> np.ndarray:
    return np.ndarray((count,), dtype=dtype, buffer=mem.buf)


def run_policy_episode(ticks: int, cube_xy: tuple[float, float]) -> dict:
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    state_mem = SharedMemory(create=True, size=STATE_N * 8)
    action_mem = SharedMemory(create=True, size=ACTION_N * 8)
    frame_mem = SharedMemory(create=True, size=2 * IMAGE_H * IMAGE_W * 3)
    state_ready = Event()
    frame_ready = Event()
    stop = Event()
    action_ready = Event()
    control = Process(
        target=_control,
        args=(state_mem.name, action_mem.name, state_ready, action_ready, stop, ticks, cube_xy),
    )
    render = Process(
        target=_render,
        args=(state_mem.name, frame_mem.name, state_ready, frame_ready, stop),
    )
    infer = Process(
        target=_infer,
        args=(state_mem.name, frame_mem.name, action_mem.name, frame_ready, action_ready, stop),
    )
    infer.start()
    render.start()
    control.start()
    control.join()
    stop.set()
    state_ready.set()
    frame_ready.set()
    render.join()
    infer.join()
    cube = _array(state_mem, STATE_N, np.float64)[6:9].copy()
    for mem in (state_mem, action_mem, frame_mem):
        mem.close()
        mem.unlink()
    return {"final_cube": cube.tolist(), "ticks": ticks}


def _control(state_name, action_name, state_ready, action_ready, stop, ticks, cube_xy) -> None:
    state_mem = SharedMemory(name=state_name)
    action_mem = SharedMemory(name=action_name)
    state = _array(state_mem, STATE_N, np.float64)
    action = _array(action_mem, ACTION_N, np.float64)
    sim = ArmSim()
    sim.reset(cube_xy)
    last = sim.read_state()
    state[:6] = last
    state[6:9] = sim.cube_pos()
    state[9] = 1.0
    state_ready.set()
    seen = 0.0
    holds = 0
    try:
        action_ready.wait()
        for step in range(ticks):
            started = time.perf_counter()
            if action[-1] > seen:
                last = action[:6].copy()
                seen = float(action[-1])
            else:
                holds += 1
            sim.command(last)
            sim.step(TICK_STEPS)
            state[:6] = sim.read_state()
            state[6:9] = sim.cube_pos()
            state[9] = float(step + 1)
            state_ready.set()
            if time.perf_counter() - started > 1.0 / 30.0:
                holds += 0
        print(f"control holds {holds} actions_seen {seen}", flush=True)
    finally:
        stop.set()
        state_ready.set()
        sim.close()
        state_mem.close()
        action_mem.close()


def _render(state_name, frame_name, state_ready, frame_ready, stop) -> None:
    import mujoco

    state_mem = SharedMemory(name=state_name)
    frame_mem = SharedMemory(name=frame_name)
    state = _array(state_mem, STATE_N, np.float64)
    frames = np.ndarray((2, IMAGE_H, IMAGE_W, 3), dtype=np.uint8, buffer=frame_mem.buf)
    sim = ArmSim()
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    seen = 0.0
    try:
        while not stop.is_set():
            if not state_ready.wait(timeout=1.0):
                continue
            state_ready.clear()
            seq = float(state[9])
            if seq <= seen:
                continue
            seen = seq
            for index, name in enumerate(names):
                sim.data.qpos[sim.joint_qadr[name]] = np.deg2rad(state[index])
            sim.data.qpos[sim.joint_qadr["gripper"]] = sim.gripper_to_rad(float(state[5]))
            sim.data.qpos[sim.cube_qadr : sim.cube_qadr + 3] = state[6:9]
            mujoco.mj_forward(sim.model, sim.data)
            frames[0] = sim.render("top")
            frames[1] = sim.render("wrist")
            frame_ready.set()
    finally:
        sim.close()
        state_mem.close()
        frame_mem.close()


def _infer(state_name, frame_name, action_name, frame_ready, action_ready, stop) -> None:
    from so101_smolvla.policy import load_policy, select_action

    state_mem = SharedMemory(name=state_name)
    frame_mem = SharedMemory(name=frame_name)
    action_mem = SharedMemory(name=action_name)
    state = _array(state_mem, STATE_N, np.float64)
    frames = np.ndarray((2, IMAGE_H, IMAGE_W, 3), dtype=np.uint8, buffer=frame_mem.buf)
    action = _array(action_mem, ACTION_N, np.float64)
    policy, pre, post, device = load_policy()
    sequence = 0.0
    try:
        while not stop.is_set():
            if not frame_ready.wait(timeout=1.0):
                continue
            frame_ready.clear()
            top = frames[0].copy()
            wrist = frames[1].copy()
            values, _elapsed = select_action(policy, pre, post, state[:6].copy(), top, wrist, device)
            sequence += 1.0
            action[:6] = values
            action[-1] = sequence
            action_ready.set()
    finally:
        state_mem.close()
        frame_mem.close()
        action_mem.close()


def main() -> None:
    from so101_smolvla.evaluate import grasp_attempt

    result = grasp_attempt()
    print(result, flush=True)
    if not result["grasped"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
