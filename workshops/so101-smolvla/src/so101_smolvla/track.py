"""Play the sim-real checkpoint and record joint lag.

This loader is separate from ``load_policy``. That function stays on the base
checkpoint. ``rollout`` is 500 steps, one new chunk every 50, which is the
checkpoint's published eval length. It still uses this scene.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from so101_smolvla.record import _write_clip
from so101_smolvla.sim import TICK_STEPS, ArmSim

CHECKPOINT = "Harrysunshine/so101-smolvla-sim-real-10task"
TASK = "Pick up a cube and place in the bin"
POWER_ON = np.array([-5.76, -100.0, 92.97, 63.38, -0.53, 1.90], dtype=np.float64)
CHUNK = 50
EVAL_STEPS = 500
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "track"


def hold_power_on(sim: ArmSim) -> np.ndarray:
    sim.reset()
    sim.hold_pose(POWER_ON)
    return sim.read_state()


def play(sim: ArmSim, targets: np.ndarray) -> dict[str, np.ndarray]:
    """Command each target for one tick. Return target, clipped command, achieved, and frames."""
    rows = np.asarray(targets, dtype=np.float64).reshape(-1, 6)
    commanded = np.zeros_like(rows)
    achieved = np.zeros_like(rows)
    top = []
    wrist = []
    for index, target in enumerate(rows):
        sim.command(target)
        sim.step(TICK_STEPS)
        commanded[index] = sim.commanded()
        achieved[index] = sim.read_state()
        top.append(sim.render("top"))
        wrist.append(sim.render("wrist"))
    return {
        "target": rows,
        "commanded": commanded,
        "achieved": achieved,
        "top": np.stack(top),
        "wrist": np.stack(wrist),
    }


def _image(frame: np.ndarray) -> torch.Tensor:
    # The policy pads this frame to 512x512. Squashing it to a square first
    # changes where the cube appears.
    return torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0


def load_checkpoint(device: str = "cuda"):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(CHECKPOINT).to(device).eval()
    policy.config.device = device
    if policy.config.n_action_steps != 50 or policy.config.empty_cameras != 0:
        raise RuntimeError(
            f"checkpoint config n_action_steps={policy.config.n_action_steps} "
            f"empty_cameras={policy.config.empty_cameras}"
        )
    pre, post = make_pre_post_processors(policy.config, pretrained_path=CHECKPOINT)
    policy.reset()
    return policy, pre, post


def predict_chunk(policy, pre, post, state: np.ndarray, top: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    batch = {
        "observation.state": torch.from_numpy(state.astype(np.float32)),
        "observation.images.top": _image(top),
        "observation.images.wrist": _image(wrist),
        "task": TASK,
    }
    with torch.inference_mode():
        prepared = pre(batch)
        chunk = post(policy.predict_action_chunk(prepared))
    actions = chunk.detach().float().cpu().numpy().reshape(-1, 6)
    if actions.shape != (50, 6):
        raise RuntimeError(f"chunk shape {actions.shape}")
    return actions


def rollout(sim: ArmSim, predict, steps: int = EVAL_STEPS, chunk_size: int = CHUNK) -> dict:
    """Take a new picture only when the previous chunk has been commanded."""
    if steps < 1 or chunk_size < 1:
        raise ValueError((steps, chunk_size))
    tops = []
    wrists = []
    calls = 0
    done = 0
    while done < steps:
        actions = np.asarray(
            predict(sim.read_state(), sim.render("top"), sim.render("wrist")), dtype=np.float64
        ).reshape(-1, 6)
        calls += 1
        take = min(chunk_size, steps - done, actions.shape[0])
        played = play(sim, actions[:take])
        tops.append(played["top"])
        wrists.append(played["wrist"])
        done += take
    return {
        "top": np.concatenate(tops),
        "wrist": np.concatenate(wrists),
        "calls": calls,
        "cube": sim.cube_pos().copy(),
        "contact": bool(sim.jaw_contact()),
        "lifted": bool(sim.above_table()),
    }


def write_report(path: Path, pose: np.ndarray, played: dict[str, np.ndarray]) -> None:
    steps = []
    for index in range(played["target"].shape[0]):
        steps.append(
            {
                "target": [round(float(v), 4) for v in played["target"][index]],
                "commanded": [round(float(v), 4) for v in played["commanded"][index]],
                "achieved": [round(float(v), 4) for v in played["achieved"][index]],
            }
        )
    path.write_text(
        json.dumps(
            {
                "checkpoint": CHECKPOINT,
                "task": TASK,
                "pose": [round(float(v), 4) for v in pose],
                "steps": steps,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    sim = ArmSim()
    try:
        pose = hold_power_on(sim)
        print("loading checkpoint", flush=True)
        policy, pre, post = load_checkpoint()
        print("predicting chunk", flush=True)
        actions = predict_chunk(policy, pre, post, pose, sim.render("top"), sim.render("wrist"))
        played = play(sim, actions)
    finally:
        sim.close()
    OUT.mkdir(parents=True, exist_ok=True)
    _write_clip(played["top"], OUT / "top.mp4")
    _write_clip(played["wrist"], OUT / "wrist.mp4")
    write_report(OUT / "tracking.json", pose, played)
    lag = np.max(np.abs(played["achieved"] - played["commanded"]), axis=0)
    clip = np.max(np.abs(played["commanded"] - played["target"]), axis=0)
    print(f"pose {np.round(pose, 4).tolist()}", flush=True)
    print(f"max_abs_achieved_minus_commanded {np.round(lag, 4).tolist()}", flush=True)
    print(f"max_abs_commanded_minus_target {np.round(clip, 4).tolist()}", flush=True)


def main_rollout() -> None:
    sim = ArmSim()
    try:
        pose = hold_power_on(sim)
        print("loading checkpoint", flush=True)
        policy, pre, post = load_checkpoint()

        def predict(state, top, wrist):
            print(f"chunk from {np.round(state, 1).tolist()}", flush=True)
            return predict_chunk(policy, pre, post, state, top, wrist)

        print(f"rolling {EVAL_STEPS} steps", flush=True)
        out = rollout(sim, predict)
    finally:
        sim.close()
    OUT.mkdir(parents=True, exist_ok=True)
    _write_clip(out["top"], OUT / "rollout-top.mp4")
    _write_clip(out["wrist"], OUT / "rollout-wrist.mp4")
    report = {
        "checkpoint": CHECKPOINT,
        "task": TASK,
        "scene": "scene_task.xml",
        "steps": EVAL_STEPS,
        "calls": out["calls"],
        "pose": [round(float(v), 4) for v in pose],
        "cube": [round(float(v), 4) for v in out["cube"]],
        "contact": out["contact"],
        "lifted": out["lifted"],
    }
    (OUT / "rollout.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rollout":
        main_rollout()
    else:
        main()
