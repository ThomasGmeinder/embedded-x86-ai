"""The model's answers for real and rendered pictures of the moment before the grasp.

Run with .venv, PYTHONPATH = <workshop>/src : ~/.cache/so101-stack-training,
from the workshop folder (the model runs on the graphics processor through
ROCm; no simulator):
    grasp_aim_predict.py

Second step of grasp_aim_check.py. For each recording in
analysis/grasp-aim-pictures.npz the unchanged checkpoint gets the recorded
joint state, the stacking sentence and one pair of pictures:

  real top + real wrist                      what it was trained on
  scene top + scene wrist (measured camera)  the stack2-recording-rig scene
  scene top + scene wrist (simulator camera) the same scene before the wrist camera was measured
  real top + scene wrist (measured camera)   only the wrist picture is rendered
  real top + scene wrist (simulator camera)
  scene top + real wrist                     only the top picture is rendered

Each answer is the model's next 50 joint targets. Writes
analysis/grasp-aim-predictions.npz.
"""

from __future__ import annotations

import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd

from compare_real_vs_sim_images import predict
from measure_objects import FPS, real_first_frame
from so101_smolvla.policy_env import PolicyObservation
from so101_smolvla.sapien_config import task_spec
from so101_smolvla.sapien_rollout import _load_policy

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
SEED = 0
# label -> (top picture, wrist picture)
PAIRS = {
    "real top + real wrist": ("real_top", "real_wrist"),
    "scene top + scene wrist (measured camera)": ("top", "wrist_measured"),
    "scene top + scene wrist (simulator camera)": ("top", "wrist_simulator"),
    "real top + scene wrist (measured camera)": ("real_top", "wrist_measured"),
    "real top + scene wrist (simulator camera)": ("real_top", "wrist_simulator"),
    "scene top + real wrist": ("top", "real_wrist"),
}


def main() -> None:
    stored = np.load(ANALYSIS / "grasp-aim-pictures.npz")
    pictures = {name: stored[name] for name in ("top", "wrist_measured", "wrist_simulator")}
    episodes = pd.concat(
        pd.read_parquet(path)
        for path in sorted(glob.glob(str(ROOT / "meta" / "episodes" / "chunk-000" / "*.parquet")))
    ).set_index("episode_index")
    for camera in ("top", "wrist"):
        column = f"videos/observation.images.{camera}/from_timestamp"
        pictures[f"real_{camera}"] = np.stack([
            real_first_frame(camera, float(episodes.loc[number][column]) + frame / FPS)
            for number, frame in zip(stored["episode"].tolist(), stored["frame"].tolist())
        ])

    text = task_spec("stack2-recording-rig").instruction
    print(f"task sentence: {text!r}", flush=True)
    checkpoint, policy, pre, post = _load_policy()
    print(f"checkpoint revision {checkpoint.name}", flush=True)

    started = time.monotonic()
    chunks: dict[str, list[np.ndarray]] = {label: [] for label in PAIRS}
    for index, state in enumerate(stored["state"]):
        for label, (top, wrist) in PAIRS.items():
            observation = PolicyObservation(
                state=state.astype(np.float32),
                top=pictures[top][index], wrist=pictures[wrist][index],
            )
            chunks[label].append(predict(policy, pre, post, observation, text, SEED))
        print(f"recording {int(stored['episode'][index])} answered", flush=True)
    count = len(stored["state"]) * len(PAIRS)
    print(f"{count} answers, {(time.monotonic() - started) / count:.2f} s per answer", flush=True)
    np.savez_compressed(
        ANALYSIS / "grasp-aim-predictions.npz",
        episode=stored["episode"], frame=stored["frame"],
        **{label: np.stack(values) for label, values in chunks.items()},
    )


if __name__ == "__main__":
    main()
