"""One training frame from the cube dataset, then the model's own chunk.

The recorded actions are the training target. The prediction is what this
loader returns for that frame's images and joint state. Both are then played
in this scene so the pictures can be compared.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from so101_smolvla.sim import ArmSim
from so101_smolvla.track import OUT, predict_chunk, load_checkpoint, play

REPO = "Harrysunshine/so101-sim-pickplace-v2"
EPISODE = 484
NAMES = ("pan", "lift", "elbow", "wflex", "wroll", "grip")


def _hub(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id=REPO, filename=filename, repo_type="dataset")
    )


def _frames(path: Path, start: int, count: int) -> np.ndarray:
    completed = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-vf", f"trim=start_frame={start}:end_frame={start + count},setpts=PTS-STARTPTS",
            "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        check=True,
        capture_output=True,
    )
    frame = 480 * 640 * 3
    raw = completed.stdout
    count_got = len(raw) // frame
    if count_got != count:
        raise RuntimeError(f"{path.name} gave {count_got} frames from {start}, wanted {count}")
    return np.frombuffer(raw[: count * frame], dtype=np.uint8).reshape(count, 480, 640, 3).copy()


def _write(frames: np.ndarray, path: Path) -> None:
    height, width = frames.shape[1:3]
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "30", "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(path)


def _episode():
    import pyarrow.parquet as pq

    meta = _hub("cube40/meta/episodes/chunk-000/file-000.parquet")
    table = pq.read_table(meta).to_pylist()
    row = next(item for item in table if item["episode_index"] == EPISODE)
    data = pq.read_table(_hub("cube40/data/chunk-000/file-000.parquet"))
    sl = data.slice(row["dataset_from_index"], row["length"])
    state = np.stack(sl.column("observation.state").to_pylist()).astype(np.float64)
    action = np.stack(sl.column("action").to_pylist()).astype(np.float64)
    return row, state, action


def _moving_frame(action: np.ndarray) -> int:
    last = min(len(action) - 50, 150)
    travel = [float(np.ptp(action[index : index + 50], axis=0).max()) for index in range(last)]
    return int(np.argmax(travel))


def main() -> None:
    print("downloading one cube episode", flush=True)
    row, state, action = _episode()
    index = _moving_frame(action)
    recorded = action[index : index + 50]
    print(
        f"episode {EPISODE} frame {index} task {row['tasks'][0]} "
        f"state {np.round(state[index], 2).tolist()}",
        flush=True,
    )
    top_path = _hub(
        "cube40/videos/observation.images.top/chunk-000/"
        f"file-{row['videos/observation.images.top/file_index']:03d}.mp4"
    )
    wrist_path = _hub(
        "cube40/videos/observation.images.wrist/chunk-000/"
        f"file-{row['videos/observation.images.wrist/file_index']:03d}.mp4"
    )
    top_start = int(round(row["videos/observation.images.top/from_timestamp"] * 30)) + index
    wrist_start = int(round(row["videos/observation.images.wrist/from_timestamp"] * 30)) + index
    theirs_top = _frames(top_path, top_start, 50)
    theirs_wrist = _frames(wrist_path, wrist_start, 50)

    print("loading checkpoint", flush=True)
    policy, pre, post = load_checkpoint()
    predicted = predict_chunk(policy, pre, post, state[index], theirs_top[0], theirs_wrist[0])
    error = np.abs(predicted - recorded)
    print(f"recorded_travel {[round(float(v), 3) for v in np.ptp(recorded, axis=0)]}", flush=True)
    print(f"mean_abs_error {[round(float(v), 3) for v in error.mean(0)]}", flush=True)
    print(f"max_abs_error {[round(float(v), 3) for v in error.max(0)]}", flush=True)
    print(f"recorded_last {np.round(recorded[-1], 2).tolist()}", flush=True)
    print(f"predicted_last {np.round(predicted[-1], 2).tolist()}", flush=True)

    sim = ArmSim()
    try:
        sim.reset()
        sim.hold_pose(state[index])
        held = sim.read_state()
        our_top = sim.render("top")
        our_wrist = sim.render("wrist")
        print(f"held_in_this_sim {np.round(held, 2).tolist()}", flush=True)
        print(f"clip_from_dataset_state {np.round(held - state[index], 2).tolist()}", flush=True)
        recorded_play = play(sim, recorded)
        sim.reset()
        sim.hold_pose(state[index])
        predicted_play = play(sim, predicted)
    finally:
        sim.close()

    gap_w = np.full((480, 6, 3), 255, dtype=np.uint8)

    def panel(left: np.ndarray, mid: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.concatenate([left, gap_w, mid, gap_w, right], axis=1)

    def both(index: int) -> np.ndarray:
        top = panel(theirs_top[index], recorded_play["top"][index], predicted_play["top"][index])
        wrist = panel(theirs_wrist[index], recorded_play["wrist"][index], predicted_play["wrist"][index])
        seam = np.full((6, top.shape[1], 3), 255, dtype=np.uint8)
        return np.concatenate([top, seam, wrist], axis=0)

    held_view = np.concatenate(
        [
            panel(theirs_top[0], our_top, our_top),
            np.full((6, panel(theirs_top[0], our_top, our_top).shape[1], 3), 255, dtype=np.uint8),
            panel(theirs_wrist[0], our_wrist, our_wrist),
        ],
        axis=0,
    )
    video = np.stack([held_view] * 30 + [both(index) for index in range(50)])
    dest = OUT / "training-compare.mp4"
    _write(video, dest)
    report = {
        "episode": EPISODE,
        "frame": index,
        "task": row["tasks"][0],
        "dataset_state": [round(float(v), 4) for v in state[index]],
        "held_state": [round(float(v), 4) for v in held],
        "mean_abs_error": [round(float(v), 4) for v in error.mean(0)],
        "max_abs_error": [round(float(v), 4) for v in error.max(0)],
        "names": list(NAMES),
        "video": str(dest),
    }
    (OUT / "training-compare.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
