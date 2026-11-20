"""How close are the model's answers to its own training data: the simulated small-cube bin task against real stacking?

The checkpoint's model card reports success only for three simulated bin
tasks (small cube into the bin: 45 of 50) and says the real tasks, stacking
among them, were not measured. This script asks how exactly the model
reproduces the grasp on pictures it was trained on, for a task it is known
to do and for stacking, measured the same way.

Three steps for one moment (a number of frames before the demonstrator
closes the gripper), from the workshop folder, PYTHONPATH = <workshop>/src :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, and on
the simulator's side also /tmp/Xbotics-SO101-Sim:
    answers_on_own_training_pictures.py pick <before>      (.venv-sapien)
    answers_on_own_training_pictures.py ask <before>       (.venv)
    answers_on_own_training_pictures.py report <before>    (.venv-sapien)
"pick <before> trial" and "ask <before> trial" handle two episodes and write
nothing. No step writes a file that exists.

Simulated side ("Pick up a small cube and place in the bin", the 20 mm cube
of Harrysunshine/so101-sim-pickplace-v2): 38 episodes spread evenly over the
episode numbers. The closing frame is the data set author's own rule
(recipe.close_frame). The model gets the two pictures of the data set's
videos and the recorded joint reading of the frame <before> frames earlier,
and the task's sentence.
Real side (stacking): the stored answers of early_reach_check.py to the real
pictures of 38 recordings at the same number of frames before the human
closes (analysis/early-reach-<before>-per-recording.npz).

Measured for both, with the same rule for where a list of 50 targets closes
the gripper (early_reach_check.closing_index), on the robot model, nothing
simulated forward:
  the gripper tip at the target where the model's answer closes, against the
  tip at the target where the demonstrator's own next 50 commands close, and
  the same for the last of the 50 targets: as one distance on the table, as
  one distance with the height, and split into along the reach, across it
  and up. "Along the reach" is the line from the robot base through the
  demonstrator's tip on the simulated side and through the red cube on the
  stacking side; the distances do not depend on that choice.

Writes into analysis/:
  pick    own-training-pictures-sim-cube20-<before>-pictures.npz, -top.png, -wrist.png
  ask     own-training-pictures-sim-cube20-<before>-answers.npz
  report  own-training-pictures-<before>.json, own-training-pictures-<before>-per-episode.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from release_aim_check import ANALYSIS, ANSWER_LENGTH

SCENE = "cube20"
TASK_MODE = "pick-place-cube20"
EPISODES = 38
TRIAL_EPISODES = 2
STACKING_ANSWERS = "real top + real wrist"
# Half the edge of the 20 mm cube that both tasks grip.
WITHIN_MM = 10.0
SHEET_PICTURES = 12


def new_file(name: str) -> Path:
    """Path of a result file in analysis/ that does not exist yet."""
    path = ANALYSIS / name
    if path.exists():
        raise SystemExit(f"{path} exists; it is not written again")
    return path


def stem(before: str) -> str:
    return f"own-training-pictures-sim-{SCENE}-{int(before)}"


def is_trial(word: str | None) -> bool:
    if word not in (None, "trial"):
        raise SystemExit(f"unknown word {word!r}")
    return word == "trial"


def pick(before: str, trial: str | None = None) -> None:
    """Choose the episodes; store their moment, reading, next 50 commands and pictures."""
    from wrist_roll_at_grasp import VIDEO_INDEX, sheet, sim_picture, spread_choice

    trying = is_trial(trial)
    targets = {} if trying else {
        kind: new_file(f"{stem(before)}-{kind}") for kind in ("pictures.npz", "top.png", "wrist.png")}
    index = np.load(VIDEO_INDEX)
    snapshot = Path(str(index["snapshot"]))
    sys.path.insert(0, str(snapshot / "source" / "0_dataset_gen"))
    import recipe

    data = np.load(ANALYSIS / f"states-sim-{SCENE}.npz")
    close_pct = recipe.CLOSE_PCT[recipe.SCENES[SCENE]["real_task"]]
    numbers = np.unique(data["episode"])
    usable = []
    for number in numbers:
        action = data["action"][data["episode"] == number]
        close = recipe.close_frame(action, close_pct)
        frame = None if close is None else close - int(before)
        if frame is not None and frame >= 0 and frame + ANSWER_LENGTH <= len(action):
            usable.append((int(number), frame))
    print(f"{len(usable)} of {len(numbers)} episodes have the moment and "
          f"{ANSWER_LENGTH} commands after it", flush=True)
    picks = spread_choice(
        np.array([number for number, _ in usable]), TRIAL_EPISODES if trying else EPISODES)

    rows: dict[str, list] = {
        "episode": [], "frame": [], "state": [], "demonstrated": [], "top": [], "wrist": []}
    for at in picks:
        number, frame = usable[at]
        chosen = data["episode"] == number
        rows["episode"].append(number)
        rows["frame"].append(frame)
        rows["state"].append(data["state"][chosen][frame])
        rows["demonstrated"].append(data["action"][chosen][frame:frame + ANSWER_LENGTH])
        for camera in ("top", "wrist"):
            rows[camera].append(sim_picture(index, snapshot, SCENE, camera, number, frame))
        print(f"episode {number}, frame {frame}: reading "
              f"{np.round(rows['state'][-1], 1).tolist()}", flush=True)
    if trying:
        print(f"trial: {len(picks)} episodes picked, pictures "
              f"{rows['top'][0].shape} and {rows['wrist'][0].shape}, nothing written", flush=True)
        return
    np.savez_compressed(
        targets["pictures.npz"], sentence=np.array(recipe.SCENES[SCENE]["task_text"]),
        **{name: np.array(values) for name, values in rows.items()})
    shown = np.round(np.linspace(0, len(picks) - 1, SHEET_PICTURES)).astype(int)
    for camera in ("top", "wrist"):
        sheet([rows[camera][at] for at in shown],
              [f"ep {rows['episode'][at]}  frame {rows['frame'][at]}" for at in shown],
              targets[f"{camera}.png"])
    print(f"wrote {targets['pictures.npz']}", flush=True)


def ask(before: str, trial: str | None = None) -> None:
    """The model's 50 targets for every picked moment."""
    import time

    from compare_real_vs_sim_images import predict as answer
    from so101_smolvla.policy_env import PolicyObservation
    from so101_smolvla.sapien_config import task_spec
    from so101_smolvla.sapien_rollout import _load_policy

    trying = is_trial(trial)
    target = None if trying else new_file(f"{stem(before)}-answers.npz")
    stored = np.load(ANALYSIS / f"{stem(before)}-pictures.npz")
    text = task_spec(TASK_MODE).instruction
    if text != str(stored["sentence"]):
        raise RuntimeError(
            f"the product's sentence {text!r} is not the data set's {str(stored['sentence'])!r}")
    print(f"task sentence: {text!r}", flush=True)
    checkpoint, policy, pre, post = _load_policy()
    print(f"checkpoint revision {checkpoint.name}", flush=True)

    count = TRIAL_EPISODES if trying else len(stored["episode"])
    started = time.monotonic()
    answers = []
    for at in range(count):
        observation = PolicyObservation(
            state=stored["state"][at].astype(np.float32),
            top=stored["top"][at], wrist=stored["wrist"][at])
        answers.append(answer(policy, pre, post, observation, text, 0))
        print(f"episode {int(stored['episode'][at])} answered", flush=True)
    print(f"{count} answers, {(time.monotonic() - started) / count:.2f} s per answer", flush=True)
    if trying:
        print(f"trial: answers of shape {np.array(answers).shape}, nothing written", flush=True)
        return
    np.savez_compressed(
        target, episode=stored["episode"], frame=stored["frame"], answer=np.array(answers))
    print(f"wrote {target}", flush=True)


def bin_task_robot():
    """The robot model of the simulated bin task: (environment, tip in mm of real-units joint numbers, base in mm)."""
    import torch
    from so101_sim.lerobot_env import So101SimEnv
    from so101_smolvla.sapien_config import environment_arguments

    env = So101SimEnv(**environment_arguments(TASK_MODE), episode_length=10)
    env._env.reset(seed=0)
    inner = env._env.unwrapped

    def tip(numbers: np.ndarray) -> np.ndarray:
        qpos = env._to_sim(np.asarray(numbers, dtype=np.float32))
        inner.agent.robot.set_qpos(
            torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
        return 1000.0 * np.mean(
            [finger.pose.p[0].cpu().numpy()
             for finger in (inner.agent.finger1_tip, inner.agent.finger2_tip)], axis=0)

    base = 1000.0 * inner.agent.robot.pose.p[0].cpu().numpy()
    return env, tip, base


def offsets(model: np.ndarray, demonstrator: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Model's tip minus the demonstrator's: (along the line from the base through
    the demonstrator's tip, across it to the left, up), one row per episode, mm."""
    outward = demonstrator[:, :2] - base[:2]
    outward /= np.linalg.norm(outward, axis=1)[:, None]
    leftward = np.column_stack([-outward[:, 1], outward[:, 0]])
    off = model - demonstrator
    return np.column_stack(
        [(off[:, :2] * outward).sum(axis=1), (off[:, :2] * leftward).sum(axis=1), off[:, 2]])


def summary(off: np.ndarray) -> dict:
    """What a set of offsets (along, across, up) amounts to."""
    flat = np.hypot(off[:, 0], off[:, 1])
    whole = np.linalg.norm(off, axis=1)

    def middle(values: np.ndarray) -> list[float]:
        return [round(float(v), 1) for v in np.percentile(values, [25, 50, 75, 90])]

    return {
        "answers": len(off),
        "quarter_median_three_quarters_ninety_percent": {
            "distance_on_the_table_mm": middle(flat),
            "distance_with_height_mm": middle(whole),
            "along_the_reach_mm": middle(off[:, 0]),
            "across_the_reach_mm": middle(off[:, 1]),
            "up_mm": middle(off[:, 2]),
        },
        "within_10_mm_on_the_table": int((flat <= WITHIN_MM).sum()),
        "within_10_mm_with_height": int((whole <= WITHIN_MM).sum()),
    }


def simulated_side(before: str) -> dict[str, np.ndarray]:
    """Offsets and closing targets of the simulated episodes, read on the bin task's robot model."""
    from early_reach_check import closing_index

    stored = np.load(ANALYSIS / f"{stem(before)}-pictures.npz")
    answers = np.load(ANALYSIS / f"{stem(before)}-answers.npz")
    if answers["episode"].tolist() != stored["episode"].tolist():
        raise RuntimeError("the answers are not of these pictures")
    lists = {"demonstrated": stored["demonstrated"], "answer": answers["answer"]}
    when = {name: [closing_index(chunk) for chunk in chunks] for name, chunks in lists.items()}
    env, tip, base = bin_task_robot()
    try:
        # A list that never closes is posed at its last target and not counted.
        at_closing = {
            name: np.array([tip(chunk[index if index is not None else -1])
                            for chunk, index in zip(lists[name], when[name])])
            for name in lists}
        last = {name: np.array([tip(chunk[-1]) for chunk in chunks])
                for name, chunks in lists.items()}
        now = np.array([tip(reading) for reading in stored["state"]])
    finally:
        env.close()
    print(f"simulated side: robot base at {np.round(base, 1).tolist()} mm; the demonstrator's "
          f"tip where it closes is {np.round(np.median(at_closing['demonstrated'], axis=0), 1).tolist()} "
          f"mm in the median, {np.round(np.median(np.linalg.norm(at_closing['demonstrated'] - now, axis=1)), 1)} "
          f"mm from the tip at the moment asked", flush=True)
    closes = {name: np.array([-1 if index is None else index for index in when[name]])
              for name in when}
    return {
        "episode": stored["episode"],
        "closing": offsets(at_closing["answer"], at_closing["demonstrated"], base),
        "last": offsets(last["answer"], last["demonstrated"], base),
        "demonstrator_closes": closes["demonstrated"],
        "model_closes": closes["answer"],
    }


def stacking_side(before: str) -> dict[str, np.ndarray]:
    """The same of the real stacking recordings, from the stored result of early_reach_check.py.

    The stored places are (past the red cube, left of it, above the table)
    for the human's commands and for the model's answer, so their difference
    is the model's tip minus the human's along those three lines.
    """
    kept = np.load(ANALYSIS / f"early-reach-{int(before)}-per-recording.npz")
    places = "{}: {}, past / left / above table mm"
    return {
        "episode": kept["episode"],
        "closing": kept[places.format(STACKING_ANSWERS, "at closing")]
        - kept[places.format("human", "at closing")],
        "last": kept[places.format(STACKING_ANSWERS, "last target")]
        - kept[places.format("human", "last target")],
        "demonstrator_closes": kept["human: closing target"],
        "model_closes": kept[f"{STACKING_ANSWERS}: closing target"],
    }


def report(before: str) -> None:
    targets = {kind: new_file(f"own-training-pictures-{int(before)}{kind}")
               for kind in (".json", "-per-episode.npz")}
    sides = {"simulated small cube into the bin": simulated_side(before),
             "real stacking": stacking_side(before)}
    result: dict = {"frames_before_the_demonstrator_closes": int(before)}
    per_episode: dict[str, np.ndarray] = {}
    for name, rows in sides.items():
        both = (rows["demonstrator_closes"] >= 0) & (rows["model_closes"] >= 0)
        later = (rows["model_closes"] - rows["demonstrator_closes"])[both]
        result[name] = {
            "episodes": len(both),
            "demonstrator_closes_within_its_50_commands": int((rows["demonstrator_closes"] >= 0).sum()),
            "demonstrator_closes_at_command_median": float(np.median(
                rows["demonstrator_closes"][rows["demonstrator_closes"] >= 0])),
            "model_closes_within_its_50_targets": int((rows["model_closes"] >= 0).sum()),
            "both_close": int(both.sum()),
            "model_closes_this_many_targets_after_the_demonstrator_quarter_median_three_quarters":
                [float(v) for v in np.percentile(later, [25, 50, 75])] if both.any() else None,
            "tip_where_the_model_closes_minus_where_the_demonstrator_closes":
                summary(rows["closing"][both]) if both.any() else None,
            "tip_at_the_last_target_minus_the_demonstrators": summary(rows["last"]),
        }
        short = "sim" if name.startswith("simulated") else "stacking"
        per_episode[f"{short}: episode"] = rows["episode"]
        per_episode[f"{short}: at closing, along / across / up mm"] = rows["closing"]
        per_episode[f"{short}: last target, along / across / up mm"] = rows["last"]
        per_episode[f"{short}: both close"] = both
        per_episode[f"{short}: demonstrator's closing target"] = rows["demonstrator_closes"]
        per_episode[f"{short}: model's closing target"] = rows["model_closes"]
    print(json.dumps(result, indent=1), flush=True)
    targets[".json"].write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    np.savez(targets["-per-episode.npz"], **per_episode)
    print(f"wrote {targets['.json']}", flush=True)


if __name__ == "__main__":
    {"pick": pick, "ask": ask, "report": report}[sys.argv[1]](*sys.argv[2:])
