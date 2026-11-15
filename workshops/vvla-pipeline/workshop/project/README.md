# Workshop project - voice→vision→language→action on Ryzen AI

This is the hands-on project for the workshop: a spoken command, Gemma 4
on the iGPU, and MoveIt on the arm. YOLO-pose and MediaPipe Hands are how
"copy my movements" sees you. Everything generic is provided by the pipeline
(`../../vla_pipeline/`); the parts where code meets the **AMD Ryzen AI
stack**, the **ROS 2 transport**, and the **pipeline's control flow** are
TODOs that you implement. Each TODO has the name and parameters of a
finished pipeline function, its answer key (`TODO_INDEX` in
`../common/feedback.py` names it). Object detection and the reach are not
TODOs. They stay in the pipeline to record a demonstrated pick for a later
VLA finetune.

## Happy path

1. **Read and run the notebooks first** (`../notebooks/`, in order:
   `01_ai_models` → `02_ros2` → `03_integration`). They teach exactly what
   each TODO needs, and each lesson's "Your turn" section has a cell that
   checks your functions.
2. **Implement component 1** (`models/npu.py`, `models/hands.py`,
   `models/llm.py`), checking yourself as you go:

       python -m vvla_workshop.selftest_models

3. **Implement component 2** (`ros/camera_client.py`, `ros/moveit_arm.py`, `ros/intent_node.py`):

       python -m vvla_workshop.selftest_ros        # over ROS 2 (required); no hardware needed

4. **Implement component 3** (`integration/dispatch.py`,
   `integration/mimic.py`, `integration/behaviors.py`):

       python -m vvla_workshop.selftest_integration    # offline - scripted pose/hands, dry-run arm

The commands run from any directory: `bootstrap.sh` installs this folder as
the `vvla_workshop.project` package.

## Check everything at once

    python -m vvla_workshop.selftest

runs all three component harnesses against synthetic backends (component 2
over ROS 2, which the workshop requires), tallies every
PASS / FAIL / TODO / SKIP into one scoreboard, and then re-runs them with
`--reference`, on the pipeline's finished functions, to prove the harness
itself is trustworthy. Read the result as: **zero FAILs means nothing you've
written so far conflicts with the pipeline - the only gap left is the TODOs
it lists.** A FAIL is different from a TODO: it is code that ran and did the
wrong thing, so fix it before moving on. Useful flags: `--strict` exits
nonzero until every TODO is done (good for "am I finished?" and CI),
`--quiet` prints just the scoreboards.

Want to watch the hardware while a notebook runs? Pop the always-on-top
resource HUD - **CPU %**, **GPU %**, and **NPU inferences/sec** (or Idle):

    ../launch_monitor.sh                              # from a shell, or in a notebook:
    from vvla_workshop.monitor import open_monitor    # open_monitor() / close_monitor()

## Layout

    models/        component 1 TODOs: the NPU session factory, MediaPipe Hands, Gemma on llama-server
    ros/           component 2 TODOs: the camera client, the intent node
    integration/   component 3 TODOs: the registry and the gauntlet, the mimic tick, the dance
    _artifacts/    what the self-tests write (created on first run)

## Rules of the game

- Every TODO stub raises `NotImplementedError` with a pointer to the
  notebook that teaches it. The harnesses report per-TODO status
  (PASS / FAIL / TODO / SKIP) with targeted hints.
- `_artifacts/` collects the proof: overlay images and round-trip snapshots.
- Two debugging lessons to keep taped to your monitor:
  - **Verify the execution provider.** Asking for the NPU isn't getting it.
  - **Mind the NPU load order.** The first model opened sizes the NPU's
    shared buffers while it stays open. Open YOLO-pose first and keep it
    open, so the second vision model can load after it.
