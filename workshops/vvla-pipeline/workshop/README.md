# Ryzen AI VVLA workshop - hands-on materials

A 3-hour workshop that takes a CS/engineering student from zero to a
running **vision -> language -> action** pipeline on an AMD Ryzen AI
APU driving a LeRobot SO-101 arm: YOLOv26s x2 on the **NPU** (VitisAI EP),
Gemma 4 E4B on the **iGPU** (llama.cpp + ROCm, one request per spoken sentence),
Silero on the **CPU**, MediaPipe Hands on the **CPU**, all wired over **ROS 2**.

The workshop depends on [Installation](../README.md#installation). That
creates `.venv`, the model files in `../models`, and the YOLO NPU compile
cache in `../cache`. The two YOLO models load from that cache. Gemma 4
and Silero are downloaded by `scripts/ensure_voice.sh`.

The digital twin is the MuJoCo bringup in `launch/so101_control.launch.py`: upstream ros2_so_arm controllers, `mujoco_ros2_control`, and `move_group`. `robot.use_ros2` starts that bringup. The arm client is `MoveItArm` (MoveItPy, a Task Constructor pick and place, the gripper action). TODOs 2.7 and 2.8 are MoveItPy `go_to_named` and `go_to_joints`. Servo calibration is `lerobot-calibrate`, which `bootstrap.sh` installs. `camera_calibration` and `easy_handeye2` are not installed, and this repo does not run calibration. The ROS driver does not read the LeRobot JSON.

## What you learn

The subject is a vision-language-action loop. You speak, Gemma 4 turns your sentence into one command about half a second after your last word, and MoveIt carries that command out on the arm. When the command is "copy my movements", YOLOv26s-pose and MediaPipe Hands read your body and your hand. The twin notebook (`notebooks/03_integration/02_digital_twin.ipynb`) is that loop.

Motion planning in the lesson is MoveIt's. YOLO object detection and the reach that turns a camera box into a grasp stay in the pipeline. They are how a demonstrated pick is recorded as training data for a later VLA finetune. The notebooks do not walk through that recorder.

## What's in the box

```
notebooks/               phase 1 - read & run these first (each is self-contained)
  01_ai_models/          NPU session factory - MediaPipe on CPU - Gemma 4 on iGPU
  02_ros2/               camera subscriber - MoveIt arm - Gemma intent node
  03_integration/        registry dispatch - the mimic composition - the live UI -
                         the digital twin (MuJoCo arm, cameras, grasp; live console over ROS 2)
common/                  PROVIDED helpers, installed as the vvla_workshop package
                         (fixtures, self-tests and their reporting, the twin over ROS 2)
project/                 phase 2 - 16 TODO stubs you implement (vvla_workshop.project)
  models/  ros/  integration/
launch_monitor.sh        always-on-top resource HUD: CPU% · GPU% · NPU inf/s | Idle
MONITOR.md               how to open/close that HUD (from a shell or a notebook)
```

The answer key is the pipeline itself, `../vla_pipeline/`: each TODO in
`project/` has the name and parameters of a pipeline function that is
finished (`TODO_INDEX` in `common/feedback.py` names it). The notebooks
run the pipeline, and a cell in each lesson's "Your turn" section imports
your functions and checks them.

**The rule that decided what's a TODO:** if code would be byte-for-byte
identical in any non-AMD, non-ROS project (letterboxing, decoders,
overlays, message codecs, grammar files), it's provided. If
it's where code meets the **Ryzen AI NPU/iGPU**, the **ROS transport**, or
the **pipeline's control flow**, it's yours.

## Student happy path

Before the steps below, install the Ryzen AI SDK and prerequisites as described in [Installation](../README.md#installation).

```bash
cd workshop
./run_notebooks.sh                        # starts the Gemma 4 server and the resource
                                          # HUD, then serves Jupyter Lab. This shell
                                          # stays here until you quit it.
```

Jupyter opens `notebooks/`. Read and run them in order: `01_ai_models`,
`02_ros2`, `03_integration`. In VS Code, open a notebook and pick the
**Python (VVLA workshop)** kernel.

`bootstrap.sh` registers that kernel with the NPU runtime, ROS 2 Jazzy and
`ros_ws/` in its `kernel.json`, so JupyterLab and VS Code start it the same
way and the `02_ros2` notebooks and section 4 of
`03_integration/02_digital_twin.ipynb` can import `rclpy`.

After the notebooks, the TODOs are in `project/`. Follow [project/README.md](project/README.md).

## Live resource monitor

A tiny always-on-top window shows the three numbers that matter while a cell
runs - **CPU %** (burgundy), **GPU %** (orange), and **NPU inferences/sec or
Idle** (cyan) - floating over the browser and everything else. `run_notebooks.sh`
opens it for you (pass `--no-monitor` to skip); to open or reopen it on its own:

```bash
./launch_monitor.sh                          # from the workshop folder
```

```python
from vvla_workshop.monitor import open_monitor, close_monitor
open_monitor()       # floats on top; close_monitor() dismisses it
```

Drag to move it; close it with its ✕, the Esc key, or a right-click. It reads
`top`, the amdgpu `gpu_busy_percent` sysfs gauge (`radeontop` fallback), and
`xrt-smi`; any tool that's missing just shows
`N/A` / `Idle`, so it runs on a plain laptop too. Details in `MONITOR.md`.

## Instructor's corner

```bash
python -m vvla_workshop.selftest --quiet     # no FAIL on the stubs, and the pipeline passes every check
```

`./run_notebooks.sh --test` runs every notebook and the command above
headless, after the same setup as the happy path. See
[Automated tests](../README.md#automated-tests).

AMD-specific lessons for the session:

**Choose the device on purpose.** YOLO-pose is the vision model this workshop
teaches, and it runs on the NPU. A second vision model shares that NPU, so
the session lesson is load order: open pose first and keep it open. Gemma 4
stays on the iGPU and Silero stays on the CPU. Pass `device="cpu"` or
`device="npu"` for a vision session, then print the provider it got.

Generated dirs (`_artifacts/`, `logs/`, `__pycache__/`) are disposable -
see `.gitignore`.
