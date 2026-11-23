# SO-101 SmolVLA digital twin

Standalone project. The demo is a simulated SO-101 arm whose motion comes from a fine-tuned SmolVLA policy. There is no physical arm in this project.

Implementation starts at Stage 1 only after this plan is accepted. Complete one stage, run its acceptance checks, report results, and stop for review.

## Decisions already made

- The product is the MuJoCo digital twin. A later real-arm port is out of scope.
- The demo policy is SmolVLA fine-tuned on demonstrations recorded in this scene. `lerobot/smolvla_base` is the weight initialization and the baseline, not the demo.
- Demonstrations come from a scripted expert that may read the simulator's cube pose. The learned policy never receives that pose.
- Two cameras: a fixed workspace view and a wrist view. The third SmolVLA camera slot stays empty (`empty_cameras=1`). The pinned Menagerie XML is checked for a wrist `<camera>`. If it only has the camera-mount mesh, this project's scene adds the wrist camera on that mount.
- Arm state and action are hinge degrees. The gripper is 0–100, closed to open. LeRobot's processor owns normalization. This project does not apply a second normalization.
- `n_action_steps` is 1, and `reset()` runs at each episode start. The control process never calls the model. It applies the latest action, or holds the last target when inference has not finished. Training still uses the checkpoint `chunk_size`.
- ROCm 10.1.0 is installed with `uv pip` into this directory's `.venv` from `https://stable.repo.amd.com/rocm/whl-next/`. The system ROCm 7.2.4 install stays. This machine is gfx1151, so the extra is `device-gfx1151`. Pins: `rocm[libraries,device-gfx1151]==10.1.0`, `torch[device-gfx1151]==2.14.0+rocm10.1.0`, `torchvision[device-gfx1151]==0.29.0a0+rocm10.1.0`, `torchaudio==2.11.0.3+rocm10.1.0`. The environment check reads those venv packages. It does not read `/opt/rocm/.info/version`. Official pages: https://rocm.docs.amd.com/en/docs-10.1.0/install/rocm.html and https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?rocm-ver=10.1.0
- This directory does not import, link, or mention any other path in the monorepo. Upstream URLs are allowed: MuJoCo Menagerie `robotstudio_so101`, Hugging Face LeRobot, and `lerobot/smolvla_base`.

## What this project owns

```text
control process, one CPU core
  physics + expert IK, 16 steps of 1/480 s, budget 1/30 s wall clock
  publish latest joint state
        | ping-pong state slot
render process, one CPU core, OSMesa
  top + wrist at 640x480 from that state
  publish one observation
        | ping-pong observation slot, drop the older frame
inference process, iGPU via ROCm, host threads unpinned
  preprocess -> select_action (n_action_steps 1) -> postprocess
        | ping-pong action slot
control applies the new target, or holds the last one
        | ping-pong record slot, drop the older frame
record process, one CPU core
  video, dataset, timing log
```

Physics stays on a CPU core. Cameras are rendered on another CPU core with OSMesa, so the iGPU stays free for SmolVLA. Inference uses the PyTorch build for ROCm 10.1, selected as `torch.cuda` / `--policy.device=cuda`. A CPU forward pass is a diagnostic. The recorded demo device is that ROCm 10.1 device when Stage 1 shows that the policy runs there.

## Joint contract

Order, for `observation.state` and `action`:

1. `shoulder_pan`
2. `shoulder_lift`
3. `elbow_flex`
4. `wrist_flex`
5. `wrist_roll`
6. `gripper`

The five arm values are degrees of the MJCF hinge of the same name (`<name>_joint`), sign +1, offset 0. The Menagerie model is the mid-range calibration, so those degrees match LeRobot's `use_degrees=True` body convention. Hinge limits are read from the vendored XML at runtime and tested. They are not copied into a second table.

The gripper value is 0 when closed and 100 when open, the LeRobot `RANGE_0_100` convention. It maps linearly onto the gripper hinge range read from that XML. Stage 2 measures which hinge end closes by closing the jaw on the cube, and that measurement sets the 0 end and the 100 end in `configs/sim.toml`.

The adapter converts the five arm values with `deg2rad` and the gripper with the 0–100 map. A test feeds a known mixed vector, including a gripper value such as 15, and checks that the gripper command is the interpolated hinge angle. Applying `deg2rad` to all six values fails that test.

Actions are absolute positions. Stage 1 reads `use_delta_joint_actions_aloha` from the loaded config. A true value means a delta action head and stops the project until this contract is rewritten.

## Camera and language contract

SmolVLA's robotics pretraining used real community demonstrations, not synthetic robot trajectories. The [SmolVLA blog](https://huggingface.co/blog/smolvla) and [paper](https://arxiv.org/pdf/2506.01844) describe about 481 to 487 SO-100 datasets, about 10 million frames, collected in labs and homes. Cameras were not recorded with one shared extrinsics file. Hugging Face manually labeled each view as top (`camera1`), wrist (`camera2`), or an extra side view (`camera3`). LIBERO and Meta-World appear only as later evaluation fine-tunes. The paper states the base model was not pretrained on SO-101 recordings. The SO-101 paper evaluation used a top camera and a side camera. The SO-100 pick-place set that the fine-tune guide names, [lerobot/svla_so100_pickplace](https://huggingface.co/datasets/lerobot/svla_so100_pickplace), uses a top camera and a wrist camera at 640×480 and 30 fps, and the task text is `Pick up the cube and place it in the box.`

No published number gives the camera's xyz or orientation. The closest mechanical standard is The Robot Studio's overhead mount, fixed to the arm base so a 640×480 webcam sees the whole workspace, and the wrist module screwed to the jaw replacement so it looks at the fingertips. Stage 2 vendors those mount meshes, measures the camera seat from the mesh, and writes that pose into `configs/sim.toml`. Until that measurement exists, the scene does not invent a camera position.

| Our key | Policy key | Source |
|---|---|---|
| `observation.images.top` | `observation.images.camera1` | overhead camera on the base mount, looking down at the table |
| `observation.images.wrist` | `observation.images.camera2` | camera on the jaw mount, looking at the fingertips |
| (absent) | `observation.images.camera3` | `empty_cameras=1` |

Rendered frames are RGB, 640×480, uint8. The LeRobot processor resizes and normalizes. This project does not resize to 224, 256, or 512 on its own.

The world is an indoor tabletop: a light matte table, a small cube, and an open box as the place target. Lighting is a diffuse ceiling source with soft shadows, in the range of a lab or living room. The scene does not use a dark studio, outdoor sun, or scanned household props. One task string is used for recording, training, and evaluation: `Pick up the cube and place it in the box.`

`configs/contract.yaml` is the only rename map. Recording and evaluation both load it.

## Control timing

After the model loads, its timestep is set to `1/480` s. Each control tick runs exactly 16 `mj_step` calls, which is `1/30` s of simulation. The dataset `fps` is the integer 30, and frame timestamps are `k/30`. LeRobot checks consecutive timestamps to within `1e-4` s of `1/fps`, so a 0.032 s tick must not be labeled 30 fps. Position-actuator gains stay as vendored. The timestep change is recorded next to the Menagerie commit pin. There is no sleep in the learner or the evaluator.

The stack is four processes so the Ryzen AI CPU cores and the iGPU do not wait on each other. Stage 1 prints `os.cpu_count()`. The machine must have at least four cores. Control, render, and record are each pinned to one core after that count is known. The core numbers are not hardcoded. The inference process is left unpinned so ROCm can use the remaining host threads.

- **Control**, one CPU core. Owns the stepping `MjModel`. Runs 16 physics steps, the expert Jacobian when recording, and the actuator write. Publishes the latest joint state, simulation time, and cube pose for the evaluator. Its wall-clock budget is `1/30` s. It does not render, encode, or call the model.
- **Render**, one CPU core. Owns a second `MjModel` used only to draw. Copies the latest joint state into it and renders the workspace and wrist cameras with OSMesa. Publishes one observation: both images plus the joint state from that same copy. OSMesa keeps this work off the iGPU. If a render takes longer than `1/30` s, the newer state replaces the unconsumed one and the tick is counted. The budget is not raised.
- **Inference**, iGPU. Preprocess, `select_action`, postprocess on the ROCm device. Its budget is the Stage 1 forward-pass time. The host process is not pinned to one core.
- **Record**, one CPU core. Writes video, the dataset, and the timing log from the latest published frame. A newer frame replaces one that has not been written. This process cannot stall control.

Handoff is a shared-memory ping-pong slot per boundary, preallocated, one writer and one reader. Image slots are 640×480×3 bytes, two of them. There is no frame queue and no lock held across physics, render, or the model call.

If no new action is waiting, control applies the last position target and counts a hold tick. Logs are written only by the record process. Each episode calls `reset()` on the policy before the first `select_action`.

The in-process simulator tests in Stages 2 and 3 call the same `sim.py` functions without spawning these four processes. Stage 4 is the first run of the four-process path.

SmolVLA is trained to predict a chunk (`chunk_size` from the checkpoint, 50 on the published base config). At runtime this project sets `n_action_steps` to 1, so each inference returns one action. The published default of 50 would hold one observation for about 1.7 s inside the policy. That default is overwritten. Training still uses the checkpoint `chunk_size`. The control process does not add a second queue.

The expert's Jacobian solve runs in the control process and must finish inside the same `1/30` s wall-clock budget. Policy episodes report how many ticks repeated the last target while inference was in flight. Expert episodes are one new target per tick.

Stage 5 recording and Stage 6 training are long jobs. Before either launch, name one tmux session, one log path, one exit file, and one `tmux wait-for` event. Completion is that event plus the exit file. A sleep or a poll of the log is not completion.

## Expert

The expert exists to fill a LeRobot dataset. It is not a second demo controller.

Privileged inputs, expert only: cube position and the fixed target position. Outputs: the same action vector the policy emits (five arm degrees and a gripper value in 0–100).

The solver is damped least squares on the MuJoCo gripper-site Jacobian. It tracks the site position and a preferred tool-down orientation, with orientation weighted below position. A position-only target can leave the jaws pointing away from the cube.

Sequence:

1. Open the gripper.
2. Approach from above along a chosen azimuth.
3. Descend to the grasp.
4. Close the gripper.
5. Lift.
6. Move above the target zone.
7. Descend.
8. Open.
9. Retreat.

An episode is kept only when the grasp-hold check below passes and the place check passes. Failed attempts are discarded and counted.

Training set: 5 fixed cube positions in the reachable workspace, 10 kept episodes each. The ten takes at one position use different approach azimuths, plus ±1 cm XY jitter. Hugging Face's 10 repeats worked because a person varied the motion; a repeated spline does not. Held-out evaluation uses 20 episodes whose seeds and positions are disjoint from those 50. The held-out list is written down before recording and is never passed to the recorder.

If the expert cannot produce 50 kept episodes, stop. Do not train on a short or failed set.

## Simulator tests

Automated tests assert on MuJoCo state. They do not judge the grasp from camera pixels. `sim.py` owns the grasp-hold reading, and both `tests/test_sim.py` and the expert keep-filter call it.

The reading has two facts from `MjData` after `mj_step`:

- Contact between the jaw collision geoms and the red cube.
- Gripper actuator force. The Menagerie model has no current sensor. Actuator force is the stand-in for gripper current.

The design rules in `~/GitHub/claude_code_embedded_dev/design-rules.md` forbid a sleep whose only purpose is to let time pass, and they forbid timeout-and-retry loops. This grasp-hold test is the stated exception, because the thing under test is whether a grasp survives a duration. The wait is simulated time, realized as one fixed step count.

Stage 2 tests the reading on the simulator actuator API: an empty close, and a close that touches the cube. The full pick-then-hold test is `tests/test_sim.py` once the adapter exists. It uses that adapter, the same command path as the demo, in one episode. It does not write the cube pose into the jaws, and it does not reset between the lift and the hold.

1. Command the gripper closed with no cube between the jaws and record that actuator force. That is the empty-close baseline.
2. Pick up the red cube through the adapter. Assert jaw-cube contact, and assert the gripper actuator force is above the empty-close baseline.
3. Hold the close command and advance the simulator by exactly 3.0 s, which is 1440 steps at timestep `1/480` s. This is not `time.sleep`, and it is not a loop that retries until the cube settles.
4. Read MuJoCo again. Jaw-cube contact is still present, and the cube center is still above the table by the lift margin in `configs/sim.toml`.

An exception in this test fails it. A missed contact is a failed assertion, not a skipped test.

A second case closes on empty space. Actuator force stays at the empty-close baseline and there is no jaw-cube contact.

The policy observation never includes actuator force, contacts, or the cube pose.

## Evaluation

The evaluator reads the cube free-joint pose. That pose is not part of the policy observation, and the policy module has no parameter that turns it on.

An episode records:

- lift: cube center rises above the table by a margin written in `configs/sim.toml`
- place: cube center is inside the open box, below the lift height, and the gripper command is open
- success: lift then place in that order
- inference time per `select_action`, and control-tick rate

The comparison runs the base checkpoint and the fine-tuned checkpoint on the same 20 seeds. Both runs use `n_action_steps: 1`. The headline number is the fine-tuned place rate. The base rate is reported beside it. A low base rate is an expected result.

## Isolation

```text
so101-smolvla/
├── PLAN.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── contract.yaml          # checkpoint facts plus the project choices below
│   └── sim.toml               # timestep, success geometry, gripper ends, expert positions
├── assets/
│   ├── robotstudio_so101/     # vendored Menagerie model, scenes, and meshes
│   └── scene.xml              # red cube, target, workspace camera, wrist camera if needed
├── src/so101_smolvla/
│   ├── contract.py
│   ├── sim.py
│   ├── adapter.py
│   ├── expert.py
│   ├── policy.py
│   ├── record.py
│   ├── evaluate.py
│   └── run_demo.py
├── scripts/
│   ├── check_environment.py
│   ├── fetch_robot.sh
│   ├── run_demo.sh
│   └── train.sh
└── tests/
    ├── test_adapter.py
    ├── test_sim.py
    └── test_expert.py
```

Python imports resolve inside `so101_smolvla` or third-party packages. Shell scripts use paths relative to this directory. README setup instructions clone or install upstream projects into this environment. They do not point at another directory of this monorepo.

Licenses to record in the README after Stage 1: Menagerie SO-101 (Apache-2.0), LeRobot, and the SmolVLA weight license as published on the Hub.

## Stage 1 result

The system stack on this machine is ROCm 7.2.4 at `/opt/rocm`. That file is not the Stage 1 gate. ROCm 10.1 userspace is installed into `.venv` and leaves the system stack alone.

## Stage 1 — Contract and environment

1. Create `.venv` with `uv` and Python 3.12. Install, in order, from `https://stable.repo.amd.com/rocm/whl-next/`: `rocm[libraries,device-gfx1151]==10.1.0`, then `torch[device-gfx1151]==2.14.0+rocm10.1.0`, `torchvision[device-gfx1151]==0.29.0a0+rocm10.1.0`, and `torchaudio==2.11.0.3+rocm10.1.0`. Then install the LeRobot `smolvla` extra with a constraint file that keeps those pins. Do not run `apt`, `amdgpu-install`, or any other command that changes the system ROCm 7.2.4 packages. Run `python -m rocm_sdk test` in the venv. Record the versions that import. The Stage 1 forward-pass time is written into `configs/contract.yaml` as the inference budget.
2. `scripts/fetch_robot.sh` downloads MuJoCo Menagerie `robotstudio_so101` at a pinned commit, verifies file hashes, and writes `assets/robotstudio_so101/`. Record the timestep override (`1/480` s) next to that pin.
3. Load `lerobot/smolvla_base` through the installed LeRobot API. Write `configs/contract.yaml` in two groups.
   - From the checkpoint: state shape, action shape, camera keys, resize, normalization mapping, `chunk_size`, and the absolute-versus-delta flag.
   - Project choices: the unit map above, `n_action_steps: 1`, `empty_cameras: 1`, and the rename map.
4. Stop the stage only when the checkpoint disagrees with a fact it owns: state width 6, action width 6, absolute actions, image keys `camera1` / `camera2` / `camera3`, or a processor that does not resize. A loaded `n_action_steps` of 50 is overwritten by the project choice. It is not a mismatch.
5. `scripts/check_environment.py` runs with `.venv/bin/python`. It prints Python, MuJoCo, LeRobot, the venv `rocm` version, torch, `torch.version.hip`, the GPU name, the gfx target, and `os.cpu_count()`. It runs one SmolVLA forward pass on `torch.cuda` when that device is available, otherwise on CPU, and prints which one ran. Fewer than four CPU cores stops the stage. A venv whose torch build is not `rocm10.1` stops the stage.

Acceptance: a clean environment imports; the venv reports ROCm 10.1 and the gfx1151 torch build; the system `/opt/rocm` version is unchanged; the contract file contains the checkpoint facts and the project choices; one forward pass completes and names its device. A missing venv wheel does not become a CPU pass.

## Stage 2 — Scene and simulator

1. Start from the vendored Menagerie `scene.xml` and `scene_box.xml`. Add a light matte table, a small cube, and an open box. Vendor the SO-ARM100 overhead-mount and wrist-mount meshes, measure the camera seat on each mesh, and write those poses into `configs/sim.toml`. The overhead camera looks down across the workspace. The wrist camera sits on the jaw mount and looks at the fingertips. Keep a wrist `<camera>` from `so101.xml` when its seat matches that mount. Otherwise replace it.
2. Implement reset, the 16-step tick at timestep `1/480` s, camera rendering, joint readback in the unit contract, and position-actuator commands in radians.
3. Measure which gripper hinge end closes, and store the 0 and 100 ends in `configs/sim.toml`.
4. Tests: each of the six actuators moves toward a commanded target and stays inside the XML range; both images are 640×480×3; the cube moves when the jaw pushes it; the same seed resets to the same joint and cube state; 16 steps consume exactly `1/30` s of simulation time and finish in under `1/30` s of wall clock. The empty-close and cube-contact readings are tested on the actuator API. The full 3.0 s pick-then-hold test is Stage 3, through the adapter.

Acceptance: those tests pass. A viewer is optional and is not part of acceptance.

## Stage 3 — Adapter and loop

1. Build the observation dict from the two cameras, the five arm degrees, the gripper 0–100 value, and the task string.
2. Convert a postprocessed action with the split map. Reject non-finite actions. Clamp to the XML ranges after conversion.
3. Run one dummy action per 16-step tick. The dummy sequence moves two arm joints and the gripper.
4. The control side of the later policy loop applies the latest action or holds the last target. This stage's dummy sequence does not build a queue. In the same episode as the lift, the grasp-hold test advances 3.0 s and checks that the cube is still held.

Acceptance: tests cover a known arm degree, gripper value 15 mapped to the interpolated hinge angle, joint order, image keys, and clamping. The dummy sequence produces the commanded motion and synchronized frames. The grasp-hold test passes through the adapter with no reset between lift and hold.

## Stage 4 — Base policy, integration only

1. Load the base checkpoint and its preprocess and postprocess pipelines. Apply `configs/contract.yaml`, including `n_action_steps: 1` and the rename map.
2. `scripts/run_demo.sh --dry-run` checks shapes and writes one predicted action per observation without stepping physics.
3. One episode runs the four processes. An exception fails the episode. Grasp success is recorded and is not the gate. The timing report lists control time, render time, inference time, each budget, and the number of hold ticks. The record process writes that report.

Acceptance: the episode finishes with no device, shape, or processor error. A control, render, or inference overrun fails the timing check. Grasp success is not the gate. Save the log.

## Stage 5 — Demonstrations

1. Implement the oriented expert, the per-take approach azimuth, and the keep-or-discard check (jaws closed on the cube, then the place check).
2. Write the held-out seed list to `configs/sim.toml` before recording.
3. Record 50 kept training episodes into a local LeRobot dataset at `fps: 30` with timestamps `k/30`. Validate episode bounds, timestamp spacing, camera keys, the mixed unit vector, and the task string.
4. A test runs one expert episode and requires a successful place.

Acceptance: 50 kept episodes, 5 positions × 10, with more than one approach azimuth among the takes at a position, and a dataset report. Training is blocked until this passes.

## Stage 6 — Fine-tune and comparison

1. Fine-tune with LeRobot's trainer from `lerobot/smolvla_base`, local dataset root, `push_to_hub` off, `policy.device=cuda`, the contract rename map, and `empty_cameras=1`. Train the action expert with the vision encoder frozen. Run 20 000 optimizer steps. Set `scheduler_decay_steps` to that same step count.
2. Use micro-batch 64 when a forward-and-backward step fits in GPU memory. Otherwise lower the micro-batch and set gradient accumulation so the effective batch stays 64. Write the micro-batch and accumulation that ran. If micro-batch 1 does not fit, stop. Do not train on CPU.
3. Evaluate base and fine-tuned checkpoints on the same 20 held-out seeds, both with `n_action_steps: 1`.
4. Write per-episode video, seed, checkpoint revision, timing, and outcome. Write a summary with lift rate, place rate, success rate, and inference latency for both checkpoints.
5. `scripts/run_demo.sh` loads the fine-tuned checkpoint, runs one episode, and writes a video plus the outcome JSON.

Acceptance: both checkpoints have a full outcome trace on the same seeds. The README quotes the measured rates, the effective batch, and the command. Improvement is whatever the table shows.

If the fine-tuned held-out place rate is indistinguishable from the base rate, stop and report the dataset and the training log. Do not raise the 20 000 optimizer steps inside this stage.

## Definition of done

```bash
./scripts/run_demo.sh
```

That command resets the SO-101 and the cube, loads the fine-tuned SmolVLA checkpoint, runs a closed-loop pick-and-place attempt at 30 Hz with one `select_action` per tick, and writes a video and an outcome report. On a supported AMD GPU the forward pass uses the PyTorch build for ROCm 10.1 through `torch.cuda`. The README's comparison table is from the Stage 6 run.

## First implementation instruction

Implement Stage 1 only. Inspect the installed LeRobot interfaces and the Hub checkpoint before writing `configs/contract.yaml`. Report the checkpoint facts, the project choices, the tested versions, the ROCm version, whether the ROCm 10.1 torch build survived installation, the forward-pass device, and any checkpoint-owned mismatch. Stop before Stage 2.
