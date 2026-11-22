# SO-101 SmolVLA digital twins

Two separate simulated SO-101 environments evaluate a SmolVLA checkpoint
fine-tuned in the upstream SAPIEN training environment.

## Models evaluated: what works and what does not

Simulation only. Every result is from the published checkpoint as
downloaded; nothing was fine-tuned here. The counts come from the run folders
under `vla_testing/`. Test details are in `TESTING_LEDGER.md`, the reasons in
`DEVELOPMENT_LEDGER.md`. State of 2026-11-20. The two twins (SAPIEN and
MuJoCo) are described under "SAPIEN policy backend" below.

| Model | Use case and command sentence | Tested in | Result | Verdict |
|---|---|---|---|---|
| `Harrysunshine/so101-smolvla-sim-real-10task` | Put the 40 mm cube into the bin: `Pick up a cube and place in the bin` | SAPIEN twin, task `pick-place`, 10 random layouts | 10 of 10 (`random-sapien-20261120-090344`) | Works |
| | Put the 20 mm cube into the bin: `Pick up a small cube and place in the bin` | SAPIEN twin, task `pick-place-cube20`, 10 random layouts | 8 of 10 (`random-sapien-cube20-20261119-113801`) | Works |
| | Put the 40 mm cube into the bin, same sentence as the first row | MuJoCo twin, 10 random layouts | 4 of 10 in the bin, 7 of 10 picked up; the arm visibly jitters (`random-MuJoCo-20261118-221817`) | Unreliable |
| | Stack the red 20 mm cube on the 40 mm cube: `Stack the smaller cube on the larger one` | SAPIEN twin, task `stack2-recording-rig` (the scene rebuilt from the real training recordings), three runs of 20 random layouts | 24 of 60 with the cube placement in the product, cube centres at least 40 mm apart (10, 7 and 7 of 20; the 10 repeated in `random-stacking-recording-rig-sapien-20261120-180023`). A placement that kept the cubes at least 73.3 mm apart, tried on 2026-11-20 and taken out again, gave 17 of 60 (6, 5 and 6) | Unreliable |
| | Same stacking sentence | SAPIEN twin with the simulator's own camera views (tasks `stack2-training`, `stack2-sim-view`), 10 random layouts per run | 0 to 1 of 10 per run | Does not work |
| | Choosing one of several objects by the sentence | SAPIEN scenes with two cubes, with the bin sentence and with the stacking sentence | A second cube pulls the gripper 30 to 44 % of the way towards itself, whichever sentence is given; on 5 of 10 layouts the red cube was never lifted in nine runs | Does not work |
| | Three cubes chosen by colour (task `stack3-color`, for example `put green cube on red cube`) | – | Never run | Not tested. The training set has no scene in which the sentence chooses between objects |
| | Can into the bin (trained in simulation, not set up here). Battery, eraser, golf, medicine bottle and plush toy into the bin, and cube onto can (trained on the real robot only) | – | No scene in this workshop | Not tested |
| `lerobot/smolvla_base` (the base model, not fine-tuned) | Stack red on blue: `Pick up the red cube and put it on top of the blue cube.` | MuJoCo twin, one probe of 30 control steps on 2026-11-17 | No jaw contact with the cube and the cube not moved when the probe stopped (`scripted_testing/base_probe/result.json`); the file holds no result for the bin task | Did not work in this one probe |

What follows for a demo:

- Put one pickup object on the table and give its sentence exactly as in the
  table. The stacking pair (red 20 mm cube and 40 mm cube) is the only
  trained scene with two objects.
- Choosing between objects by the sentence is a known limit of this class of
  models, not a fault of the twin. The measurements and the published
  sources are in `DEVELOPMENT_LEDGER.md`, section ["Several objects in one
  scene"](DEVELOPMENT_LEDGER.md#d-2026-11-20-several-objects-in-one-scene).

Compared from their model cards only and not run here:
`Project-IRA/TPSoSe2026_SmolVLA_LeRobot_SO101_Finetuning_V6_Full`,
`io-intelligence/smolvla_so101_stack_cups` and
`dobri420/smolvla-so101-pick-cube`.

## Environment

Run these commands from this directory. The system ROCm (Radeon Open Compute) install is left as it is. This machine's integrated GPU is gfx1151. TorchCodec, LeRobot's video decoder, needs the system `ffmpeg` package.

The pins come from the [ROCm 10.1.0 pip install](https://rocm.docs.amd.com/en/docs-10.1.0/install/rocm.html) and the [PyTorch for ROCm 10.1.0](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?rocm-ver=10.1.0) page for gfx1151. The constraint file keeps those wheels when LeRobot is installed.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "rocm[libraries,device-gfx1151]==10.1.0" \
  "torch[device-gfx1151]==2.14.0+rocm10.1.0" \
  "torchvision[device-gfx1151]==0.29.0a0+rocm10.1.0" \
  "torchaudio==2.11.0.3+rocm10.1.0"

.venv/bin/python - <<'PY' > constraints.txt
import importlib.metadata as m
for pkg in ("torch", "torchvision", "torchaudio", "numpy", "triton"):
    print(f"{pkg}=={m.version(pkg)}")
PY

export UV_OVERRIDE="$PWD/constraints.txt"
export PIP_CONSTRAINT="$PWD/constraints.txt"
uv pip install --python .venv/bin/python "lerobot[smolvla,training]"

.venv/bin/python -m rocm_sdk test
.venv/bin/python -c "import torch, torchvision; assert torch.version.hip; assert '+rocm' in torchvision.__version__; assert torch.cuda.is_available()"
```

## SAPIEN policy backend

MuJoCo and SAPIEN are two separate digital twins:

- The MuJoCo twin uses this repository's MJCF robot, scene, collision model,
  physics, position controller, cameras, and placement check.
- The SAPIEN twin uses the pinned upstream KIT URDF, independently constructed
  scene and collisions, PhysX physics, damped proportional-derivative
  controller, cameras, and native success check.

They share the checkpoint-facing six-value state/action convention and nominal
camera calibration. They do not share geometry, simulator state, controllers,
physics, or task state. MuJoCo remains the default. The complete original
`SO101PickPlaceCube40-v1` SAPIEN environment is available as an alternative
policy-evaluation backend:

```bash
./scripts/setup_sapien_env.sh
make random BACKEND=sapien COUNT=10 STEPS=300
make random BACKEND=mujoco COUNT=10 STEPS=300
```

Random regressions are written to timestamped `random-sapien-*` and
`random-MuJoCo-*` directories.

On this AMD machine the two engines run in separate processes:

- SmolVLA inference runs on the integrated GPU through ROCm.
- SAPIEN/PhysX physics, control, and rendering run on the CPU. SAPIEN GPU
  acceleration requires NVIDIA CUDA and is not used here.

The setup script pins `Xbotics-SO101-Sim` commit
`239b38e076387b11d5c0b0da29487edcbde5f905` in `/tmp/Xbotics-SO101-Sim` and
creates `.venv-sapien`. The SAPIEN worker owns the complete simulation and
returns the state and both 640×480 camera observations to the ROCm policy
process through a versioned binary protocol.

For explicit output and seeds:

```bash
./scripts/run_sapien_rollout.sh \
  --count 1 \
  --steps 300 \
  --environment-seed-base 0 \
  --policy-seed-base 0 \
  --output vla_testing/sapien-seed0
```

Each completed episode contains `result.json`, `trajectory.npz`,
`policy-observations.npz`, top/wrist videos, a combined `rollout.mp4`, and a
`COMPLETE.json` marker written last. The batch directory also contains a report
and accumulated videos.

## Published environment

The published LeRobot scene is four cubes. This project's demo remains the one-cube SO-101 digital twin, simulated in MuJoCo. Set `MUJOCO_GL=osmesa` so rendering stays off the integrated GPU.

```python
import os

os.environ["MUJOCO_GL"] = "osmesa"

from lerobot.envs import HubEnvConfig, make_env

envs = make_env(
    HubEnvConfig(
        hub_path="lerobot/so101-tabletop-mujoco@4684f500b94d07194f2c860314c4e27cb9349e8c",
        trust_remote_code=True,
    ),
    n_envs=1,
)
```
