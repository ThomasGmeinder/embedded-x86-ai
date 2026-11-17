# Twin compared with the Harrysunshine training rig

This scene is a MuJoCo twin of the arm. The pictures the checkpoint was trained on come from a different simulator. This note records what was copied from that rig, what was measured, and what is still different. The renderer reads `assets/robotstudio_so101/scene_task.xml` and `so101.xml`. `configs/sim.yaml` repeats the camera numbers so they can be read without opening the XML.

The side-by-side frame is `outputs/track/scene-match.png`. Left is episode 484, frame 13, of the training video. Right is this scene at the same recorded joint angles.

## Sources

The training code is public. It is not on a mounted NAS path.

- Environment: [Xbotics-Embodied-AI-club/Xbotics-SO101-Sim](https://github.com/Xbotics-Embodied-AI-club/Xbotics-SO101-Sim), files `so101_sim/envs.py` and `so101_sim/robots/kit_assets/kit_v1_so101.urdf`.
- Checkpoint: `Harrysunshine/so101-smolvla-sim-real-10task`.
- Dataset: `Harrysunshine/so101-sim-pickplace-v2`, config `cube40`. Snapshot `96a125d64bc3fd0d50234fad77e87a10c46020b8`. 498 episodes, robot type `so_follower`. The task string in the data is `Pick up a cube and place in the bin`.
- Published env name: `SO101PickPlaceCube40-v1`. An episode there is at most 400 control steps. Episode 484 is 296 frames, dataset rows 163155–163451. The small video `cube40/videos/.../file-001.mp4` starts at this episode.

`src/so101_smolvla/policy.py` `load_policy` still loads `lerobot/smolvla_base` with `n_action_steps` 1 and `empty_cameras` 1. The checkpoint loader in `src/so101_smolvla/track.py` is a separate function. It keeps `n_action_steps` 50 and `empty_cameras` 0.

## What matches

| | Training rig | This twin |
|---|---|---|
| Arm | KIT SO-101, URDF `kit_v1_so101.urdf` | Menagerie SO-101 meshes, with the gripper link set to the KIT wrist-roll zero |
| Cameras | 640×480, both vertical fields 59.17° | 640×480, both vertical fields 59.17° |
| Overhead mount | `top_camera_optical_frame` on `base_link`, plus a shift of (−0.002, 0.027, 0) m in the optical frame | Same position in the base: (0.049013, 0.121504, 0.528474) m |
| Wrist mount | `wrist_camera_optical_frame` on `gripper_link`, (0.0025, 0.072673, 0.005115) m | Same position in the gripper body. The old lens housing is carried with it |
| Control rate | 30 Hz | 30 Hz (`TICK_STEPS` 16 at 480 Hz) |
| Action | Absolute arm degrees, gripper 0–100 | Same, through `command` |
| Task string used by the checkpoint path | `Pick up a cube and place in the bin` | `track.py` uses that string, with no final period |
| Cube size | `cube_4`, 40 mm | Box half-size 0.02 m |
| Cube color | (0.612, 0.733, 0.820), `#9cbbd1` | Same RGBA |
| Arm color | (0.05, 0.05, 0.05) | Same |
| Table color | (0.9, 0.9, 0.9) | Same on the table top |

The overhead field of 59.17° is a lens calibration (ChArUco board, 233 frames, RMS 0.72 px). It is the vertical field of view. A separate script in that repo uses 0.54 m as the optical-axis depth from this camera to the table, to convert pixels to metres. That depth is not a camera position. The camera is the URDF mount above, tilted, not a camera at z = 0.54 m looking straight down.

## How the camera orientation was converted

SAPIEN, which ManiSkill uses, looks along +X (Y left, Z up). MuJoCo looks along −Z (X right, Y up). The URDF roll-pitch-yaw is the optical frame. Their code then applies one more rotation, `_OPTICAL_CONV`: the ROS optical frame (Z forward, X right, Y down) mapped into SAPIEN, then 180° about the optical axis so the image is upright. The MuJoCo quaternion is that composed rotation with the axes remapped. Pasting either the URDF roll-pitch-yaw or the SAPIEN quaternion into a MuJoCo camera points the lens the wrong way.

The wrist XML still shows the old local camera numbers. Those numbers are in the `camera_mount` body, and that body is shifted so the lens lands on the KIT optical frame. The pose in the gripper is the published one.

The Menagerie gripper quaternion was 90° about the wrist-roll axis away from the KIT `gripper_link`. The same joint angle then put the jaws on their side and aimed this wrist camera at the arm. The gripper body quaternion is now the KIT joint origin, `0.51202 -0.512021 0.487685 0.487682`. The other arm-joint origins already matched the URDF.

## Episode 484 object positions

The cube40 parquet stores joint state and action. It does not store the cube pose or the bin pose. The positions in the scene were read by setting each recorded joint state into this model and reading the gripper site:

- Frame 120: gripper command 1.27% (shut), site on the table at (0.302, −0.037, 0.005) m. That is the cube.
- Frame 200: jaws open after the carry, site at (0.249, 0.071, 0.066) m. That is the bin.
- Frame 13, the comparison frame: gripper site near (0.185, 0.013, 0.017) m. The state is [−5.71, −101.19, 94.51, 61.98, −2.20, 2.71] degrees and percent.

Those two positions are where the recorded hand was. They are not a log of the object. The cube in the XML is at (0.302, −0.037, 0.02) m. The bin body is at (0.249, 0.071, 0.025) m.

## What the picture still shows

`outputs/track/scene-match.png` is the same kind of shot on both sides. The overhead view looks from behind the arm toward the bin and the cube. The wrist view looks out between the fingers at the bin.

Two differences are still visible:

- The overhead training frame has the KIT base plate and the frame extrusion across the bottom. This scene has neither mesh, so that part of the view is empty table.
- The wrist training frame looks into the opening of their `bin_2` mesh, with the cube beside it. This wrist view is on the near wall of a box built from five geoms. The cube is only at the top-right edge.

A search over 1344 overhead poses peaked at a correlation of 0.46 and was not written. A wrist move that put the lens into the finger was rendered, looked worse, and was not written.

## What is still different

- Simulator. Theirs is SAPIEN through ManiSkill at 120 Hz physics and 30 Hz control. This twin is MuJoCo at 480 Hz physics and 30 Hz control.
- Episode cap. Their env stops at 400 steps. `track.py` rolls out 500 steps on this scene.
- Overhead jitter. On each of their resets the overhead camera is shifted by up to ±0.023 m in optical X and Y. This camera stays on the published mean offset.
- Bin. Theirs is the `bin_2` mesh, 80 × 100 × 30 mm. This bin is a 100 mm square box with walls about 50 mm tall, dark gray `(0.12, 0.12, 0.12)`.
- Cube mass. Their density is 200 kg/m³, so a 40 mm cube is 12.8 g. This geom mass is 0.04 kg.
- Shoulder lift. Their URDF limit is −110° to +100°. This hinge is ±110°.
- Wrist-roll range. Their URDF range is about −67° to +253°. This hinge stays at the Menagerie range, about ±160°. Episode 484 stays inside −33° to −2°, so that episode is not clipped.
- Gripper-link frames are close and not identical. Their fixed fingertip is at (−0.002, 0, −0.092) m in `gripper_link`. This fixed-jaw tip marker is at (−0.0081, 0, −0.101) m in the gripper body.
- `src/so101_smolvla/task.py` and `configs/sim.yaml` append a period to the task sentence. The dataset and `track.py` do not. The checkpoint path uses the dataset sentence.
- `load_policy` is unchanged and still serves the base SmolVLA checkpoint.

## Checkpoint on their own pixels

`scripts/training_frame.py` feeds episode 484, frame 13, of their video to the checkpoint and compares the 50 predicted targets with the 50 recorded actions. On those pixels the mean absolute error is about 0.7° pan, 8.6° lift, 8.8° elbow, 2.3° wrist flex, 2.0° wrist roll, 3.3 gripper percent. The largest single error is about 13° on lift and elbow. That run held the lift at −100° because the hinge limit at the time was −100°, and the recorded lift is −101.19°. The numbers are in `outputs/track/training-compare.json`. The video is `outputs/track/training-compare.mp4`.

Playing the same model on this scene is a separate question. The arm tracks a commanded target within a few degrees. The scene the policy sees is the one in `scene-match.png`.
