# Motion tracking check

The checkpoint `Harrysunshine/so101-smolvla-sim-real-10task` has `chunk_size` 50 and `n_action_steps` 50 in its `config.json`. `empty_cameras` is 0. One inference is 50 absolute joint targets. This run plays them at 30 Hz and records how far the arm lags. It does not score a grasp.

The pickup in `scripted_testing/harry/poweron-wrist.mp4` set `n_action_steps` to 1, kept only the first target, held it for 4 ticks (0.133 s), and inferred again. The position actuator in `assets/robotstudio_so101/so101.xml` is `kp="998.22"` with `kv="2.731"`, joint `damping="0.60"`, and `forcerange="-2.94 2.94"`. The proportional term hits that force limit at about 0.17 degrees of error. Past that, speed is limited to about 50 degrees per second, so a large step is unfinished when the next picture is taken and the new first target yaws the hand past the cube.

## Loop

1. Load that checkpoint. Leave `n_action_steps` at 50 and `empty_cameras` at 0. Do not copy either 1 from `configs/contract.yaml`, and do not call `load_policy`. That loader stays on `lerobot/smolvla_base` for the base-checkpoint demo.
2. The task string is `Pick up a cube and place in the bin`.
3. The published power-on pose is `[-5.76, -102.68, 92.97, 63.38, -0.53, 1.90]`. `shoulder_lift` ends at -1.7453293 rad, so the pose this model can hold is `[-5.76, -100, 92.97, 63.38, -0.53, 1.90]`. That vector is arm degrees and a gripper percent. `command` converts it and clips it into `ctrl`. Copy those `ctrl` radians into `qpos`, zero those joint velocities, and run `mj_forward`. The state sent to the policy is `read_state()` after that. Leave the cube where the scene puts it.
4. Run that observation through this checkpoint's own preprocessor, then call `predict_action_chunk` once. Unnormalize all 50 outputs with this checkpoint's own postprocessor.
5. Command the 50 targets in order. Hold each target for exactly one control tick (`TICK_STEPS` at 480 Hz, which is 1/30 s). Do not wait for the joint error to fall before the next target.
6. The policy sees no later picture. Record top and wrist after every tick, for all 50.

## Not in this change

The MuJoCo model stays `so101_new_calib`. The gripper percent stays 0% = -10 degrees and 100% = +100 degrees. Actuator torque stays 2.94 Nm. A grasp number belongs to a later run, on the checkpoint's own scene and on the order of its 500-step eval, with a new chunk each time the 50 targets are used up.

## Check

One chunk, one video, and at each of the 50 steps the policy target, the clipped command, and the achieved joint. The result is that tracking error. A target past a hinge limit is a clip, not lag.
