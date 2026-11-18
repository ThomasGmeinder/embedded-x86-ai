# Development ledger

What was learned while making the SO-101 digital twin, simulated in MuJoCo, run the checkpoint `Harrysunshine/so101-smolvla-sim-real-10task`. Test counts stay in `TESTING_LEDGER.md`. This file is the progress and the lessons. An entry names a commit when one existed at the time of writing. It does not track whether a later commit absorbed the work.

The task string the checkpoint path uses is `Pick up a cube and place in the bin`. One inference is 50 absolute joint targets, played one per tick at 30 Hz. Nothing is averaged before the command.

## 2026-11-17 — Digital twin, cameras, bin, first pickup

Commits `d0cdaca` through `80a4ec2`.

The SO-101 digital twin copies the KIT training rig: cameras 640×480 with a 59.17° vertical field, the published bin mesh, and the published lights (ambient 0.3, directions (1, 1, −1) and straight down, shadows off). The wrist lens is the KIT optical frame on the gripper, position (0.0025, 0.072673, 0.005115) m. The gripper body quaternion is the KIT wrist-roll zero. The Menagerie quaternion was 90° off and aimed the wrist camera at the arm.

A 500-step rollout of the digital twin puts the cube in the bin. The arm jitters. Across those 500 targets the lift reverses 167 times and the elbow 181 times, in steps of about 1–2°. The position actuator follows the targets. That note is in `COMPARISON.md`.

Lesson: the jitter is in the commanded targets. Smoothing them was tried and removed. The project rule forbids a filter the published controller does not run.

## 2026-11-18 — Randomized runs, then the wrist picture

Commits `a5fc0b2`, `cb43e02`, `bda51df` existed when this entry was written. The rest of the entry had no commit on 2026-11-18.

Scripted runs live in `scripted_testing/`. Checkpoint runs live in `vla_testing/`. The first 100 seeds are in `vla_testing/random/`, recorded in `TESTING_LEDGER.md`: 56 of 100 cubes finished in the bin. That draw moved only the cube, inside the published spawn box, and left the bin fixed, so some cubes spawned in the tray. Seed 19 is one of those.

Work done the same day. It had no commit when this entry was written:

- The bin is placed first, inside the top camera’s view of the table and inside the measured finger area. The cube is placed outside the bin, in that same area. The finger area is the real-robot range from `so101_sim/envs.py`: x 0.172–0.368 m, y −0.099–0.190 m. Seed 0 of the old camera-only draw put the bin at (0.457, −0.139), 0.478 m from the base. With the fingers pointing down at the table, the farthest fingertip in the digital twin is 0.381 m. The nearest downward fingertip was 0.10 m short of that bin.
- A start whose cube is cut by the edge of the power-on wrist picture is rejected. The old seed 0 cube, (0.291, 0.112), was clipped in the corner: 4,400 pale-blue pixels, centered at (58, 17), touching the top and left edges. The replacement draw for that seed puts the cube at (0.206, 0.150) and the bin at (0.297, −0.021). The cube ends in the bin. The opening wrist frame has no pale-blue cube and nothing touches the picture edge.
- `render` raises if the cube center is inside a jaw solid. A test wrote the cube onto a camera ray and rendered one frame before physics could separate it. The cube overlapped the moving jaw by 7.4 mm. That picture was deleted. The overlap is not in the scene and not in the placement code.
- The table was enlarged so the top camera’s frame is the table gray. The cube material was given emission so the published pale blue is one color instead of a white face and a gray face. Neither change removed the jitter. The wrist camera pose was not edited.

`make random COUNT=1` is the local launcher (`Makefile`, `scripts/run_random_regression.sh`). It writes a timestamped directory.

## Finding — the model is smooth on training pictures and jagged on the digital twin’s wrist picture

The published sampler draws one normal value at every step of the 50 and integrates the checkpoint’s 10 Euler steps. On 14 training episodes (484–497), at the window where the lift and elbow move the most, that sampler reverses on 4.4% of steps and misses the recorded action by about 4°. The direction matches the recording every time.

The same sampler, same joints, episode 486 frame 246:

| Wrist picture | Top picture | Reversals | Lift error | Elbow error |
|---|---|---:|---:|---:|
| Training | Training | 1.7% | 2.0° | 1.5° |
| Training | Digital twin | 1.0% | 6.0° | 4.2° |
| Digital twin | Training | 42.7% | 55.7° | 58.8° |
| Digital twin | Digital twin | 55.6% | 65.5° | 70.4° |

The top picture is not the switch. Replacing only the wrist picture is what moves the reversal from about 1% to about 40%. The arm jitters because it is commanded through that reversing list, one target per tick.

Checked, and not the cause:

- The wrist lens. Recomputed from the URDF optical frame and `_OPTICAL_CONV`. Position matches to 0.001 mm. Quaternion matches to 0.000001. The published look misses the URDF fingertip by 12.0°. The digital twin’s fingertip is 9 mm from that URDF tip, so the same look misses it by 10.5°. The 73 mm Y offset is the published coordinate.
- Zeroing the sampler noise. One digital-twin seed went from 38% lift reversal to 4% and still traveled the same angle. That draw is not the published sampler. It was taken back out.
- More Euler steps, 20 and 50, on a digital-twin picture. The reversal stayed near 37%.
- A flat cube color and a table that fills the top frame. Grasp-picture reversal went from 57.6% to 55.9%.
- Matching the wrist blob’s size and pixel without putting the cube inside a jaw. Blob 10,190 pixels at (349, 264), training blob 10,502 pixels at (344, 260). Reversal stayed at 46%.

The dataset stores no cube pose and no bin pose. The digital twin’s cube and bin are the hand positions inferred for episode 484. A wrist frame can therefore show a different arrangement than the training frame at the same joints, even though the lens is on the published mount.

Lesson: the checkpoint follows a smooth demonstration when the wrist picture is one it was trained on. It does not carry that smooth trajectory over to the digital twin’s wrist picture. A pick can still finish. With the cube at (0.206, 0.150) and the bin at (0.297, −0.021), seed 0 ends in the bin and the arm is still jittery.

## 2026-11-18 — Finger contact, ten seeds, the pause after the drop

### Mistake — the finger contact was turned off, then the wrong file was restored

The Robot Studio file `so101_new_calib.xml` collides with the finger meshes `wrist_roll_follower_so101_v1` and `moving_jaw_so101_v1`. Those geoms use class `collision`. They are the pinch surfaces.

Commit `d0cdaca` replaced that with a local class `collision_gripper_mesh` whose geoms are `contype="0"` and `conaffinity="0"`. The comment in that commit says the hull was batting the cube out of the jaws, and that the finger boxes are the pinch surfaces. The boxes that actually met the cube were 1 mm thick (`fixed_jaw_box4`–`fixed_jaw_box7`, `moving_jaw_box2`, `moving_jaw_box3`). The visible finger mesh could pass into the cube. The thin box stopped on the cube face. That was a local change. It was not in the published model.

Asked to put `so101.xml` back to the original, the restore used `git checkout`. That brought back the commit, which still had the disabled mesh. The original is the Robot Studio file, not `HEAD`. The file was then edited again on top of that commit. That edit was the wrong baseline.

`so101.xml` is now the Robot Studio file. The finger collision geoms are those two meshes, class `collision`. The `contype="0"` finger class and the 1 mm pinch boxes are gone. Four pieces from this digital twin stay on that file, because the published file does not have them and the twin already uses them: the KIT gripper quaternion, the KIT wrist camera and its mount, the two 0.75 mm fingertip spheres the code reads, and the physics timestep of 1/480 s.

The disabled collision was hiding a real grasp failure. `Expert.run((0.22, -0.08), azimuth=0.4)` returns false. The cube is at z 0.085 m and the jaws are not touching it. The mesh knocks the cube out of the fingers. `tests/test_expert.py` fails on that. The collision disable is not going back in.

### Ten seeds on the published finger mesh

`./scripts/run_random_regression.sh 10` exited 0 in 39 s. Seeds 0–9. Six cubes ended in the bin. All ten were picked up. The jaws touched the cube on every seed. Seed 3 was the only one not held. Finger-to-face angle at grip: mean 8.4°, median 7.4°, max 19.0° on seed 9. Those rows used the old hold flag, which became true on a squeeze even while the cube was still on the table. A hold now requires the cube off the table on the same step, still in the jaws.

| Seed | In bin | Held | Finger angle |
|---:|---|---|---:|
| 0 | fail | pass | 7.1° |
| 1 | pass | pass | 5.8° |
| 2 | pass | pass | 7.7° |
| 3 | fail | fail | 9.8° |
| 4 | pass | pass | 6.4° |
| 5 | pass | pass | 1.3° |
| 6 | fail | pass | 9.6° |
| 7 | fail | pass | 10.4° |
| 8 | pass | pass | 6.8° |
| 9 | pass | pass | 19.0° |

### The arm sits still after the cube is in the bin

`rollout` plays until `EVAL_STEPS` (500) and does not look at `in_box`. `in_box` is recorded after the loop. 500 steps is 16.7 s on every seed. On seed 1 the top picture matches the final frame from 8.9 s on, so the last 7.8 s are the open gripper over the cube in the bin. Seeds 5 and 8 hold the same way for 6.9 s and 8.3 s. The later 50-step chunks leave the arm there, and the loop keeps commanding them. The published training env stops at 400 steps. This runner uses 500, and it does not stop when the cube is down.

## Conclusion — the finetuned model does not generalize to this digital twin

The checkpoint `Harrysunshine/so101-smolvla-sim-real-10task` was trained in the SAPIEN simulation that produced `Harrysunshine/so101-sim-pickplace-v2`. It does not generalize to this MuJoCo digital twin.

Shown a training picture, it follows the recorded shoulder command to about 1°. Shown this twin’s pictures, the same checkpoint reverses direction by 1–2° and the arm jiggles. A pick can still finish. In the ten-seed run, 6 of 10 cubes ended in the bin. That success count has not improved.

## Comparison — the training simulation and this simulation

The training simulation is SAPIEN through ManiSkill, environment `SO101PickPlaceCube40-v1`. Physics is 120 Hz. Control is 30 Hz. An episode stops at 400 steps. The arm color in that rig is (0.05, 0.05, 0.05). A training top frame has no yellow pixels. The cube is 40 mm and 12.8 g.

This simulation is MuJoCo. Physics is 480 Hz. Control is 30 Hz. A rollout here is 500 steps. `so101.xml` is the Robot Studio model, so the arm is yellow. One top frame from the ten-seed run has 18,432 yellow pixels, about (247, 242, 54). Painting those pixels the training black and asking the checkpoint again left the lift reversals at 54% and 67%. The yellow arm is a visible difference. It is not what makes the targets reverse.

Same joints, episode 490. On the training pictures the lift error is 0.88° at frame 80 and 1.04° at frame 160. On this twin’s pictures at those same joints the lift error is 6.71° and 4.85°. The dataset stores no cube pose and no bin pose, so the cube in this render is not where the training frame shows it. The training wrist shows the cube between black fingers. This wrist shows the cube off to the side of yellow fingers.

The cameras match: 640×480, 59.17° vertical field, wrist lens on the KIT optical frame. Both use absolute joint targets, the same task string, and the same 40 mm pale-blue cube. The recorded training motion does not jiggle. The lift reverses on 1.45% of 167,804 steps, and on 0% of the busiest 50 steps in all 498 episodes. A training clip’s picture never reverses direction (0 of 129 moving frames). This twin’s ten-seed video reverses on 27–53% of moving frames.

## 2026-11-18 — I reported that the pickups did not get better

That report was my failure. I had been asked what causes the jitter, and then what got better. I had not changed a pickup. The ten-seed run was still 6 of 10 cubes in the bin because I left it alone. I wrote that unchanged run down as the bottom line.

The picture test below started as a measurement on one frame. It was later put into the random layout. The rerun is the last section.

## One reach frame, with the cube and the bin moved

The first ten-seed run on the published finger meshes put 6 of 10 cubes in the bin. The arm jiggled. The picture test did not change that run.

The test asked one question. Freeze the arm on a training pose. Show the checkpoint the original training photos, then show it this digital twin. How far is the commanded shoulder from the recorded demonstration?

Episode 490, frame 80 is the arm reaching toward the cube, before the grasp.

- Training photos: the shoulder command is 1.22° off the demonstration.
- This digital twin, cube and bin left where the scene puts them: 7.86° off.
- Same arm pose, cube and bin moved so the wrist photo shows them where the training wrist photo shows them: 2.41° off.

That is the only improvement, and it is one frame of a test render. The cube was at (0.308, 0.033) m. The bin was at (0.326, 0.151) m. In the wrist photo the cube sits on the same spot as in training, and the bin is within about 1 pixel. The fingers already covered the same part of that photo. Painting the yellow arm black does not remove the jiggle.

Two limits:

- The hand-on-the-cube moment does not improve. Episode 490, frame 160 stays about 6° off, against 2° on the training photos. Episode 486, frame 246 stays about 51° off, against 1.3° on the training photos.
- The bin position that looks right to the wrist camera looks wrong to the overhead camera, by about 60 pixels across and 45 pixels down. Moving the bin to fix the overhead view makes the wrist view wrong again. There is no single spot on the table that looks right in both photos, which is why 2.41° did not reach 1.22°.

## Reach placement put into the pickup, then ten seeds again

The random layout now starts from that cube and bin, with a few millimetres of variation per seed. A cube that puts only one corner into the power-on wrist is kept. The old corner bug put two or more corners in, and that is still rejected. The bin center stays inside the measured finger area. The bin corners of this pose stick about 2 cm outside that box, so the old all-corners rule was blocking it.

Seeds 0 through 9 were run again. 5 of 10 cubes ended in the bin. The previous run put 6 of 10 in the bin. The arm reversed direction on 23% to 53% of the moving frames. The previous run was 27% to 53%. The fingers met the cube at 22° on average. The previous run was 8°.

The jitter did not stop. The grasp got worse. The second ten-seed run used this layout.

## Why the arm jitters

The checkpoint does not choose one path. For each photo it draws 50 separate random numbers, one for each of the next 50 steps, and tries to turn each number into the shoulder angle from a demonstration. Those 50 steps do not look at each other. A bump in the random number for step 10 moves that step by about 17° and the next step by about 3°. The position servo then performs every step.

A training wrist photo is enough to replace those 50 random numbers with one smooth demonstration. The shoulder stays within about 1° of the recording, and the recording itself almost never reverses.

This twin's wrist photo is not enough. Each random number is only partly removed, so the next step often points the other way by 1–2°. The arm does that. That is the jitter.

The wrist photo fails because, at the same arm joints, it does not show the cube and the bin where the demonstration had them. The dataset does not record those positions. At the reach, episode 490 frame 80, putting them where the training wrist shows them cut the shoulder error from 7.86° to 2.41°. The training photo at that frame is 1.22°. The new videos look a bit calmer for that reason, and only for that part of the motion. The overhead photo of the same bin is still about 60 pixels off, so the photo is still not the demonstration.

The grasp is a different photo. The hand is on the cube, and moving the cube and the bin does not turn that photo into a training photo. Episode 490 frame 160 stays about 6° off. Episode 486 frame 246 stays about 51° off. The pickup takes a new photo every 50 steps, so those later photos put the random numbers back into the shoulder command. That is the jitter still in the video. The yellow arm is not the cause: painting it black left the reversals in place, and the fingers already cover the same part of the photo as in training.
