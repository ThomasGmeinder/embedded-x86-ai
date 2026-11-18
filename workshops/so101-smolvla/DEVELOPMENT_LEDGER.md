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
