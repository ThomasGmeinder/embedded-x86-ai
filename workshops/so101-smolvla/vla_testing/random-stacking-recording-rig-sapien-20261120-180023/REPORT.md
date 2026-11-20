# SAPIEN stacking regression random-stacking-recording-rig-sapien-20261120-180023

Task: `stack2-recording-rig`
Complete stacks: 10/20
Command 1 success: 10/20

Accumulated video: [top](overall-top.mp4), [wrist](overall-wrist.mp4).

The finger angle is the angle between the fingertip line and the nearest cube face
at the first jaw contact. Zero is square onto the face. If the jaws never touch,
the angle is taken at the closest approach.
A test with several commands lists one value per command.

Finger-to-face angle at grip: mean 10.6°, median 6.2°, max 34.7°. Ideal is 0°.

| Environment seed | Policy seed | Complete | Command results | Failure stage | Jaw contact | Finger angle ° | Wrist roll at grip ° |
|---:|---:|---|---|---|---|---:|---:|
| 0 | 0 | fail | put red cube on blue cube: fail | dropped cube | pass | 6.1 | -18.0 |
| 1 | 1 | pass | put red cube on blue cube: pass | - | pass | 0.3 | -14.3 |
| 2 | 2 | fail | put red cube on blue cube: fail | dropped cube | pass | 6.3 | 0.3 |
| 3 | 3 | pass | put red cube on blue cube: pass | - | pass | 10.7 | -25.4 |
| 4 | 4 | pass | put red cube on blue cube: pass | - | pass | 19.1 | 3.4 |
| 5 | 5 | pass | put red cube on blue cube: pass | - | pass | 34.7 | 12.0 |
| 6 | 6 | fail | put red cube on blue cube: fail | no lift | pass | 32.5 | 4.9 |
| 7 | 7 | pass | put red cube on blue cube: pass | - | pass | 7.7 | -14.1 |
| 8 | 8 | pass | put red cube on blue cube: pass | - | pass | 5.2 | -18.1 |
| 9 | 9 | pass | put red cube on blue cube: pass | - | pass | 5.1 | -15.9 |
| 10 | 10 | fail | put red cube on blue cube: fail | no lift | fail | 17.8 | -4.5 |
| 11 | 11 | fail | put red cube on blue cube: fail | cube displaced without grasp | pass | 5.6 | -19.0 |
| 12 | 12 | fail | put red cube on blue cube: fail | cube displaced without grasp | pass | 22.3 | -3.5 |
| 13 | 13 | pass | put red cube on blue cube: pass | - | pass | 13.6 | -4.0 |
| 14 | 14 | fail | put red cube on blue cube: fail | no lift | pass | 1.6 | -3.1 |
| 15 | 15 | pass | put red cube on blue cube: pass | - | pass | 0.8 | -8.4 |
| 16 | 16 | fail | put red cube on blue cube: fail | no lift | pass | 9.1 | -12.8 |
| 17 | 17 | fail | put red cube on blue cube: fail | dropped cube | pass | 4.6 | -7.0 |
| 18 | 18 | pass | put red cube on blue cube: pass | - | pass | 4.2 | -34.5 |
| 19 | 19 | fail | put red cube on blue cube: fail | no lift | fail | 5.1 | -5.5 |
