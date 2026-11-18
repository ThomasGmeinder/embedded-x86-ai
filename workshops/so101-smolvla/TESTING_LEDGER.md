# Testing ledger

Checkpoint runs live in `vla_testing/`. Scripted runs live in `scripted_testing/`. Each entry names the command, the seeds, and the counts from that run's own report.

## 2026-11-18 — 100 randomized pickups

Command: `PYTHONPATH=src .venv/bin/python -u -m so101_smolvla.vla_test random 100`

Checkpoint: `Harrysunshine/so101-smolvla-sim-real-10task`. Task: `Pick up a cube and place in the bin`. 500 steps, a new picture every 50. Raw joint targets, no average.

Seeds 0 through 99. Each seed draws one cube inside the published spawn box, center (0.27, 0.045) m, half-size 0.10 m, and one yaw about vertical. The same seed draws the same cube. The bin stays where the scene puts it.

Reports: `vla_testing/random/summary.json` and `vla_testing/random/seed-00000/result.json` through `seed-00099`. Each report stores `seed`, `cube_start_xy`, `cube_yaw_deg`, `cube_end_xyz`, `contact`, `lifted`, and `in_box`. Videos `top.mp4` and `wrist.mp4` are next to the report and are not in git. The reports are in commit `cb43e02`.

| Count | Number |
|---|---|
| Runs | 100 |
| Cube in the bin at the last step | 56 |
| Jaw contact at the last step | 2 |
| Cube above the table at the last step | 0 |

`in_box` is the cube center inside the tray after the run. `lifted` is that same last step, so a cube that was picked up and then set in the bin counts as not lifted. The 44 seeds that did not finish in the bin: 0, 1, 5, 6, 7, 11, 12, 13, 15, 19, 22, 23, 27, 31, 32, 36, 37, 40, 41, 42, 43, 46, 47, 50, 53, 55, 57, 58, 59, 60, 62, 63, 65, 67, 73, 76, 78, 82, 83, 84, 86, 93, 95, 96.

## 2026-11-18 — SAPIEN policy-backend acceptance

Focused command:
`PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_sapien_protocol.py tests/test_sapien_env_integration.py tests/test_vla_test.py tests/test_track.py -v --tb=short`

Result: 25 passed in 8.81 s.

Full project command:
`PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python -m pytest -v --tb=short`

Result: 44 passed in 18.71 s.

Fixed-seed acceptance command:
`./scripts/run_sapien_rollout.sh --count 1 --steps 300 --environment-seed-base 0 --policy-seed-base 0 --output vla_testing/sapien-backend-acceptance-20261118-2215`

Result: 1 of 1 episodes succeeded; first success was step 244. Every declared
artifact hash, both 300×6 joint trajectories, both 640×480 RGB policy
observations, and the episode and batch completion markers were verified.

The pinned `Xbotics-SO101-Sim` upstream suite produced 12 passes and 9 failures.
Eight failures require its NVIDIA CUDA GPU backend. One CPU gripper test failed:
the commanded 60% position reached 1.697% after 20 steps. This failure was not
excluded. The repository's real CPU SAPIEN reset/step/shutdown test passed.

## 2026-11-18 — Clean two-backend baseline

After removing the hybrid SAPIEN renderer, analytical action-padding patch,
failed smoothness-training code, and unvalidated domain-training expansion:

Full project command:
`PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python -m pytest -v --tb=short`

The first run exposed the already documented stale expert input
`(0.22, -0.08)`: 30 passed and 1 failed in 10.17 s. The unchanged expert
succeeds at the scene's established cube position `(0.302, -0.037)` and ends
in the bin. After correcting that test input, the full suite passed: 31 passed
in 9.94 s.

Clean MuJoCo command:
`./scripts/run_random_regression.sh 1 clean-mujoco-20261118-2211`

Result: seed 0 picked up, held, and placed the cube in the bin. The result names
backend `mujoco`. Raw shoulder-lift targets reversed on 49.7% of opportunities;
elbow targets reversed on 43.8%.

Clean SAPIEN command:
`./scripts/run_sapien_rollout.sh --count 1 --steps 300 --environment-seed-base 0 --policy-seed-base 0 --output vla_testing/sapien-clean-acceptance-20261118-2212`

Result: seed 0 first succeeded at step 244. Raw lift/elbow targets reversed
40.1%/33.2%; SAPIEN achieved joints reversed 5.6%/3.4%. The result names backend
`sapien`, records the pinned simulator commit and complete environment
arguments, and has episode and batch completion markers.

## 2026-11-18 — Ten random layouts per backend, 300 steps

One command shape now selects either backend:

`make random BACKEND=mujoco COUNT=10 STEPS=300`

`make random BACKEND=sapien COUNT=10 STEPS=300`

The naming and 300-step forwarding changes passed the complete project suite:
32 passed in 9.99 s.

MuJoCo wrote `vla_testing/random-MuJoCo-20261118-221817/`. Four of ten cubes
ended in the bin; seven were picked up and held; all ten had jaw contact.

SAPIEN wrote `vla_testing/random-sapien-20261118-221953/`. Nine of ten episodes
reached native task success. All ten picked up the item; nine held it.

Both runs used environment seeds 0–9 and policy seeds 10000–10009. Each twin
generated its own deterministic random layouts, so equal seed numbers do not
mean equal cube/bin coordinates across the two independently implemented
environments.
