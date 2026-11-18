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
