# Testing ledger

What was run or measured in this workshop, with the command or the log it was
read from. The newest entry is last. An entry is corrected with a marked
amendment, not rewritten.

## 2026-11-27 — The faster shoulder lift, read from the twin's controller logs: 2.2 times less time in planned moves, no sign of lost accuracy

**What changed.** Commit `4d820e4` (2026-11-26 10:08:53) raised the shoulder
lift's speed limit from 0.30543 to 2.14152 rad/s
(`so_arm101_moveit_config/config/joint_limits.yaml`) and made a streamed joint
brake before its target (`vla_pipeline/robot/moveit_arm.py`).

**What this entry is.** A reading of logs that already existed, done on
2026-11-27 at 17:05. No test was run for it: the graphics processor and the
NPU were in use. Every run in these logs is the MuJoCo twin (the controller
logs load `so101_sim/models/scene_vvla.xml`). None is the physical arm.

**Where the numbers come from.** `~/.ros/log/ros2_control_node_*.log`. For
every planned move the joint trajectory controller writes "Received new action
goal" and then "Goal reached, success!" or a cancel. The time of a move is the
time between those two lines. The stream warnings are the lines "Joint
position limit reached" in the `servo_node_*.log` of the same session.

**The comparison.** One session of 335 planned moves and 164 gripper goals
occurs five times: four times with the old limit and once with the new one, 10
minutes before the commit. Which test file produces that session was not read;
that it is the same one is taken from the equal counts of moves, cancels and
gripper goals.

| Session start | Limit | Controller log | Time in planned moves | Median move | Longest move | Moves of 3 s or more | Whole session | Stream warnings |
|---|---|---|---|---|---|---|---|---|
| 11-25 19:30:24 | old | `ros2_control_node_3749533_1795631424350.log` | 513.1 s | 0.69 s | 8.00 s | 50 | 929 s | 935 |
| 11-25 22:13:45 | old | `ros2_control_node_4037175_1795641225469.log` | 513.6 s | 0.69 s | 7.98 s | 51 | 923 s | 1,140 |
| 11-26 00:14:09 | old | `ros2_control_node_40090_1795648449719.log` | 512.1 s | 0.69 s | 7.87 s | 50 | 947 s | 1,119 |
| 11-26 00:45:26 | old | `ros2_control_node_125048_1795650326830.log` | 512.9 s | 0.69 s | 7.98 s | 50 | 991 s | 1,443 |
| 11-26 09:58:59 | new | `ros2_control_node_668049_1795683539088.log` | 233.3 s | 0.55 s | 2.30 s | 0 | 578 s | 578 |

In all five sessions: 331 of the 335 moves reached their goal, 4 were
cancelled, and no line reports an abort or a path or goal tolerance.

**Speed.**

- The time spent in planned moves is 2.2 times shorter (512.1 to 513.6 s
  against 233.3 s).
- The gain is in the long moves: 50 or 51 moves took 3 s or more before, none
  does now, and the longest move went from about 8 s to 2.30 s. The median
  move went from 0.69 to 0.55 s.
- The whole session is 1.6 to 1.7 times shorter (923 to 991 s against 578 s).
  Gripper moves, planning and perception are not affected by the limit.
- The one session after the commit (11-26 10:50:39,
  `ros2_control_node_762414_1795686639175.log`, 61 moves, 60 reached, 1
  cancelled) has the same picture: median 0.54 s, longest 2.30 s, no move of
  3 s or more.

**Accuracy: what the logs show.**

- The same moves reach their goal as before (331 of 335, the same 4 cancels).
- No move was aborted for a tolerance in any session.
- MoveIt Servo's position-limit warnings went down, from 935 to 1,443 per
  session to 578. The same commit added the braking of streamed joints, so
  this cannot be put down to the speed limit alone.

**Accuracy: what the logs do not show.**

- **Position error.** The controller log holds no position error. The only
  figure is the one the session that made the change wrote into
  `joint_limits.yaml`: the arm was at most 0.003 rad behind the commanded path
  with the old limit and at most 0.011 rad with the new one, nothing in the
  jaws. The log of that measurement was in `/tmp` and is gone, so the figure
  was not read again.
- **Where the gripper ends up.** The per-test result files in
  `_artifacts/sim_tests/tier1/` were written on 2026-11-25 at 17:11 to 17:16,
  before the change. There is no pair of result files to compare.
- **The controller's tolerance.** Where the goal tolerance is set was not
  found, so "goal reached" means "within a tolerance that was not read".
- **The physical arm.** No run after the change. The old limit was the speed
  the STS3215 servo can still hold in its worst pose with the heaviest prop;
  the twin cannot show whether the real servo keeps up at 2.14152 rad/s under
  load.

**Not used for the comparison.** The sessions of 2026-11-26 between 08:17 and
09:58: several limits were being tried that morning (the comment in
`joint_limits.yaml` lists four), and which limit each session ran with is not
in the logs.
