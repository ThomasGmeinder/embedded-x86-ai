# Development ledger

What was decided or found in this workshop and why, with the measurement
behind it. The measurements themselves, with their commands and logs, are in
`TESTING_LEDGER.md`. The newest entry is last. An entry is corrected with a
marked amendment, not rewritten.

## 2026-11-27 — The faster shoulder lift holds on the twin; the physical arm is not measured

**The change.** Since commit `4d820e4` (2026-11-26) the shoulder lift may run
at 2.14152 rad/s, the pan joint's limit, instead of 0.30543 rad/s. A planned
move runs all joints in step, so the lift had paced most of the time the arm
spent moving. The elbow and the wrist keep their limits; the reason is in the
comment of `so_arm101_moveit_config/config/joint_limits.yaml`.

**What the logs say** (`TESTING_LEDGER.md`, entry of 2026-11-27, read from
the controller logs of the twin):

- The same session of 335 planned moves spends 233.3 s in those moves instead
  of 512.1 to 513.6 s: 2.2 times less. No move takes 3 s or more any longer;
  50 or 51 did.
- Nothing in the logs points to lost accuracy: the same 331 of 335 moves
  reach their goal, the same 4 are cancelled, no move is aborted for a
  tolerance, and Servo's position-limit warnings are fewer, not more.

**How far that goes.** It is a statement about the twin and about what the
controller reports. It is not a measurement of position error, and it says
nothing about the physical arm.

**Open.**

1. The physical arm at the new limit with the heaviest prop in the jaws:
   commanded against measured joint positions over one pick. The old limit
   was the speed the servo can still hold in that case.
2. The tier-1 twin tests run again, so that their result files can be set
   beside those of 2026-11-25 for where the gripper ends up.
3. The controller's goal tolerance: where it is set and what it is.
4. The lag of 0.011 rad behind the commanded path (0.003 rad before) is
   recorded in `joint_limits.yaml` only; its log is gone. It was measured with
   nothing in the jaws.
