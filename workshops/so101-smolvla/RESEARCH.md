# Research: picking the named object, stacking, and rebuilding a memorised stack

<a id="r-main-conclusion"></a>
## Main conclusion

> **Something other than the policy decides which object is meant, and
> hands the policy that object as a mask.**

"Policy" is the vision-language-action model (VLA; defined below) that
moves the arm. In every test read for this file in which the named
object is picked reliably among several objects or look-alikes, the
policy does not choose the object. A separate part chooses it; the
policy keeps the motion. Where the policy is left to choose from the
sentence, it takes the wrong object or fails in most trials.

| Who decides, and how the policy is told | Test | Decided outside the policy | Policy left to choose |
|---|---|---|---|
| A segmentation model (SAM 3) makes a mask of the target from a text phrase, a click or a box; the policy gets the mask with the first picture. MaskWAM, [arXiv 2606.13515](https://arxiv.org/abs/2606.13515) | Real two-arm robot, about 100 demonstrations per task, four tasks in which the sentence alone does not say which object is meant, 60 trials per task and setting | 92.9 % under trained conditions, 90.4 % with unseen extra objects on the table | Target written into the sentence as coordinates, trained conditions: 41.7 % (π0) and 26.3 % (FastWAM) |
| A planner names the next object by its place in the row, SAM 3 makes the mask, a vision-language model (Qwen3.5) checks the mask before the motion and the result after it, at every step. MaskHarness-WAM, [arXiv 2609.19974](https://arxiv.org/abs/2609.19974) | Real two-arm robot, 3, 4 and 5 look-alike blocks to be handled in a commanded order, 25 trials each. Counted: the whole sequence in the right order | 88.0, 84.0 and 84.0 % (22, 21 and 21 of 25) | π0: 20.0, 28.0 and 0.0 % (5, 7 and 0 of 25) |
| A person clicks the target in the camera picture, a segmentation model (SAM2) makes the mask and follows it. VCA, [arXiv 2602.23583](https://arxiv.org/abs/2602.23583) | Real two-arm robot, four orange and four blue blocks, 100 trials. Counted: the clicked block ends in the right bin | 96 of 100, no wrong target. With seven rings of one colour and different sizes: 94 of 100, 3 wrong targets, each time the neighbouring ring | – (the comparison policy in that paper is given no target) |
| The same idea with the mask used to blank out every other object in the policy's pictures. OBEYED-VLA, [Finding 6](#r-finding-6) | Real arm, 100 runs per setting | "above 90%" with one other object, "around 80%" with seven | "below 10%" |
| Decided outside, and no policy in the pick: the arm goes to the measured place. RefGuard, [Finding 2](#r-finding-2) | Real arm, objects on the table twice, 90 trials | 81 of 90, no wrong object | π0.5: 42 of 90, 36 wrong objects |
| Decided outside, policy not told by a mask: an agent finds the target with SAM 3, lets the policy grasp on unchanged pictures, watches and can take over. SO-101 Bench, Findings 5 and 6 | Real SO-101, 45 layouts | 29 of 45, no wrong object | Best policy alone: 24 of 45 |
| No outside decision at all. [arXiv 2606.08881](https://arxiv.org/abs/2606.08881) | Real SO-101, 100 demonstrations per task, 20 runs per model, "Put the pink block in the plate and remove the others" | – | ACT 0 %, SmolVLA 5 %, Wall-X 0 %, π0.5 10 % |

Three limits that belong to this conclusion:

- A mask handed over once is not enough for a sequence over look-alikes.
  In the MaskHarness-WAM test the mask-taking policy without the planner
  (MaskWAM, rebuilt by those authors because its code "had not released"
  at the time) completes 56.0, 8.0 and 4.0 % of the sequences (14, 2 and
  1 of 25). The mask has to be made anew and checked at every step.
- Two identical objects cannot be told apart by a sentence. The part
  that decides needs a place word ("the third object from the left"), a
  click, or a question back to the user.
- None of the three mask tests was run on an SO-101 or with SmolVLA, and
  nothing of this was measured in this workshop. The workshop's
  checkpoint was trained neither to take a mask (MaskWAM) nor on
  blanked-out pictures (OBEYED-VLA).

  Amendment, 2026-11-21 12:27: this bullet first ended "A policy has to
  be trained to take a mask (MaskWAM) or trained on blanked-out pictures
  (OBEYED-VLA); the workshop's checkpoint was trained on neither." The
  first half was wrong. Two published methods remove the other objects
  from the pictures of an unchanged policy, without any training, and
  raise its success (CGVD and BYOVLA, [Finding 8](#r-finding-8)). The user pointed this
  out on 2026-11-21.

This section was added on 2026-11-21 at 11:46. The numbers of the three
mask tests and of the SO-101 colour test were read in the saved full
texts of the papers at that time; the other rows point to the findings
below.

Started 2026-11-21. This file collects what published tests say about
three robot tasks and what follows for this workshop. It holds literature
findings and decisions; measurements made in this workshop stay in
`DEVELOPMENT_LEDGER.md` and `TESTING_LEDGER.md` and are only linked here.

Every count below was read in the full text of the source on 2026-11-21
unless a line says otherwise. A count is written as "k of n", with what
was counted and in which setting. Nothing in this file was measured on
the SO-101 arm of this workshop.

Findings 5 to 7, the section on SO-101 Bench and the second table under
[Finding 3](#r-finding-3) were added later on 2026-11-21. Their numbers were read in the
full text of each paper, README, issue thread or page linked next to
them. Where a source gives only a percentage, the percentage is quoted
as the source gives it.

Two terms used throughout:

- **VLA (vision-language-action model):** one neural network that takes
  camera pictures and a sentence and gives robot commands. SmolVLA
  (450 million parameters), π0.5 and GR00T N1.7 are such models. In this
  file "policy" means such a model used alone.
- **Wrong-object pick:** the arm carries out a clean grasp, but on an
  object the sentence did not name.

<a id="r-the-three-tasks"></a>
## The three tasks

1. Objects of different kinds and colours (cubes, balls, cylinders) lie
   anywhere on the table. The robot picks the one a spoken or written
   command names. The same object may be there twice: two red cubes of
   the same size, or two red cubes of different sizes.
2. Stack cubes of different colours.
3. The robot is shown a stack of cubes. The user scatters the cubes and
   turns a marker object (a blue cylinder joined to a red cuboid). The
   robot rebuilds the stack. The reference for this task is the project
   `~/GitHub/VLA-VLM`.

<a id="r-finding-1"></a>
## Finding 1: a policy alone picks the named object unreliably once objects can lie anywhere or occur twice

It is not true that such models cannot do it. With objects at nearly
fixed places they do it well. The published failures appear when the
place no longer gives the object away.

| Test | Setting | Result |
|---|---|---|
| "Robust Skills, Brittle Grounding", [arXiv 2602.24143](https://arxiv.org/abs/2602.24143), Table 2 | Simulation, Franka Panda arm, five different household objects, grasp only, 100 test episodes per row. Counted: the named object is held at the end of the episode | SmolVLA, each object only slightly moved from its own region: 90 of 100. SmolVLA, objects anywhere on the table: 2 of 100 (12 grasped any object at some point, 4 ended within 5 cm of the named object). π0.5, objects anywhere: 6 of 100 (15 grasped any object, 16 ended within 5 cm); by Table 4 this is its result with 100,000 demonstrations |
| Same paper, Table 4 | Objects anywhere; training set raised from 10,000 to 100,000 demonstrations | SmolVLA 2 → 1 of 100. π0.5 4 → 6 of 100 |
| Same paper, Table 5 | Objects anywhere, SmolVLA, 10,000 demonstrations, fewer objects on the table | One object: 15 of 100. Two objects: 2 of 100. Three objects: 1 of 100 |
| Same paper, Table 3 | SmolVLA, objects only slightly moved; each object shown in three regions in training, tested in the two regions it was never shown in | 44 of 100 in the trained regions, 0 of 100 in the untrained ones |
| RefGuard, [arXiv 2609.06221](https://arxiv.org/abs/2609.06221), Fig. 4 | Real UF850 arm, fruit and vegetables of which five kinds are on the table twice, commands such as "pick the bell pepper to the right of the can and place it into the bowl near the cup". 90 trials per method (three settings of 30). π0.5 fine-tuned on 300 demonstrations of that robot, GR00T N1.7 fine-tuned on the same data | π0.5: 42 correct, 36 wrong-object picks, 12 other failures. GR00T N1.7: 45 correct, 30 wrong-object picks, 15 other failures |
| TiPToP, [arXiv 2603.09971](https://arxiv.org/abs/2603.09971), Table I | Eight scenes with extra objects on the table, one in simulation and seven on a real arm, 45 trials. Policy: π0.5-DROID | 12 of 45 |

The single-object row of Table 5 matters for reading the "2 of 100": with
one object and free placement SmolVLA still succeeds only 15 times in
100, so in that paper the whole grasp is weak with free placement, not
only the choice of object.

Ten further sources on the same weakness, among them a test on a real
SO-101 (picking the pink block among other blocks, four models, 20 trials
each, all between 0 and 10 %; SmolVLA 5 %), are in
`DEVELOPMENT_LEDGER.md`, section ["2026-11-20 — Several objects in one
scene"](DEVELOPMENT_LEDGER.md#d-2026-11-20-several-objects-in-one-scene) (line 4283). They were read on 2026-11-20 and not read again for
this file.

What was measured in this workshop agrees with it (`README.md`, table
"Models evaluated"): with a second cube in the scene the gripper is
pulled 30 to 44 % of the way towards it whichever sentence is given, and
on 5 of 10 layouts the red cube was never lifted in nine runs.

<a id="r-finding-2"></a>
## Finding 2: deciding outside the policy which object is meant gives better results in the published tests

| Test | Policy alone | Object decided outside the policy |
|---|---|---|
| RefGuard, Fig. 4, same 90 real trials with duplicate objects | π0.5: 42 correct, 36 wrong-object picks | RefGuard: 81 correct, 0 wrong-object picks, 6 times declined to act (asked back, looked again or stopped), 3 other failures |
| RefGuard, same figure, the paper's baseline built on a language model ("LLM-based"; how it is built was not read for this file) | – | 45 correct, 41 wrong-object picks, 4 other failures |
| TiPToP, Table I, all 165 trials in 28 scenes | π0.5-DROID: 55 of 165 | TiPToP: 98 of 165 |
| TiPToP, the eight scenes with extra objects | 12 of 45 | 27 of 45 |
| TiPToP, scene "sort the blocks into opposite color plates" | 0 of 5 | 5 of 5 |
| TiPToP, scene "put 3 cubes into the bowl" (simulation) | 0 of 10 | 9 of 10 |
| TiPToP, scene "put the red cubes onto the white plate" | 5 of 5 | 1 of 5 |
| TiPToP, the five scenes with one object and nothing else | 27 of 40 | 22 of 40 |

Reading these:

- Moving the decision outside the policy is not enough by itself. The
  language-model baseline in RefGuard still takes the wrong object 41
  times in 90. RefGuard's authors put their result down to keeping
  several candidate objects open until the evidence decides between
  them, and to declining to act when it does not; their controlled
  comparison is stated to show that the option to decline alone does not
  explain it (the comparison's table was not read for this file).
- The outside-the-policy systems lose where the grasp itself is hard.
  TiPToP plans once and then moves without looking again; its authors
  trace most of its failures to grasps that slip during that blind
  execution. The two rows at the bottom of the table show it.
- TiPToP uses no robot training data. RefGuard's two policies were
  fine-tuned on its robot.

<a id="r-the-design-these-systems-share"></a>
### The design these systems share

["The design"](#r-the-design-these-systems-share) in this file means: which object to pick is decided by
separate parts, not inside the policy network.

1. A language model turns the command into a description of the target
   (kind, colour, size, position relative to other objects), once per
   command. RefGuard uses GPT-4o for this, TiPToP one query to Gemini
   Robotics-ER 1.5.
2. A segmentation model finds every object in the camera picture, and
   depth gives each object's position and size. RefGuard uses SAM 3
   (Segment Anything Model 3) on colour-and-depth pictures; TiPToP uses a
   depth model for stereo pictures (FoundationStereo).
3. A program matches the description to exactly one object. If several
   objects fit, it asks back or declines. Two identical red cubes cannot
   be told apart by "the red cube"; two red cubes of different sizes can,
   by the measured size.
4. The arm picks at the measured position, and the camera checks the
   result afterwards.

A policy can still carry out the motion in step 4. One test of that:
Pointing-VLA ([arXiv 2608.23138](https://arxiv.org/abs/2608.23138), real
AgileX PiPER arm, 50 trials in each of three scenes) keeps π0.5 as the
motion policy and adds a pointed grasp place and a pointed set-down
place. Completed tasks rise from 79 of 150 to 121 of 150. That test has
one target part and at most one other object on the table; it shows that
a pointed place helps the motion, not that it solves duplicates.

Examples of this design on an SO-101, found and not evaluated (their
success counts were not read):
[so101-gap](https://github.com/kabilankb/so101-gap),
[Gemini Robotics-ER 2 on an SO-101](https://ekumenlabs.com/blog/posts/give-robot-arm-a-brain/).
A third example,
[so101-vlm-agent](https://github.com/daniiarabdiev/so101-vlm-agent), was
read later; its counts are under [Finding 5](#r-finding-5).

<a id="r-finding-3"></a>
## Finding 3: stacking two blocks works at moderate rates with a fine-tuned policy; taller stacks are where policies fail

| Test | Setting | Result |
|---|---|---|
| PriorVLA, [arXiv 2605.10925](https://arxiv.org/abs/2605.10925), Table 4, task "Stack Blocks" | Real Franka arm, 100 demonstrations, 20 trials under training conditions and 20 under changed conditions (lighting and background among them) | π0.5: 80 % (16 of 20) and 50 % (10 of 20). GR00T N1.7: 40 % (8 of 20) and 20 % (4 of 20) |
| Same paper, Table 5 | Same task with 10 demonstrations | π0.5: 15 % (3 of 20) and 5 % (1 of 20) |
| [NovaPlan](https://rai-inst.com/wp-content/uploads/2026/10/1147_NovaPlan_Zero_Shot_Long_H.pdf), its section on long tasks | Tower of four 2-inch blocks, ten trials. π0.5 as released, not fine-tuned, given the correct order of steps as text | π0.5 "stacks up to two layers but fails beyond that". NovaPlan, which plans from a generated video: 7 of 10 |

This workshop's own number for one fine-tuned SmolVLA checkpoint, in
simulation: 24 of 60 two-cube stacks (`README.md`, table "Models
evaluated").

<a id="r-added-later-on-2026-11-21"></a>
### Added later on 2026-11-21: cube stacking on a real SO-101, and in the ManiSkill simulator

Read in the project's README, in the ManiSkill task documentation and
issue threads, and in the pages linked in each row.

| Test | Setting | Result |
|---|---|---|
| [so101-cube-stacking](https://github.com/GGOMOONG/so101-cube-stacking) (MIT licence) | Real SO-101, two 3 cm cubes, orange onto red, one camera above the table and one on the wrist, 10 trials per condition. A classical pipeline (camera calibrated with a printed marker grid on the table, cube position from its colour, inverse kinematics, correction from the wrist camera) against a fine-tuned SmolVLA (the project's version 4, 216 demonstrations) | Cube places inside the demonstrated range: classical 10 of 10, SmolVLA 4 of 10. Places outside that range: classical 7 of 10, SmolVLA 0 of 10 |
| Same project, the same SmolVLA weights | Where the setup stood during the demonstrations, then after it was moved | 9 of 10 at the old place, 4 of 10 at the new place, 0 of 10 with the grid paper under the cubes |
| Same project, a policy trained on simulated demonstrations only | On the real arm | 2 of 10 inside the demonstrated range, 0 of 10 outside |
| Same project, a five-level stack | ACT policy (Action Chunking with Transformers), 100 demonstrations | 0 of 10 |
| ManiSkill StackCube-v1 (red cube onto green cube), Diffusion Policy from camera pictures, [issue 1004](https://github.com/haosulab/ManiSkill/issues/1004) and [issue 1325](https://github.com/haosulab/ManiSkill/issues/1325) | Simulation. Numbers reported by users in the threads; number of test episodes not stated | 23 % in one user's run with 100 demonstrations. Another user cites "~7%" as the authors' own result and "close to 0%" for his runs. A third reports ">50% success rate with 300 demos" using a pretrained picture encoder, without a run to show |
| [mini-pi0](https://github.com/Thehunk1206/mini-pi0), one person's small policy that works from camera pictures, trained on ManiSkill's motion-planner demonstrations; self-reported in its README | StackCube-v1, 200 episodes. StackPyramid-v1 (red cube next to the green one, blue cube on top of both), 50 episodes | StackCube "95.5% success" (191 of 200). StackPyramid "26.0% success" (13 of 50) |
| [FRMD paper](https://doi.org/10.3389/frobt.2026.1751688), its Diffusion Policy baseline | Simulation, 12 tasks from the MetaWorld and ManiSkill benchmarks sorted into three difficulty levels, 100 demonstrations per task; which tasks are in which level and the number of test episodes were not read | 99.3 % on the easy tasks, 41.0 % on the medium ones, 10.1 % on the hard ones |

Reading these:

- The classical pipeline is the only entry here that reaches 10 of 10 on
  a real SO-101, and it does so with a printed grid on the table that it
  cannot work without. The README notes that with 10 trials the
  intervals are wide (7 of 10 stands for 35 to 93 %).
- The learned policy is tied to the table place it was demonstrated at:
  the same weights fall from 9 to 4 of 10 when the setup is moved.
- The maintainer's answer in issue 1004 on stacking from camera pictures:
  "That task is just much harder in general."

A claim seen elsewhere, "Diffusion Policy, ManiSkill PickCube, 99.2 %",
is a real row on the
[SOTA2 leaderboard](https://www.sota2.com/research/sota/pick-cube-on-maniskill)
("DP, total trainable parameters=12M, 2026.02, 99.2, std 0.002"; the
next rows are 94.5, 92 and 88). It does not carry over to the tasks of
this file:

- PickCube-v1 has one red cube and a goal point marked by a green sphere;
  success is the cube within 2.5 cm of the goal with the arm still
  ([task documentation](https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html)).
  There is no command and no second object.
- The page does not say how many episodes or demonstrations stand behind
  the row, or whether the policy saw camera pictures or the simulator's
  own numbers. The paper behind the row was not found.
- A commenter in issue 1325 writes that the PickCube baseline is given
  the goal position and a "cube is grasped" flag as numbers even when it
  is trained on camera pictures, and that success approaches 0 once he
  removes them. A
  [Scientific Reports paper](https://doi.org/10.1038/s41598-026-76821-1)
  describes the same: in PickCube-v1 the goal "is hidden before sensor
  capture and enters only as the low-dimensional goal_pos vector", next
  to a grasp flag; StackCube-v1 has no such goal vector.
- A picture shown with that claim (Franka Panda arm, a red, a green and a
  blue cube) fits StackPyramid-v1 by the task documentation, not
  PickCube.

<a id="r-finding-4"></a>
## Finding 4: rebuilding a memorised stack needs the memory outside the policy

A policy that sees only the current pictures has no record of the stack
it was shown.

| Test | Setting | π0.5 | With added memory (Mem-0) |
|---|---|---|---|
| RMBench, [arXiv 2603.01229](https://arxiv.org/abs/2603.01229), Table I, "Rearrange Blocks" | Simulation; number of trials per task not found in the text | 13 % | 89 % |
| Same table, "Put Back Block" | Simulation | 11 % | 90 % |
| Same paper, Table IV, "Rearrange Blocks" | Real two-arm robot, 100 demonstrations, 40 trials | 7.5 % (3 of 40) | 37.5 % (15 of 40) |
| Same table, "Put Back Block" | Real two-arm robot | 10 % (4 of 40) | 17.5 % (7 of 40) |

The real-robot numbers are low for both. The authors write that most
failures of their memory model on the real robot are imprecise block
handling, not wrong planning, and that they read the experiment "as
evidence that the memory mechanism transfers to hardware rather than as a
demonstration of deployable reliability".

How `~/GitHub/VLA-VLM` does this task today (commit `194e70d`): a plain
program holds the memory.

- `arm/planner/armpi_builder.py:1070` `learn` stores the stack it sees.
- `arm/planner/armpi_builder.py:1239-1285` `rebuild` moves one cube,
  goes back to the viewing pose, looks again, and repeats; a missed
  grasp is tried again, and the third miss stops the rebuild
  (`MISSES = 2`, line 127).
- `arm/planner/armpi_builder.py:1033-1034` `fits` matches a cube to a
  stored cube by colour and by size class.
- `arm/perception/armpi_perceive.py:31-35` reads the marker: the end
  that stands up says what to do. That project's marker has a dark red
  triangular end ("learn") and a blue cylinder end ("build").
- `arm/perception/armpi_perceive.py:36` and `:42` know four fixed
  colours, each with one cube size: `light-blue` 40 mm, `green`,
  `orange` and `blue` 30 mm.

For a fixed protocol triggered by the marker, a program that holds the
layout is enough. A language or vision model adds something where that
program is limited: colours and sizes outside the four fixed ones, and
commands in free wording. Whether an AI agent does the whole task better
than the program was not tested (see ["Unfinished"](#r-unfinished) below).

<a id="r-finding-5"></a>
## Finding 5: a distinct object named directly, after fine-tuning on that table, is picked 82 to 89 times in 100 on a real SO-101; descriptions, untrained rooms and released models without fine-tuning stay weak

Added later on 2026-11-21, after a search for existing projects. The
numbers of [Finding 1](#r-finding-1) are from the hardest settings; these are the
easier ones. Counts were read in the saved full texts of the two papers
and in the projects' own pages.

| Test | Setting | Result |
|---|---|---|
| Spa-Bench, [arXiv 2610.12696](https://arxiv.org/abs/2610.12696) | Real SO-101, a middle and a wrist camera. MolmoAct2, π0.5 and GR00T N1.7 (the variant with a frozen language model), each fine-tuned on 1,200 demonstrations of that table, 200 per task. 120 scenes per policy with a command that names the object directly | 107, 100 and 98 of 120 |
| Same, the same 120 scenes with the object given by a spatial description instead | | 73, 54 and 50 of 120 |
| Same, 300 runs per policy with spatial descriptions in combinations withheld from training | | 156, 122 and 133 of 300. All three fail every withheld counting run |
| Same, 120 runs per policy with spatial descriptions in trained combinations | | 88, 82 and 81 of 120 |
| SO-101 Bench, [arXiv 2610.36420](https://arxiv.org/abs/2610.36420), its Table 6 | Real SO-101 in a room not seen in training, 45 layouts: 3 for each of 15 combinations of task and kind of object (seen in training, unseen object of a seen class, unseen class). Tasks: put into the bin, place next to, place between, move in a direction; the named object has to be picked out among others of its class. Policies fine-tuned on 3,629 simulated demonstrations only; real runs scored by the operator under the benchmark's rules | Galaxea 0.5: 24 of 45. GR00T N1.7: 11. π0.5: 8. MolmoAct2-SO101: 7. In simulation on the same layouts: 30, 25, 22 and 11 |
| Same, an agent: the language model GPT-6.1 Sol writes Python in the PyRUA-Lean harness (segmentation with SAM 3, 3D position, inverse kinematics) and hands the grasp to Galaxea 0.5 | Same 45 real layouts | 29 of 45 (simulation 38). "Place next to": 8 of 9 against 2 of 9 for Galaxea 0.5 alone. None of its 16 failures was a wrong object: 7 on the bin tasks, 3 moved an object that was not the target, 6 missed the placement rule |
| Same, time per real run until it stops | Median over the 45 layouts | Agent 197 s, Galaxea 0.5 38 s |
| Same, 60 paired simulated episodes | What the agent gains from each part | Language model with direct tool calls: 21 of 60. With the code harness: 40. With the hand-off of the grasp: 51. Galaxea 0.5 alone: 32 |
| [so101-vlm-agent](https://github.com/daniiarabdiev/so101-vlm-agent), README | A vision-language model (Qwen3.8-27B with a trained adapter, trained in simulation only) picks one of nine skills per step and points at a pixel; ordinary code moves the arm by inverse kinematics. Real SO-101, 28 recorded runs over two days; the README calls it "a field report: the adapter changed between runs" | One ball into the container: 6 of 9 valid runs. Two balls with one command: 1 of 4. A ball out of the container: 0 of 10. In simulation: 440 of 544 solvable test episodes, stacking 30 of 30 without and 36 of 43 with an injected fault |
| [Manda Robotics test](https://mandarobotics.com/blog/state-of-robot-policies/index.html) | Five released policies used as released, without training for these tasks; simulation, 120 table-top tasks with 10 episodes each per policy | Best policy (Cosmos 3 Nano Policy): 35.1 % of 1,200 episodes. π0.5: 27.5 %. All five fail every episode on 30 of the 120 tasks |
| π0.7, [arXiv 2604.15483](https://arxiv.org/abs/2604.15483) | The authors' own summary, no counts in the sentence | "seen tasks often have success rates in excess of 90%, unseen tasks or unseen task-robot combinations have success rates in the 60-80% range" |

Reading these:

- Fine-tuned on the own table and asked for a distinct object by name, a
  policy fails 13 to 22 times in 120. Asked by a description, the same
  policies on the same scenes fail 47 to 70 times in 120.
- In a room it was not trained in, the best policy of SO-101 Bench
  succeeds 24 times in 45. The paper states that simulation overstates
  success by 9 to 31 percentage points.
- The agent with segmentation never took the wrong object in its 45 real
  runs, and a run takes it about five times as long as the policy alone
  (197 s against 38 s).
- In what was read for this file, no project reaches 9 in 10 on
  duplicates or on commands by description with a low-cost arm.

Unitree's humanoid demonstrations were raised as a comparison. What was
read: Unitree publishes software for steering its humanoids' arms and
hands by a person wearing a headset
([xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate):
Apple Vision Pro, PICO 4 Ultra Enterprise, Meta Quest 3), and its own
manipulation model
([UnifoLM-VLA-0](https://github.com/unitreerobotics/unifolm-vla)) is
described as completing "12 categories of complex manipulation tasks",
with no success count in its README.

<a id="r-so-101-bench"></a>
## SO-101 Bench: what taking it over would need

Added later on 2026-11-21. Read in the repository
[5hadytru/so101_bench](https://github.com/5hadytru/so101_bench) (Apache
2.0, last push 2026-11-04) and in
[DAGroup-PKU/PyRUA-Lean](https://github.com/DAGroup-PKU/PyRUA-Lean)
(Apache 2.0). Nothing was installed.

- **The agent cannot be run from published code.** The repository holds
  only the bridge between arm and agent. It starts
  `pyrualean play --robot so101_bench` from a folder on the author's
  machine, `~/workspace/PyRUA-Lean-so101-v96-matched`, and refuses to
  run when a file in that folder does not have a fixed checksum
  (`scripts/so101_real_pyrua.py:51`, `:156-162`, `:350-351`). The public
  PyRUA-Lean repository has no file for the SO-101 or SO-100 on any of
  its six branches; the author's 13 public GitHub repositories and his
  26 models and 37 datasets on Hugging Face hold no such port. The
  author was not asked.
- **The simulator needs an NVIDIA card.** The README requires Isaac Sim
  5.1 and names "one RTX GPU" as the tested setup (`README.md:20`). The
  workshop machine has an AMD graphics device, ROCm 7.2.4 (AMD's
  software for running such models on its graphics chips) and no
  `nvidia-smi`. The workshop's twin is built on SAPIEN.
- **Its tasks cover only task 1 of this file.** There is no stacking
  task. Its memory subset is about objects that get hidden from view,
  not about rebuilding a remembered stack.
- **Its real setup:** two colour cameras at 640×480, one above the table
  and one on the wrist, no depth camera
  (`scripts/so101_real_pyrua.py:10`, `:44`). The camera above the table
  is an iPhone aligned to the simulator's camera, and the calibration
  is the simulator's (`:11-13`). SAM 3 needs a 3.4 GB checkpoint and
  about 8 GB of graphics memory by the PyRUA-Lean documentation, and
  its download has to be requested on Hugging Face. The Galaxea 0.5
  model is 13.5 GB in full precision and shared a 24 GB graphics card
  with the simulator (`scripts/g05_server.py:56-57`, `:75`).
- **The language model can be exchanged.** The default is GPT-6.1 Sol
  through the Codex command-line tool; the harness also takes any
  endpoint that speaks the OpenAI chat protocol
  (`scripts/pyrua_eval.py:49-52`). The one comparison of models on the
  whole benchmark: score 83.2 with GPT-6.1 Sol against 66.8 with
  DeepSeek V4.1 Flash-1017; the two best policies alone score 61.4 and
  60.8.
- **What can be used:** the PyRUA-Lean framework, whose README points
  to a guide "Adding a Robot" (not read for this file); the published
  rules for
  when the agent looks while the policy drives
  (`scripts/pyrua_eval.py:194-209`); the fine-tuned policy checkpoints
  and the two data sets.

<a id="r-finding-6"></a>
## Finding 6: four ways to keep a policy from being pulled to the other objects; only one makes it see a single object

Added later on 2026-11-21. Nothing in a policy by itself hides the other
objects. The systems read for this file do one of four things.

1. **No policy in the pick.** A program measures where the chosen object
   lies and the arm goes there by inverse kinematics. RefGuard does it
   (its planner "samples grasps on the selected object"), TiPToP, the
   classical cube-stacking pipeline under [Finding 3](#r-finding-3) and `VLA-VLM` too.
2. **Blank out everything else before the policy gets the picture.**
   OBEYED-VLA, [arXiv 2512.22519](https://arxiv.org/abs/2512.22519): a
   segmentation model outlines every object, a vision-language model
   (Qwen3-VL 8B) picks the outlines that belong to the command, and all
   other pixels are removed in the fixed and in the wrist picture. The
   policy (π0) is then fine-tuned on such pictures, from demonstrations
   with a single object. Real UR10e arm, 100 runs per setting: the
   unchanged policies fall "below 10%" once other objects are added,
   OBEYED stays "above 90%" with one other object and "around 80%" with
   seven. The exact values are only in a chart. One control step takes
   0.62 to 0.78 s with π0. Stated weak point: the segmentation merging
   objects that lie close together. No test with two identical objects
   was found in it.
3. **Let the policy see everything and watch it.** SO-101 Bench: the
   grasp policy gets the unchanged pictures and the whole command
   (`scripts/pyrua_eval.py:272`, `:303`); the agent looks at least every
   5 seconds and on motion triggers and can take the arm over
   (`:194-209`). Result under [Finding 5](#r-finding-5): no wrong object in 45 real
   runs, 3 runs in which another object was moved. How the agent
   prepares the hand-off, for example whether it first brings the
   gripper over the target, was not found in the published files.
4. **Give the policy a hint and leave the picture alone.** Pointing-VLA
   (under [Finding 2](#r-finding-2)): 79 to 121 of 150, with at most one other object on
   the table.

For this workshop: the checkpoint in use puts a single 40 mm cube into
the bin in 10 of 10 layouts of the twin (`README.md`, table "Models
evaluated"). It was trained on unchanged pictures; how it behaves on
blanked-out pictures is not measured.

<a id="r-finding-7"></a>
## Finding 7: in commercial use, such models run one narrow task, are trained on that site's own data, and keep people as fallback; the picking that runs at the largest scale is not such a model

Added later on 2026-11-21. Every figure here was read on the linked
page or in the linked paper. All of them are the companies' own
statements; no independent audit of any of them was found.

| Who | What runs | What is published |
|---|---|---|
| Nomagic, [Fortune, 2026-07-08](https://fortune.com/2026/07/08/nomagics-new-ai-lab-headed-by-former-google-deepmind-researcher-claims-success-in-early-deployment-of-ai-brain-for-warehouse-robots/) | A VLA for the cases in which its warehouse picking robots got stuck, at the paying customer Brack.Alltron | Calls for human help "roughly halved". The company: "Our VLAs aren't at 99.9% success on their own yet — no one's customer-deployed VLAs are there yet"; its older classical software catches errors around the model |
| Physical Intelligence at Ultra, [company blog](https://www.pi.website/blog/partner) | π0.6 packing orders at a customer site | One full shift "at 96.4% autonomy"; a remote intervention team covers the rest |
| Physical Intelligence, π*0.6, [arXiv 2511.14759](https://arxiv.org/abs/2511.14759) | Espresso making, box assembly, laundry folding, after further training on the robot's own runs of that task | "On all of the tasks except diverse laundry, the success rate … is in the 90%+ range" |
| Figure at BMW, [company page](https://www.figure.ai/news/production-at-bmw) | Humanoid loading three sheet-metal parts into a welding fixture | More than 90,000 parts in more than 1,250 hours. Targets: above 99 % correct per shift and zero interventions per shift. The achieved values are not published |
| Figure in parcel sorting, [company page](https://www.figure.ai/news/scaling-helix-logistics) | Its model Helix turning parcels so the label can be scanned | Label correctly turned "~95%" of the time |
| Siemens factory trial, [arXiv 2605.27461](https://arxiv.org/abs/2605.27461) | π0.5 fine-tuned on 2,535 episodes (about 10 hours) recorded at the factory, packing an accessory bag into a box | "the final success rates of the finetuned policy did not meet expectations" |
| Sereact, [company site](https://sereact.ai/), [funding post](https://sereact.ai/posts/series-b) | Its model Cortex in warehouse picking and returns handling | Home page: "2B+" real production picks, "300+" live systems, ">99%" pick success, remote support intervention "~1 per 53,000 picks". The same page also gives "98% first-attempt pick success" and a customer quote, "After only a few weeks, the first robot achieved a picking success rate of over 96%". The funding post gives "1B+" picks and "200+" systems |
| Sereact, the model itself, [arXiv 2604.20246](https://arxiv.org/abs/2604.20246) | Cortex 2.0: a VLA with a second model in front that predicts several possible futures and takes the best-scored one before the arm moves. Inputs: wrist cameras, robot state and, where fitted, force and vacuum pressure. The fleet recorded over 500 million manipulation interactions; the model is trained on a selected 10 million of them and on other data | The paper's own four tests (16 trials of pick-and-place after fine-tuning on 160 episodes; 10 runs each of item sorting, screw sorting and shoebox handling): "success rates beyond 90% on all tasks", a model "that reaches 99% after continued operation on in-domain data", and zero human interventions |
| Amazon's parcel-picking fleet, [arXiv 2305.10272](https://arxiv.org/abs/2305.10272) | Not a VLA: a small learned model ranks candidate grasp points, then collisions are checked | 98 % success over more than 200 million picks |
| Amazon Vulcan, [Amazon Science](https://www.amazon.science/blog/how-amazons-vulcan-robots-use-touch-to-plan-and-execute-motions) | Not a VLA: a check decides whether the robot may try an item at all, suction pressure is watched during the pick, and after too many failures the item goes to a person | No success rate found on the page |
| Ocado, [company page](https://solutions.ocadogroup.com/en-us/products/on-grid-robotic-pick) | Robot arms picking groceries; a remote person guides the arm in exceptions | More than 350 million picks with more than 600 arms |

Reading these:

- Sereact is the one company found that claims a VLA at production
  reliability with rare human help. Not found on its pages or in the
  paper: how "pick success" is counted (its
  [Cortex page](https://sereact.ai/cortex) says "In picking, a miss is
  usually just a regrasp"); how many of the picks were done by
  Cortex 2.0, which the funding post announces as new; and any number
  for choosing a named item among look-alikes.
- Every other entry either keeps classical software or people around the
  model, or does not publish what it achieved.
- These are closed products or trials on a company's own site, trained
  further on that site's data: 2,535 episodes at Siemens, a selected
  10 million interactions at Sereact. What can be downloaded and run on
  an SO-101 are the models under Findings 1, 3 and 5.

<a id="r-finding-8"></a>
## Finding 8: the other objects can be taken out of an unchanged policy's pictures at run time, if they are painted over with background and not blacked out; the wrist camera is the weak spot

Added on 2026-11-21 at 12:27, after the user's objection that a policy
which picks a single object reliably should also pick it when the other
objects are hidden from it. Every number was read in the saved full text
of the paper at that time.

| Method | What it does to the pictures | Policy trained on such pictures? | Test | Result |
|---|---|---|---|---|
| CGVD, [arXiv 2603.10340](https://arxiv.org/abs/2603.10340), its Table II | SAM 3 outlines what is to stay (target, a second object the paper calls the anchor, robot) and the other objects; an inpainting model (LaMa) paints background over the others. This clean picture is made once per episode; every later picture is blended with it | No ("training-free") | Simulation (SimplerEnv), WidowX arm, one fixed camera, π0, task "spoon on towel" with 18 look-alike objects added, 200 episodes per row | Unchanged pictures 43.0 %. CGVD 77.5 %. The same with the other objects filled with their mean colour instead of inpainted: 56.5 %. Without its check that the target is not painted over: 65.0 % |
| BYOVLA, [arXiv 2410.01971](https://arxiv.org/abs/2410.01971) | GPT-4o names the regions that do not belong to the task, once per episode; Grounded-SAM2 outlines them at every step; a probe finds which of them the policy reacts to; those objects are inpainted (Inpaint Anything) | No ("without model fine-tuning or access to the model's weights") | Real WidowX 250S arm. Octo-Base, 30 trials per method | Other objects lower the success rate "by 40%"; with BYOVLA the policy keeps "the nominal task success rate of 67%" it has without them |
| Same paper | | No | OpenVLA, 15 trials per method, a more cluttered scene | Without other objects a "perfect task success rate"; other objects lower it "by 40%"; BYOVLA improves that "by 20-25%" and does not get back to the rate without them. Stated reason: GPT-4o "often fails to locate all task-irrelevant objects" |
| ARRO, [arXiv 2505.08627](https://arxiv.org/abs/2505.08627), its section on language guidance | An object detector (Grounding DINO) finds the target objects in the first picture, a tracker follows them and the gripper fingers; everything else becomes black or a drawn grid. With a wrist camera: black in the wrist picture, grid only in the side picture | Yes: the edit "is applied to all frames" of the training set and at run time | Real Franka arm, π0 fine-tuned with a side and a wrist camera, command to pick the green cuboid with a red cuboid of the same shape added; number of trials not found in that section | Unchanged pictures 30 % ("often grasps the wrong cuboid"). Everything else black: 70 %. Grid: 90 % |

Reading these:

- The sentence under ["Main conclusion"](#r-main-conclusion) that a policy has to be trained
  for it was wrong (see the amendment there). CGVD and BYOVLA leave the
  policy as it is.
- How the other objects are removed matters. In CGVD's own comparison a
  plain colour fill reaches 56.5 % where inpainting reaches 77.5 %; the
  authors put it down to "stark, unnatural region boundaries". ARRO's
  black background works with a policy that was trained on it.
- Removing objects is not always a gain. On CGVD's second task (carrot
  on plate) it "consistently underperforms the baseline"; the authors
  suspect that this policy uses the other objects as visual anchors.
- CGVD's clean picture is made once. Its authors name the limit: "If a
  distractor is moved dynamically, the cached background will
  desynchronize from the physical scene." Its cost: 4,914 ms once at the
  start of an episode, then 421 ms per step against 317 ms for π0
  alone (its Table III).
- The wrist camera is the weak spot. CGVD uses "a single fixed
  third-person camera". ARRO, the one of the three with a wrist camera,
  writes that its result "depends heavily on the underlying segmentation
  model, especially for wrist cameras, where objects can appear or
  disappear abruptly" and that "task-relevant objects were occasionally
  not tracked reliably".
- A later paper that compares itself with ARRO on a WidowX arm (PEEK,
  [arXiv 2509.18282](https://arxiv.org/abs/2509.18282)) reports that
  "ARRO does not improve overall success rates of either base model as
  its pre-trained object detection module often fails to identify
  correct objects in clutter". The part that finds the target decides
  the result.
- None of the three was run on an SO-101 or with SmolVLA. The workshop's
  10 of 10 is one 40 mm cube into the bin (`README.md`, table "Models
  evaluated"); with the other objects painted over it has not been
  measured.

  Amendment, 2026-11-22: measured since, in simulation only, with the
  workshop's unchanged SmolVLA checkpoint on a three-cube bin test
  (three 40 mm cubes, one command per cube, layout seeds 0 to 9 in every
  run; `TESTING_LEDGER.md`, sections "2026-11-22 — Three cubes into one
  bin" and "… by SAM 3's outlines"). All three cubes in the bin at the
  end:

  | Pictures given to SmolVLA | Layouts | Commands whose cube was in the bin |
  |---|---:|---:|
  | as the cameras take them | 0 of 10 | 1 of 30 |
  | the two other cubes painted over with table, by the simulator's outlines | 7 of 10 | 26 of 30 |
  | the same, by SAM 3's outlines (phrase "cube", tracked) | 9 of 10 | 29 of 30 |
  | the same, by the outlines of the fast SAM 3 reference (`sam3-tracker-rocm`, 0.3 s per picture instead of 7 to 9 s) | 9 of 10 | 29 of 30 |
  | painted over on the table only, a cube already in the bin left visible | 1 of 10 | 16 of 30 |

  Ten layouts do not show that 9 differs from 7. They do show the step
  from 0 to 7 and 9 layouts, and that most of it is lost when a cube
  already in the bin stays visible. Not run on the real arm.

  Added after 17:39 on 2026-11-22: the last row. The two SAM 3 routes
  give the same counts on the same ten layouts, with two layouts
  changing outcome in opposite directions; the reference is now the
  workshop's SAM 3. Detail, layout by layout and per call:
  [`TESTING_LEDGER.md`](TESTING_LEDGER.md#t-2026-11-22-fast-sam3-run);
  what it means for the work: [`DEVELOPMENT_LEDGER.md`](DEVELOPMENT_LEDGER.md#d-2026-11-22-fast-sam3).

<a id="r-finding-9"></a>
## Finding 9: playing all 50 predicted actions before the next look is the shipped default, not the best published method; a prediction at every step is not smoother by itself

Added on 2026-11-22 after 14:13, on the user's question whether "50
actions, then new pictures" is the current method and whether it makes
the arm jitter in MuJoCo. Numbers were read that day in the saved full
texts of the two papers; the two rows marked "excerpt" were read only
in a search excerpt of the page.

Two terms: a **chunk** is the list of future actions one prediction
gives (50 for SmolVLA). **Actions played** is how many of them the robot
carries out before new pictures are taken (`n_action_steps` in LeRobot).

| Source | What it says | Numbers |
|---|---|---|
| SmolVLA paper, [arXiv 2506.01844](https://arxiv.org/abs/2506.01844), Table 13 | Average success on the simulated LIBERO benchmark, by actions played before new pictures | 1: 80.3 %. 10: 82.8 %. 30: 70.8 %. 50: 51.8 % |
| Same, Table 12 | The same benchmark, by chunk size | 1: 50.0 %. 10: 84.0 %. 30: 78.5 %. 50: 80.3 %. 100: 74.5 % |
| Same, sections 3.3 and 4.6, Figure 5 | Playing the whole chunk ("sync") "reflects a standard evaluation setting in robotics" and "inherently hinders the responsiveness". The paper's alternative ("async") asks for the next chunk once the queue falls below 70 % and joins the overlap | Real robot, three tasks, sync against async: success 78.3 against 73.3 % on average (pick-place 75 / 80, stacking 90 / 90, sorting 70 / 50); 13.75 against 9.70 s per pick-place; 9 against 19 cubes in a fixed time |
| Real-Time Chunking (RTC), [arXiv 2506.07339](https://arxiv.org/abs/2506.07339), section 2 | Actions played is "often … shorter than the prediction horizon, but still much greater than 1 (e.g., s≈H/2)". "A long execution horizon reduces a policy's responsiveness to new information, while a short one increases the likelihood of mode-jumping, jerky behavior resulting from discontinuities between chunks" | – |
| Same, sections 4.2 and 6, appendix A.5 | RTC computes the next chunk while the current one plays and holds its beginning to the actions already under way; "applicable to any diffusion- or flow-based VLA out of the box with no re-training". Tested on a real two-arm robot with π0.5 | Chunk 50, at least 25 played. Averaging overlapping predictions (temporal ensembling, from the ACT policy): both variants "do not run at all due to causing such high oscillations that the robot's protective stop is triggered". Stated cost of RTC: "adds significant computational overhead". Success counts are in charts and were not read |
| [LeRobot documentation of RTC](https://huggingface.co/docs/lerobot/rtc) (excerpt) | Works with SmolVLA without retraining | `execution_horizon`: "Typical values: 8-12 steps" |
| [A blog's list of defaults](https://www.ay-robots.com/blog/en/action-chunking-explained) (excerpt; the primary sources were not checked) | – | Diffusion Policy predicts 16 and plays 8. GR00T N1.7 predicts 40 and plays 8 in NVIDIA's example. LeRobot's SmolVLA and π0.5 predict 50 and play 50 |

Reading these:

- "All 50, then look again" is what the workshop's checkpoint ships
  with (`MOTION.md:3`) and what its authors call standard. In their own
  table it is the worst of four settings (51.8 % against 82.8 % at 10).
  That table is LIBERO in simulation, not an SO-101.
- A prediction at every step is not smoother by itself. Each chunk is
  drawn from fresh random numbers, so two predictions can disagree, and
  the arm jumps where one takes over from the other. The settings these
  sources recommend play 8 to 25 actions; playing 1 scores 80.3 % in
  Table 13, below the 82.8 % at 10.
- The published way to look more often without those jumps is RTC. The
  LeRobot 0.7.0 installed in this workshop's `.venv` contains it for
  SmolVLA (`lerobot/policies/rtc/`; `rtc_config` at
  `configuration_smolvla.py:105`). It runs through
  `predict_action_chunk`, not through `select_action`
  (`modeling_smolvla.py:270-272`), which is the call the workshop's
  runner uses (`src/so101_smolvla/sapien_rollout.py:98`). Not tried
  here.
- The jitter in MuJoCo is not caused by the 50. This workshop measured
  it inside each chunk: 167 and 181 direction reversals in 500 targets
  with only 9 chunk boundaries (`DEVELOPMENT_LEDGER.md:13`); 33.3 to
  58.7 % reversals inside single predictions on an exact training
  picture (`:181-185`); part of it traced to a LeRobot training fault in
  the padded action dimensions (`:218-228`); hidden in SAPIEN by its
  more damped controller (`:245-249`).
- Asking this checkpoint more often has not helped so far, on the
  stacking test in SAPIEN: every 10 steps 0 stacks of 20
  (`DEVELOPMENT_LEDGER.md:1529-1536`); every 25 steps 21 of 60 against
  17 of 60, with overlapping ranges (`:3474-3477`). Jitter was not
  counted in those runs.

<a id="r-what-follows-for-this-workshop"></a>
## What follows for this workshop

- The plan to fine-tune SmolVLA so that it chooses one of four objects by
  the sentence
  (`~/.cursor/plans/four-object_sorting_fine-tune_608f9494.plan.md`) asks
  the policy for exactly what [Finding 1](#r-finding-1) shows to be unreliable. The user
  dropped that plan on 2026-11-21.
- Whatever replaces it follows the main conclusion at the top of this
  file: a separate part decides which object is meant, and the policy,
  if one is kept, is told by a mask.
- Open, and the user's decision: which form of it. Way 1 of [Finding 6](#r-finding-6)
  (no policy in the pick, as `VLA-VLM` does it), way 2 (SmolVLA kept for
  the motion, every other object blanked out of its pictures), or a
  policy that takes the mask as an input (MaskWAM, MaskHarness-WAM).
  None has been tried in this workshop.

  Amendment, 2026-11-21 12:27: the user decided on 2026-11-21 for way 2
  in the form of [Finding 8](#r-finding-8). The arm is moved by SmolVLA only; Gemma 4
  E4B (the model already used for voice control) hears, reads the scene
  and plans; SAM 3 makes and follows the outlines; the other objects are
  painted over in SmolVLA's pictures. Not tried in this workshop yet.

  Amendment, 2026-11-21 13:08: the spec for it is `SPEC.md` in this
  folder (draft 1). After the test under "Unfinished" (Gemma 4 E4B: 0 of
  12 pairs read correctly from raw pictures) the spec has Gemma hear the
  command and turn it into fields, and has a program read colour, size
  and place inside SAM 3's outlines. That change from the line above is
  put to the user in the spec's section 11.
- Task 3 does not need a policy with memory. The memory belongs in the
  program or agent that calls the pick and place steps.
- Added 2026-11-22 ([Finding 9](#r-finding-9)): the checkpoint's 50 actions per
  prediction stay until a test says otherwise. Every extra prediction
  also costs one SAM 3 call per camera, so fewer actions per prediction
  are only practical with the fast SAM 3. The test that would decide is
  listed as open in `TESTING_LEDGER.md`, section ["2026-11-22 — How often
  SmolVLA is asked for a new plan"](TESTING_LEDGER.md#t-2026-11-22-how-often-smolvla-is-asked-for); whether to run it is the user's
  decision.
- Added 2026-11-22: way 2 in the form of [Finding 8](#r-finding-8) has now been tried in
  simulation (amendment under [Finding 8](#r-finding-8)), so "Not tried in this workshop
  yet" in the amendment of 2026-11-21 12:27 above no longer holds for
  the simulation. What stands in the way of using it is SAM 3's time:
  through plain PyTorch a call took 6.93 s (top camera) and 9.18 s
  (wrist camera) in the median, 3,901 s of SAM 3 in a run that lasted
  from 13:25:52 to 14:49:56. The fast reference the user named,
  `sam3-tracker-rocm`, publishes 89.77 to 90.15 ms per frame for one
  object at 504 pixels (its `README.md`). It is being set up without
  Docker in a Python environment of its own; it has not yet run on this
  machine's graphics processor.

  Amendment, 2026-11-22 after 17:17: it runs now, without Docker, in
  `.venv-sam3` (model files built at 504 pixels; one MIOpen setting
  needed, `TESTING_LEDGER.md`, ["The fast SAM 3 model built"](TESTING_LEDGER.md#t-2026-11-22-fast-sam3-built)). On the
  stored plan pictures of two layouts (48 per camera, 640 by 480) it
  took 0.25 s per picture in the median against 6.55 s (top camera) and
  8.61 s (wrist) for the PyTorch route, and outlined the same cubes in
  250 of the 254 (plan, cube) pairs in which either route outlined a
  cube, with a median overlap of 0.95 (top) and 0.98 (wrist) between the
  two routes' outlines (`TESTING_LEDGER.md`, ["The fast SAM 3 connected"](TESTING_LEDGER.md#t-2026-11-22-fast-sam3-connected)).
  The 10-layout run with it, same layouts and seeds as the 9 of 10 above,
  was in progress when this was written; its result goes into the table
  under [Finding 8](#r-finding-8) when it is in.

  Amendment, 2026-11-22 after 17:23: on the user's word the reference
  now is the workshop's SAM 3 (`--outlines sam3`); the plain PyTorch
  route is removed from the code. The user holds separate evidence that
  the plain PyTorch SAM 3 can also be made significantly faster, should
  it ever be needed again; that evidence is not part of this repository
  and was not read for this file.

<a id="r-not-verified-not-measured"></a>
## Not verified, not measured

- Nothing here was run on this workshop's SO-101, in simulation or real.
- How fast and how well SAM 3 (on this machine as `facebook/sam3`) finds
  cubes, balls and cylinders in this workshop's camera pictures.
- Whether SAM 3 runs on this machine's AMD graphics under ROCm.
- How the workshop's SmolVLA checkpoint behaves on pictures in which
  every object but one is blanked out.

  Amendment, 2026-11-22, to the four bullets above: answered in part,
  in simulation. SAM 3 runs on this machine's AMD graphics under ROCm
  through plain PyTorch, at seconds per picture, and the checkpoint was
  run on pictures with the other cubes painted over (amendment under
  [Finding 8](#r-finding-8); counts and times in `TESTING_LEDGER.md`, sections of
  2026-11-22). Still not measured: anything on the real SO-101; SAM 3 on
  balls and cylinders; the fast SAM 3 reference on this machine.

  Amendment, 2026-11-22 after 17:17: the fast SAM 3 reference has now
  run on this machine (the amendment at the end of ["What follows for
  this workshop"](#r-what-follows-for-this-workshop)). Still not measured: the real SO-101, and SAM 3 on
  balls and cylinders.
- The number of simulation trials per task in RMBench.
- Success counts of two of the SO-101 example projects (so101-gap and
  the Gemini Robotics-ER 2 post).
- For the "99.2 %" PickCube row on the SOTA2 leaderboard: the paper
  behind it, the number of episodes, and whether the policy saw camera
  pictures.
- The exact success values of OBEYED-VLA; the paper's text gives rounded
  values and the rest is in charts.
- Whether the SO-101 port of PyRUA-Lean can be had from its author.
- Any independent figure for Sereact's pick success or intervention
  rate.
- For [Finding 9](#r-finding-9): whether fewer actions per prediction, or RTC, reduces
  the jitter or raises the success of this workshop's checkpoint; RTC's
  success counts; the defaults in the blog's list against their primary
  sources.
- Read earlier in the same session and not read again for this file, so
  left out: results on memory held by an agent
  ([arXiv 2609.11308](https://arxiv.org/abs/2609.11308),
  [arXiv 2606.07723](https://arxiv.org/abs/2606.07723)) and the
  comparison of SmolVLA with larger models under changed conditions
  ([arXiv 2611.13160](https://arxiv.org/abs/2611.13160)).

<a id="r-unfinished"></a>
## Unfinished

A test of whether a vision-language model can read a stack and plan the
rebuild from two pictures was begun on 2026-11-20 and stopped before any
result. Its files are in `~/.cache/stack-memory-agent-test/` (24 rendered
pictures of 12 scene pairs, a scorer, saved answers of one
vision-language model).

Amendment, 2026-11-21 11:46: the saved answers were scored on
2026-11-21 at 11:24 (`logs/score-session-1795256665.log` in that
folder): 12 of 12 pairs fully correct, 42 of 42 pick points on the right
cube. The pictures are rendered, the test covers reading and planning
only, and the plain program it is to be compared with has not been run.
No conclusion on agent against program may be drawn yet.

Amendment, 2026-11-21 13:06: the same 12 pairs were put to Gemma 4 E4B
(the 4-bit file `gemma-4-E4B-it-Q4_0.gguf`, the model the voice control
of `workshops/vvla-pipeline` uses), through that workshop's own
`llama-server`, started by the client of branch
`gemma4-e4b-it_voice_intent` (commit `1d231e1`). Job
`gemma-12-pairs-1795262096`, 12:54:56 to 13:00:11, exit status 0; log
`logs/gemma-12-pairs-1795262096.log` in the test folder. Each pair was
one request with both pictures and the same question as for the other
subject, answered once, scored by the same scorer.

| Subject | Tokens per picture (about) | Seconds per request | Pairs fully correct | Stack colours in the right order | Pick points on the right cube | Marker end, picture 1 / picture 2 |
|---|---|---|---|---|---|---|
| The model of this Cursor session (a large hosted model), scored at 11:24 | – | not measured | 12 of 12 | 12 of 12 | 42 of 42 | 12 / 12 of 12 |
| Gemma 4 E4B, server as configured for voice (context 2,048) | 136 | 5.2 to 8.3 | 0 of 12 | 0 of 12 | 0 of 42 | 7 / 9 of 12 |
| Gemma 4 E4B, 560 tokens per picture, context 4,096 | 594 | 7.0 to 10.2 | 0 of 12 | 2 of 12 | 0 of 42 | 11 / 10 of 12 |
| Gemma 4 E4B, 1,120 tokens per picture, context 4,096 | 1,137 | 10.2 to 13.3 | 0 of 12 | 0 of 12 | 0 of 42 | 12 / 10 of 12 |

Reading these:

- A picture request works on the existing server build: every one of
  the 36 requests was answered (HTTP 200) with a readable JSON object.
  "Tokens per picture" is half of what is left of the request's prompt
  tokens (691, 1,607 and 2,693) after the 419 tokens of its text, so it
  includes a few tokens of the chat template.
- Gemma 4 E4B does not read this task from the raw pictures at any of
  the three picture sizes. It gave the right number of moves in 4, 3 and
  1 of 12 pairs, and in several pairs it reported the marker's two
  colours (red, blue) as the stack.
- Its pick points are not a matter of how the two numbers are written.
  Read in four ways (x before y or y before x, in pixels or on a grid of
  1,000 steps), at most 6 of the 30 to 34 points given in a run land on
  any cube at all.
- The marker has two possible answers, so 7 to 12 of 12 is between
  guessing and right.
- Not run: the plain program. Its file was not written, because the
  editor refused Python files in this session. Not tested: Gemma 4 E4B
  on a picture in which SAM 3's outlines are drawn and numbered, or on a
  cut-out of a single object.
- What can be said: for this task, on rendered pictures, a large hosted
  model as agent reads and plans without error, and the small local
  model the user wants as the one language model does not. Whether a
  plain program does it is still not measured.
