# Agent Mistakes

This branch (`vvla-moveit-arm`) had no journal before 2026-11-25. The voice branches
(`gemma4-e4b-it_voice_intent`, `d1-omni-2b-voice_intent`) carry their own, with earlier entries.

## 2026-11-25 — Said a stashed design was "never merged" from its file list alone

- **Agent/tool:** Claude Code (Claude Fable 5.2)
- **Task:** Describe what a stash made on `vvla-moveit-arm` contained
- **Mistake:** Told the user that none of the stash's new modules exist on the current
  branches and that the three-process design was never merged
- **Evidence:** `git log 4aa837e..vvla-moveit-arm`: the same work was committed on that branch
  three and a half hours after the stash (`b81471b` to `fa5516d`); 20 of the stash's 31 source
  files are identical on the branch tip
- **Cause:** Read only `git stash show --stat` and did not compare the stash with the branch
  it was made on
- **Correction:** Compared every stashed file with the branch tip; reported the stash as
  obsolete; the user had it dropped
- **Prevention:** Before describing a stash as unmerged work, diff it against the tip of the
  branch named in its title
- **Status:** corrected

## 2026-11-25 — Started a test run while the setup script was still running

- **Agent/tool:** Claude Code (Claude Fable 5.2)
- **Task:** Run `./run_sim_tests.sh` on `vvla-moveit-arm` after the user ran `./bootstrap.sh`
- **Mistake:** Launched the suite when the user wrote that they had run bootstrap; the script
  was still running in their terminal
- **Evidence:** The user: "looks like the bootstrap.sh is still running in the other terminal";
  the process `bash ./bootstrap.sh --skip-models` was alive during the run of 19:20 to 19:21
- **Cause:** Checked that the packages and the Python module were installed, not that the
  setup script had exited
- **Correction:** Discarded that run as evidence and reran once the script's process was gone
- **Prevention:** The readiness check before a test run includes "no setup script of this
  repository is running", by its process id
- **Status:** corrected

## 2026-11-25 — Two test commands of mine did not do what their wrapper promised

- **Agent/tool:** Claude Code (Claude Fable 5.2)
- **Task:** Run tests in tmux with a log, an exit file and a completion event
- **Mistake:** (1) Sourced `scripts/test_env.sh` in the wrapper shell; it turns on
  exit-on-error, so the shell ended at the failing pytest and the exit file was never written.
  (2) Guarded a run with `pgrep -f 'bootstrap.sh'`, which matched the guard's own command
  line, so the run did not start
- **Evidence:** `cat: /tmp/vvla-moveit-port-1.exit: No such file or directory` after the 19:05
  run; "bootstrap still running - not starting" at 19:23 when the script had ended
- **Cause:** Did not read what the sourced file sets; matched a process by a pattern that the
  matching command itself contains
- **Correction:** Later wrappers run the repository's own script as a child and write its
  status unconditionally; the guard checks the script's process id with `kill -0`
- **Prevention:** Never source a repository script into the shell that writes the exit file;
  never match a process with `pgrep -f` on text that is in the command doing the matching
- **Status:** corrected

## 2026-11-25 — Started a second twin while the user's own session was running

- **Agent/tool:** Claude Code (Claude Fable 5.2)
- **Task:** Measure how fast the twin's clock runs, to explain "the robot moves slowly"
- **Mistake:** The command listed the user's running `run_pipeline.sh --digital-twin` processes
  and then started a second arm launch anyway, on the same pinned CPU cores; the probe also
  ended with `os._exit` and lost its printed result
- **Evidence:** The process list in that command's own output (`so101_sim.stack serve`,
  perception, console, control, `ros2_control_node`) followed by an empty probe result
- **Cause:** The check for running sessions and the start of the probe were one command, so
  the check could not stop the start; `os._exit` does not flush a piped stdout
- **Correction:** Confirmed the probe left nothing running; measured the user's running stack
  passively instead (a subscriber on `/clock`, nothing started): 1.000 x real time
- **Prevention:** A check for running sessions is its own command, read before anything is
  started; a probe that must measure a live system only listens
- **Status:** corrected

## 2026-11-25 — Ported two of the Gemma branch's fixes when told to proceed with the list

- **Agent/tool:** Claude Code (Claude Fable 5.2)
- **Task:** Apply the Gemma 4 branch's fixes to `vvla-moveit-arm`
- **Mistake:** After listing which fixes apply, and the user's "proceed", ported only the launch
  wrapper and the stop bound; left the voice fix (`58fab42`: sentence sent at end of speech,
  command before transcript) as "your decision" and closed the work without it
- **Evidence:** The user tested `run_pipeline.sh --digital-twin`: voice slower, transcript before
  command; then: "I earlier told you to apply all the fixes from the gemma branch to this branch"
- **Cause:** Read "proceed" as approval of the first numbered item only, and did not restate in
  the closing report that the voice path was still the old one
- **Correction:** The voice fix is being ported; see the commits after this entry
- **Prevention:** When a "proceed" follows a list, either do every item or say in the same
  reply which items are not being done and why
- **Status:** unresolved

  **Amendment, 2026-11-26:** corrected. The voice fix is ported in `d759f20` (listener,
  streamed request, command before its words across the three processes, the voice server
  off its starter's one core), the documents in `8722efc`, the camera waits in `80964c8`.
  `tests/sim/test_voice_live.py` against the real Gemma: command 0.48 / 0.49 / 0.86 s after
  the last word. `./run_sim_tests.sh` on `d759f20`: exit 0. Still not ported, as the plan the
  user approved says: the Gemma branch's controller bring-up, whose spawners here live in
  `ros2_so_arm`.

## 2026-11-26 — Let a test suite go on opening a webcam whose driver was deadlocked

- **Agent/tool:** Claude Code (Claude Fable 5.2)
- **Task:** Run the notebook suite as soon as the user's own pipeline session had ended
- **Mistake:** Three things. (1) Armed a watcher that started the suite the instant the user's
  session released the webcam; the first test opened it eight seconds later, the driver failed
  to start the stream and deadlocked. (2) Having seen the camera node stuck in the kernel,
  wrote "I am letting the suite finish"; two more processes opened the webcam after that and
  are stuck too. (3) Told the user the node was stuck at shutdown by a signal arriving during
  the driver's stop call, and that a kernel worker had been stuck since 06:52
- **Evidence:** `journalctl -k`: 07:33:17 `amd_isp_capture: fail to get out prop`, `fail to setup
  output`; hung-task trace of `v4l2_camera_node` in `VIDIOC_STREAMON` →
  `isp4vid_qops_start_streaming` → `isp4sd_enable_streams` → `isp4sd_stop_stream` →
  `mutex_lock`. `ps`: four processes in state D, three of them in `_vb2_fop_release`, started
  07:35, 07:45 and 08:00. The kernel worker was idle on the next look; its elapsed time is its
  age
- **Cause:** (1) Treated "the session has ended" as "the webcam is ready", although this
  branch's notebooks say the camera can stay busy just after it is released. (2) Wanted the
  remaining results and did not weigh that every further open adds a process that cannot be
  killed. (3) Named a cause from the process state and this branch's comments before reading
  the kernel log, and read a worker's age as time spent stuck
- **Correction:** Stopped the suite; reported the kernel trace and withdrew both wrong
  statements. The webcam stays held until the machine is rebooted
- **Prevention:** A process in uninterruptible sleep on a device means: stop everything that
  touches that device, at once. Read the kernel log before naming the cause of a stuck
  process. Do not open the webcam right after another process has released it
- **Status:** unresolved (needs a reboot; the trigger is likely, not proven)
