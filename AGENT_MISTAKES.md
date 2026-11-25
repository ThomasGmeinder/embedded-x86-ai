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
