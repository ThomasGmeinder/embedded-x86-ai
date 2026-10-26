# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""End-to-end workshop check (PROVIDED) - run: ``python -m vvla_workshop.selftest``

One command to answer the question you actually care about: *is the only thing
standing between me and the finished pipeline the TODOs I haven't written yet?*

It does three things, all without a robot, an NPU, a GPU, or a webcam. ROS 2
is required (``source /opt/ros/jazzy/setup.bash``): component 2 runs on it.

1. **TODO scan** - reads ``workshop/project/`` and lists which TODOs are still stubs.
2. **The three component harnesses** - runs ``selftest_models``,
   ``selftest_ros`` and ``selftest_integration`` on your files and tallies
   every PASS / FAIL / TODO / SKIP into one scoreboard. A FAIL means code you
   wrote does the wrong thing; a TODO means a stub you haven't filled in yet.
3. **Reference parity (the oracle)** - runs those same harnesses with
   ``--reference``, on the pipeline's finished functions, and confirms they
   come back green. That's the proof the harness itself is trustworthy: if
   the reference passes every check this machine can run, then "no FAILs on
   your files" really does mean the only gap left is your remaining TODOs.

Verdict and exit code:
  - any FAIL            -> exit 1  ("fix this - it's not a missing TODO")
  - no FAIL, TODOs left -> exit 0  (or 1 with --strict; "here's what's left")
  - everything passes   -> exit 0  ("matches the pipeline end to end")

Flags:
  --strict        treat remaining TODOs as failure (handy for CI / "am I done?")
  --no-reference  skip the reference oracle
  --quiet         show only the scoreboards, not each harness's full output
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

from vvla_workshop import todos

_TALLY = re.compile(r"PASS\s+(\d+)\s+FAIL\s+(\d+)\s+TODO\s+(\d+)\s+SKIP\s+(\d+)")

_COMPONENTS = [
    ("Component 1  models       ", "vvla_workshop.selftest_models"),
    ("Component 2  ros          ", "vvla_workshop.selftest_ros"),
    ("Component 3  integration  ", "vvla_workshop.selftest_integration"),
]


def _run_harness(module: str, reference: bool, echo: bool):
    """Run one harness as a subprocess. Returns (returncode, pass, fail, todo, skip).

    Counts are parsed from its summary line; a crash (no summary) is surfaced
    as a synthetic FAIL so it can never be silently swallowed.
    """
    argv = [sys.executable, "-m", module] + (["--reference"] if reference else [])
    proc = subprocess.run(argv, capture_output=True, text=True)
    out = proc.stdout + ("\n" + proc.stderr if proc.stderr.strip() else "")
    if echo:
        print(out.rstrip("\n"))
    matches = _TALLY.findall(out)
    if not matches:
        if echo:
            print("  (!) no summary line - the harness crashed before finishing")
        return proc.returncode, 0, 1, 0, 0  # count the crash as a FAIL
    p, f, t, s = (int(x) for x in matches[-1])
    return proc.returncode, p, f, t, s


def _run_suite(reference: bool, *, echo: bool):
    """Run all three harnesses; return the summed (pass, fail, todo, skip) and the rows."""
    tot = [0, 0, 0, 0]
    rows = []
    for label, module in _COMPONENTS:
        if echo:
            print(f"\n{'=' * 74}\n>>> {label.strip()}   (python -m {module}{' --reference' if reference else ''})"
                  f"\n{'=' * 74}")
        _, p, f, t, s = _run_harness(module, reference, echo)
        rows.append((label, p, f, t, s))
        for i, v in enumerate((p, f, t, s)):
            tot[i] += v
    return tot, rows


def _scoreboard(title: str, rows, tot) -> None:
    """Print a compact PASS/FAIL/TODO/SKIP table for a suite run."""
    print(f"\n{title}")
    print(f"  {'':26}{'PASS':>6}{'FAIL':>6}{'TODO':>6}{'SKIP':>6}")
    for label, p, f, t, s in rows:
        print(f"  {label:26}{p:>6}{f:>6}{t:>6}{s:>6}")
    print(f"  {'-' * 50}")
    print(f"  {'TOTAL':26}{tot[0]:>6}{tot[1]:>6}{tot[2]:>6}{tot[3]:>6}")


def main() -> int:
    """Scan TODOs, run the three harnesses, check the reference oracle, print a verdict."""
    ap = argparse.ArgumentParser(description="Run the whole workshop self-test suite end to end.")
    ap.add_argument("--strict", action="store_true", help="exit nonzero if any TODO is still unimplemented")
    ap.add_argument("--no-reference", action="store_true", help="skip the reference parity (oracle) check")
    ap.add_argument("--quiet", action="store_true", help="show only the scoreboards, not each harness's full output")
    args = ap.parse_args()
    echo = not args.quiet

    print("=" * 74)
    print("  VVLA workshop - end-to-end self test")
    print("  (ROS 2 required; no NPU / GPU / camera needed - those backends are synthetic)")
    print("=" * 74)

    # -- 1. static TODO scan ------------------------------------------------
    left = todos.remaining()
    print("\n-- TODO scan --")
    print("  " + todos.progress_line())
    print(todos.format_list(left) if left else "  every TODO is implemented - nice.")

    # -- 2. the three component harnesses on YOUR files --------------------
    tot, rows = _run_suite(False, echo=echo)
    _scoreboard("-- your scoreboard --", rows, tot)
    your_fail = tot[1]

    # -- 3. reference parity (the oracle) -----------------------------------
    oracle_ok = None
    if not args.no_reference:
        print(f"\n{'=' * 74}\n>>> reference parity (oracle): the same harnesses on the pipeline\n{'=' * 74}")
        ref_tot, ref_rows = _run_suite(True, echo=False)
        _scoreboard("-- reference scoreboard --", ref_rows, ref_tot)
        oracle_ok = ref_tot[1] == 0 and ref_tot[2] == 0  # no FAIL, no TODO
        if oracle_ok:
            print(f"  the reference passes all {ref_tot[0]} runnable checks here -> the harness is trustworthy.")
        else:
            print(f"  (!) the reference did NOT come back clean on this machine (FAIL {ref_tot[1]}, TODO {ref_tot[2]}).")
            print("      that points at the environment or the harness, not your code - investigate before "
                  "trusting the verdict.")

    # -- verdict ------------------------------------------------------------
    print("\n" + "=" * 74)
    if your_fail:
        print(f"  X  {your_fail} check(s) FAIL. That is code that RAN but did the wrong thing -")
        print("     not a missing TODO. Read the FAIL hint above and fix it before moving on.")
        code = 1
    elif left:
        print(f"  OK so far: 0 failures. What's left is exactly {len(left)} TODO(s):")
        print("     " + todos.compact(left))
        print("     Implement them (each TODO's reference in vla_pipeline/ is listed in feedback.TODO_INDEX)")
        print("     and re-run. Nothing you've written so far conflicts with the pipeline's behavior.")
        code = 1 if args.strict else 0
    else:
        print("  DONE: every check passes and no TODOs remain - your functions match the pipeline's")
        print("     behavior end to end.")
        code = 0
    if oracle_ok is False:
        code = code or 1
    print("=" * 74)
    return code


if __name__ == "__main__":
    sys.exit(main())
