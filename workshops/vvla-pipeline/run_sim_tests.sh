#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# Twin and NPU tests. Notebooks are a different command:
#   cd workshop && ./run_notebooks.sh --test
#
#   ./run_sim_tests.sh                              # NPU load order, then the twin stacks
#   ./run_sim_tests.sh -k toothbrush                # args go to the twin stacks only
#   ./run_sim_tests.sh --sim-yolo-device cpu
#   ./run_sim_tests.sh --sim-video failed
#   ./run_sim_tests.sh --sim-eyes seg
#
# The first pytest run that fails ends this script with its exit status; the
# runs after it are not started.
#
# ./bootstrap.sh installs everything these tests import.
set -euo pipefail

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

say "run_sim_tests.sh${*:+ $*}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/test_env.sh"
LAST_TWIN_DOMAIN=""

# A MoveIt process that just exited can leave its DDS participant on that
# domain until the lease ends. The next twin then never sees /joint_states
# inside the bring-up bound. Pick a free domain, and not the one just used.
# so101_sim.domain_lock.owner is the lock the twin holds; this does not wait.
pick_twin_domain() {
  "$PY" - "$LAST_TWIN_DOMAIN" <<'PY'
import sys
from so101_sim.domain_lock import owner
avoid = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else None
for domain in range(90, 110):
    if avoid is not None and domain in (avoid - 1, avoid, avoid + 1):
        continue
    if owner(domain) is None and owner(domain + 1) is None:
        print(domain)
        break
else:
    sys.exit("no free ROS domain for a twin")
PY
}
say "NPU load order"
"$PY" -m pytest tests/npu
say "Stacks on the twin"
# A second MoveItPy in one process segfaults when its context is destroyed.
# The two twin modules each construct one, so each gets its own process.
"$PY" -m pytest --rootdir "$REPO_ROOT" tests/sim -m "sim and not docs" \
  --ignore=tests/sim/test_twin_infra.py --ignore=tests/sim/test_twin_ros.py "$@"
say "MoveIt twin, one MoveItPy per process"
# The shared fixtures keep one MoveItPy. Tests that construct their own need
# a free domain, so they are deselected here and run below. Collecting them
# together exits at once (tests/sim/conftest.py). Each process gets its own
# ROS domain so it does not meet the DDS participant the previous one left.
run_on_fresh_domain() {
  local domain
  domain="$(pick_twin_domain)" || die "no free ROS domain for a twin"
  say "ROS domain ${domain}"
  LAST_TWIN_DOMAIN="$domain"
  TWIN_TEST_DOMAIN_ID="$domain" "$@"
}
run_on_fresh_domain "$PY" -m pytest --rootdir "$REPO_ROOT" tests/sim/test_twin_infra.py -m "sim and not docs" \
  --deselect tests/sim/test_twin_infra.py::test_the_twin_domain_is_private \
  --deselect tests/sim/test_twin_infra.py::test_the_node_dies_with_its_kernel \
  --deselect tests/sim/test_twin_infra.py::test_reset_service_restores_the_scene_and_parks_the_arm \
  "$@"
for node in \
  tests/sim/test_twin_infra.py::test_the_twin_domain_is_private \
  tests/sim/test_twin_infra.py::test_the_node_dies_with_its_kernel \
  tests/sim/test_twin_infra.py::test_reset_service_restores_the_scene_and_parks_the_arm
do
  run_on_fresh_domain "$PY" -m pytest --rootdir "$REPO_ROOT" "$node" -m "sim and not docs" "$@"
done
run_on_fresh_domain "$PY" -m pytest --rootdir "$REPO_ROOT" tests/sim/test_twin_ros.py -m "sim and not docs" "$@"
say "MoveIt 2 on the twin"
"$PY" -m pytest --rootdir "$REPO_ROOT" tests/moveit
