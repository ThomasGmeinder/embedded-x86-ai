# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# The twin and NPU test environment, sourced by run_sim_tests.sh and
# digital_twin_auto_testing.sh. It also works when sourced directly. This sets
# PY. Order: the Ryzen AI runtime, ROS 2, the ros_ws overlay.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; return 1; }
PY="${REPO_ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || die "no venv at ${REPO_ROOT}/.venv - run ./bootstrap.sh first"

say "Sourcing scripts/ryzen_ai_env.sh"
# shellcheck disable=SC1091
VIRTUAL_ENV="${REPO_ROOT}/.venv" source "${REPO_ROOT}/scripts/ryzen_ai_env.sh" \
  || die "ryzen_ai_env.sh failed"

ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
[[ -f "$ROS_SETUP" ]] || die "ROS 2 is required and is not at $ROS_SETUP - run ./bootstrap.sh"
say "Sourcing ROS 2 ($ROS_SETUP)"
set +u +e
# shellcheck disable=SC1090
source "$ROS_SETUP" || die "ROS 2 setup failed"
ROS_WS_SETUP="${REPO_ROOT}/ros_ws/install/setup.bash"
[[ -f "$ROS_WS_SETUP" ]] || die "no ros_ws overlay at $ROS_WS_SETUP - run ./bootstrap.sh"
say "Sourcing the ros_ws overlay ($ROS_WS_SETUP)"
# shellcheck disable=SC1090
source "$ROS_WS_SETUP" || die "ros_ws setup failed"
set -e -u

export VVLA_TEST_STRICT=1
