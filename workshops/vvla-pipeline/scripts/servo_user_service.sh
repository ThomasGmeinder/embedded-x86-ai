#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# Run the given command as a user service and stop that service when this
# process is signalled.
#
# Servo sets SCHED_FIFO before it loads the arm model. This desktop session
# inherited rtkit's 200 ms realtime budget, and a child cannot raise that hard
# limit. The user service manager's budget is unlimited.
#
# Launch signals the process group of the process it started. systemd-run dies
# from that signal without stopping the service it created, so the service is
# started in its own session and this process stops the unit.

set -u

unit="vvla-servo-$$"
keys=(
    ROS_DOMAIN_ID
    ROS_LOCALHOST_ONLY
    ROS_DISTRO
    RMW_IMPLEMENTATION
    AMENT_PREFIX_PATH
    LD_LIBRARY_PATH
    PATH
    FASTRTPS_DEFAULT_PROFILES_FILE
    CYCLONEDDS_URI
    FASTDDS_BUILTIN_TRANSPORTS
)

stop_unit() {
    systemctl --user stop "$unit" || true
    exit 0
}
trap stop_unit INT TERM

args=(systemd-run --user --unit "$unit" --wait --pipe --collect --property=TimeoutStopSec=5s)
for key in "${keys[@]}"; do
    value="${!key-}"
    if [[ -n "$value" ]]; then
        args+=("--setenv=${key}=${value}")
    fi
done

setsid "${args[@]}" "$@" &
child=$!
wait "$child"
exit $?
