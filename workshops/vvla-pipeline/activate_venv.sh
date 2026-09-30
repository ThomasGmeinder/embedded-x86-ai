# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# source this file: source ./activate_venv.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/jazzy/setup.bash      # if using the ROS 2 transport
source "$HERE/.venv/bin/activate"
source "$HERE/scripts/ryzen_ai_env.sh"        # NPU: puts voe/lib on LD_LIBRARY_PATH

#"$HERE/workshop/launch_monitor.sh" &
