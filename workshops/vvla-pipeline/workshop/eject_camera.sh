# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

CAM_PID=$(sudo lsof -t /dev/video0)
echo $CAM_PID
kill -9 $CAM_PID