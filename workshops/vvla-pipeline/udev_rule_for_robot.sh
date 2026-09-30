# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

SUBSYSTEM=="ttyACM0", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", MODE="0666"
sudo udevadm control --reload-rules
ls -l /dev/ttyACM0

