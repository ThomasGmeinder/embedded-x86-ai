#!/usr/bin/env bash
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# One-time setup step (run once with sudo).
#
# Installs a systemd service that puts the Ryzen AI NPU into turbo power mode on
# every boot. Turbo resets to "Default" on each reboot, so without this you would
# have to run `sudo xrt-smi configure --pmode turbo` by hand every session. Baking
# it into the boot process means the NPU is always at full clock and no further
# sudo is needed to run the workshop.
#
# Usage:
#   sudo ./provisioning/install-npu-turbo.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "This provisioning script must be run as root (sudo)." >&2
  exit 1
fi

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/npu-turbo.service"
DEST="/etc/systemd/system/npu-turbo.service"

install -m 0644 "$SRC" "$DEST"
systemctl daemon-reload
systemctl enable --now npu-turbo.service

echo
echo "Installed and enabled npu-turbo.service."
echo "Verify current NPU power mode with:"
echo "  /opt/xilinx/xrt/bin/xrt-smi examine -r platform | grep -i mode"
