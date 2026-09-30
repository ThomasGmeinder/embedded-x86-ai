#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

set -e

STATE_FILE="/tmp/igpu_install_state"

echo "Installing ROCm stack..."

# Step 1: Kernel (if needed)
sudo apt update
sudo apt install -y linux-oem-24.04c

# Step 2: AMD installer
cd /tmp
wget -q https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb

sudo apt install -y ./amdgpu-install_7.2.1.70201-1_all.deb

# Step 3: Install ROCm (no dkms for iGPU systems)
sudo amdgpu-install -y --usecase=rocm --no-dkms

# Step 4: GPU device access (/dev/kfd, /dev/dri) — takes effect after reboot
sudo usermod -a -G render,video $USER

echo "STEP=ENV" > $STATE_FILE

echo "===== REBOOT REQUIRED ====="

echo "Run install.sh after reboot:"

