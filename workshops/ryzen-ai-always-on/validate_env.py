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

"""Validate that the Ryzen AI environment has everything the workshop needs."""

import importlib
import sys


def check_import(name: str, attr: str | None = None) -> bool:
    try:
        mod = importlib.import_module(name)
        if attr and not hasattr(mod, attr):
            print(f"FAIL  {name}: missing attribute {attr}")
            return False
        ver = getattr(mod, "__version__", "?")
        print(f"OK    {name:22s} {ver}")
        return True
    except Exception as e:
        print(f"FAIL  {name:22s} {type(e).__name__}: {str(e)[:60]}")
        return False


def check_providers() -> bool:
    import onnxruntime as ort

    providers = ort.get_available_providers()
    needed = ["VitisAIExecutionProvider", "CPUExecutionProvider"]
    missing = [p for p in needed if p not in providers]
    if missing:
        print(f"FAIL  ORT providers missing: {missing}")
        return False
    print(f"OK    ORT providers          {providers}")
    return True


def check_opencl() -> bool:
    # OpenCV OpenCL availability (optional sanity check). On Linux the AMD iGPU
    # OpenCL driver is often absent; this is a non-fatal warning, not a hard failure.
    import cv2

    have, use = cv2.ocl.haveOpenCL(), cv2.ocl.useOpenCL()
    status = "OK   " if have else "WARN "
    print(
        f"{status} cv2.ocl                haveOpenCL={have} useOpenCL={use} (optional)"
    )
    return True


def check_devices() -> bool:
    # Webcam + iGPU render-node access (for the live demo and SO-101 renderer).
    # On Linux this is granted to the user at the physical desktop via logind seat
    # ACLs, so a WARN here usually just means you're on SSH — non-fatal.
    import os

    for label, path in [
        ("webcam", "/dev/video0"),
        ("iGPU render", "/dev/dri/renderD128"),
    ]:
        if not os.path.exists(path):
            print(f"WARN  {label:14s} {path} not found")
            continue
        ok = os.access(path, os.R_OK | os.W_OK)
        status = "OK   " if ok else "WARN "
        note = "" if ok else " (no access — see README 'Device access')"
        print(f"{status} {label:14s} {path}{note}")
    return True


def main() -> int:
    ok = True
    modules = [
        "onnxruntime",
        "quark",
        "ultralytics",
        "cv2",
        "psutil",
        "matplotlib",
        "ipywidgets",
        "netron",
    ]
    for name in modules:
        ok &= check_import(name)
    ok &= check_providers()
    ok &= check_opencl()
    ok &= check_devices()
    print()
    if ok:
        print("All checks passed.")
        return 0
    print("Some checks failed — see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
