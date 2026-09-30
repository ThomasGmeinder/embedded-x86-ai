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

"""Tiny registry so only one live pipeline runs at a time.

Each live demo entry point (``run_webcam_demo``, ``run_arm_teleop_in_notebook``)
starts daemon threads that keep reading the camera and the NPU until you click
Stop. If you switch demo cells *without* stopping, the old threads keep running
and split the camera + NPU session with the new demo — which silently halves the
live frame rate. Each entry point calls ``stop_active_pipelines()`` before
starting, then ``register()``s the new pipeline, so the notebook stays foolproof.
"""

from __future__ import annotations

import threading

_active: list = []
_lock = threading.Lock()


def stop_active_pipelines() -> None:
    """Stop and release every currently-registered pipeline."""
    with _lock:
        entries, _active[:] = list(_active), []
    for pipe, cap in entries:
        try:
            pipe.stop()
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


def register(pipe, cap=None) -> None:
    with _lock:
        _active.append((pipe, cap))
