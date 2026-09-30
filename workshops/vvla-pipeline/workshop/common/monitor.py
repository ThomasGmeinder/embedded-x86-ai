# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""PROVIDED: pop the always-on-top resource monitor over your notebook.

A tiny window that floats above every other window (this browser included) and
shows just three numbers, colour-coded:

    CPU %   (burgundy)   ·   GPU %   (orange)   ·   NPU inf/s | Idle (cyan)

It runs as its own process, so the notebook kernel stays free. Use it from any
notebook (the setup cell already puts ``common`` on the path)::

    from common.monitor import open_monitor, close_monitor
    open_monitor()      # floats on top; watch the devices while a cell runs
    close_monitor()     # dismiss it (or click the window's ✕ / press Esc)

The window itself lives in ``common/resource_hud.py`` and can also be launched
straight from the workshop folder with ``./launch_monitor.sh``.
"""

from __future__ import annotations

import os
import subprocess
import sys

# The HUD lives right next to this module inside the shared common/ package.
_HUD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource_hud.py")
_PROC = None


def _say(msg: str) -> None:
    """Print a short status line; render it nicely inside a notebook if we can."""
    try:
        from IPython.display import HTML, display

        display(HTML(f"<div style='font-family:system-ui;font-size:13px'>{msg}</div>"))
    except Exception:
        # strip the tiny bit of markup we use for notebooks
        print(msg.replace("<code>", "").replace("</code>", ""))


def open_monitor():
    """Pop up the always-on-top resource HUD; returns the child process handle.

    The HUD floats above every window so you can watch CPU/GPU/NPU light up
    while a cell runs. Dismiss it with :func:`close_monitor`, or the window's
    own ✕ / Esc / right-click. Calling this again while it's open is a no-op.
    """
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        _say("Resource monitor is already open.")
        return _PROC
    if not os.path.isfile(_HUD):
        _say(f"Couldn't find the HUD at {_HUD}")
        return None
    try:
        _PROC = subprocess.Popen(
            [sys.executable, _HUD], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        _say(
            "Resource monitor opened — it floats on top of every window. "
            "Close it with its ✕, Esc, or <code>close_monitor()</code>."
        )
    except Exception as e:
        _PROC = None
        _say(f"Couldn't launch the resource monitor: {e}")
    return _PROC


def close_monitor() -> None:
    """Close the always-on-top resource HUD if it's open."""
    global _PROC
    if _PROC is None or _PROC.poll() is not None:
        _PROC = None
        _say("Resource monitor isn't open.")
        return
    try:
        _PROC.terminate()
        _say("Resource monitor closed.")
    except Exception as e:
        _say(f"Couldn't close the resource monitor: {e}")
    _PROC = None


def toggle_monitor() -> None:
    """Open the HUD if it's closed, close it if it's open."""
    if _PROC is not None and _PROC.poll() is None:
        close_monitor()
    else:
        open_monitor()
