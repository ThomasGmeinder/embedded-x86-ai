# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

#!/usr/bin/env python3
"""
demo_v3_rocm.py - AMD Advancing AI  .  Physical AI Agent
=============================================================
AMD  .  Advancing AI
Hackathon-grade GUI: dark glassmorphism, AMD red/orange accent palette,
gradient telemetry bars with sparkline history, rounded panel chrome,
animated status pulse, and a cinematic letterboxed scene view.
"""

import os
import sys
import json
import time
import queue
import threading
import subprocess
import re
import warnings
import socket
import signal
import shutil
import glob
import collections
import numpy as np
import cv2
import requests

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
os.environ["TI_LOG_LEVEL"] = "error"
warnings.filterwarnings("ignore")
# =============================================================================
# CONFIGURATION
# =============================================================================
# VK_HEADLESS=1 forces a fully headless run (no live viewer, and frames route
# to the video-capture queue instead of a live window) — same as running with
# no display at all.
HAS_DISPLAY = (
    bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    and os.environ.get("VK_HEADLESS", "") != "1"
)
_HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA_BIN = os.environ.get(
    "LLAMA_BIN", os.path.expanduser("~/llama-vulkan/bin/llama-server")
)
LLAMA_MODEL = os.environ.get(
    "LLAMA_MODEL", os.path.expanduser("~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
)
SERVER_URL = "http://127.0.0.1:8081"
SERVER_PORT = 8081
CREDIT_LINE = "AMD  .  Advancing AI"
# Scene geometry
CUBE_SIDE = 0.04
CUBE_Z_REST = CUBE_SIDE / 2
INITIAL_CUBES = {
    "red": np.array([0.50, 0.15, CUBE_Z_REST]),
    "green": np.array([0.50, -0.15, CUBE_Z_REST]),
    "blue": np.array([0.65, 0.00, CUBE_Z_REST]),
}
DROP_OFF_POS = np.array([0.30, 0.30, CUBE_Z_REST])
SORT_ZONES = {
    "red": np.array([0.30, 0.30, CUBE_Z_REST]),
    "green": np.array([0.30, 0.00, CUBE_Z_REST]),
    "blue": np.array([0.30, -0.30, CUBE_Z_REST]),
}
WORKSPACE_R_MIN = 0.25
WORKSPACE_R_MAX = 0.60
WORKSPACE_X_MIN = 0.22
WORKSPACE_Y_ABS = 0.40


def clamp_to_workspace(xyz: np.ndarray, label: str = "target") -> np.ndarray:
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    orig = (x, y)
    x = max(WORKSPACE_X_MIN, x)
    y = max(-WORKSPACE_Y_ABS, min(WORKSPACE_Y_ABS, y))
    r = (x * x + y * y) ** 0.5
    if r > WORKSPACE_R_MAX:
        s = WORKSPACE_R_MAX / r
        x, y = x * s, y * s
    elif r < WORKSPACE_R_MIN and r > 1e-6:
        s = WORKSPACE_R_MIN / r
        x, y = x * s, y * s
    if abs(x - orig[0]) > 1e-4 or abs(y - orig[1]) > 1e-4:
        print(
            f"[SAFE] {label} ({orig[0]:+.2f},{orig[1]:+.2f}) -> "
            f"({x:+.2f},{y:+.2f}) (clamped)"
        )
    return np.array([x, y, z])


DIRECTION_ZONES = {
    "left": np.array([0.40, 0.32, CUBE_Z_REST]),
    "right": np.array([0.40, -0.32, CUBE_Z_REST]),
    "front": np.array([0.55, 0.00, CUBE_Z_REST]),
    "back": np.array([0.30, 0.00, CUBE_Z_REST]),
    "center": np.array([0.45, 0.00, CUBE_Z_REST]),
    "middle": np.array([0.45, 0.00, CUBE_Z_REST]),
    "forward": np.array([0.55, 0.00, CUBE_Z_REST]),
    "backward": np.array([0.30, 0.00, CUBE_Z_REST]),
}
CUBE_COLORS_RGBA = {
    "red": (0.9, 0.1, 0.1, 1.0),
    "green": (0.1, 0.8, 0.1, 1.0),
    "blue": (0.1, 0.2, 1.0, 1.0),
}
ANNO_BGR = {
    "red": (60, 60, 235),
    "green": (80, 220, 80),
    "blue": (235, 140, 60),
}
# Camera
# Pulled back + raised + tilted up (and slightly wider FOV) so the WHOLE Franka
# arm stays in frame at full extension. Projected checks: arm top (z~1.0-1.1)
# lands near v~30px (inside the top edge), base ~v245, cubes ~v287-344 -- all
# inside the 640x480 frame. world_to_pixel / pixel_to_world use these same
# constants, so cube dots + K3b back-projection stay aligned.
CAM_RES = (640, 480)
CAM_POS = (2.2, 0.0, 1.6)
CAM_LOOKAT = (0.45, 0.0, 0.35)
CAM_FOV = 50
# Composite frame (slightly larger for a more cinematic feel)
COMP_W = 1440
COMP_H = 810
MAIN_W = 900
MAIN_H = COMP_H
PANEL_W = COMP_W - MAIN_W
PANEL_H = COMP_H // 2
# =============================================================================
# THEME  -  HACKATHON DARK + AMD ACCENT
# =============================================================================
# Everything is BGR (OpenCV).
THEME = {
    "bg_deep": (10, 12, 18),  # near-black backdrop
    "bg_panel": (22, 24, 32),  # panel base
    "bg_panel_alt": (30, 33, 44),  # alternating section
    "bg_chip": (38, 42, 56),  # chip / pill background
    "border": (60, 65, 82),  # subtle panel border
    "border_hi": (95, 110, 145),  # highlight border
    "divider": (45, 50, 65),
    "text_primary": (236, 240, 248),
    "text_secondary": (165, 175, 195),
    "text_muted": (110, 118, 138),
    "amd_red": (35, 35, 230),  # AMD signature red (BGR)
    "amd_red_dim": (55, 55, 150),
    "accent_orange": (40, 140, 255),  # warm orange
    "accent_cyan": (235, 200, 80),  # electric cyan/teal
    "accent_violet": (220, 90, 175),
    "accent_green": (110, 220, 130),
    "accent_amber": (60, 195, 245),
    "ok": (110, 220, 130),
    "warn": (60, 195, 245),
    "err": (60, 60, 235),
}
# Per-bar history for sparklines (last 60 samples ~30s @ 2Hz)
GPU_HIST = collections.deque(maxlen=60)
VRAM_HIST = collections.deque(maxlen=60)
# Robot control
JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "finger_joint1",
    "finger_joint2",
]
KP = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
KV = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])
F_LO = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100])
F_HI = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])
HOME_QPOS = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8, 0.04, 0.04])
GRIPPER_OPEN = 0.04
GRIPPER_GRIP = 0.012
# Arm colour: flat gold/yellow like the classic Genesis Franka (RGB 0-1). The MJCF
# asset is white by default; setup_scene overrides its surface with this colour.
ARM_COLOR_RGB = (1.0, 0.80, 0.15)
FINGER_FORCE = 2.0  # squeeze force (N); was 0.5 -> too weak, cube slipped during carry
FINGER_OPEN_FORCE = 0.5
HAND_TO_TIP = 0.105
PRE_GRASP_OFFSET = 0.28
LIFT_OFFSET = 0.30
RELEASE_OFFSET = 0.05
PLACE_CLEARANCE = 0.020
GRASP_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
SAFE_Z = 0.32
PLANNER_PROMPT = (
    "You are a robot-arm task planner. Convert the user request into a JSON plan.\n"
    'Output ONLY a single JSON object of the form: {"plan": [step1, step2, ...]}\n'
    "Each step is one of these dictionaries (use EXACT keys):\n"
    '  {"action": "pick",  "color": "<red|green|blue>", "to": "<zone?>"}\n'
    '  {"action": "stack", "top": "<color>", "bottom": "<color>"}\n'
    '  {"action": "sort",  "color": "<color>"}\n'
    '  {"action": "home"}\n'
    "Valid zones: left, right, front, back, center.\n"
    "Rules:\n"
    ' - \'pick X\' -> {"plan":[{"action":"pick","color":"X"}]}\n'
    ' - \'pick X and put it on the left\' -> add "to":"left"\n'
    " - 'pick X then Y' -> list both pick steps in order\n"
    " - 'stack X on Y' -> one stack step\n"
    " - 'stack X on Y on Z' -> two stack steps: [stack Y on Z, stack X on Y]\n"
    " - 'sort the cubes' -> three sort steps for red, green, blue\n"
    ' - \'home\' or \'reset\' -> {"plan":[{"action":"home"}]}\n'
    ' - If unparseable, return {"plan":[]}\n'
    "Valid colors: red, green, blue. NO other keys, NO prose."
)
# =============================================================================
# GLOBAL STATE
# =============================================================================
_llm_proc: subprocess.Popen | None = None
_ffmpeg_proc: subprocess.Popen | None = None
WINDOW = "AMD Advancing AI  |  Physical AI Agent"
CUBE_ENTITIES: dict[str, object] = {}
AGENT_STATE = {
    "user_input": "(awaiting command)",
    "llm_raw": "-",
    "plan": [],
    "plan_index": -1,
    "plan_status": "idle",
    "exec_log": [],
    "status": "Idle - awaiting command",
}
GPU_STATE = {
    "available": False,
    "source": "(none)",
    "card_path": "",
    "gpu_pct": 0.0,
    "vram_used_mb": 0.0,
    "vram_total_mb": 0.0,
    "vram_pct": 0.0,
    "temp_c": 0.0,
    "last_update": 0.0,
}
_state_lock = threading.Lock()
_START_TIME = time.time()


def push_log(msg: str):
    with _state_lock:
        AGENT_STATE["exec_log"].append(msg)
        AGENT_STATE["exec_log"] = AGENT_STATE["exec_log"][-6:]
    print(msg)


# =============================================================================
# LLM AGENT
# =============================================================================
def start_llm_server():
    global _llm_proc
    try:
        if requests.get(f"{SERVER_URL}/health", timeout=2).status_code == 200:
            print("[LLM] llama-server already running")
            return
    except Exception:
        pass
    if not LLAMA_BIN or not os.path.exists(LLAMA_BIN):
        print(
            "[LLM] llama-server not configured — set LLAMA_BIN/LLAMA_MODEL to enable. Using regex planner."
        )
        return
    if not LLAMA_MODEL or not os.path.exists(LLAMA_MODEL):
        print("[LLM] Model not found — set LLAMA_MODEL to enable. Using regex planner.")
        return
    result = subprocess.run(
        ["pgrep", "-f", "llama-server"], capture_output=True, text=True
    )
    for pid_str in result.stdout.strip().split():
        try:
            pid = int(pid_str)
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except ValueError:
            pass
    subprocess.run(["fuser", "-k", f"{SERVER_PORT}/tcp"], capture_output=True)
    for _ in range(10):
        time.sleep(0.5)
        try:
            with socket.create_connection(("127.0.0.1", SERVER_PORT), timeout=0.3):
                pass
        except OSError:
            break
    cmd = [
        LLAMA_BIN,
        "-m",
        LLAMA_MODEL,
        "-ngl",
        "99",
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVER_PORT),
        "--ctx-size",
        "2048",
    ]
    log_path = os.path.join(_HERE, "llama_server.log")
    log_file = open(log_path, "w")
    print("[LLM] Starting llama-server (Vulkan, AMD iGPU)...")
    _llm_proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
    for i in range(90):
        time.sleep(1)
        if _llm_proc.poll() is not None:
            print(f"\n[LLM] X llama-server crashed (exit {_llm_proc.returncode})")
            with open(log_path) as lf:
                for line in lf.readlines()[-15:]:
                    print(f"   {line.rstrip()}")
            sys.exit(1)
        try:
            if requests.get(f"{SERVER_URL}/health", timeout=1).status_code == 200:
                print(f"[LLM] Server ready after {i+1}s")
                log_file.close()
                return
        except Exception:
            print(".", end="", flush=True)
    sys.exit("[LLM] X Server didn't become healthy after 90s")


def parse_plan(user_text: str) -> dict:
    fast = _fallback_parse(user_text)
    if fast.get("plan"):
        with _state_lock:
            AGENT_STATE["llm_raw"] = "[regex fast-path] " + json.dumps(
                fast, separators=(",", ":")
            )
        return fast
    raw = "(no response)"
    try:
        resp = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json={
                "model": "local",
                "messages": [
                    {"role": "system", "content": PLANNER_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                "max_tokens": 220,
                "temperature": 0.0,
            },
            timeout=30,
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            if isinstance(data, dict) and isinstance(data.get("plan"), list):
                with _state_lock:
                    AGENT_STATE["llm_raw"] = json.dumps(data, separators=(",", ":"))
                return data
    except Exception as e:
        print(f"[LLM] parse error: {e}")
    with _state_lock:
        AGENT_STATE["llm_raw"] = raw[:120]
    return {"plan": []}


def _fallback_parse(text: str) -> dict:
    t = text.lower()
    if any(k in t for k in ("home", "reset")):
        return {"plan": [{"action": "home"}]}
    if "sort" in t:
        return {
            "plan": [{"action": "sort", "color": c} for c in ("red", "green", "blue")]
        }
    m = re.search(
        r"stack\s+(red|green|blue)\s+on\s+(red|green|blue)(?:\s+on\s+(red|green|blue))?",
        t,
    )
    if m:
        a, b, c = m.group(1), m.group(2), m.group(3)
        if c:
            return {
                "plan": [
                    {"action": "stack", "top": b, "bottom": c},
                    {"action": "stack", "top": a, "bottom": b},
                ]
            }
        return {"plan": [{"action": "stack", "top": a, "bottom": b}]}
    zone_re = r"(left|right|front|back|center|middle|forward|backward)"
    m = re.search(
        r"(?:pick|place|put|move|drop|set)\s+(?:up\s+)?(?:the\s+)?"
        r"(red|green|blue)(?:\s+cube)?"
        r".*?(?:on|to|at|in|toward|towards)(?:\s+the)?\s+" + zone_re,
        t,
    )
    if m:
        color, where = m.group(1), m.group(2)
        return {"plan": [{"action": "pick", "color": color, "to": where}]}
    picks = re.findall(r"(red|green|blue)", t)
    if picks and ("pick" in t or "grab" in t):
        return {"plan": [{"action": "pick", "color": c} for c in picks]}
    return {"plan": []}


# =============================================================================
# GPU MONITOR
# =============================================================================
def _find_amd_card() -> str | None:
    for path in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        if "/renderD" in path:
            continue
        busy = os.path.join(path, "gpu_busy_percent")
        if not os.path.exists(busy):
            continue
        try:
            with open(os.path.join(path, "vendor")) as f:
                vid = f.read().strip().lower()
            if vid not in ("0x1002", "1002"):
                continue
        except Exception:
            pass
        return path
    return None


def gpu_monitor(stop_event: threading.Event):
    card_path = _find_amd_card()
    if card_path is not None:
        busy_path = os.path.join(card_path, "gpu_busy_percent")
        vused_path = os.path.join(card_path, "mem_info_vram_used")
        vtot_path = os.path.join(card_path, "mem_info_vram_total")
        temp_path = ""
        for hw in glob.glob(os.path.join(card_path, "hwmon", "hwmon*", "temp1_input")):
            temp_path = hw
            break
        cardname = os.path.basename(os.path.dirname(card_path))
        GPU_STATE["available"] = True
        GPU_STATE["source"] = f"sysfs/{cardname}"
        GPU_STATE["card_path"] = card_path
        print(f"[GPU] Monitoring via sysfs: {card_path}")
        while not stop_event.is_set():
            try:
                with open(busy_path) as f:
                    GPU_STATE["gpu_pct"] = max(0.0, min(100.0, float(f.read().strip())))
                if os.path.exists(vused_path) and os.path.exists(vtot_path):
                    with open(vused_path) as f:
                        used = int(f.read().strip())
                    with open(vtot_path) as f:
                        total = int(f.read().strip())
                    GPU_STATE["vram_used_mb"] = used / (1024.0 * 1024.0)
                    GPU_STATE["vram_total_mb"] = total / (1024.0 * 1024.0)
                    GPU_STATE["vram_pct"] = (used / total * 100.0) if total > 0 else 0.0
                if temp_path:
                    try:
                        with open(temp_path) as f:
                            GPU_STATE["temp_c"] = int(f.read().strip()) / 1000.0
                    except Exception:
                        pass
                GPU_STATE["last_update"] = time.time()
                GPU_HIST.append(GPU_STATE["gpu_pct"])
                VRAM_HIST.append(GPU_STATE["vram_pct"])
            except Exception:
                pass
            stop_event.wait(0.5)
        return
    rocm = shutil.which("rocm-smi")
    if rocm is not None:
        GPU_STATE["available"] = True
        GPU_STATE["source"] = "rocm-smi"
        print(f"[GPU] Monitoring via rocm-smi: {rocm}")
        while not stop_event.is_set():
            try:
                out = subprocess.run(
                    [rocm, "--showuse", "--showmemuse", "--showtemp", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                ).stdout
                data = json.loads(out)
                for key, info in data.items():
                    if not isinstance(info, dict):
                        continue
                    for k in ("GPU use (%)", "GPU Use (%)", "GPU_Use_(%)"):
                        if k in info:
                            try:
                                GPU_STATE["gpu_pct"] = float(str(info[k]).rstrip("%"))
                            except ValueError:
                                pass
                            break
                    for k in ("GPU memory use (%)", "GPU Memory Allocated (VRAM%)"):
                        if k in info:
                            try:
                                GPU_STATE["vram_pct"] = float(str(info[k]).rstrip("%"))
                            except ValueError:
                                pass
                            break
                    for k in ("Temperature (Sensor edge) (C)", "Temperature (C)"):
                        if k in info:
                            try:
                                GPU_STATE["temp_c"] = float(info[k])
                            except ValueError:
                                pass
                            break
                    break
                GPU_STATE["last_update"] = time.time()
                GPU_HIST.append(GPU_STATE["gpu_pct"])
                VRAM_HIST.append(GPU_STATE["vram_pct"])
            except Exception:
                pass
            stop_event.wait(0.5)
        return
    GPU_STATE["available"] = False
    GPU_STATE["source"] = "unavailable"
    print("[GPU] No sysfs gpu_busy_percent and no rocm-smi - bar disabled")


# =============================================================================
# DRAWING PRIMITIVES  (modern look)
# =============================================================================
def _put_text(
    img,
    text,
    org,
    scale=0.45,
    color=(220, 220, 220),
    thick=1,
    font=cv2.FONT_HERSHEY_SIMPLEX,
):
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


def _filled_rect(img, p1, p2, color):
    cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)


def _rounded_rect(img, p1, p2, color, radius=10, thickness=-1):
    """Pillow-free rounded rectangle via two rects + four filled circles."""
    x1, y1 = p1
    x2, y2 = p2
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(
            img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA
        )
        cv2.ellipse(
            img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA
        )
        cv2.ellipse(
            img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA
        )
        cv2.ellipse(
            img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA
        )


def _vgradient(img, p1, p2, c_top, c_bot):
    """Vertical gradient fill between two BGR colors."""
    x1, y1 = p1
    x2, y2 = p2
    h = max(1, y2 - y1)
    for i in range(h):
        t = i / float(h - 1) if h > 1 else 0.0
        b = int(c_top[0] * (1 - t) + c_bot[0] * t)
        g = int(c_top[1] * (1 - t) + c_bot[1] * t)
        r = int(c_top[2] * (1 - t) + c_bot[2] * t)
        cv2.line(img, (x1, y1 + i), (x2, y1 + i), (b, g, r), 1, cv2.LINE_AA)


def _hgradient(img, p1, p2, c_left, c_right):
    x1, y1 = p1
    x2, y2 = p2
    w = max(1, x2 - x1)
    for i in range(w):
        t = i / float(w - 1) if w > 1 else 0.0
        b = int(c_left[0] * (1 - t) + c_right[0] * t)
        g = int(c_left[1] * (1 - t) + c_right[1] * t)
        r = int(c_left[2] * (1 - t) + c_right[2] * t)
        cv2.line(img, (x1 + i, y1), (x1 + i, y2), (b, g, r), 1, cv2.LINE_AA)


def _section_header(panel, x, y, w, label, accent):
    """Section header with left accent stripe."""
    cv2.rectangle(panel, (x, y), (x + 3, y + 16), accent, -1, cv2.LINE_AA)
    _put_text(panel, label.upper(), (x + 10, y + 13), 0.42, THEME["text_secondary"], 1)


def _chip(panel, x, y, text, bg=None, fg=None, pad=6):
    bg = bg or THEME["bg_chip"]
    fg = fg or THEME["text_primary"]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    _rounded_rect(panel, (x, y), (x + tw + pad * 2, y + th + pad + 2), bg, radius=6)
    _put_text(panel, text, (x + pad, y + th + pad // 2), 0.38, fg, 1)
    return x + tw + pad * 2 + 4


def _draw_panel(canvas, p1, p2, label=None, accent=None):
    """Rounded glass-style panel with title bar."""
    accent = accent or THEME["accent_orange"]
    _rounded_rect(canvas, p1, p2, THEME["bg_panel"], radius=14)
    _rounded_rect(canvas, p1, p2, THEME["border"], radius=14, thickness=1)
    if label:
        x1, y1 = p1
        x2, _ = p2
        # accent bar on top edge
        cv2.line(canvas, (x1 + 14, y1 + 2), (x2 - 14, y1 + 2), accent, 2, cv2.LINE_AA)
        _put_text(
            canvas,
            label,
            (x1 + 16, y1 + 24),
            0.52,
            THEME["text_primary"],
            1,
            cv2.FONT_HERSHEY_DUPLEX,
        )


def _draw_gradient_bar(panel, x, y, w, h, pct, label, history=None):
    """Sleek telemetry bar: track + gradient fill + sparkline behind."""
    pct = max(0.0, min(100.0, float(pct)))
    # background track
    _rounded_rect(panel, (x, y), (x + w, y + h), THEME["bg_chip"], radius=h // 2)
    # sparkline behind the fill (very subtle)
    if history and len(history) > 1:
        pts = list(history)
        n = len(pts)
        for i in range(1, n):
            x0 = x + int((i - 1) * (w / max(1, n - 1)))
            x1_ = x + int(i * (w / max(1, n - 1)))
            y0 = y + h - int(pts[i - 1] * (h - 2) / 100.0) - 1
            y1_ = y + h - int(pts[i] * (h - 2) / 100.0) - 1
            cv2.line(panel, (x0, y0), (x1_, y1_), (70, 80, 100), 1, cv2.LINE_AA)
    # gradient fill: amber -> orange -> AMD red as pct grows
    fill_w = int(w * pct / 100.0)
    if fill_w > 2:
        if pct < 55.0:
            c_l, c_r = THEME["accent_green"], THEME["accent_amber"]
        elif pct < 80.0:
            c_l, c_r = THEME["accent_amber"], THEME["accent_orange"]
        else:
            c_l, c_r = THEME["accent_orange"], THEME["amd_red"]
        # use a temporary mask so the fill is rounded
        tmp = np.zeros_like(panel)
        _hgradient(tmp, (x, y), (x + fill_w, y + h), c_l, c_r)
        mask = np.zeros(panel.shape[:2], dtype=np.uint8)
        _rounded_rect(mask, (x, y), (x + fill_w, y + h), 255, radius=h // 2)
        panel[mask > 0] = tmp[mask > 0]
    # outline
    _rounded_rect(
        panel, (x, y), (x + w, y + h), THEME["border"], radius=h // 2, thickness=1
    )
    # label / pct text overlay
    _put_text(panel, label, (x + 10, y + h - 6), 0.40, THEME["text_primary"], 1)
    pct_txt = f"{pct:5.1f}%"
    (tw, _), _ = cv2.getTextSize(pct_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    _put_text(
        panel, pct_txt, (x + w - tw - 10, y + h - 6), 0.40, THEME["text_primary"], 1
    )


def _draw_bar(panel, x, y, w, h, pct, label):
    """Compat shim for old call sites."""
    _draw_gradient_bar(panel, x, y, w, h, pct, label)


# =============================================================================
# HIP / ROCm VISION + TACTILE KERNELS  (Strix Halo iGPU)
# =============================================================================
# 8 HIP kernels wired INTO the live agent: K1-K3 build the depth/normal/seg
# perception thumbnails, K3b localises the target cube in image space (replaces
# the ground-truth pixel cheat), K4 reduces the ElastomerTaxel field on the GPU
# to CLOSE the grasp control loop, and K5-K7 are the classic image-processing
# trio (RGB->gray, Gaussian blur, Sobel edges) run on the live camera frame.
HIP_DEVICE = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
_HIP_RUNTIME = None
if _HAS_TORCH:
    try:
        _HIP_RUNTIME = torch.version.hip
    except Exception:
        _HIP_RUNTIME = None
CONTACT_THRESH_M = 2.5e-5  # taxel displacement to count as contact; was 1e-4 (too strict -> never SECURE)
CONTACT_SECURE_TAXELS = 3  # taxels needed to declare SECURE; was 6
GRIP_STIFFNESS_N_PER_M = 8.0e3
SECURE_HOLD_STEPS = 5
GRASP_MIN_STEPS = 40
GRASP_MAX_STEPS = 160
TACTILE_READ_EVERY = 4
KERNEL_PIP_EVERY = 4
COLOR_LOCK_TOL = 0.36
TARGET_COLOR_RGB = {c: tuple(v[:3]) for c, v in CUBE_COLORS_RGBA.items()}


def hip_normalize_depth(depth_np, near=0.1, far=5.0):
    d = torch.from_numpy(np.asarray(depth_np, dtype=np.float32)).to(HIP_DEVICE)
    if d.ndim == 3:
        d = d[:, :, 0]
    return ((d - near) / (far - near)).clamp(0, 1).mul(255).byte().cpu().numpy()


def hip_colorize_normal(normal_np):
    n = torch.from_numpy(np.asarray(normal_np, dtype=np.float32)).to(HIP_DEVICE)
    if n.ndim == 4:
        n = n[0]
    out = (n + 1.0) / 2.0 if n.min().item() < -0.05 else n
    mn, mx = out.min(), out.max()
    if (mx - mn).item() > 1e-4:
        out = (out - mn) / (mx - mn)
    return (out * 255).clamp(0, 255).byte().cpu().numpy()


def hip_colorize_seg(seg_np):
    arr = np.asarray(seg_np)
    if arr.ndim == 4:
        arr = arr[0]
    ids = arr[..., 0].astype(np.int32) if arr.ndim == 3 else arr.astype(np.int32)
    s = torch.from_numpy(ids).to(HIP_DEVICE)
    r = ((s * 37 + 11) % 256).byte()
    g = ((s * 79 + 43) % 256).byte()
    b = ((s * 131 + 97) % 256).byte()
    return torch.stack([r, g, b], dim=-1).cpu().numpy()


def hip_color_lock(rgb_uint8, target, tol=COLOR_LOCK_TOL):
    a = np.asarray(rgb_uint8)
    if a.ndim != 3 or a.shape[-1] < 3:
        return {"n": 0, "bbox": None, "cx": None, "cy": None}
    t = torch.as_tensor(a[..., :3].astype(np.float32)).to(HIP_DEVICE) / 255.0
    tgt = torch.tensor(target, dtype=torch.float32, device=t.device)
    mask = torch.linalg.norm(t - tgt, dim=-1) < tol
    n = int(mask.sum().item())
    if n < 8:
        return {"n": n, "bbox": None, "cx": None, "cy": None}
    ys, xs = torch.where(mask)
    return {
        "n": n,
        "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        "cx": int(xs.float().mean().item()),
        "cy": int(ys.float().mean().item()),
    }


def hip_tactile_reduce(
    left_disp,
    right_disp,
    contact_thresh_m=CONTACT_THRESH_M,
    secure_taxels=CONTACT_SECURE_TAXELS,
):
    l = (
        torch.as_tensor(np.asarray(left_disp, dtype=np.float32))
        .to(HIP_DEVICE)
        .reshape(-1, 3)
    )
    r = (
        torch.as_tensor(np.asarray(right_disp, dtype=np.float32))
        .to(HIP_DEVICE)
        .reshape(-1, 3)
    )
    lmag = torch.linalg.norm(l, dim=-1)
    rmag = torch.linalg.norm(r, dim=-1)
    n_contact = int(
        (lmag > contact_thresh_m).sum().item() + (rmag > contact_thresh_m).sum().item()
    )
    grip_force_N = float((lmag.sum() + rmag.sum()).item() * GRIP_STIFFNESS_N_PER_M)
    peak_mm = float(max(lmag.max().item(), rmag.max().item()) * 1000.0)
    return {
        "n_contact": n_contact,
        "n_taxels": int(lmag.numel() + rmag.numel()),
        "grip_force_N": grip_force_N,
        "peak_mm": peak_mm,
        "secure": n_contact >= secure_taxels,
    }


# --- classic image-processing kernels (K5-K7), the canonical GPU "hello world"
#     trio, run every frame on the live RGB camera image. Chained gray -> blur ->
#     edges so the panel visibly shows the pipeline. All three are plain HIP/torch
#     tensor ops (RGB2GRAY reduction, separable Gaussian conv2d, Sobel conv2d).
def hip_rgb_to_gray(rgb_uint8):
    """K5: RGB -> BT.601 luminance grayscale on the GPU. Returns HxW uint8."""
    a = np.asarray(rgb_uint8)
    if a.ndim == 4:
        a = a[0]
    t = torch.as_tensor(a[..., :3].astype(np.float32)).to(HIP_DEVICE)
    w = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32, device=t.device)
    return (t * w).sum(-1).clamp(0, 255).byte().cpu().numpy()


def hip_gaussian_blur(gray_uint8, ksize=5, sigma=1.4):
    """K6: separable Gaussian blur via two depthwise conv2d passes. HxW uint8."""
    a = np.asarray(gray_uint8, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    t = torch.as_tensor(a).to(HIP_DEVICE)[None, None]  # 1,1,H,W
    ax = torch.arange(ksize, device=t.device, dtype=torch.float32) - (ksize - 1) / 2.0
    k1 = torch.exp(-(ax * ax) / (2.0 * sigma * sigma))
    k1 = k1 / k1.sum()
    pad = ksize // 2
    t = torch.nn.functional.conv2d(t, k1.view(1, 1, 1, ksize), padding=(0, pad))
    t = torch.nn.functional.conv2d(t, k1.view(1, 1, ksize, 1), padding=(pad, 0))
    return t[0, 0].clamp(0, 255).byte().cpu().numpy()


def hip_sobel_edges(gray_uint8):
    """K7: Sobel gradient magnitude (two conv2d + hypot), normalised. HxW uint8."""
    a = np.asarray(gray_uint8, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    t = torch.as_tensor(a).to(HIP_DEVICE)[None, None]
    gx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=t.device
    ).view(1, 1, 3, 3)
    gy = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=t.device
    ).view(1, 1, 3, 3)
    ex = torch.nn.functional.conv2d(t, gx, padding=1)
    ey = torch.nn.functional.conv2d(t, gy, padding=1)
    mag = torch.sqrt(ex * ex + ey * ey)
    mag = mag / (mag.max() + 1e-6) * 255.0
    return mag[0, 0].clamp(0, 255).byte().cpu().numpy()


# ---- kernel-driven runtime state (consumed by the HUD) ----
TACTILE_SENSORS = None
_FRAME_IDX = 0
KERNEL_STATE = {
    "thumbs": {
        "depth": None,
        "normal": None,
        "seg": None,
        "gray": None,
        "blur": None,
        "edges": None,
    },
    "lock": None,
    "grip": None,
    "n_live": 6,
    "tactile": False,
}


def _cam_basis():
    cam_pos = np.array(CAM_POS, dtype=float)
    fwd = np.array(CAM_LOOKAT, dtype=float) - cam_pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return cam_pos, fwd, right, up


def pixel_to_world_on_plane(u, v, z_plane=CUBE_Z_REST, w=CAM_RES[0], h=CAM_RES[1]):
    cam_pos, fwd, right, up = _cam_basis()
    f = (h / 2.0) / np.tan(np.radians(CAM_FOV) / 2.0)
    a = (u - w / 2.0) / f
    b = (h / 2.0 - v) / f
    dir_vec = fwd + a * right + b * up
    if abs(dir_vec[2]) < 1e-6:
        return None
    t = (z_plane - cam_pos[2]) / dir_vec[2]
    if t <= 0:
        return None
    world = cam_pos + t * dir_vec
    return np.array([float(world[0]), float(world[1]), float(z_plane)])


def _grab_modalities(cam):
    out = cam.render(rgb=True, depth=True, segmentation=True, normal=True)
    if isinstance(out, dict):
        rgb = out.get("rgb")
        depth = out.get("depth")
        seg = out.get("segmentation")
        nrm = out.get("normal")
    else:
        seq = list(out)
        rgb = seq[0] if len(seq) > 0 else None
        depth = seq[1] if len(seq) > 1 else None
        seg = seq[2] if len(seq) > 2 else None
        nrm = seq[3] if len(seq) > 3 else None

    def _np(x):
        return (
            x.cpu().numpy()
            if hasattr(x, "cpu")
            else (None if x is None else np.asarray(x))
        )

    return _np(rgb), _np(depth), _np(seg), _np(nrm)


def _run_perception_kernels(cam):
    if not _HAS_TORCH:
        return
    try:
        rgb, depth, seg, nrm = _grab_modalities(cam)
        if depth is not None:
            d = hip_normalize_depth(depth)
            KERNEL_STATE["thumbs"]["depth"] = cv2.applyColorMap(
                np.asarray(d, np.uint8), cv2.COLORMAP_INFERNO
            )
        if nrm is not None:
            nn = hip_colorize_normal(nrm)
            KERNEL_STATE["thumbs"]["normal"] = cv2.cvtColor(
                np.asarray(nn, np.uint8), cv2.COLOR_RGB2BGR
            )
        if seg is not None:
            ss = hip_colorize_seg(seg)
            KERNEL_STATE["thumbs"]["seg"] = cv2.cvtColor(
                np.asarray(ss, np.uint8), cv2.COLOR_RGB2BGR
            )
        if rgb is not None:
            # K5 -> K6 -> K7 chained: gray, then blur the gray, then Sobel the blur
            g = hip_rgb_to_gray(rgb)
            KERNEL_STATE["thumbs"]["gray"] = cv2.cvtColor(
                np.asarray(g, np.uint8), cv2.COLOR_GRAY2BGR
            )
            b = hip_gaussian_blur(g)
            KERNEL_STATE["thumbs"]["blur"] = cv2.cvtColor(
                np.asarray(b, np.uint8), cv2.COLOR_GRAY2BGR
            )
            e = hip_sobel_edges(b)
            KERNEL_STATE["thumbs"]["edges"] = cv2.applyColorMap(
                np.asarray(e, np.uint8), cv2.COLORMAP_TURBO
            )
        # 6 image kernels run every frame (K1-K3 + K5-K7); K4 tactile adds a 7th
        # only during a grasp. Library total is 8 (K3b color-lock is data-only).
        KERNEL_STATE["n_live"] = 6 + (1 if KERNEL_STATE["tactile"] else 0)
    except Exception:
        pass


def _kernel_target_xy(cam, color, fallback_xy):
    # K3b: localise the cube in image space, back-project to the table plane.
    if not _HAS_TORCH:
        KERNEL_STATE["lock"] = None
        return fallback_xy
    try:
        rgb, _, _, _ = _grab_modalities(cam)
        if rgb is None:
            return fallback_xy
        res = hip_color_lock(
            np.asarray(rgb, np.uint8)[..., :3],
            TARGET_COLOR_RGB.get(color, (1.0, 1.0, 1.0)),
        )
        if res["bbox"] is None:
            KERNEL_STATE["lock"] = None
            return fallback_xy
        KERNEL_STATE["lock"] = {
            "color": color,
            "bbox": res["bbox"],
            "cx": res["cx"],
            "cy": res["cy"],
            "n": res["n"],
        }
        world = pixel_to_world_on_plane(res["cx"], res["cy"], CUBE_Z_REST)
        if world is None:
            return fallback_xy
        if np.linalg.norm(world[:2] - np.asarray(fallback_xy, dtype=float)) > 0.12:
            return fallback_xy  # trust physics prior if vision fix is implausible
        return np.array([world[0], world[1]])
    except Exception:
        return fallback_xy


def _draw_kernel_overlays(canvas, fx, fy, scale, src_w, src_h, overlay_w, overlay_h):
    st = KERNEL_STATE
    # NOTE: the K1-K3 perception thumbnails now live in the right "AI BRAIN"
    # panel (render_thinking_panel), not over the scene. The K3b color-lock box
    # overlay was removed at the user's request. K3b still runs -- it just
    # refines the grasp target silently instead of drawing a box on the render.
    # K4 grip-secure banner (top-left of render area)
    grip = st["grip"]
    if grip is not None:
        secure = grip.get("secure", False)
        col = THEME["ok"] if secure else THEME["accent_orange"]
        txt = (
            f"{'GRIP SECURE' if secure else 'CLOSING'}   K4  "
            f"{grip.get('grip_force_N', 0.0):.1f} N   "
            f"contact {grip.get('n_contact', 0)}/{grip.get('n_taxels', 128)}"
        )
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        bx = fx + 10
        by = fy + 10
        _rounded_rect(
            canvas, (bx, by), (bx + tw + 26, by + th + 16), (14, 16, 22), radius=8
        )
        cv2.circle(canvas, (bx + 12, by + (th + 16) // 2), 5, col, -1, cv2.LINE_AA)
        _put_text(
            canvas, txt, (bx + 24, by + th + 6), 0.5, col, 1, cv2.FONT_HERSHEY_DUPLEX
        )
    # kernel-count chip (bottom-left of render area)
    _dev = "HIP/ROCm" if _HIP_RUNTIME else HIP_DEVICE.upper()
    # n_live = kernels executing THIS frame, out of the 8-kernel library
    # (K1-K3 + K5-K7 perception always run; K4 tactile only during a grasp).
    kc = f"{st['n_live']}/8 HIP kernels active  |  {_dev}"
    if _HIP_RUNTIME:
        kc += f" {_HIP_RUNTIME}"
    ky = fy + int(src_h * scale) - 10
    _put_text(canvas, kc, (fx + 12, ky), 0.4, THEME["accent_cyan"], 1)


# =============================================================================
# PERCEPTION
# =============================================================================
def get_cube_positions() -> dict[str, np.ndarray]:
    return {
        color: entity.get_pos().cpu().numpy().flatten().astype(float)
        for color, entity in CUBE_ENTITIES.items()
    }


def world_to_pixel(world_pt: np.ndarray, w: int = CAM_RES[0], h: int = CAM_RES[1]):
    cam_pos = np.array(CAM_POS, dtype=float)
    lookat = np.array(CAM_LOOKAT, dtype=float)
    p = world_pt - cam_pos
    fwd = lookat - cam_pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    z_cam = float(np.dot(p, fwd))
    if z_cam <= 1e-3:
        return None
    x_cam = float(np.dot(p, right))
    y_cam = float(np.dot(p, up))
    f = (h / 2.0) / np.tan(np.radians(CAM_FOV) / 2.0)
    u = int(w / 2.0 + x_cam * f / z_cam)
    v = int(h / 2.0 - y_cam * f / z_cam)
    if 0 <= u < w and 0 <= v < h:
        return (u, v)
    return None


# =============================================================================
# COMPOSITE DISPLAY  -  redesigned
# =============================================================================
def _status_color(status: str):
    s = status.lower()
    if "fail" in s or "abort" in s or "error" in s:
        return THEME["err"]
    if "plan complete" in s or "done" in s or "ok" in s:
        return THEME["ok"]
    if "idle" in s or "await" in s:
        return THEME["text_secondary"]
    return THEME["accent_orange"]


def annotate_main(
    rgb: np.ndarray, status: str, highlight: str | None, overlay_w: int, overlay_h: int
) -> np.ndarray:
    """Cinematic scene panel: top bar with branding, letterboxed render,
    bottom HUD with status pill + plan progress."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    src_h, src_w = bgr.shape[:2]
    canvas = np.full((overlay_h, overlay_w, 3), THEME["bg_deep"], dtype=np.uint8)
    # subtle vertical vignette
    _vgradient(canvas, (0, 0), (overlay_w, overlay_h), THEME["bg_deep"], (4, 5, 9))
    # ---- top branding bar ----
    top_h = 56
    _filled_rect(canvas, (0, 0), (overlay_w, top_h), THEME["bg_panel"])
    cv2.line(canvas, (0, top_h), (overlay_w, top_h), THEME["divider"], 1, cv2.LINE_AA)
    # AMD red accent block (logo-ish)
    _rounded_rect(canvas, (16, 12), (52, 44), THEME["amd_red"], radius=8)
    _put_text(
        canvas, "AMD", (22, 35), 0.55, THEME["text_primary"], 2, cv2.FONT_HERSHEY_DUPLEX
    )
    _put_text(
        canvas,
        "ADVANCING AI",
        (64, 28),
        0.62,
        THEME["text_primary"],
        1,
        cv2.FONT_HERSHEY_DUPLEX,
    )
    _put_text(
        canvas,
        "Physical AI Agent  .  AMD Strix Halo iGPU",
        (64, 47),
        0.40,
        THEME["text_secondary"],
        1,
    )
    # uptime + recording chip on the right
    uptime = int(time.time() - _START_TIME)
    mm, ss = divmod(uptime, 60)
    hh, mm = divmod(mm, 60)
    chip_x = overlay_w - 16
    rec_txt = "REC"
    (tw, th), _ = cv2.getTextSize(rec_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    chip_x -= tw + 26
    _rounded_rect(
        canvas,
        (chip_x, 16),
        (chip_x + tw + 24, 16 + th + 12),
        THEME["bg_chip"],
        radius=8,
    )
    pulse = np.sin(time.time() * 4) * 0.5 + 0.5
    rec_col = (60, 60, int(180 + 70 * pulse))
    cv2.circle(canvas, (chip_x + 10, 16 + (th + 12) // 2), 4, rec_col, -1, cv2.LINE_AA)
    _put_text(
        canvas, rec_txt, (chip_x + 18, 16 + th + 6), 0.40, THEME["text_primary"], 1
    )
    up_txt = f"{hh:02d}:{mm:02d}:{ss:02d}"
    (tw2, _), _ = cv2.getTextSize(up_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    chip_x -= tw2 + 22
    _rounded_rect(
        canvas,
        (chip_x, 16),
        (chip_x + tw2 + 18, 16 + th + 12),
        THEME["bg_chip"],
        radius=8,
    )
    _put_text(canvas, up_txt, (chip_x + 9, 16 + th + 6), 0.42, THEME["text_primary"], 1)
    # ---- scene render (letterboxed, framed) ----
    hud_h = 110
    img_area_y0 = top_h + 14
    img_area_y1 = overlay_h - hud_h - 14
    img_area_x0 = 18
    img_area_x1 = overlay_w - 18
    avail_w = img_area_x1 - img_area_x0
    avail_h = img_area_y1 - img_area_y0
    scale = min(avail_w / src_w, avail_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    resized = cv2.resize(bgr, (new_w, new_h))
    fx = img_area_x0 + (avail_w - new_w) // 2
    fy = img_area_y0 + (avail_h - new_h) // 2
    # render frame backdrop
    _rounded_rect(
        canvas,
        (img_area_x0, img_area_y0),
        (img_area_x1, img_area_y1),
        (5, 6, 10),
        radius=12,
    )
    canvas[fy : fy + new_h, fx : fx + new_w] = resized
    _rounded_rect(
        canvas,
        (img_area_x0, img_area_y0),
        (img_area_x1, img_area_y1),
        THEME["border"],
        radius=12,
        thickness=1,
    )
    # corner brackets - cinematic feel
    bl = 18

    def bracket(p, dx, dy):
        x, y = p
        cv2.line(
            canvas, (x, y), (x + dx * bl, y), THEME["accent_orange"], 2, cv2.LINE_AA
        )
        cv2.line(
            canvas, (x, y), (x, y + dy * bl), THEME["accent_orange"], 2, cv2.LINE_AA
        )

    bracket((img_area_x0 + 6, img_area_y0 + 6), +1, +1)
    bracket((img_area_x1 - 6, img_area_y0 + 6), -1, +1)
    bracket((img_area_x0 + 6, img_area_y1 - 6), +1, -1)
    bracket((img_area_x1 - 6, img_area_y1 - 6), -1, -1)
    # cube annotations
    for color, world_pt in get_cube_positions().items():
        px = world_to_pixel(world_pt, src_w, src_h)
        if px is None:
            continue
        u = fx + int(px[0] * scale)
        v = fy + int(px[1] * scale)
        bgr_c = ANNO_BGR[color]
        is_hi = color == highlight
        radius = 20 if is_hi else 13
        thick = 3 if is_hi else 2
        if is_hi:
            # animated pulse halo
            pulse_r = radius + int(6 + 4 * np.sin(time.time() * 6))
            cv2.circle(canvas, (u, v), pulse_r, bgr_c, 1, cv2.LINE_AA)
        cv2.circle(canvas, (u, v), radius, bgr_c, thick, cv2.LINE_AA)
        cv2.circle(canvas, (u, v), 3, bgr_c, -1, cv2.LINE_AA)
        # label chip
        lbl = color.upper()
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        cx = u + radius + 6
        cy = v - th // 2 - 4
        _rounded_rect(
            canvas, (cx, cy), (cx + tw + 12, cy + th + 10), (15, 18, 24), radius=6
        )
        _put_text(
            canvas, lbl, (cx + 6, cy + th + 4), 0.45, bgr_c, 1, cv2.FONT_HERSHEY_DUPLEX
        )
    # ---- HIP kernel overlays (PiP + K3b lock + K4 grip banner) ----
    _draw_kernel_overlays(canvas, fx, fy, scale, src_w, src_h, overlay_w, overlay_h)
    # ---- bottom HUD ----
    hud_y0 = overlay_h - hud_h
    _filled_rect(canvas, (0, hud_y0), (overlay_w, overlay_h), THEME["bg_panel"])
    cv2.line(canvas, (0, hud_y0), (overlay_w, hud_y0), THEME["divider"], 1, cv2.LINE_AA)
    # status pill
    s_col = _status_color(status)
    pill_w = min(overlay_w - 240, 16 + 8 * len(status))
    _rounded_rect(
        canvas,
        (16, hud_y0 + 14),
        (16 + pill_w, hud_y0 + 42),
        THEME["bg_chip"],
        radius=14,
    )
    cv2.circle(canvas, (30, hud_y0 + 28), 5, s_col, -1, cv2.LINE_AA)
    _put_text(canvas, status[:80], (44, hud_y0 + 34), 0.48, THEME["text_primary"], 1)
    # tech stack chips row
    cx = 16
    cy = hud_y0 + 56
    cx = _chip(canvas, cx, cy, "Simulator", bg=(34, 38, 52), fg=THEME["accent_orange"])
    cx = _chip(
        canvas, cx, cy, "8 HIP Kernels", bg=(34, 38, 52), fg=THEME["accent_amber"]
    )
    cx = _chip(canvas, cx, cy, "Llama-3.2-3B", bg=(34, 38, 52), fg=THEME["accent_cyan"])
    cx = _chip(
        canvas, cx, cy, "LLM on iGPU", bg=(34, 38, 52), fg=THEME["accent_violet"]
    )
    return canvas


def render_thinking_panel(w: int, h: int) -> np.ndarray:
    panel = np.full((h, w, 3), THEME["bg_deep"], dtype=np.uint8)
    # header
    head_h = 44
    _filled_rect(panel, (0, 0), (w, head_h), THEME["bg_panel"])
    cv2.line(panel, (0, head_h), (w, head_h), THEME["divider"], 1, cv2.LINE_AA)
    cv2.line(panel, (14, 6), (w - 14, 6), THEME["amd_red"], 2, cv2.LINE_AA)
    _put_text(
        panel,
        "AI BRAIN",
        (16, 28),
        0.55,
        THEME["text_primary"],
        1,
        cv2.FONT_HERSHEY_DUPLEX,
    )
    _put_text(
        panel,
        "Llama-3.2-3B  -  AMD Strix Halo iGPU",
        (120, 28),
        0.38,
        THEME["text_secondary"],
        1,
    )
    with _state_lock:
        user_in = AGENT_STATE["user_input"]
        llm_raw = AGENT_STATE["llm_raw"]
        plan = list(AGENT_STATE["plan"])
        plan_idx = AGENT_STATE["plan_index"]
        plan_stat = AGENT_STATE["plan_status"]
        log_lines = list(AGENT_STATE["exec_log"])
    gpu_pct = float(GPU_STATE.get("gpu_pct", 0.0))
    vram_pct = float(GPU_STATE.get("vram_pct", 0.0))
    vram_used_mb = float(GPU_STATE.get("vram_used_mb", 0.0))
    vram_total_mb = float(GPU_STATE.get("vram_total_mb", 0.0))
    temp_c = float(GPU_STATE.get("temp_c", 0.0))
    gpu_src = GPU_STATE.get("source", "(none)")
    gpu_avail = bool(GPU_STATE.get("available", False))
    # ===== Telemetry section =====
    pad = 14
    y = head_h + 12
    _section_header(panel, pad, y, w - pad * 2, "AMD iGPU TELEMETRY", THEME["amd_red"])
    if temp_c > 0.5:
        temp_str = f"{temp_c:4.1f} C"
        (tw, _), _ = cv2.getTextSize(temp_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        _put_text(
            panel, temp_str, (w - tw - pad, y + 13), 0.40, THEME["accent_orange"], 1
        )
    y += 24
    bar_x = pad
    bar_w = w - pad * 2
    bar_h = 22
    _draw_gradient_bar(panel, bar_x, y, bar_w, bar_h, gpu_pct, "GPU", history=GPU_HIST)
    y += bar_h + 8
    vlabel = (
        f"VRAM  {vram_used_mb:5.0f} / {vram_total_mb:5.0f} MB"
        if vram_total_mb > 0
        else "VRAM"
    )
    _draw_gradient_bar(
        panel, bar_x, y, bar_w, bar_h, vram_pct, vlabel, history=VRAM_HIST
    )
    y += bar_h + 6
    if not gpu_avail:
        _put_text(
            panel,
            f"source: {gpu_src} (bar inactive)",
            (pad, y + 10),
            0.36,
            THEME["text_muted"],
            1,
        )
        y += 14
    else:
        _put_text(
            panel, f"source: {gpu_src}", (pad, y + 10), 0.34, THEME["text_muted"], 1
        )
        y += 14
    # divider
    y += 8
    cv2.line(panel, (pad, y), (w - pad, y), THEME["divider"], 1, cv2.LINE_AA)
    y += 10
    # ===== Input =====
    _section_header(panel, pad, y, w - pad * 2, "USER INPUT", THEME["accent_cyan"])
    y += 22
    _rounded_rect(panel, (pad, y), (w - pad, y + 26), THEME["bg_chip"], radius=8)
    _put_text(
        panel, f">  {user_in[:58]}", (pad + 10, y + 18), 0.46, THEME["text_primary"], 1
    )
    y += 32
    # ===== LLM JSON =====
    _section_header(panel, pad, y, w - pad * 2, "PLANNER OUTPUT", THEME["accent_green"])
    y += 22
    box_h = 40
    _rounded_rect(
        panel, (pad, y), (w - pad, y + box_h), THEME["bg_panel_alt"], radius=8
    )
    raw = llm_raw[:200]
    line1 = raw[:60]
    line2 = raw[60:120] if len(raw) > 60 else ""
    _put_text(panel, line1, (pad + 10, y + 16), 0.38, THEME["accent_green"], 1)
    if line2:
        _put_text(panel, line2, (pad + 10, y + 32), 0.38, THEME["accent_green"], 1)
    y += box_h + 8
    # ===== Plan (compact: status + progress + current step only) =====
    status_col = {
        "running": THEME["accent_orange"],
        "done": THEME["ok"],
        "failed": THEME["err"],
        "planning": THEME["accent_cyan"],
        "idle": THEME["text_muted"],
    }.get(plan_stat, THEME["text_secondary"])
    _section_header(
        panel,
        pad,
        y,
        w - pad * 2,
        f"EXECUTION PLAN  -  {plan_stat.upper()}",
        status_col,
    )
    y += 22
    if plan:
        done = max(0, min(len(plan), plan_idx))
        bar_pct = (done / len(plan)) * 100.0 if len(plan) > 0 else 0.0
        _rounded_rect(panel, (pad, y), (w - pad, y + 6), THEME["bg_chip"], radius=3)
        fill_w = int((w - pad * 2) * bar_pct / 100.0)
        if fill_w > 0:
            tmp = np.zeros_like(panel)
            _hgradient(
                tmp,
                (pad, y),
                (pad + fill_w, y + 6),
                THEME["accent_orange"],
                THEME["amd_red"],
            )
            mask = np.zeros(panel.shape[:2], dtype=np.uint8)
            _rounded_rect(mask, (pad, y), (pad + fill_w, y + 6), 255, radius=3)
            panel[mask > 0] = tmp[mask > 0]
        y += 16
        _cur_i = (
            plan_idx
            if 0 <= plan_idx < len(plan)
            else max(0, min(plan_idx, len(plan) - 1))
        )
        _cur = plan[_cur_i]
        _step_txt = (
            f"step {min(_cur_i + 1, len(plan))}/{len(plan)}:  {_step_summary(_cur)}"
        )
        _put_text(
            panel, _step_txt[:52], (pad + 4, y + 12), 0.42, THEME["text_primary"], 1
        )
        y += 20
    else:
        _put_text(
            panel, "(no plan yet)", (pad + 8, y + 12), 0.40, THEME["text_muted"], 1
        )
        y += 18
    # ===== Exec log (fills only the space that's left; never overflows the panel) =====
    # Compute the room remaining between the plan block (y) and the panel bottom,
    # then fit as many log lines as physically fit. This guarantees the white log
    # text stays inside the panel instead of spilling past the bottom-right edge.
    # The kernel-output thumbnails occupy a fixed band at the bottom of the
    # panel; the exec log fills only the space ABOVE that band so nothing spills
    # into the thumbnails or past the panel edge.
    # Exec log: capped to a few lines, drawn right under the plan block. We track
    # where it actually ends (log_bottom) so the kernel band can float up to meet
    # it instead of being pinned to the panel's bottom edge.
    log_top = y + 8
    LINE_DY = 14
    log_lines = log_lines[-6:]
    _section_header(
        panel, pad, log_top, w - pad * 2, "EXEC LOG", THEME["accent_violet"]
    )
    log_y = log_top + 20
    for line in log_lines:
        _put_text(panel, line[:60], (pad + 6, log_y), 0.34, THEME["text_secondary"], 1)
        log_y += LINE_DY
    log_bottom = log_y  # y just past the last log line
    # ===== HIP kernel outputs — 2 rows x 3 thumbnails (K1-K3 + K5-K7) =====
    # Row 1 = depth / normal / seg; row 2 = the classic gray -> blur -> edge trio.
    # Thumbnails are sized to FIT the free space between the exec log and the panel
    # bottom (4:3 aspect preserved), and the whole band is centered in that space
    # so it floats with even padding instead of hugging the bottom edge.
    _rows = [
        [
            ("K1 depth", KERNEL_STATE["thumbs"]["depth"]),
            ("K2 normal", KERNEL_STATE["thumbs"]["normal"]),
            ("K3 seg", KERNEL_STATE["thumbs"]["seg"]),
        ],
        [
            ("K5 gray", KERNEL_STATE["thumbs"]["gray"]),
            ("K6 blur", KERNEL_STATE["thumbs"]["blur"]),
            ("K7 sobel", KERNEL_STATE["thumbs"]["edges"]),
        ],
    ]
    n_rows, n_cols = len(_rows), 3
    gap, rowgap, header_h = 10, 12, 26
    region_top = log_bottom + 16
    region_bot = h - 14
    avail_h = region_bot - region_top
    # per-thumbnail height that lets both rows + header + gaps fit; clamp sane.
    th_budget = (avail_h - 10 - header_h - rowgap * (n_rows - 1)) // n_rows
    th_ = int(max(58, min(104, th_budget)))
    tw_ = int(th_ * 4 / 3)
    max_tw = (w - pad * 2 - gap * (n_cols - 1)) // n_cols
    if tw_ > max_tw:  # width-limited -> shrink to fit
        tw_ = max_tw
        th_ = int(tw_ * 3 / 4)
    band_h = 10 + header_h + th_ * n_rows + rowgap * (n_rows - 1) + 4
    kern_top = region_top + 10 + max(0, (avail_h - band_h) // 2)
    kern_top = min(kern_top, region_bot - band_h + 14)  # never run off the bottom
    row_w = tw_ * n_cols + gap * (n_cols - 1)
    x_start = pad + max(0, (w - pad * 2 - row_w) // 2)  # center the columns
    cv2.line(
        panel,
        (pad, kern_top - 10),
        (w - pad, kern_top - 10),
        THEME["divider"],
        1,
        cv2.LINE_AA,
    )
    # Only these 6 kernels produce a viewable image; K3b (color-lock) + K4
    # (tactile) are data kernels, so 6 thumbnails is correct even when 7 run.
    _section_header(
        panel,
        pad,
        kern_top,
        w - pad * 2,
        "HIP OUTPUTS  -  K1-K3 + K5-K7 IMAGE  (K3b/K4 = DATA)",
        THEME["accent_orange"],
    )
    ty = kern_top + header_h
    for row in _rows:
        tx = x_start
        for label, img in row:
            x0, y0, x1, y1 = tx, ty, tx + tw_, ty + th_
            if img is not None:
                try:
                    panel[y0:y1, x0:x1] = cv2.resize(img, (tw_, th_))
                except Exception:
                    _filled_rect(panel, (x0, y0), (x1, y1), THEME["bg_panel_alt"])
            else:
                _filled_rect(panel, (x0, y0), (x1, y1), THEME["bg_panel_alt"])
                _put_text(
                    panel,
                    "(idle)",
                    (x0 + 8, y0 + th_ // 2),
                    0.34,
                    THEME["text_muted"],
                    1,
                )
            _rounded_rect(
                panel,
                (x0 - 1, y0 - 1),
                (x1 + 1, y1 + 1),
                THEME["accent_orange"],
                radius=6,
                thickness=1,
            )
            _filled_rect(panel, (x0, y1 - 14), (x1, y1), (12, 14, 20))
            _put_text(panel, label, (x0 + 5, y1 - 4), 0.34, THEME["text_primary"], 1)
            tx += tw_ + gap
        ty += th_ + rowgap
    return panel


def _step_summary(step: dict) -> str:
    a = step.get("action", "?")
    if a == "pick":
        to = step.get("to")
        return f"pick {step.get('color', '?')}" + (f" -> {to}" if to else "")
    if a == "stack":
        return f"stack {step.get('top', '?')} on {step.get('bottom', '?')}"
    if a == "sort":
        return f"sort {step.get('color', '?')}"
    if a == "home":
        return "home"
    return a


def compose_frame(
    scene_rgb: np.ndarray, status: str, highlight: str | None
) -> np.ndarray:
    canvas = np.full((COMP_H, COMP_W, 3), THEME["bg_deep"], dtype=np.uint8)
    # global ambient gradient
    _vgradient(canvas, (0, 0), (COMP_W, COMP_H), (14, 16, 22), (4, 5, 9))
    # 2-panel layout: scene on the left, the "AI BRAIN" telemetry/plan panel
    # fills the FULL-height right column. (YOLO vision panel removed -- it was
    # decorative only and hallucinated COCO classes on synthetic renders.)
    main_panel = annotate_main(scene_rgb, status, highlight, MAIN_W, MAIN_H)
    thinking_panel = render_thinking_panel(PANEL_W, COMP_H)
    canvas[0:MAIN_H, 0:MAIN_W] = main_panel
    canvas[0:COMP_H, MAIN_W : MAIN_W + PANEL_W] = thinking_panel
    # subtle vertical separator between left and right column
    cv2.line(canvas, (MAIN_W, 0), (MAIN_W, COMP_H), THEME["divider"], 1, cv2.LINE_AA)
    return canvas


_display_stop = threading.Event()


def display_loop(display_q: queue.Queue):
    global _ffmpeg_proc
    out_path = os.path.join(_HERE, "demo_output.mkv")
    _ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{COMP_W}x{COMP_H}",
            "-pix_fmt",
            "bgr24",
            "-r",
            "30",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-f",
            "matroska",
            out_path,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    print(f"[DISPLAY] Recording composite to {out_path}")
    while not _display_stop.is_set():
        try:
            frame = display_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if _display_stop.is_set():
            break
        proc = _ffmpeg_proc
        if proc is None or proc.stdin is None or proc.stdin.closed:
            break
        try:
            proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, ValueError, OSError):
            break


def stop_video():
    global _ffmpeg_proc
    _display_stop.set()
    proc = _ffmpeg_proc
    if proc is None:
        return
    time.sleep(0.2)
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    _ffmpeg_proc = None


def push_frame(
    cam, display_q: queue.Queue, status: str = "", highlight_color: str | None = None
):
    global _FRAME_IDX
    _FRAME_IDX += 1
    if _HAS_TORCH and (_FRAME_IDX % KERNEL_PIP_EVERY == 0):
        _run_perception_kernels(cam)
    rgb, *_ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    rgb_np = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
    frame = compose_frame(rgb_np, status=status, highlight=highlight_color)
    if HAS_DISPLAY:
        cv2.imshow(WINDOW, frame)
        cv2.waitKey(1)
    else:
        try:
            display_q.put_nowait(frame)
        except queue.Full:
            pass


# =============================================================================
# EXECUTOR
# =============================================================================
def move_to_qpos(
    franka,
    scene,
    cam,
    display_q,
    qpos_goal,
    motors_dof,
    fingers_dof,
    status: str,
    highlight_color: str | None = None,
    n_waypoints: int = 200,
    render_every: int = 4,
    finger_pos: float | None = None,
    finger_force: float | None = None,
):
    path = franka.plan_path(qpos_goal=qpos_goal, num_waypoints=n_waypoints)
    use_force = finger_force is not None
    if use_force:
        f_force_arr = np.array([finger_force, finger_force])
        f_target = None
    else:
        if finger_pos is None:
            f_target = qpos_goal[-2:]
        else:
            f_target = np.array([finger_pos, finger_pos])
    for i, waypoint in enumerate(path):
        franka.control_dofs_position(waypoint[:-2], motors_dof)
        if use_force:
            franka.control_dofs_force(f_force_arr, fingers_dof)
        else:
            franka.control_dofs_position(f_target, fingers_dof)
        scene.step()
        if i % render_every == 0:
            push_frame(cam, display_q, status=status, highlight_color=highlight_color)
    for _ in range(40):
        franka.control_dofs_position(qpos_goal[:-2], motors_dof)
        if use_force:
            franka.control_dofs_force(f_force_arr, fingers_dof)
        else:
            franka.control_dofs_position(f_target, fingers_dof)
        scene.step()
    push_frame(cam, display_q, status=status, highlight_color=highlight_color)


def go_home(franka, scene, cam, display_q, motors_dof, fingers_dof):
    push_log("[EXEC] Going home...")
    move_to_qpos(
        franka,
        scene,
        cam,
        display_q,
        HOME_QPOS,
        motors_dof,
        fingers_dof,
        status="Returning home...",
        n_waypoints=150,
        finger_pos=GRIPPER_OPEN,
    )


def hold_target(
    franka,
    scene,
    cam,
    display_q,
    qpos_arm_target,
    motors_dof,
    n_steps: int,
    render_every: int = 6,
    status: str = "",
    highlight_color: str | None = None,
    grip_force: float | None = None,
    fingers_dof=None,
):
    # grip_force (N): if set, actively re-assert the finger squeeze EVERY step so
    # the cube can't slip out during lift/carry/place. Without this the fingers
    # relax during the fast lateral carry and the cube gets flung.
    franka.control_dofs_position(qpos_arm_target[:-2], motors_dof)
    _hold_grip = grip_force is not None and fingers_dof is not None
    if _hold_grip:
        _f = np.array([-abs(grip_force), -abs(grip_force)])
    for i in range(n_steps):
        franka.control_dofs_position(qpos_arm_target[:-2], motors_dof)
        if _hold_grip:
            franka.control_dofs_force(_f, fingers_dof)
        scene.step()
        if i % render_every == 0:
            push_frame(cam, display_q, status=status, highlight_color=highlight_color)


def _set_state(**kw):
    with _state_lock:
        AGENT_STATE.update(kw)


def pick_and_place(
    franka,
    scene,
    cam,
    display_q,
    motors_dof,
    fingers_dof,
    color: str,
    place_xyz: np.ndarray,
    place_above_cube: bool = False,
) -> bool:
    if color not in CUBE_ENTITIES:
        push_log(f"[EXEC] X Unknown color: {color}")
        return False
    end_effector = franka.get_link("hand")
    cube = CUBE_ENTITIES[color]
    cube_pos = cube.get_pos().cpu().numpy().flatten().astype(float)
    tx, ty, tz = float(cube_pos[0]), float(cube_pos[1]), float(cube_pos[2])
    push_log(f"[PERCEPT] {color} @ ({tx:+.3f},{ty:+.3f},{tz:+.3f})")
    _fb = np.array([tx, ty])
    _txy = _kernel_target_xy(cam, color, _fb)
    if not np.allclose(_txy, _fb):
        push_log(
            f"[K3b] color-lock refined {color} -> " f"({_txy[0]:+.3f},{_txy[1]:+.3f})"
        )
    tx, ty = float(_txy[0]), float(_txy[1])
    grasp_z = tz + HAND_TO_TIP
    pre_z = tz + PRE_GRASP_OFFSET
    lift_z = tz + LIFT_OFFSET
    qpos_hover = franka.inverse_kinematics(
        link=end_effector, pos=np.array([tx, ty, pre_z]), quat=GRASP_QUAT
    )
    qpos_hover[-2:] = GRIPPER_OPEN
    move_to_qpos(
        franka,
        scene,
        cam,
        display_q,
        qpos_hover,
        motors_dof,
        fingers_dof,
        status=f"Hovering above {color}",
        highlight_color=color,
        n_waypoints=180,
        finger_pos=GRIPPER_OPEN,
    )
    cube_pos = cube.get_pos().cpu().numpy().flatten().astype(float)
    tx, ty = float(cube_pos[0]), float(cube_pos[1])
    qpos_grasp = franka.inverse_kinematics(
        link=end_effector, pos=np.array([tx, ty, grasp_z]), quat=GRASP_QUAT
    )
    qpos_grasp[-2:] = GRIPPER_OPEN
    move_to_qpos(
        franka,
        scene,
        cam,
        display_q,
        qpos_grasp,
        motors_dof,
        fingers_dof,
        status=f"Descending on {color}",
        highlight_color=color,
        n_waypoints=120,
        render_every=3,
        finger_pos=GRIPPER_OPEN,
    )
    push_log(f"[EXEC] Close gripper on {color} (force={FINGER_FORCE}N)")
    franka.control_dofs_position(qpos_grasp[:-2], motors_dof)
    franka.control_dofs_force(np.array([-FINGER_FORCE, -FINGER_FORCE]), fingers_dof)
    if TACTILE_SENSORS is not None and _HAS_TORCH:
        # K4 CLOSED LOOP: squeeze until the GPU tactile reduction reports SECURE
        _secure_run = 0
        _grip = None
        for s in range(GRASP_MAX_STEPS):
            franka.control_dofs_position(qpos_grasp[:-2], motors_dof)
            franka.control_dofs_force(
                np.array([-FINGER_FORCE, -FINGER_FORCE]), fingers_dof
            )
            scene.step()
            if s % TACTILE_READ_EVERY == 0:
                try:
                    _ls, _rs = TACTILE_SENSORS
                    _grip = hip_tactile_reduce(_ls.read(), _rs.read())
                    KERNEL_STATE["grip"] = _grip
                    _secure_run = _secure_run + 1 if _grip["secure"] else 0
                except Exception:
                    pass
            if s % 10 == 0:
                _gN = _grip["grip_force_N"] if _grip else 0.0
                _gc = _grip["n_contact"] if _grip else 0
                _gt = _grip["n_taxels"] if _grip else 128
                push_frame(
                    cam,
                    display_q,
                    status=f"Grasping {color} | K4 {_gN:.1f}N {_gc}/{_gt}",
                    highlight_color=color,
                )
            if s >= GRASP_MIN_STEPS and _secure_run >= SECURE_HOLD_STEPS:
                push_log(
                    f"[K4] grip SECURE @ step {s} "
                    f"({_grip['n_contact']} taxels, {_grip['grip_force_N']:.1f}N)"
                )
                break
        else:
            push_log("[K4] max squeeze reached without SECURE (continuing)")
    else:
        for s in range(120):
            scene.step()
            if s % 10 == 0:
                push_frame(
                    cam, display_q, status=f"Grasping {color}", highlight_color=color
                )
    qpos_lift = franka.inverse_kinematics(
        link=end_effector, pos=np.array([tx, ty, lift_z]), quat=GRASP_QUAT
    )
    hold_target(
        franka,
        scene,
        cam,
        display_q,
        qpos_lift,
        motors_dof,
        n_steps=200,
        render_every=6,
        status=f"Lifting {color}",
        highlight_color=color,
        grip_force=FINGER_FORCE,
        fingers_dof=fingers_dof,
    )
    cube_z = cube.get_pos().cpu().numpy().flatten().astype(float)[2]
    if cube_z < CUBE_SIDE * 1.5:
        push_log(f"[EXEC] X grasp failed (cube z={cube_z:.3f})")
        franka.control_dofs_position(
            np.array([GRIPPER_OPEN, GRIPPER_OPEN]), fingers_dof
        )
        for _ in range(50):
            scene.step()
        return False
    dx, dy, dz_center = float(place_xyz[0]), float(place_xyz[1]), float(place_xyz[2])
    push_log(f"[EXEC] Carry {color} -> ({dx:+.2f},{dy:+.2f})")
    qpos_carry = franka.inverse_kinematics(
        link=end_effector, pos=np.array([dx, dy, SAFE_Z]), quat=GRASP_QUAT
    )
    hold_target(
        franka,
        scene,
        cam,
        display_q,
        qpos_carry,
        motors_dof,
        n_steps=360,
        render_every=6,
        status=f"Carrying {color}",
        highlight_color=color,
        grip_force=FINGER_FORCE,
        fingers_dof=fingers_dof,
    )
    place_hand_z = dz_center + HAND_TO_TIP + PLACE_CLEARANCE
    qpos_place = franka.inverse_kinematics(
        link=end_effector, pos=np.array([dx, dy, place_hand_z]), quat=GRASP_QUAT
    )
    hold_target(
        franka,
        scene,
        cam,
        display_q,
        qpos_place,
        motors_dof,
        n_steps=200,
        render_every=6,
        status=f"Placing {color}",
        highlight_color=color,
        grip_force=FINGER_FORCE,
        fingers_dof=fingers_dof,
    )
    push_log("[EXEC] Release (open fingers, arm latched)")
    franka.control_dofs_position(np.array([GRIPPER_OPEN, GRIPPER_OPEN]), fingers_dof)
    for s in range(120):
        scene.step()
        if s % 10 == 0:
            push_frame(
                cam, display_q, status=f"Releasing {color}", highlight_color=color
            )
    qpos_retreat = franka.inverse_kinematics(
        link=end_effector, pos=np.array([dx, dy, SAFE_Z]), quat=GRASP_QUAT
    )
    hold_target(
        franka,
        scene,
        cam,
        display_q,
        qpos_retreat,
        motors_dof,
        n_steps=150,
        render_every=6,
        status=f"Retreating after {color}",
    )
    for _ in range(40):
        scene.step()
    final = cube.get_pos().cpu().numpy().flatten().astype(float)
    push_log(
        f"[EXEC] OK {color} placed @ ({final[0]:+.2f},{final[1]:+.2f},{final[2]:+.2f})"
    )
    return True


def execute_step(
    step: dict, franka, scene, cam, display_q, motors_dof, fingers_dof
) -> bool:
    a = step.get("action")
    if a == "home":
        go_home(franka, scene, cam, display_q, motors_dof, fingers_dof)
        return True
    if a == "pick":
        color = (step.get("color") or "").lower()
        if color not in CUBE_ENTITIES:
            push_log(f"[EXEC] X unknown color '{color}'")
            return False
        zone = (step.get("to") or "").lower()
        if zone:
            target = DIRECTION_ZONES.get(zone)
            if target is None:
                push_log(f"[EXEC] X unknown zone '{zone}', using default drop")
                target = DROP_OFF_POS
            else:
                push_log(
                    f"[EXEC] Drop zone: {zone} -> "
                    f"({target[0]:+.2f},{target[1]:+.2f})"
                )
        else:
            target = DROP_OFF_POS
        target = clamp_to_workspace(target, label=f"pick {color} drop")
        ok = pick_and_place(
            franka, scene, cam, display_q, motors_dof, fingers_dof, color, target
        )
        go_home(franka, scene, cam, display_q, motors_dof, fingers_dof)
        return ok
    if a == "sort":
        color = (step.get("color") or "").lower()
        if color not in CUBE_ENTITIES:
            push_log(f"[EXEC] X unknown color '{color}'")
            return False
        target = clamp_to_workspace(SORT_ZONES[color], label=f"sort {color}")
        ok = pick_and_place(
            franka, scene, cam, display_q, motors_dof, fingers_dof, color, target
        )
        go_home(franka, scene, cam, display_q, motors_dof, fingers_dof)
        return ok
    if a == "stack":
        top = (step.get("top") or "").lower()
        bot = (step.get("bottom") or "").lower()
        if top not in CUBE_ENTITIES or bot not in CUBE_ENTITIES:
            push_log(f"[EXEC] X stack needs valid colors (got top={top}, bot={bot})")
            return False
        if top == bot:
            push_log("[EXEC] X cannot stack a cube on itself")
            return False
        bot_pos = CUBE_ENTITIES[bot].get_pos().cpu().numpy().flatten().astype(float)
        target = np.array([bot_pos[0], bot_pos[1], bot_pos[2] + CUBE_SIDE])
        target = clamp_to_workspace(target, label=f"stack {top}-on-{bot}")
        ok = pick_and_place(
            franka, scene, cam, display_q, motors_dof, fingers_dof, top, target
        )
        go_home(franka, scene, cam, display_q, motors_dof, fingers_dof)
        return ok
    push_log(f"[EXEC] X unknown action '{a}'")
    return False


def execute_plan(plan: list, franka, scene, cam, display_q, motors_dof, fingers_dof):
    if not plan:
        push_log("[PLAN] Empty plan - nothing to do.")
        _set_state(plan_status="failed")
        return
    _set_state(plan=plan, plan_index=0, plan_status="running")
    success = 0
    for i, step in enumerate(plan):
        _set_state(
            plan_index=i, status=f"Step {i+1}/{len(plan)}: {_step_summary(step)}"
        )
        push_log(f"[PLAN] Step {i+1}/{len(plan)}: {_step_summary(step)}")
        ok = execute_step(step, franka, scene, cam, display_q, motors_dof, fingers_dof)
        if ok:
            success += 1
        else:
            push_log(f"[PLAN] X step {i+1} failed - aborting plan")
            _set_state(plan_status="failed", plan_index=i + 1)
            return
    _set_state(
        plan_status="done",
        plan_index=len(plan),
        status=f"Plan complete ({success}/{len(plan)} ok)",
    )


# =============================================================================
# SCENE SETUP
# =============================================================================
def setup_scene():
    import genesis as gs

    gs.init(backend=gs.amdgpu, logging_level="warning")
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -1.8, 1.8),
            camera_lookat=(0.5, 0.0, 0.2),
            camera_fov=35,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=HAS_DISPLAY,
    )
    scene.add_entity(gs.morphs.Plane())
    for color, pos in INITIAL_CUBES.items():
        scene.add_entity(
            gs.morphs.Box(
                size=(CUBE_SIDE, CUBE_SIDE, CUBE_SIDE),
                pos=tuple(pos),
            ),
            surface=gs.surfaces.Default(color=CUBE_COLORS_RGBA[color]),
        )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        # Recolor the arm flat gold/yellow (classic Genesis Franka look). The MJCF
        # asset ships photo-real white/black textures; a flat Default surface overrides
        # every link's material WITHOUT renaming links, so tactile sensors, wrist cam
        # and IK ("hand"/"left_finger"/"right_finger") all keep working.
        surface=gs.surfaces.Default(color=ARM_COLOR_RGB),
    )
    cam = scene.add_camera(
        res=CAM_RES,
        pos=CAM_POS,
        lookat=CAM_LOOKAT,
        fov=CAM_FOV,
        GUI=False,
    )
    global TACTILE_SENSORS
    TACTILE_SENSORS = None
    try:
        import genesis.utils.geom as gu

        _probe_normal = (0.0, -1.0, 0.0)
        _probe_local_pos = gu.generate_grid_points_on_plane(
            lo=(-0.006, 0.0, 0.04),
            hi=(0.008, 0.0, 0.05),
            normal=_probe_normal,
            nx=8,
            ny=8,
        ).reshape(-1, 3)
        _tk = dict(
            entity_idx=franka.idx,
            probe_local_pos=_probe_local_pos,
            probe_local_normal=_probe_normal,
            probe_radius=0.002,
            draw_debug=False,
        )
        _left = scene.add_sensor(
            gs.sensors.ElastomerTaxel(
                link_idx_local=franka.get_link("left_finger").idx_local,
                track_link_idx=(franka.get_link("left_finger").idx_local,),
                **_tk,
            )
        )
        _right = scene.add_sensor(
            gs.sensors.ElastomerTaxel(
                link_idx_local=franka.get_link("right_finger").idx_local,
                track_link_idx=(franka.get_link("right_finger").idx_local,),
                **_tk,
            )
        )
        TACTILE_SENSORS = (_left, _right)
        KERNEL_STATE["tactile"] = True
        print("[SCENE] ElastomerTaxel tactile sensors attached (K4 grasp gating ON)")
    except Exception as _e:
        print(f"[SCENE] tactile sensors unavailable ({_e}); K4 -> fixed squeeze")
    scene.build()
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    franka.set_dofs_kp(KP)
    franka.set_dofs_kv(KV)
    franka.set_dofs_force_range(F_LO, F_HI)
    entity_list = list(scene.entities)
    for i, color in enumerate(["red", "green", "blue"]):
        CUBE_ENTITIES[color] = entity_list[1 + i]
        print(f"[SCENE] {color} cube -> entity idx {CUBE_ENTITIES[color].idx}")
    return scene, franka, cam, motors_dof, fingers_dof


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 64)
    print("  AMD Advancing AI  .  Physical AI Agent (v3 + HIP/ROCm kernels)")
    print("  AMD Strix Halo  |  Llama-3.2-3B  |  Genesis Physics")
    print(f"  {CREDIT_LINE}")
    print("=" * 64)
    if not HAS_DISPLAY:
        print("[DISPLAY] No $DISPLAY - running headless, frames -> demo_output.mkv\n")
    start_llm_server()
    gpu_stop = threading.Event()
    threading.Thread(target=gpu_monitor, args=(gpu_stop,), daemon=True).start()
    print("[SCENE] Building Genesis scene...")
    scene, franka, cam, motors_dof, fingers_dof = setup_scene()
    print("[SCENE] Ready")
    display_q: queue.Queue = queue.Queue(maxsize=4)
    if HAS_DISPLAY:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, COMP_W, COMP_H)
    else:
        threading.Thread(target=display_loop, args=(display_q,), daemon=True).start()
    go_home(franka, scene, cam, display_q, motors_dof, fingers_dof)
    print("\n[SCENE] Initial cube positions (from physics):")
    for color, pos in get_cube_positions().items():
        print(f"   {color:<6}: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
    cmd_q: queue.Queue[str] = queue.Queue()

    def input_loop():
        print("\nReady! Try natural-language commands such as:")
        print("   pick the red cube")
        print("   pick red and put it on the left")
        print("   stack red on green")
        print("   sort the cubes")
        print("   home          (reset arm)")
        print("   quit          (exit)\n")
        while True:
            try:
                txt = input("[YOU] > ").strip()
                if txt:
                    cmd_q.put(txt)
            except EOFError:
                break

    threading.Thread(target=input_loop, daemon=True).start()
    _set_state(status="Idle - awaiting command")
    while True:
        if not cmd_q.empty():
            raw = cmd_q.get()
            if raw.lower() in ("quit", "exit", "q"):
                print("[AGENT] Shutting down.")
                break
            _set_state(
                user_input=raw,
                llm_raw="(thinking...)",
                plan=[],
                plan_index=-1,
                plan_status="planning",
                status=f"Planning: {raw[:40]}",
            )
            print(f"\n[LLM] Parsing: '{raw}'")
            parsed = parse_plan(raw)
            plan = parsed.get("plan", [])
            print(f"[LLM] -> {plan}")
            if not plan:
                print("[AGENT] Couldn't parse. Try: 'pick the red cube'")
                _set_state(plan_status="failed", status="Command not understood")
                continue
            execute_plan(plan, franka, scene, cam, display_q, motors_dof, fingers_dof)
            _set_state(status="Idle - awaiting command")
        else:
            scene.step()
            with _state_lock:
                cur_status = AGENT_STATE.get("status", "Idle")
            push_frame(cam, display_q, status=cur_status)
    gpu_stop.set()
    if _llm_proc:
        try:
            _llm_proc.terminate()
        except Exception:
            pass
    stop_video()
    if not HAS_DISPLAY:
        print(f"[DISPLAY] Video -> {os.path.join(_HERE, 'demo_output.mkv')}")
    if HAS_DISPLAY:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[AGENT] Interrupted.")
        if _llm_proc:
            try:
                _llm_proc.terminate()
            except Exception:
                pass
        stop_video()
        if not HAS_DISPLAY:
            print(f"[DISPLAY] Video -> {os.path.join(_HERE, 'demo_output.mkv')}")
