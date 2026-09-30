# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""_labkit — the mini-lab's plumbing, hidden from attendees.

The `lab/lab.ipynb` notebook is meant to show attendees ONLY the task cells and
the editable `# >>> TRY ME <<<` knobs. All the setup noise (imports, sys.path
juggling, Ollama streaming, robot connect/shutdown, the ipywidgets chat bar,
camera capture) lives here so the notebook stays clean:

    from _labkit import *
    connect()

Shared mutable state (the connected robot + the two controllers) lives on a
single `state` singleton. `from _labkit import *` cannot write module globals
back from a notebook cell (name binding rebinds the notebook's own name, not the
module's), so instead the notebook NEVER touches these directly: it calls the
lazy helpers (`get_controller_v1`, `get_controller_offline`, `look_and_describe`)
which read/write `state`, keeping the module and the notebook on the SAME object.
"""

import os
import sys

# sys.path bootstrap FIRST — before the emo_v* imports below — so that importing
# _labkit also makes emo_v1/2/3 (in lab/) and utils/ (at repo root) importable,
# whether the kernel CWD is the repo root or lab/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../lab
_REPO_ROOT = os.path.dirname(_THIS_DIR)  # repo root
for _p in (_THIS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _resolve(path: str) -> str:
    """Resolve a possibly-relative asset path against the repo root, so notebook
    cells work no matter what CWD the Jupyter kernel was launched from.

    Repo-relative asset paths (e.g. `models/...onnx`) would otherwise resolve
    against the kernel's CWD; if that's `lab/` the Piper model isn't found and
    Task 2 goes silent. Absolute paths and empty strings are passed through
    untouched so user overrides and "no path" cases behave as before."""
    if not path or os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


import json
import time
import base64
import threading
from typing import Any, Callable, Dict, Optional

import requests

# Reuse the existing lab code (import chain: emo_v3 -> emo_v2 -> emo_v1). None of
# these touch the robot/network at import time, so importing _labkit is cheap and
# side-effect-free — connect() is the only thing that talks to hardware.
from emo_v1 import EmotionControllerV6, strip_emojis
from emo_v2 import EmotionControllerV71

# WebPreview is reused as the single-owner camera reader for the notebook live
# feed (see ask_live below): it already runs ONE ffmpeg that owns the device and
# keeps a latest-JPEG buffer we can pull from, so we don't reinvent capture.
from emo_v3 import find_camera_device, capture_jpeg, CAMERA_NAME_HINT, WebPreview

# IPython.display is available inside a notebook kernel; guard it so a plain
# `import _labkit` (e.g. in a py_compile / import smoke test) doesn't blow up.
try:
    from IPython.display import Image, display
except Exception:  # pragma: no cover - only hit outside a notebook kernel
    Image = None

    def display(*_args, **_kwargs):  # type: ignore[misc]
        """No-op display fallback when IPython isn't present."""
        pass


OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "qwen3.5:0.8b"
VLM_MODEL = "qwen2.5vl:3b"

# Ollama streams in token/word CHUNKS, so raw output looks choppy. We re-emit each
# chunk one CHARACTER at a time so the reply reads like a terminal teletype.
# Referenced as a module global inside stream_ollama() so staff/attendees can
# disable it live from the notebook with `_labkit.TYPEWRITER_DELAY = 0`.
TYPEWRITER_DELAY = 0.01  # seconds per character; set 0 to disable

# Canonical defaults so the notebook's Reset cell can restore the TRY ME knobs
# between attendees (booth hygiene: one attendee's pirate persona / voice swap
# shouldn't carry over to the next). These MUST stay byte-for-byte identical to
# the initial values in the notebook's `# >>> TRY ME <<<` config cells.
DEFAULT_PERSONA_1 = (
    "You are a cute desktop robot assistant. Respond with enthusiasm and warmth, "
    "in two or three short sentences. Keep it brief and conversational."
)
DEFAULT_VOICE_1 = "en-US-JennyNeural"
DEFAULT_PERSONA_2 = (
    "You are a cute desktop robot assistant. Respond with enthusiasm and warmth, "
    "in two or three short sentences. Keep it brief and conversational."
)
DEFAULT_PIPER_MODEL = "models/en-us-blizzard_lessac-medium.onnx"
# Task 3's editable knob. Attendees now TYPE their question live in the bar, so a
# static describe-prompt is no longer the central knob. Instead this ANSWER STYLE
# steers Reachy's tone/persona for every answer; it's folded into the VLM prompt
# alongside the attendee's typed question by build_vision_prompt() below.
DEFAULT_VISION_STYLE = (
    "You are a cheerful, curious desktop robot. Answer in one or two short, "
    "upbeat sentences."
)


class _State:
    """The single shared, mutable handle bag for the whole lab.

    Kept as an OBJECT (not module globals) precisely because the notebook imports
    via `*`: attribute writes on this shared instance are visible everywhere,
    whereas rebinding a module global from a cell would not write back.
    """

    reachy: Any = None  # the connected robot (or None if the daemon is down)
    reachy_ctx: Any = None  # the ReachyMini context manager we entered
    controller_v1: Any = None  # Task 1 controller (Edge-TTS), built lazily
    controller_offline: Any = None  # Tasks 2 & 3 controller (Piper), built lazily

    # Task 3 live feed (see ask_live). Stored on the singleton so re-running the
    # notebook cell can tear down the previous feed first — otherwise each run
    # would stack another ffmpeg reader + pump thread and leak the camera device.
    live_reader: Any = None  # the WebPreview instance owning ffmpeg + latest frame
    live_thread: Any = None  # the frame-pump thread pushing bytes into the widget
    live_stop: Any = None  # threading.Event; set to stop the pump thread


state = _State()


def stream_ollama(endpoint: str, payload: Dict[str, Any]) -> Optional[str]:
    """POST to Ollama with stream=True and print tokens live as they arrive,
    exactly like the terminal scripts (emo_v1 /api/generate, emo_v3 VLM).
    Returns the accumulated text, or None on a friendly error (no traceback).
    Chunks carry `response`/`thinking` (/api/generate) or `message.content`
    (/api/chat); we read whichever is present."""
    payload = {**payload, "stream": True}
    try:
        resp = requests.post(
            f"{OLLAMA_URL}{endpoint}", json=payload, stream=True, timeout=300
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"❌ Ollama not reachable at {OLLAMA_URL}. Is `ollama serve` running?")
        return None
    except Exception as e:
        print(f"⚠️ Ollama request failed: {e}")
        return None
    full_text = ""
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line.decode("utf-8"))
        except Exception:
            continue
        if chunk.get("error"):
            print(f"\n⚠️ Ollama error: {chunk['error']}")
            return None
        msg = chunk.get("message") or {}
        piece = (
            chunk.get("response") or msg.get("content") or chunk.get("thinking") or ""
        )
        if piece:
            # Typewriter: emit the chunk char-by-char so streamed output reads
            # like a teletype. TYPEWRITER_DELAY <= 0 prints the chunk normally.
            if TYPEWRITER_DELAY > 0:
                for ch in piece:
                    print(ch, end="", flush=True)
                    time.sleep(TYPEWRITER_DELAY)
            else:
                print(piece, end="", flush=True)
            full_text += piece
    print()
    return full_text.strip()


def _react(controller: Any, text: str) -> None:
    """Shared: analyze emotion on the streamed text and speak it via the
    controller's SYNC speak_with_expression (talk + animate together)."""
    if controller is None:
        print("   (No robot: text only — run setup with the daemon up to hear + move.)")
        return
    spoken = strip_emojis(text)
    emotion, intensity, level = controller.analyze_emotion(text)
    print(f"🎭 emotion={emotion} intensity={intensity} level={level:.2f}")
    controller.speak_with_expression(spoken, emotion, intensity, level)


def check_ollama() -> None:
    """Ping Ollama and report which lab models are present. No traceback if down."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"❌ Ollama not reachable at {OLLAMA_URL}. Is `ollama serve` running?")
        return
    except Exception as e:
        print(f"⚠️ Could not query Ollama: {e}")
        return
    print(f"✅ Ollama is reachable at {OLLAMA_URL}")
    names = [m.get("name", "") for m in resp.json().get("models", [])]
    for wanted in (CHAT_MODEL, VLM_MODEL):
        present = wanted in names or f"{wanted}:latest" in names
        mark = "✅" if present else "⚠️"
        hint = "" if present else f"  (pull it: `ollama pull {wanted}`)"
        print(f"   {mark} {wanted}{hint}")


def connect() -> None:
    """Connect to Reachy ONCE and stash the handle on `state`. Friendly ⚠️ if the
    daemon is down (no traceback).

    The scripts use `with ReachyMini(...) as reachy:` — a context manager. A `with`
    block can't span notebook cells, so we enter the context manager manually here
    and exit it in shutdown() at the end.
    """
    if state.reachy is not None:
        print("✅ Reachy already connected (reusing it).")
        return
    try:
        from reachy_mini import ReachyMini
        from reachy_mini.utils import create_head_pose

        state.reachy_ctx = ReachyMini(media_backend="no_media")
        state.reachy = state.reachy_ctx.__enter__()
        state.reachy.goto_target(head=create_head_pose(), duration=1.0)
        print("✅ Connected to Reachy Mini (motion only, media_backend='no_media').")
    except Exception as e:
        state.reachy, state.reachy_ctx = None, None
        print(f"⚠️ Could not connect to Reachy Mini: {e}")
        print("   Start the daemon in a terminal, then re-run this cell:")
        print("       reachy-mini-daemon --no-media")
        print("   (The Ollama chat/vision cells still work without the robot.)")


def get_controller_v1(voice: str) -> Optional[EmotionControllerV6]:
    """Build the Task 1 controller once (Edge-TTS); rebuild if the voice changed.
    Cached on `state.controller_v1`. Returns None (with a ⚠️) if no robot."""
    if state.reachy is None:
        print("⚠️ Robot not connected — run the setup cell (start the daemon first).")
        return None
    if (
        state.controller_v1 is None
        or state.controller_v1.tts_engine.default_voice != voice
    ):
        state.controller_v1 = EmotionControllerV6(
            state.reachy,
            debug=False,
            gentle_mode=True,
            voice=voice,
        )
    return state.controller_v1


def get_controller_offline(piper_model: str) -> Optional[EmotionControllerV71]:
    """Build the offline (Piper) controller once; reused by Tasks 2 & 3.
    Cached on `state.controller_offline`. Returns None (with a ⚠️) if no robot."""
    if state.reachy is None:
        print("⚠️ Robot not connected — run the setup cell (start the daemon first).")
        return None
    # Resolve to an absolute path against the repo root so the Piper model is
    # found regardless of the kernel's CWD (the notebook knob stays relative).
    resolved_model = _resolve(piper_model)
    # Cache/compare on the RESOLVED path so re-entry with the same knob doesn't
    # trigger a rebuild loop, and a genuine voice-model swap does rebuild.
    if (
        state.controller_offline is None
        or getattr(state.controller_offline, "_labkit_piper_model", None)
        != resolved_model
    ):
        state.controller_offline = EmotionControllerV71(
            state.reachy,
            resolved_model,
            piper_config=None,
            speaker_id=0,
            debug=False,
            gentle_mode=True,
        )
        state.controller_offline._labkit_piper_model = resolved_model
    return state.controller_offline


def chat_bar(
    get_controller: Callable[[], Any],
    build_payload: Callable[[str], Dict[str, Any]],
    primer: str = "🤖 Reachy: ",
) -> None:
    """Render an interactive chat input bar for the chat tasks (1 & 2).

    - get_controller(): zero-arg callable returning the right controller (or None).
    - build_payload(user_text): returns the /api/generate payload; it reads the
      CURRENT notebook globals (PERSONA_1/VOICE_1/etc.) at CALL time — the closure
      is created in the notebook namespace — so editing the persona/voice cell
      above takes effect on the next Send (no rebuild needed).

    On Send (button click or Enter): prints "🧑 You: ...", streams the reply with
    the typewriter effect inside an Output widget, then calls _react() so the robot
    speaks. Falls back to a single input() prompt if ipywidgets is unavailable.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display as _display
    except Exception:
        # Graceful fallback: no widgets, just one text prompt this run.
        print("ℹ️ ipywidgets not available — using a simple text prompt.")
        print("   (For the interactive bar: pip install ipywidgets)")
        try:
            user_text = input("🧑 You: ").strip()
        except Exception:
            print("⚠️ Could not read input.")
            return
        if not user_text:
            print("   (Nothing typed.)")
            return
        print(primer, end="", flush=True)
        reply = stream_ollama("/api/generate", build_payload(user_text))
        if reply:
            _react(get_controller(), reply)
        return

    text = widgets.Text(
        placeholder="Type a message and press Enter (or click Send)…",
        layout=widgets.Layout(width="70%"),
        continuous_update=False,  # fire `value` change on Enter/blur, not per keystroke
    )
    send = widgets.Button(description="Send", button_style="primary")
    out = widgets.Output()

    def _handle(_=None):
        user_text = (text.value or "").strip()
        if not user_text:
            return
        text.value = ""
        send.disabled = True  # avoid double-fire while streaming
        text.disabled = True
        try:
            with out:
                print(f"🧑 You: {user_text}")
                print(primer, end="", flush=True)
                reply = stream_ollama("/api/generate", build_payload(user_text))
                if reply:
                    _react(get_controller(), reply)
                print()
        finally:
            send.disabled = False
            text.disabled = False

    # Button click is the primary path. For Enter-to-send we observe the `value`
    # trait (the modern ipywidgets 8 replacement for the deprecated on_submit).
    # With continuous_update=False, `value` fires on Enter/blur. We ignore the
    # change fired by our own `text.value = ""` reset so it can't re-trigger.
    send.on_click(_handle)

    def _on_value_change(change):
        if (change.get("new") or "").strip():
            _handle()

    text.observe(_on_value_change, names="value")

    _display(widgets.HBox([text, send]), out)


def build_vision_prompt(question: str, style: str = DEFAULT_VISION_STYLE) -> str:
    """Combine the attendee's typed QUESTION with an ANSWER STYLE into the prompt
    sent to the VLM alongside the current frame.

    The style comes from the notebook's `VISION_STYLE` knob (tone/persona); the
    question is what the attendee just typed in the bar. Kept minimal on purpose:
    a short instruction so a bare question ("what am I holding?") still yields a
    good, self-contained spoken answer about the current camera view."""
    style = (style or DEFAULT_VISION_STYLE).strip()
    question = (question or "").strip() or "What do you see right now?"
    return (
        f"{style} You are looking through your own camera. "
        f"Answer this about what you currently see: {question}"
    )


def look_and_describe(
    vision_prompt: str, piper_model: str = DEFAULT_PIPER_MODEL
) -> None:
    """Task 3: capture ONE frame from the robot camera, show it inline, stream the
    local VLM's description, then speak + react via the offline (Piper) controller.

    Kept whole here so the notebook action cell is a single call. Friendly ⚠️ if
    the camera or VLM is unavailable (no traceback)."""
    # 1) Capture ONE frame from the robot camera (auto-detects the Arducam).
    jpeg = None
    try:
        device, _ = find_camera_device(CAMERA_NAME_HINT)
        print(f"📷 Using camera device: {device}")
        jpeg = capture_jpeg(device)
        print(f"✅ Captured a frame ({len(jpeg)} bytes).")
    except Exception as e:
        print(f"⚠️ Could not capture a camera frame: {e}")
        print("   Task 3 needs the REAL robot camera (no sim camera). Check that the")
        print("   daemon runs with --no-media so it isn't holding the device, and see")
        print("   cameras with: v4l2-ctl --list-devices")
        return

    # 2) Show the frame, then STREAM the local VLM's description + speak it.
    if Image is not None:
        display(Image(data=jpeg, format="jpeg"))
    b64 = base64.b64encode(jpeg).decode("ascii")
    print("🤖 Reachy sees: ", end="", flush=True)
    # /api/generate + images streaming — same payload emo_v3 uses (temp 0.5 /
    # num_predict 120). Tokens print live so the heavy VLM feels alive.
    description = stream_ollama(
        "/api/generate",
        {
            "model": VLM_MODEL,
            "prompt": vision_prompt,
            "images": [b64],
            "keep_alive": "30m",
            "options": {"temperature": 0.5, "num_predict": 120},
        },
    )
    if description:
        # Reuse the SAME offline Piper controller from Task 2 (build if needed).
        _react(get_controller_offline(piper_model), description)


def _start_live_reader(device: str) -> Any:
    """Start ONE ffmpeg reader that owns `device` for the session and return it.

    We reuse emo_v3's WebPreview because it already runs a single ffmpeg process
    that owns the camera and keeps a latest-JPEG buffer (encode_latest_jpeg()),
    plus an interrupt-resilient _terminate_ffmpeg/stop(). We deliberately DO NOT
    call WebPreview.start() — that would also spin up its HTTP server. Inside a
    notebook we don't want a browser server; we only need the ffmpeg reader +
    latest-frame buffer, so we launch just the reader thread here (mirroring the
    reader-launch half of WebPreview.start) and pull frames on a timer instead.

    Returns the started WebPreview, or raises RuntimeError if ffmpeg couldn't
    produce a first frame (no camera / busy / sim)."""
    reader = WebPreview(device)
    # Launch ONLY the ffmpeg reader thread (not _start_http). This is the same
    # thread WebPreview.start() launches; we stop before the _start_http() call.
    reader._reader = threading.Thread(
        target=reader._run_ffmpeg, name="reachy-nb-cam", daemon=True
    )
    reader._reader.start()
    if not reader._ready.wait(timeout=12.0):
        reader.stop()
        raise RuntimeError("timed out waiting for the first camera frame")
    if reader.error:
        err = reader.error
        reader.stop()
        raise RuntimeError(err)
    return reader


def stop_live() -> None:
    """Stop the Task 3 live feed: signal the pump thread, kill ffmpeg (releasing
    the camera), and clear the handles on `state`.

    Safe to call any number of times (idempotent). Called by shutdown() too, so
    releasing the robot also frees the camera — no orphan ffmpeg left holding the
    device, which is what otherwise causes "Device or resource busy" next run."""
    if state.live_stop is not None:
        state.live_stop.set()
    if state.live_thread is not None:
        try:
            state.live_thread.join(timeout=2.0)
        except (Exception, KeyboardInterrupt):
            pass
    if state.live_reader is not None:
        # WebPreview.stop() is interrupt-resilient and always terminates ffmpeg.
        try:
            state.live_reader.stop()
        except (Exception, KeyboardInterrupt):
            pass
    state.live_reader, state.live_thread, state.live_stop = None, None, None


# Suggestion chips prefilled in the Task 3 question bar. Kept short + few so the
# bar stays uncluttered; clicking one submits it immediately.
VISION_SUGGESTIONS = (
    "What do you see?",
    "What do I look like?",
    "Describe where I am?",
)


def ask_live(
    style_getter: Optional[Callable[[], str]] = None,
    piper_model: str = DEFAULT_PIPER_MODEL,
    fps: int = 8,
) -> None:
    """Task 3 (live): render a live camera feed inside the notebook with a
    QUESTION BAR (type + Enter, or click "Send") and a few suggestion chips. The
    attendee TYPES A QUESTION and Reachy answers about what it currently sees,
    mirroring the Tasks 1 & 2 chat_bar UX. The camera is released automatically
    on shutdown() or when the cell is re-run (no explicit Stop button).

    A SINGLE ffmpeg reader (WebPreview) owns the camera for the session; a
    background thread pumps its latest JPEG into an ipywidgets.Image at ~`fps`.
    Each Ask reads the SAME shared latest frame (never a second capture), so the
    two never collide on "Device or resource busy".

    `style_getter` is an optional zero-arg callable read at each Ask, returning
    the ANSWER STYLE (tone/persona) from the notebook's VISION_STYLE knob so cell
    edits apply to the next question without a rebuild. Defaults to
    DEFAULT_VISION_STYLE when omitted.

    Falls back to a single input() question routed through look_and_describe() if
    ipywidgets is unavailable, and prints friendly guidance (no traceback/crash)
    if the camera can't open."""
    if style_getter is None:
        style_getter = lambda: DEFAULT_VISION_STYLE

    # ipywidgets fallback: no widgets → ask ONE question via input() and route it
    # through the proven one-shot path (single capture + describe + react).
    try:
        import ipywidgets as widgets
        from IPython.display import display as _display
    except Exception:
        print("ℹ️ ipywidgets not available — asking a single question this run.")
        print("   (For the live feed + question bar: pip install ipywidgets)")
        try:
            question = input("🧑 Ask Reachy about what it sees: ").strip()
        except Exception:
            print("⚠️ Could not read input.")
            return
        look_and_describe(build_vision_prompt(question, style_getter()), piper_model)
        return

    # Guard against stacking: if a live view is already running (re-run of this
    # cell), tear the old one down first so we never leak ffmpeg/threads.
    stop_live()

    # Start the single-owner camera reader. Same friendly guidance as
    # look_and_describe on failure — DON'T crash the cell.
    try:
        device, _ = find_camera_device(CAMERA_NAME_HINT)
        print(f"📷 Using camera device: {device}")
        reader = _start_live_reader(device)
    except Exception as e:
        print(f"⚠️ Could not start the live camera feed: {e}")
        print("   Task 3 needs the REAL robot camera (no sim camera). Check that the")
        print("   daemon runs with --no-media so it isn't holding the device, and see")
        print("   cameras with: v4l2-ctl --list-devices")
        return

    stop_event = threading.Event()
    state.live_reader = reader
    state.live_stop = stop_event

    live_img = widgets.Image(format="jpeg", width=480)
    text = widgets.Text(
        placeholder="Ask about what Reachy sees, then press Enter (or click Send)…",
        layout=widgets.Layout(width="70%"),
        continuous_update=False,  # fire `value` change on Enter/blur, not per keystroke
    )
    send_btn = widgets.Button(description="Send", button_style="primary")
    # Suggestion chips: clicking one fills the bar and submits it.
    chips = [
        widgets.Button(description=s, layout=widgets.Layout(width="auto"))
        for s in VISION_SUGGESTIONS
    ]
    out = widgets.Output()

    # While an Ask is in progress we FREEZE the widget on the exact frame the VLM
    # is answering about: the pump skips updates when this is set, so the still
    # image holds until the answer finishes (then _ask clears it to resume live).
    frozen = threading.Event()

    def _pump() -> None:
        """Push the reader's latest JPEG into the Image widget at ~fps until
        stopped. Runs in a daemon thread; never opens the camera itself.
        Pauses updates while `frozen` is set so an Ask can hold a still."""
        interval = 1.0 / max(1, fps)
        while not stop_event.is_set():
            if not frozen.is_set():
                try:
                    live_img.value = reader.encode_latest_jpeg()
                except Exception:
                    # No frame yet (or reader torn down mid-loop) — just wait.
                    pass
            stop_event.wait(interval)

    pump = threading.Thread(target=_pump, name="reachy-nb-pump", daemon=True)
    state.live_thread = pump
    pump.start()

    def _set_controls(disabled: bool) -> None:
        text.disabled = disabled
        send_btn.disabled = disabled
        for c in chips:
            c.disabled = disabled

    def _ask(question: str) -> None:
        question = (question or "").strip()
        if not question:
            return
        _set_controls(True)
        # Freeze the live feed so the widget holds the exact frame being answered
        # about for the whole Ask (the pump checks `frozen` and stops overwriting).
        frozen.set()
        try:
            with out:
                print(f"🧑 You: {question}")
                # Read the CURRENT latest frame from the SHARED reader — never a
                # second capture_jpeg() (that would double-open the device).
                try:
                    jpeg = reader.encode_latest_jpeg()
                except Exception as e:
                    print(f"⚠️ No camera frame available yet: {e}")
                    return
                # Show exactly the frame the VLM will answer about (held via `frozen`).
                live_img.value = jpeg
                b64 = base64.b64encode(jpeg).decode("ascii")
                print("🤖 Reachy sees: ", end="", flush=True)
                # Wrap the typed question with the LIVE answer style so cell edits
                # apply. Same payload as look_and_describe (temp 0.5 / num_predict 120).
                answer = stream_ollama(
                    "/api/generate",
                    {
                        "model": VLM_MODEL,
                        "prompt": build_vision_prompt(question, style_getter()),
                        "images": [b64],
                        "keep_alive": "30m",
                        "options": {"temperature": 0.5, "num_predict": 120},
                    },
                )
                if answer:
                    _react(get_controller_offline(piper_model), answer)
                print()
        finally:
            # Resume the live feed (pump starts updating the widget again).
            frozen.clear()
            _set_controls(False)

    def _on_ask(_=None) -> None:
        question = (text.value or "").strip()
        if not question:
            return
        text.value = ""  # clear before asking; our own reset is ignored below
        _ask(question)

    # Button click is the primary path; Enter-to-send observes the `value` trait
    # (the ipywidgets 8 replacement for the deprecated on_submit). With
    # continuous_update=False, `value` fires on Enter/blur. We ignore the change
    # fired by our own `text.value = ""` reset so it can't re-trigger.
    send_btn.on_click(_on_ask)

    def _on_value_change(change):
        if (change.get("new") or "").strip():
            _on_ask()

    text.observe(_on_value_change, names="value")

    def _make_chip_handler(question: str):
        def _handler(_=None) -> None:
            _ask(question)

        return _handler

    for chip, suggestion in zip(chips, VISION_SUGGESTIONS):
        chip.on_click(_make_chip_handler(suggestion))

    _display(
        live_img,
        widgets.HBox([text, send_btn]),
        widgets.HBox(chips),
        out,
    )


def reset_defaults() -> Dict[str, str]:
    """Return the default TRY ME knob values (for booth hygiene between attendees).

    A function in _labkit CANNOT rebind the NOTEBOOK module's globals (name binding
    only touches the caller's own namespace), so this helper only RETURNS the canonical
    defaults; the notebook's Reset cell does the actual rebinding via
    `globals().update(reset_defaults())`. Restores ONLY the knobs — it does not rebuild
    controllers, clear chat output, or re-run setup."""
    print(
        "✅ Knobs reset to defaults (variables only). "
        "Re-run the task config cells to see the values.\n"
        "   For a FULL clean slate between attendees, the operator runs "
        "`./reset.sh` then reloads the notebook from disk."
    )
    return {
        "PERSONA_1": DEFAULT_PERSONA_1,
        "VOICE_1": DEFAULT_VOICE_1,
        "PERSONA_2": DEFAULT_PERSONA_2,
        "PIPER_MODEL": DEFAULT_PIPER_MODEL,
        "VISION_STYLE": DEFAULT_VISION_STYLE,
    }


def shutdown() -> None:
    """Cleanly exit the ReachyMini context manager we entered in connect()."""
    # Stop any Task 3 live feed FIRST so releasing the robot also frees the
    # camera (kills ffmpeg): otherwise an orphan reader keeps holding the device
    # and the next run fails with "Device or resource busy".
    stop_live()
    if state.reachy_ctx is None:
        print("ℹ️ No robot connection to close.")
        return
    try:
        state.reachy_ctx.__exit__(None, None, None)
        print("✅ Reachy released. Bye!")
    except Exception as e:
        print(f"⚠️ Error while releasing Reachy: {e}")
    finally:
        state.reachy, state.reachy_ctx = None, None
        state.controller_v1, state.controller_offline = None, None


__all__ = [
    # Constants (attendees/staff may reference or override these).
    "OLLAMA_URL",
    "CHAT_MODEL",
    "VLM_MODEL",
    "TYPEWRITER_DELAY",
    # Canonical TRY ME defaults + the Reset helper (booth hygiene).
    "DEFAULT_PERSONA_1",
    "DEFAULT_VOICE_1",
    "DEFAULT_PERSONA_2",
    "DEFAULT_PIPER_MODEL",
    "DEFAULT_VISION_STYLE",
    "VISION_SUGGESTIONS",
    "reset_defaults",
    # Shared state + lifecycle.
    "state",
    "connect",
    "shutdown",
    "check_ollama",
    # High-level task helpers (keep the notebook cells tiny).
    "get_controller_v1",
    "get_controller_offline",
    "chat_bar",
    "look_and_describe",
    "build_vision_prompt",
    "ask_live",
    "stop_live",
    # Lower-level pieces still referenced or handy in the notebook.
    "stream_ollama",
    "_react",
    "strip_emojis",
    "find_camera_device",
    "capture_jpeg",
    "CAMERA_NAME_HINT",
    "EmotionControllerV6",
    "EmotionControllerV71",
    # Re-exports the notebook uses directly.
    "base64",
    "json",
    "display",
    "Image",
]
