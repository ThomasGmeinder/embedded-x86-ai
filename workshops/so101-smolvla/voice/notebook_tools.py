"""What the voice notebook and its test need beside the Gemma client in this directory.

- ``load_voice_config`` and ``start_server``: the client's settings, and the start of ``llama-server`` (the
  program of llama.cpp that runs Gemma 4 E4B) through the client itself.
- ``open_espeak``, ``speak``, ``clip``: one sentence in a synthetic voice (espeak-ng), inside a quiet clip.
- ``CASES``, ``same``: sentences with the command, object and place each one should give.

The client (``vla_pipeline/``, ``config/``) is a copy of branch ``gemma4-e4b-it_voice_intent`` with this
workshop's changes; see the two commits that added it. The model files, the server binary, the
speech-detection model and the Python environment are not in git: they are those of the set-up voice workshop
``workshops/vvla-pipeline`` beside this one.

Import this module with this directory first on ``sys.path``, so that ``vla_pipeline`` is the copy here.
"""
from __future__ import annotations

import ctypes
import io
import os
import time
import wave

import numpy as np
import requests
from scipy.signal import resample_poly

from vla_pipeline.llm import llama_intent
from vla_pipeline.llm.llama_intent import LlamaIntentParser
from vla_pipeline.utils.config import load_config

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(llama_intent.__file__)))) != HERE:
    raise ImportError(f"vla_pipeline was loaded from {llama_intent.__file__}, not from {HERE}: "
                      f"put {HERE} first on sys.path")

BRANCH_COMMIT = "1d231e1"  # the commit of branch gemma4-e4b-it_voice_intent the client was copied from
SECTION = "gemma4-e4b-it"  # the client's section of config/pipeline.yaml
VOICE_WORKSHOP = os.path.normpath(os.path.join(HERE, "..", "..", "vvla-pipeline"))
# Settings of the section that name files of the voice workshop, relative to it.
WORKSHOP_FILES = ("server_bin", "model_gguf", "mmproj", "vad_model")

ESPEAK_LIBRARY = "libespeak-ng.so.1"
ESPEAK_VOICE = b"en-us"
WORDS_PER_MINUTE = 150
PEAK = 0.5    # loudest sample of every sentence, as a share of full scale
LEAD_S = 0.5  # quiet before the sentence in a clip

PICK = "pick with object and place"
STACK = "learn / rebuild the stack"
EXISTING = "commands the branch has today"
UNKNOWN = "must stay unknown"
# (group, sentence, expected command, expected object, expected place): the sentences of the tool-call test of
# SPEC.md section 4.5. None of them is among the examples in the instructions to the model, except
# "pick up the ball", which is the branch's own example.
CASES = [
    (PICK, "pick up the small red cube and put it in the box", "pick_place", "small red cube", "box"),
    (PICK, "put the red cube in the box", "pick_place", "red cube", "box"),
    (PICK, "put the blue ball in the box", "pick_place", "blue ball", "box"),
    (PICK, "grab the large green cylinder and place it in the box", "pick_place", "large green cylinder", "box"),
    (PICK, "pick up the yellow cylinder", "pick_place", "yellow cylinder", ""),
    (PICK, "take the small blue cube to the bin", "pick_place", "small blue cube", "bin"),
    (PICK, "place the large red ball in the bin", "pick_place", "large red ball", "bin"),
    (STACK, "learn the stack", "learn_stack", "", ""),
    (STACK, "rebuild the stack", "rebuild_stack", "", ""),
    (EXISTING, "stop robot", "stop", "", ""),
    (EXISTING, "pick up the ball", "pick_place", "ball", ""),
    (EXISTING, "go home", "home", "", ""),
    (EXISTING, "wave hello", "wave", "", ""),
    (EXISTING, "fetch the cube", "fetch_block", "", ""),
    (EXISTING, "show me your best dance moves", "dance", "", ""),
    (UNKNOWN, "do not rebuild the stack", "unknown", "", ""),
    (UNKNOWN, "the stack looks nice", "unknown", "", ""),
    (UNKNOWN, "I learned a lot today", "unknown", "", ""),
    (UNKNOWN, "who built this stack", "unknown", "", ""),
    (UNKNOWN, "the red cube is in the box", "unknown", "", ""),
    (UNKNOWN, "is the ball in the box", "unknown", "", ""),
]


def same(a: str, b: str) -> bool:
    """Two argument values, compared after trimming and lower-casing."""
    return a.strip().lower() == b.strip().lower()


def load_voice_config() -> dict:
    """The client's config/pipeline.yaml with the files of the voice workshop pointed there."""
    cfg = load_config()
    section = cfg[SECTION]
    for key in WORKSHOP_FILES:
        section[key] = os.path.join(VOICE_WORKSHOP, section[key])
        if not os.path.isfile(section[key]):
            raise RuntimeError(f"SETUP FAILURE: missing {section[key]}; set up workshops/vvla-pipeline first")
    return cfg


def start_server(log_dir: str, changes: dict, extra_args: list) -> tuple:
    """Start the voice workshop's server through the client: (client, base address, settings used).

    changes: settings of the section that differ from config/pipeline.yaml; extra_args: server arguments added
    to the section's own. Every difference from the file is printed. Refuses to use a server that already
    answers, because its settings are unknown. The caller closes the client, which stops the server."""
    cfg = load_voice_config()
    section = cfg[SECTION]
    cfg.setdefault("system", {})["log_dir"] = os.path.abspath(log_dir)
    changed = {}
    for key, value in changes.items():
        changed[key] = (section.get(key), value)
        section[key] = value
    if extra_args:
        extra = list(section.get("extra_args", []) or []) + list(extra_args)
        changed["extra_args"] = (section.get("extra_args"), extra)
        section["extra_args"] = extra
    print(f"Gemma branch commit {BRANCH_COMMIT} with this workshop's changes, client {os.path.relpath(HERE)}, "
          f"section {SECTION}", flush=True)
    print("settings used:", {k: section.get(k) for k in ("port", "n_gpu_layers", "ctx_size", "parallel", "temperature",
                                                           "no_warmup", "hsa_override_gfx_version", "extra_args")}, flush=True)
    print("differences from config/pipeline.yaml (setting: file value -> used value):", changed or "none", flush=True)
    print("HSA_OVERRIDE_GFX_VERSION in the calling environment (wins over the setting if set):",
          os.environ.get("HSA_OVERRIDE_GFX_VERSION", "<not set>"), flush=True)
    base = f"http://{section['host']}:{section['port']}"
    try:
        requests.get(f"{base}/health", timeout=0.5)
    except requests.RequestException:
        pass
    else:
        raise RuntimeError(f"SETUP FAILURE: something already answers at {base}; its settings are unknown")
    began = time.monotonic()
    parser = LlamaIntentParser(cfg, section=SECTION)  # starts the server and returns when it listens
    print(f"server ready after {time.monotonic() - began:.1f} s, "
          f"log {os.path.relpath(os.path.join(log_dir, 'llama-server.log'))}", flush=True)
    return parser, base, section


def open_espeak() -> tuple:
    """(the espeak-ng library, its sample rate), set to the voice and speaking rate of this module."""
    lib = ctypes.CDLL(ESPEAK_LIBRARY)
    lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.espeak_Initialize.restype = ctypes.c_int
    lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
    lib.espeak_SetVoiceByName.restype = ctypes.c_int
    lib.espeak_SetParameter.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.espeak_SetParameter.restype = ctypes.c_int
    lib.espeak_Synth.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_int, ctypes.c_uint,
                                 ctypes.c_uint, ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
    lib.espeak_Synth.restype = ctypes.c_int
    lib.espeak_Synchronize.restype = ctypes.c_int
    rate = lib.espeak_Initialize(2, 0, None, 0)  # 2 = AUDIO_OUTPUT_SYNCHRONOUS: samples go to the callback
    if rate <= 0:
        raise RuntimeError(f"SETUP FAILURE: espeak_Initialize returned {rate}")
    if lib.espeak_SetVoiceByName(ESPEAK_VOICE) != 0:
        raise RuntimeError(f"SETUP FAILURE: espeak-ng has no voice {ESPEAK_VOICE!r}")
    if lib.espeak_SetParameter(1, WORDS_PER_MINUTE, 0) != 0:  # 1 = espeakRATE
        raise RuntimeError("SETUP FAILURE: espeak-ng did not take the speaking rate")
    return lib, rate


def speak(lib, rate: int, text: str, out_rate: int) -> np.ndarray:
    """The sentence as int16 samples at out_rate, without the silence before and after it."""
    chunks = []
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)

    def collect(samples, count, _events):
        if samples and count > 0:
            chunks.append(np.ctypeslib.as_array(samples, shape=(count,)).copy())
        return 0

    callback = callback_type(collect)
    lib.espeak_SetSynthCallback(callback)
    data = text.encode("utf-8")
    # position 0, counted in characters (1), no end position, text is UTF-8 (flag 1)
    if lib.espeak_Synth(data, len(data) + 1, 0, 1, 0, 1, None, None) != 0:
        raise RuntimeError(f"SETUP FAILURE: espeak-ng could not speak {text!r}")
    lib.espeak_Synchronize()
    if not chunks:
        raise RuntimeError(f"SETUP FAILURE: espeak-ng gave no sound for {text!r}")
    sound = np.concatenate(chunks).astype(np.float64)
    loud = np.flatnonzero(np.abs(sound) > 0.01 * np.abs(sound).max())
    sound = sound[loud[0]:loud[-1] + 1]
    sound = resample_poly(sound, out_rate, rate)
    sound *= PEAK * 32767 / np.abs(sound).max()
    return np.round(sound).astype(np.int16)


def clip(sound: np.ndarray, rate: int, window_s: float) -> bytes:
    """The sentence, LEAD_S into a quiet clip of window_s seconds, as the bytes of a WAV file."""
    total, start = int(round(window_s * rate)), int(round(LEAD_S * rate))
    if start + len(sound) > total:
        raise RuntimeError(f"SETUP FAILURE: a sentence of {len(sound) / rate:.1f} s does not fit a {window_s} s clip")
    samples = np.zeros(total, dtype=np.int16)
    samples[start:start + len(sound)] = sound
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(samples.tobytes())
    return buffer.getvalue()
