# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Component 1 self-test (PROVIDED) - run: ``python -m vvla_workshop.selftest_models``

Checks YOUR TODOs 1.1-1.6 in ``workshop/project/models/``; ``--reference``
runs the same checks on the pipeline's finished functions instead. Order of
business, mirroring how you should iterate:

1. **Offline checks** (no NPU, no GPU, no models, no mic): your session
   factory against a tiny embedded ONNX model, your CPU-fallback path, your
   VitisAI wiring against a *recording* mock of onnxruntime, your MediaPipe
   init, your llama-server launch recipe, and your intent parsing against a
   mock server that captures exactly what your client sent.
2. **On-hardware check** (auto-detected): the real YOLO-pose model through
   your session factory, with the actual execution provider printed - the
   "did I *really* get the NPU?" receipt - and an overlay artifact.

Every check reports PASS / FAIL / TODO / SKIP with a pointer, so a fresh
project prints a clean TODO list instead of a stack trace.
"""

from __future__ import annotations

import shutil
from copy import deepcopy
from unittest import mock

import numpy as np

from vla_pipeline.utils.config import load_config, resolve
from vvla_workshop.feedback import ARTIFACTS, Reporter, implementation, reference_flag
from vvla_workshop.fixtures import MockLlamaServer, SyntheticCamera, tiny_onnx_path


class _RecordingSession:
    """Stands in for ort.InferenceSession; remembers how it was built."""

    calls: list = []

    def __init__(self, path, providers=None, provider_options=None, **kw):
        _RecordingSession.calls.append(
            {"path": str(path), "providers": providers or [], "provider_options": provider_options or []})
        self._providers = providers or ["CPUExecutionProvider"]

    def get_providers(self):
        """Return the providers this recording session was built with."""
        return list(self._providers)

    def get_inputs(self):
        """Return one stub input descriptor (mimics ort's ``.name`` attribute)."""
        class _I:
            """Stub input descriptor exposing just the ``.name`` attribute ort reads."""
            name = "x"
        return [_I()]

    def run(self, *_a, **_k):
        """Return a zeroed ``(1, 4)`` array, ignoring whatever inputs were passed."""
        return [np.zeros((1, 4), dtype=np.float32)]


def _offline_checks(rep: Reporter, cfg: dict, reference: bool) -> None:
    """Run every offline (no NPU/GPU/mic) check against ``rep``."""
    import onnxruntime as ort

    npu = implementation("1.1", reference)
    hands = implementation("1.4", reference)
    llm = implementation("1.5", reference)

    print(f"  onnxruntime providers on this machine: {ort.get_available_providers()}")

    # --- TODO 1.1: npu_available -------------------------------------------
    def check_available():
        """Verify npu_available() returns a bool that matches this onnxruntime build."""
        got = npu.npu_available()
        assert isinstance(got, bool), f"npu_available() returned {type(got).__name__}, expected bool"
        real = "VitisAIExecutionProvider" in ort.get_available_providers()
        assert got == real, (
            f"npu_available() returned {got} but this onnxruntime "
            f"{'DOES' if real else 'does NOT'} expose VitisAIExecutionProvider")
    rep.run("1.1", "npu_available() matches this onnxruntime build", check_available)

    tiny = tiny_onnx_path(ARTIFACTS)

    # --- TODO 1.2: CPU path --------------------------------------------------
    def check_cpu_session():
        """Verify device='cpu' builds a working CPU-only session."""
        sess = npu.build_npu_session(tiny, "cpu", cache_key="selftest_tiny")
        x = np.arange(4, dtype=np.float32).reshape(1, 4)
        (y,) = sess.run(None, {"x": x})
        assert np.allclose(y, 2 * x), (
            f"tiny model says y != 2x (got {y}) - the session ran, but on a wrong/odd configuration")
        provs = sess.get_providers()
        assert provs[0] == "CPUExecutionProvider", (
            f"device='cpu' must build a CPU-only session; got providers={provs}")
        return sess
    ok_cpu, _ = rep.run("1.2", "device='cpu' builds a working CPU session", check_cpu_session,
                        hint="build the session with providers=['CPUExecutionProvider']")

    # --- TODO 1.2: graceful CPU fallback ------------------------------------
    def check_fallback():
        """Verify device='npu' without the Ryzen AI stack gets a CPU session."""
        with mock.patch.object(ort, "get_available_providers", return_value=["CPUExecutionProvider"]), \
             mock.patch.object(npu, "npu_available", return_value=False):
            sess = npu.build_npu_session(tiny, "npu", cache_key="selftest_tiny")
        provs = sess.get_providers()
        assert provs[0] == "CPUExecutionProvider", (
            "with no VitisAI EP available, device='npu' must FALL BACK to a "
            f"CPU session (got providers={provs}) - never crash on a dev box")
    rep.run("1.2", "device='npu' without the Ryzen AI stack gets a CPU session", check_fallback,
            hint="check npu_available() first; warn and drop to the CPU path")

    # --- TODO 1.2: VitisAI wiring (recorded, no NPU needed) ------------------
    def check_npu_wiring():
        """Verify device='npu' wires the VitisAI EP with the right provider_options."""
        _RecordingSession.calls.clear()
        with mock.patch.object(ort, "InferenceSession", _RecordingSession), \
             mock.patch.object(npu, "npu_available", return_value=True):
            npu.build_npu_session(tiny, "npu", vitisai_config="config/vaiep_config.json",
                                  cache_dir=ARTIFACTS / "cache", cache_key="selftest_tiny")
        assert _RecordingSession.calls, "InferenceSession was never constructed"
        call = _RecordingSession.calls[-1]
        assert call["providers"] == ["VitisAIExecutionProvider"], (
            f"providers={call['providers']} - the NPU path must request exactly ['VitisAIExecutionProvider']")
        opts = call["provider_options"]
        assert opts and isinstance(opts, list), (
            "provider_options is empty - the VitisAI EP is configured through provider_options=[{...}]")
        o = opts[0]
        for key in ("config_file", "cache_dir", "cache_key"):
            assert key in o, (
                f"provider_options missing '{key}' - without it the compiler config / compile cache "
                "is not wired and every load recompiles (or collides with another model)")
        assert o.get("target") == "VAIML", (
            f"provider_options 'target' is {o.get('target')!r}, expected 'VAIML' (matches config/vaiep_config.json)")
        assert o["cache_key"] == "selftest_tiny", (
            f"cache_key {o['cache_key']!r} ignored the caller's argument - each model must get its own key")
    rep.run("1.2", "device='npu' wires the VitisAI EP correctly (mocked)", check_npu_wiring)

    # --- TODO 1.3: active_provider -------------------------------------------
    def check_active_provider():
        """Verify active_provider() reports the session's real execution provider."""
        sess = npu.build_npu_session(tiny, "cpu", cache_key="selftest_tiny")
        got = npu.active_provider(sess)
        assert got == "CPUExecutionProvider", (
            f"active_provider() returned {got!r} for a CPU session - it must report the FIRST entry "
            "of session.get_providers()")
    if ok_cpu:
        rep.run("1.3", "active_provider() reports the session's real EP", check_active_provider)
    else:
        rep.add("1.3", "active_provider() reports the session's real EP", "SKIP", "needs TODO 1.2 first")

    # --- TODO 1.4: MediaPipe on CPU ------------------------------------------
    try:
        import mediapipe  # noqa: F401
        have_mp = True
    except ImportError:
        have_mp = False
    if have_mp:
        def check_hands():
            """Verify your tracker drives the pipeline's HandTracker on a frame."""
            from vla_pipeline.vision.mediapipe_hands import HandTracker

            tracker = HandTracker(cfg, hands=hands.create_hand_tracker(cfg))
            ok, frame = SyntheticCamera(320, 240).read()
            obs, _ = tracker.process(frame)  # no hand in the pattern → None
            tracker.close()
            assert obs is None or hasattr(obs, "pinch"), "unexpected observation"
        rep.run("1.4", "MediaPipe Hands initializes on CPU and processes a frame", check_hands,
                hint="mp.solutions.hands.Hands(...) with the mediapipe: config values")
    else:
        rep.add("1.4", "MediaPipe Hands initializes on CPU", "SKIP", "mediapipe not installed - pip install mediapipe")

    # --- TODO 1.5: llama-server launch recipe --------------------------------
    scratch = ARTIFACTS / "fake_llama"
    scratch.mkdir(parents=True, exist_ok=True)
    fake_bin = scratch / "llama-server"
    fake_gguf = scratch / "model.gguf"
    fake_bin.write_text("#!/bin/sh\n")
    fake_gguf.write_text("gguf")
    cfg_llm = deepcopy(cfg)
    cfg_llm["llm"]["server_bin"] = str(fake_bin)
    cfg_llm["llm"]["model_gguf"] = str(fake_gguf)

    def check_server_cmd():
        """Verify the llama-server command and env carry the iGPU survival flags."""
        cmd, env = llm.llama_server_command(cfg_llm)
        text = " ".join(cmd)
        assert cmd[0] == str(fake_bin), f"cmd[0]={cmd[0]!r}, expected the server binary"
        assert "--model" in cmd, "missing --model"
        for flag in ("--host", "--port", "--n-gpu-layers", "--ctx-size"):
            assert flag in cmd, f"missing {flag} - see the llm: config section"
        ctx = int(cmd[cmd.index("--ctx-size") + 1])
        assert ctx <= 2048, (
            f"--ctx-size {ctx} > 2048 - the native 131072 KV cache OOMs the iGPU VRAM carve-out on Strix; keep it capped")
        assert "--no-warmup" in text, (
            "missing --no-warmup - config sets no_warmup: true because the warmup pass can segfault on gfx1151")
        want = str(cfg["llm"]["hsa_override_gfx_version"])
        assert env.get("HSA_OVERRIDE_GFX_VERSION") == want, (
            f"HSA_OVERRIDE_GFX_VERSION={env.get('HSA_OVERRIDE_GFX_VERSION')!r}, expected {want!r} - the ROCm "
            "runtime needs the iGPU presented as a supported gfx target (config: hsa_override_gfx_version)")
    with mock.patch.dict("os.environ", {}, clear=False):
        import os
        os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
        rep.run("1.5", "llama-server command + env carry the iGPU survival flags", check_server_cmd)

    # --- TODO 1.6: grammar-constrained intent parsing -------------------------
    grammar = resolve(cfg["llm"]["grammar_file"]).read_text(encoding="utf-8")

    def check_parse():
        """Verify parse_intent() sends grammar-constrained chat completions and routes correctly."""
        from vla_pipeline.llm.intents import Intent

        with MockLlamaServer() as mocksrv:
            cmd = llm.parse_intent("", mocksrv.base_url, grammar)
            assert cmd.intent == Intent.UNKNOWN, "empty transcript must be UNKNOWN"
            assert mocksrv.request_count == 0, "empty transcript still hit the server - short-circuit it"
            cases = [
                ("pick up the ball", Intent.PICK_PLACE, "ball"),
                ("mirror my hand please", Intent.GESTURE_MIMIC, None),
                ("show me your best dance moves", Intent.DANCE, None),
            ]
            for text, want, obj in cases:
                got = llm.parse_intent(text, mocksrv.base_url, grammar)
                assert got.intent == want, f"{text!r} → {got.intent.value}, expected {want.value}"
                if obj:
                    assert got.object_name == obj, (
                        f"{text!r} lost its object: got {got.object_name!r}, expected {obj!r} - read data.get('object')")
            p = mocksrv.last_payload
            assert p.get("grammar"), (
                "request payload had no 'grammar' - the GBNF constraint is what makes the output "
                "guaranteed-parseable; attach it")
            assert float(p.get("temperature", 1.0)) == 0.0, "temperature must be 0 - intent routing has to be deterministic"
            roles = [m.get("role") for m in p.get("messages", [])]
            assert "system" in roles and "user" in roles, (
                f"messages roles={roles} - send the system prompt AND the transcript as the user message")
    rep.run("1.6", "parse_intent() speaks grammar-constrained chat completions", check_parse)


def _hardware_checks(rep: Reporter, cfg: dict, reference: bool, device=None) -> None:
    """The real YOLO-pose model through your session factory - auto-skips off the rig."""
    npu = implementation("1.2", reference)
    y = cfg["yolo_pose"]
    pose_onnx = resolve(y["onnx"])
    if not pose_onnx.exists():
        rep.add("-", "on-hardware: YOLO models present", "SKIP",
                f"{pose_onnx} not found - export/download the models first")
        return
    try:
        if not npu.npu_available():
            rep.add("-", "on-hardware: VitisAI EP present", "SKIP",
                    "this onnxruntime has no VitisAI EP - CPU-only machine; running the model on CPU instead "
                    "(the NPU run happens on the rig)")
            device = "cpu"
    except NotImplementedError:
        rep.add("-", "on-hardware: YOLO-pose session", "SKIP", "needs TODO 1.1 first")
        return

    def check_pose():
        """Verify the YOLO-pose session builds and infers on real hardware."""
        import cv2

        from vla_pipeline.vision.yolo_pose_npu import decode_pose, draw_skeleton, letterbox

        requested = device or y["device"]
        session = npu.build_npu_session(
            y["onnx"], requested, vitisai_config=y.get("vitisai_config", "config/vaiep_config.json"),
            cache_dir=cfg["system"]["cache_dir"], cache_key=y["cache_key"])
        print(f"          requested: {requested}   actual provider: {npu.active_provider(session)}")
        cap = cv2.VideoCapture(0)
        ok, frame = (cap.read() if cap.isOpened() else SyntheticCamera().read())
        if cap.isOpened():
            cap.release()
        nchw, info = letterbox(frame, int(y["imgsz"]))
        outputs = session.run(None, {session.get_inputs()[0].name: nchw})
        boxes, scores, kpts = decode_pose(outputs, info, frame.shape, float(y["conf_threshold"]))
        out = draw_skeleton(frame, boxes, scores, kpts, float(y["kpt_threshold"]))
        path = ARTIFACTS / "pose_overlay.png"
        cv2.imwrite(str(path), out)
        rep.artifact(path)
        assert out.shape == frame.shape
    rep.run("1.2", "on-hardware: YOLO-pose session builds and infers", check_pose,
            hint="if this fails with 'HW context creation unsuccessful', the NPU is out of contexts - move a "
                 "model to CPU (see the NPU-context-budget section of the notebook)")


def run(reference: bool = False) -> int:
    """Run offline + hardware checks on your functions (or the reference); return the exit code."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    rep = Reporter("Component 1 - AI models on the Ryzen AI stack" + (" (reference)" if reference else ""))
    _offline_checks(rep, cfg, reference)
    _hardware_checks(rep, cfg, reference)
    code = rep.summary()
    shutil.rmtree(ARTIFACTS / "fake_llama", ignore_errors=True)
    return code


if __name__ == "__main__":
    raise SystemExit(run(reference_flag("Component 1 self-test")))
