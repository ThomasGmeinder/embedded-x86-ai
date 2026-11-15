# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Gemma 4 E4B intent parsing via llama.cpp running on ROCm.

The model runs in ``llama-server`` (built with ``-DGGML_HIP=ON`` by
``bootstrap.sh``) and is queried over its OpenAI-compatible HTTP API. A GBNF
grammar (``config/intent.gbnf``) constrains generation so the model can only
emit ``{"command": "<name>", "object": "<words>", "heard": "<words>"}`` - no
free-form parsing. The ``gemma4-e4b-it`` config section picks the model; when it
has an ``mmproj`` (the audio projector), :func:`parse_audio_clip` sends a
microphone clip and Gemma hears it directly.

:func:`llama_server_command` is the launch recipe and :func:`parse_intent`
one grammar-constrained request (the workshop's TODOs 1.5 and 1.6).
:class:`LlamaIntentParser` owns the lifecycle: it connects to a running
server or starts one, and is the only place that starts llama-server
(``--serve`` runs it for the notebooks and scripts). It keeps a persistent
HTTP session (keep-alive) to shave connection setup off the voice-command
latency budget.

Independent test
----------------
    python -m vla_pipeline.llm.llama_intent --text "hey can you pick up the ball"
    python -m vla_pipeline.llm.llama_intent --selftest
    python -m vla_pipeline.llm.llama_intent --serve   # keep a server up until Ctrl-C
    python -m vla_pipeline.llm.llama_intent           # interactive REPL
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import signal
import subprocess
import threading
import time

import requests

from vla_pipeline.llm.intents import INTENT_SYSTEM_PROMPT, Intent, ParsedCommand
from vla_pipeline.utils.config import load_config, resolve

logger = logging.getLogger(__name__)

# What llama-server logs once its model is loaded and it serves requests.
READY_LINE = "listening on http://"

# The config section that names the model, its port, and its grammar.
DEFAULT_SECTION = "gemma4-e4b-it"


def server_alive(base_url: str, http=None, timeout: float = 0.5) -> bool:
    """``GET /health`` answers HTTP 200."""
    http = http or requests
    try:
        return http.get(f"{base_url}/health", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def llama_server_command(cfg: dict, section: str = DEFAULT_SECTION):
    """Build the ``llama-server`` launch: ``(cmd list, env dict)``.

    Reads the ``section`` block of the config (default ``gemma4-e4b-it``). The
    command carries the model, host, port, ``--n-gpu-layers``, ``--ctx-size``,
    ``--parallel``, ``--mmproj`` (the audio projector) when configured, and
    ``--no-warmup`` when configured; the environment carries
    ``HSA_OVERRIDE_GFX_VERSION`` when the config sets one, without
    overriding a value the user already exported.
    """
    l = cfg[section]
    server_bin = resolve(l["server_bin"])
    model = resolve(l["model_gguf"])
    if not server_bin.exists():
        raise FileNotFoundError(f"{server_bin} not found - run ./bootstrap.sh (llama.cpp build step).")
    if not model.exists():
        raise FileNotFoundError(f"{model} not found - run ./bootstrap.sh (model download step).")
    cmd = [
        str(server_bin),
        "--model", str(model),
        "--host", l["host"],
        "--port", str(l["port"]),
        "--n-gpu-layers", str(l["n_gpu_layers"]),
        "--ctx-size", str(l["ctx_size"]),
        "--parallel", str(l.get("parallel", 1)),
    ]
    if l.get("mmproj"):
        mmproj = resolve(l["mmproj"])
        if not mmproj.exists():
            raise FileNotFoundError(
                f"{mmproj} not found - the Gemma audio projector has not finished downloading."
            )
        cmd += ["--mmproj", str(mmproj)]
    if l.get("no_warmup", False):
        cmd.append("--no-warmup")
    # Smaller compute ubatches keep each GPU submission short, so a 3-D
    # render pass sharing the iGPU waits less behind prompt processing.
    if l.get("ubatch_size"):
        cmd += ["--ubatch-size", str(int(l["ubatch_size"]))]
    cmd += list(l.get("extra_args", []) or [])

    # Deterministic GPU init on Strix: the ROCm runtime can segfault on the
    # first kernel dispatch unless the iGPU is presented as its nearest
    # supported relative.
    env = os.environ.copy()
    gfx = l.get("hsa_override_gfx_version")
    if gfx:
        env.setdefault("HSA_OVERRIDE_GFX_VERSION", str(gfx))
    # Optional compute-unit fence (e.g. "0:0-31" leaves 8 of 40 CUs to the
    # graphics queue). Off by default: it has to be measured against the twin's
    # renderer before it is turned on.
    if l.get("cu_mask"):
        env["HSA_CU_MASK"] = str(l["cu_mask"])
    return cmd, env


def _chat_command(
    user_content, base_url: str, grammar: str, http=None, temperature: float = 0.0,
    *, audio: bool = False, transcript: str = "",
) -> ParsedCommand:
    """One grammar-constrained chat completion, read into a :class:`ParsedCommand`.

    ``user_content`` is the text or, for audio, the content list with the WAV
    clip. ``heard`` (the words Gemma reports) is the transcript when Gemma gives
    none.
    """
    http = http or requests
    payload = {
        "messages": [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": 96,  # command JSON plus the short "heard" phrase
        "grammar": grammar,
    }
    if audio:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    r = http.post(f"{base_url}/v1/chat/completions", json=payload, timeout=30)
    if r.status_code >= 400:
        logger.error("llama-server rejected the request: %s", r.text[:500])
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    try:
        data = json.loads(content)
        command = data["command"]
    except (json.JSONDecodeError, KeyError):
        logger.warning("Unparseable LLM output: %r", content)
        return ParsedCommand(Intent.UNKNOWN, transcript=transcript)
    params: dict[str, str] = {}
    obj = str(data.get("object", "") or "").strip()
    if obj:
        params["object"] = obj
    heard = str(data.get("heard", "") or "").strip() or transcript
    return ParsedCommand(Intent.from_string(command), params=params, transcript=heard)


def parse_intent(
    transcript: str, base_url: str, grammar: str, http=None, temperature: float = 0.0, *, audio: bool = False,
) -> ParsedCommand:
    """One transcript → one :class:`ParsedCommand`, grammar-constrained.

    POSTs an OpenAI-style chat completion to ``{base_url}/v1/chat/completions``
    with the system prompt (:data:`INTENT_SYSTEM_PROMPT`), the transcript as
    the user message, ``temperature`` (0 = deterministic routing), a small
    ``max_tokens``, and the GBNF ``grammar``, which is what makes the output
    JSON guaranteed parseable. An empty transcript maps to UNKNOWN without
    any HTTP round trip.
    """
    if not transcript or not transcript.strip():
        return ParsedCommand(Intent.UNKNOWN, transcript=transcript or "")
    return _chat_command(
        transcript.strip(), base_url, grammar, http, temperature, audio=audio, transcript=transcript,
    )


def parse_audio_clip(wav: bytes, base_url: str, grammar: str, http=None, temperature: float = 0.0) -> ParsedCommand:
    """One WAV clip → one :class:`ParsedCommand`: Gemma hears the audio itself."""
    encoded = base64.b64encode(wav).decode("ascii")
    content = [
        {"type": "text", "text": "Classify the spoken robot instruction in this audio."},
        {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}},
    ]
    return _chat_command(content, base_url, grammar, http, temperature, audio=True)


class LlamaIntentParser:
    """Connect to (or start) llama-server and turn transcripts into :class:`ParsedCommand`.

    It owns the server process only if it started one.
    """

    def __init__(self, cfg: dict, autostart: bool | None = None, *, section: str = DEFAULT_SECTION):
        l = cfg[section]
        self._section = section
        self._audio = bool(l.get("mmproj"))
        self.base = f"http://{l['host']}:{int(l['port'])}"
        self.temperature: float = float(l.get("temperature", 0.0))
        self.grammar = resolve(l["grammar_file"]).read_text(encoding="utf-8")
        self._cfg = cfg
        self._log_dir = cfg.get("system", {}).get("log_dir", "logs")
        self._proc: subprocess.Popen | None = None
        self._http = requests.Session()  # keep-alive: saves ~5-20 ms per command

        if not server_alive(self.base, self._http):
            if l.get("autostart", True) if autostart is None else autostart:
                self._spawn_server()
            else:
                raise ConnectionError(
                    f"llama-server not reachable at {self.base} and autostart is disabled."
                )

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _spawn_server(self, ready_timeout_s: float = 120.0) -> None:
        """Launch llama-server as a subprocess and block until it logs that it
        is listening (:data:`READY_LINE`), or exits."""
        cmd, env = llama_server_command(self._cfg, self._section)
        logger.info("Starting llama-server (ROCm): %s", " ".join(cmd))
        log_dir = resolve(self._log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._server_log_path = log_dir / "llama-server.log"
        self._server_log = open(self._server_log_path, "w")
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, errors="replace",
        )
        ready = threading.Event()

        def relay() -> None:
            """The server's output to its log; its readiness line, or its exit, sets ``ready``."""
            for line in self._proc.stdout:
                self._server_log.write(line)
                self._server_log.flush()
                if READY_LINE in line:
                    ready.set()
            ready.set()

        threading.Thread(target=relay, daemon=True, name="llama-server-log").start()
        try:
            if not ready.wait(ready_timeout_s):
                raise TimeoutError(
                    f"llama-server did not log {READY_LINE!r} within {ready_timeout_s:.0f} s "
                    f"(log: {self._server_log_path})"
                )
            if self._proc.poll() is not None:
                tail = self._read_server_log_tail()
                raise RuntimeError(
                    f"llama-server exited with code {self._proc.wait()} during "
                    f"startup. Last log lines:\n{tail}\n"
                    f"(full log: {self._server_log_path})"
                )
            if not server_alive(self.base, self._http):
                raise RuntimeError(f"llama-server logged readiness but {self.base}/health is not ready")
        except BaseException:
            self.close()
            raise
        logger.info("llama-server ready at %s", self.base)

    def _read_server_log_tail(self, n: int = 20) -> str:
        """Return the last n lines of the server log, for error messages."""
        try:
            with open(self._server_log_path) as f:
                return "".join(f.readlines()[-n:])
        except OSError:
            return "(no server log available)"

    def close(self) -> None:
        """Terminate the server if this object spawned it."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        server_log = getattr(self, "_server_log", None)
        if server_log is not None and not server_log.closed:
            server_log.close()

    # ------------------------------------------------------------------
    # Intent parsing
    # ------------------------------------------------------------------

    def parse(self, transcript: str) -> ParsedCommand:
        """Map one spoken transcript to a ParsedCommand (grammar-constrained), logging the latency."""
        t0 = time.perf_counter()
        cmd = parse_intent(
            transcript, self.base, self.grammar, http=self._http, temperature=self.temperature, audio=self._audio,
        )
        logger.info(
            "Intent %r %s ← %r (%.0f ms)",
            cmd.intent.value,
            cmd.params or "",
            cmd.transcript or transcript,
            (time.perf_counter() - t0) * 1e3,
        )
        return cmd

    def parse_audio(self, wav: bytes) -> ParsedCommand:
        """Map one WAV clip to a ParsedCommand: Gemma hears the audio (needs the ``mmproj``)."""
        t0 = time.perf_counter()
        cmd = parse_audio_clip(wav, self.base, self.grammar, http=self._http, temperature=self.temperature)
        logger.info(
            "Intent %r %s ← %r (%.0f ms)",
            cmd.intent.value,
            cmd.params or "",
            cmd.transcript,
            (time.perf_counter() - t0) * 1e3,
        )
        return cmd


# =============================================================================
# Standalone component test
# =============================================================================

_SELFTEST_CASES = [
    ("can you copy my arm movements", Intent.GESTURE_MIMIC, None),
    ("mirror my hand please", Intent.GESTURE_MIMIC, None),
    ("go grab the block and bring it to me", Intent.FETCH_BLOCK, None),
    ("fetch the cube", Intent.FETCH_BLOCK, None),
    ("pick up the ball", Intent.PICK_PLACE, "ball"),
    ("grab the bottle and hand it to me", Intent.PICK_PLACE, "bottle"),
    ("show me your best dance moves", Intent.DANCE, None),
    ("hold this pencil for me", Intent.GRIP, None),
    ("release", Intent.RELEASE, None),
    ("open your hand", Intent.RELEASE, None),
    ("drop it", Intent.RELEASE, None),
    ("place it", Intent.PLACE, None),
    ("set it down on the table", Intent.PLACE, None),
    ("wave hello", Intent.WAVE, None),
    ("go home", Intent.HOME, None),
    ("stop robot", Intent.STOP, None),
    ("what's the weather like today", Intent.UNKNOWN, None),
]


def main() -> None:
    """CLI entry point: parse one utterance, run the selftest suite, keep a server up, or start a REPL."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Gemma 4 intent parser (llama.cpp + ROCm) - component test"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--text", default=None, help="single utterance to parse")
    parser.add_argument(
        "--selftest", action="store_true", help="run the built-in utterance suite"
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="start llama-server (or use the running one) and keep it until SIGINT/SIGTERM",
    )
    args = parser.parse_args()

    llm = LlamaIntentParser(load_config(args.config))
    try:
        if args.serve:
            # blocked only now: llama-server, started above, keeps them unblocked
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            print(f"llama-server ready at {llm.base}", flush=True)
            signal.sigwait({signal.SIGINT, signal.SIGTERM})
            return
        if args.selftest:
            passed = 0
            for text, expected, obj in _SELFTEST_CASES:
                got = llm.parse(text)
                ok = got.intent == expected and (obj is None or got.object_name == obj)
                passed += ok
                print(
                    f"  [{'PASS' if ok else 'FAIL'}] {text!r} -> {got.intent.value} "
                    f"{got.params} (expected {expected.value} {obj or ''})"
                )
            print(f"\n{passed}/{len(_SELFTEST_CASES)} cases passed")
            raise SystemExit(0 if passed == len(_SELFTEST_CASES) else 1)
        if args.text:
            cmd = llm.parse(args.text)
            print(cmd.intent.value, cmd.params)
            return
        print("Interactive intent REPL - Ctrl-D to exit")
        while True:
            try:
                line = input("you> ")
            except EOFError:
                break
            cmd = llm.parse(line)
            print(f"  -> {cmd.intent.value} {cmd.params}")
    finally:
        llm.close()


if __name__ == "__main__":
    main()
