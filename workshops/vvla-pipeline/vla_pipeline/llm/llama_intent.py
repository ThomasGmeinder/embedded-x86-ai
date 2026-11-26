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
emit the list ``["<name>", "<object>", "<words heard>"]`` - no free-form
parsing. The ``gemma4-e4b-it`` config section picks the model; when it has an
``mmproj`` (the audio projector), :func:`parse_audio_clip` sends a microphone
clip and Gemma hears it directly.

The answer is read as the server writes it (a stream of server-sent events).
It is written in the order command, object, heard, so the call is complete
before the words heard are: ``on_call`` receives the call at that moment, with
the seconds since the request began, while the rest is still being written.
Nothing waits for the whole answer to act on the call. A typed sentence is its
own words, so its request ends at the call.

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
from collections.abc import Callable, Iterator

import requests

from vla_pipeline.llm.intents import (
    INTENT_SYSTEM_PROMPT,
    Intent,
    ParsedCommand,
    call_from_partial,
    heard_from_partial,
)
from vla_pipeline.utils.config import load_config, resolve

logger = logging.getLogger(__name__)

# What llama-server logs once its model is loaded and it serves requests.
READY_LINE = "listening on http://"
# The longest a request may go without the server sending anything, in seconds.
REQUEST_TIMEOUT_S = 30
# The call, and the seconds from the request to it. Returning False says the words are not wanted.
OnCall = Callable[[ParsedCommand, float], bool | None]

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


def answer_pieces(lines) -> Iterator[str]:
    """The answer's text piece by piece, as the server's events carry it.

    ``lines`` are the lines of an OpenAI-style event stream; it ends at ``[DONE]``.
    """
    for line in lines:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            return
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(f"llama-server reported: {str(event['error'])[:300]}")
        choices = event.get("choices") or [{}]
        yield (choices[0].get("delta") or {}).get("content") or ""


def _chat_command(
    user_content, base_url: str, grammar: str, http=None, temperature: float = 0.0,
    *, audio: bool = False, transcript: str = "", on_call: OnCall | None = None, words: bool = False,
) -> tuple[ParsedCommand, float, float]:
    """One grammar-constrained chat completion, its answer read as it is written.

    ``user_content`` is the text or, for audio, the content list with the WAV
    clip. ``on_call`` gets the call the moment it is complete. With ``words``
    false, or when ``on_call`` returns False, the reading ends there and the
    connection is closed: for a caller that has the sentence already, or that
    drops the call. The server is then free for the next request sooner. An
    answer that ends before its call is complete is ``unknown``. ``heard``
    (the words Gemma reports) is the transcript when Gemma gives none.

    Returns the command, the seconds until its call was complete (0.0 when it
    never was), and the seconds until reading the answer ended. They are
    returned, not kept: a typed sentence and a spoken one are asked from two
    threads.
    """
    http = http or requests
    payload = {
        "messages": [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": 96,  # the command list plus the short "heard" phrase
        "grammar": grammar,
        "stream": True,
    }
    if audio:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    text, call, call_s = "", None, 0.0
    t0 = time.perf_counter()
    with http.post(
        f"{base_url}/v1/chat/completions", json=payload, stream=True, timeout=REQUEST_TIMEOUT_S,
    ) as r:
        if r.status_code >= 400:
            logger.error("llama-server rejected the request: %s", r.text[:500])
        r.raise_for_status()
        # chunk_size=1: a line is handed on when its last byte arrives. The library's
        # default reads in blocks of 512 bytes and would hold the end of the call back
        # until the server has written that much more.
        for piece in answer_pieces(r.iter_lines(chunk_size=1)):
            text += piece
            if call is None:
                call = call_from_partial(text)
                if call is not None:
                    call_s = time.perf_counter() - t0
                    # on_call answers False when it has no use for the words (a call it drops).
                    wanted = on_call(call, call_s) if on_call is not None else None
                    if not words or wanted is False:
                        break  # leaving the block closes the connection; the rest is not read
    total_s = time.perf_counter() - t0
    if call is None:
        logger.warning("Unparseable LLM output: %r", text)
        return ParsedCommand(Intent.UNKNOWN, transcript=transcript), 0.0, total_s
    heard = heard_from_partial(text) or transcript
    return ParsedCommand(call.intent, params=call.params, transcript=heard), call_s, total_s


def parse_intent(
    transcript: str, base_url: str, grammar: str, http=None, temperature: float = 0.0, *, audio: bool = False,
) -> ParsedCommand:
    """One transcript → one :class:`ParsedCommand`, grammar-constrained.

    POSTs an OpenAI-style chat completion to ``{base_url}/v1/chat/completions``
    with the system prompt (:data:`INTENT_SYSTEM_PROMPT`), the transcript as
    the user message, ``temperature`` (0 = deterministic routing), a small
    ``max_tokens``, ``stream`` on, and the GBNF ``grammar``, which is what
    makes the answer guaranteed parseable. The answer is read as it is written
    (:func:`answer_pieces`) and the request ends when the call is complete
    (:func:`~vla_pipeline.llm.intents.call_from_partial`): a typed sentence is
    its own words. An empty transcript maps to UNKNOWN without any HTTP round
    trip.
    """
    if not transcript or not transcript.strip():
        return ParsedCommand(Intent.UNKNOWN, transcript=transcript or "")
    return _chat_command(
        transcript.strip(), base_url, grammar, http, temperature, audio=audio, transcript=transcript,
    )[0]


def _audio_content(wav: bytes) -> list:
    """The user message of an audio request: the instruction and the WAV clip."""
    encoded = base64.b64encode(wav).decode("ascii")
    return [
        {"type": "text", "text": "Classify the spoken robot instruction in this audio."},
        {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}},
    ]


def parse_audio_clip(
    wav: bytes, base_url: str, grammar: str, http=None, temperature: float = 0.0, on_call: OnCall | None = None,
) -> ParsedCommand:
    """One WAV clip → one :class:`ParsedCommand`: Gemma hears the audio itself.

    ``on_call`` receives the call while Gemma still writes the words it heard;
    the returned command holds those words.
    """
    return _chat_command(
        _audio_content(wav), base_url, grammar, http, temperature, audio=True, on_call=on_call, words=True,
    )[0]


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
        # The process that starts the server may be pinned to one core (``cores`` in the pipeline
        # config pins perception), and a child inherits that. The server is not pinned: its
        # threads feed the iGPU and decode the audio, and on one shared core they wait behind the
        # cameras and the vision models. ``taskset`` sets the allowed cores before the server
        # starts its first thread.
        cmd = ["taskset", "-c", f"0-{os.cpu_count() - 1}", *cmd]
        logger.info("Starting llama-server (ROCm): %s", " ".join(cmd))
        log_dir = resolve(self._log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._server_log_path = log_dir / "llama-server.log"
        self._server_log = open(self._server_log_path, "w")
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, errors="replace",
        )
        ready = threading.Event()  # the readiness line was logged, or the output ended
        listening = threading.Event()  # the readiness line was logged

        def relay() -> None:
            """The server's output to its log; its readiness line, or the end of its output, sets ``ready``."""
            for line in self._proc.stdout:
                self._server_log.write(line)
                self._server_log.flush()
                if READY_LINE in line:
                    listening.set()
                    ready.set()
            ready.set()

        threading.Thread(target=relay, daemon=True, name="llama-server-log").start()
        try:
            if not ready.wait(ready_timeout_s):
                raise TimeoutError(
                    f"llama-server did not log {READY_LINE!r} within {ready_timeout_s:.0f} s "
                    f"(log: {self._server_log_path})"
                )
            if not listening.is_set():
                # Its output ended without the readiness line: it is exiting. Wait for that exit.
                code = self._proc.wait()
                tail = self._read_server_log_tail()
                raise RuntimeError(
                    f"llama-server exited with code {code} during "
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
        """Map one typed sentence to a ParsedCommand (grammar-constrained), logging the latency.

        The sentence is its own words, so the request ends when the call is complete.
        """
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

    def parse_audio(self, wav: bytes, on_call: OnCall | None = None) -> ParsedCommand:
        """Map one WAV clip to a ParsedCommand: Gemma hears the audio (needs the ``mmproj``).

        ``on_call`` receives the call while Gemma still writes the words it heard; the
        returned command holds those words.
        """
        call_s: list[float] = []  # this request's own, so a typed request on another thread cannot change it

        def call_written(call: ParsedCommand, seconds: float):
            call_s.append(seconds)
            return on_call(call, seconds) if on_call is not None else None

        t0 = time.perf_counter()
        cmd = parse_audio_clip(
            wav, self.base, self.grammar, http=self._http, temperature=self.temperature, on_call=call_written,
        )
        logger.info(
            "Intent %r %s ← %r (call after %.0f ms, %.0f ms in all)",
            cmd.intent.value,
            cmd.params or "",
            cmd.transcript,
            (call_s[0] if call_s else 0.0) * 1e3,
            (time.perf_counter() - t0) * 1e3,
        )
        return cmd

    def warm_up(self, wav: bytes | None = None) -> float:
        """One request whose answer is thrown away; the seconds it took.

        The first request to a freshly started server pays for the instructions being read
        into the model. Paid here, at the start, not at the user's first command.

        ``wav`` is a clip to send as that request, for a server that hears audio. The first
        clip also pays for the part of the model that hears, which a typed request does not
        run: after the typed "hello" the first spoken command's call came after 0.34 s and
        every later one after 0.16 s; after a clip of silence the first came after 0.17 s
        (2026-11-26, two fresh servers each).
        """
        t0 = time.perf_counter()
        if wav is not None and self._audio:
            self.parse_audio(wav)
        else:
            self.parse("hello")
        return time.perf_counter() - t0


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
