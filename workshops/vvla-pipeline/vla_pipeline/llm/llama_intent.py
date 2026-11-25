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
parsing.

A request returns a :class:`ParsedCommand` (intent **plus parameters**) so
commands like "pick up the ball" carry their object through to behaviors.

The answer is read as the server writes it (a stream of server-sent events).
It is written in the order command, object, heard, so the call is complete
before the words heard are: ``on_call`` receives the call at that moment, with
the seconds since the request began, while the rest is still being written.
Nothing waits for the whole answer to act on the call. A typed sentence is its
own words, so its request ends at the call.

Independent test
----------------
    python -m vla_pipeline.llm.llama_intent --text "hey can you pick up the ball"
    python -m vla_pipeline.llm.llama_intent --selftest
    python -m vla_pipeline.llm.llama_intent           # interactive REPL
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
from collections.abc import Callable, Iterator

from vla_pipeline.llm.intents import (
    INTENT_SYSTEM_PROMPT,
    SELFTEST_CASES,
    Intent,
    ParsedCommand,
    call_from_partial,
    heard_from_partial,
)
from vla_pipeline.llm.llama_server import REQUEST_TIMEOUT_S, LlamaServer
from vla_pipeline.utils.config import load_config, resolve

logger = logging.getLogger(__name__)

# The call, and the seconds from the request to it. Returning False says the words are not wanted.
OnCall = Callable[[ParsedCommand, float], bool | None]


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


class LlamaIntentParser(LlamaServer):
    """Gemma 4 on a llama.cpp server (:class:`LlamaServer`): transcripts and clips into :class:`ParsedCommand`."""

    def __init__(self, cfg: dict, *, section: str = "gemma4-e4b-it"):
        l = cfg[section]
        self._audio = bool(l.get("mmproj"))
        self.temperature: float = float(l.get("temperature", 0.0))
        self.grammar = resolve(l["grammar_file"]).read_text(encoding="utf-8")
        self.last_call_s: float | None = None  # the last request: seconds until its call was complete
        self.last_total_s: float | None = None  # the last request: seconds until reading its answer ended
        super().__init__(cfg, section=section)  # finds the server running, or starts it

    def _server_args(self) -> list[str]:
        """Gemma's own server arguments: the context, the slots, the warm-up, the batch."""
        args = ["--ctx-size", str(self._cfg["ctx_size"]), "--parallel", str(self._cfg.get("parallel", 1))]
        if self._cfg.get("no_warmup", False):
            args.append("--no-warmup")
        # Smaller compute ubatches keep each GPU submission short, so a 3-D
        # render pass sharing the iGPU waits less behind prompt processing.
        if self._cfg.get("ubatch_size"):
            args += ["--ubatch-size", str(int(self._cfg["ubatch_size"]))]
        return args + list(self._cfg.get("extra_args", []) or [])

    # ------------------------------------------------------------------
    # Intent parsing
    # ------------------------------------------------------------------

    def _complete(self, user_content, transcript: str, on_call: OnCall | None, words: bool) -> ParsedCommand:
        """One chat completion, grammar-constrained, its answer read as it is written.

        ``on_call`` gets the call the moment it is complete. With ``words`` false, or when
        ``on_call`` returns False, the reading ends there and the connection is closed: for a
        caller that has the sentence already, or that drops the call. The server is then free
        for the next request sooner. An answer that ends before its call is complete is
        ``unknown``.
        """
        payload = {
            "messages": [
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": 96,  # the command list plus the short "heard" phrase
            "grammar": self.grammar,
            "stream": True,
        }
        if self._audio:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        self.last_call_s = self.last_total_s = None
        # The seconds of this request are kept in locals: a typed sentence and a spoken one can
        # be asked from two threads, and the fields above are only the last request's record.
        text, call, call_s = "", None, 0.0
        t0 = time.perf_counter()
        with self._http.post(
            f"{self.base}/v1/chat/completions", json=payload, stream=True, timeout=REQUEST_TIMEOUT_S
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
                        call_s = self.last_call_s = time.perf_counter() - t0
                        # on_call answers False when it has no use for the words (a call it drops).
                        wanted = on_call(call, call_s) if on_call is not None else None
                        if not words or wanted is False:
                            break  # leaving the block closes the connection; the rest is not read
        total_s = self.last_total_s = time.perf_counter() - t0
        if call is None:
            logger.warning("Unparseable LLM output: %r", text)
            return ParsedCommand(Intent.UNKNOWN, transcript=transcript)
        heard = heard_from_partial(text) or transcript
        logger.info(
            "Intent %r %s ← %r (call after %.0f ms, %.0f ms in all)",
            call.intent.value,
            call.params or "",
            heard,
            call_s * 1e3,
            total_s * 1e3,
        )
        return ParsedCommand(call.intent, params=call.params, transcript=heard)

    def parse(self, transcript: str, on_call: OnCall | None = None) -> ParsedCommand:
        """Map one typed sentence to a ParsedCommand (grammar-constrained).

        The sentence is its own words, so the request ends when the call is complete.
        """
        if not transcript or not transcript.strip():
            return ParsedCommand(Intent.UNKNOWN, transcript=transcript or "")
        return self._complete(transcript.strip(), transcript, on_call, words=False)

    def parse_audio(self, wav: bytes, on_call: OnCall | None = None) -> ParsedCommand:
        """Map one WAV clip to a ParsedCommand. The clip is the user message.

        ``on_call`` receives the call while Gemma still writes the words it heard; the
        returned command holds those words.
        """
        encoded = base64.b64encode(wav).decode("ascii")
        content = [
            {"type": "text", "text": "Classify the spoken robot instruction in this audio."},
            {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}},
        ]
        return self._complete(content, "", on_call, words=True)

    def warm_up(self) -> float:
        """One request whose answer is thrown away; the seconds it took.

        The first request to a freshly started server pays for the instructions being read
        into the model. Paid here, at the start, not at the user's first command.
        """
        self.parse("hello")
        return float(self.last_total_s)


# =============================================================================
# Standalone component test
# =============================================================================

def main() -> None:
    """CLI entry point: parse one utterance, run the selftest suite, or start an interactive REPL."""
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
    args = parser.parse_args()

    llm = LlamaIntentParser(load_config(args.config))
    try:
        if args.selftest:
            passed = 0
            for text, expected, obj in SELFTEST_CASES:
                got = llm.parse(text)
                ok = got.intent == expected and (obj is None or got.object_name == obj)
                passed += ok
                print(
                    f"  [{'PASS' if ok else 'FAIL'}] {text!r} -> {got.intent.value} "
                    f"{got.params} (expected {expected.value} {obj or ''})"
                )
            print(f"\n{passed}/{len(SELFTEST_CASES)} cases passed")
            raise SystemExit(0 if passed == len(SELFTEST_CASES) else 1)
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
