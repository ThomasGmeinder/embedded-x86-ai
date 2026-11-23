"""One request to the Gemma server: typed text, or a 16 kHz clip, to a :class:`ParsedCommand`.

The two requests are the ones measured in SPEC.md: a chat completion with the
instructions of ``intents.py`` as the system message and the sentence, or the
clip as ``input_audio``, as the user message, held to the shape of one answer.
How the shape is held depends on the backend of the settings: ``llama-server``
takes the grammar of ``intent.gbnf``; FastFlowLM takes the JSON schema of
``intents.response_format`` and the model's tag, with thinking off through its
own ``think`` field.

The answer is read as the server writes it (a stream of server-sent events).
It is written in the order command, object, place, heard, so the call is
complete before the words heard are: ``on_call`` receives the call at that
moment, with the seconds since the request began, while the rest is still
being written. Nothing waits for the whole answer to act on the call.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Callable, Iterator

import requests

from so101_smolvla.voice.intents import (
    INTENT_SYSTEM_PROMPT,
    Intent,
    ParsedCommand,
    call_from_partial,
    heard_from_partial,
    response_format,
)
from so101_smolvla.voice.settings import Settings, grammar, load_settings

AUDIO_INSTRUCTION = "Classify the spoken robot instruction in this audio."
# The call, and the seconds from the request to it. Returning False says the words are not wanted.
OnCall = Callable[[ParsedCommand, float], bool | None]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def answer_pieces(lines) -> Iterator[tuple[str, dict | None]]:
    """The answer's text piece by piece, as the server's events carry it, each with the usage block of
    its event when it has one. ``lines`` are the lines of an OpenAI-style event stream; it ends at
    ``[DONE]``."""
    for line in lines:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            return
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(f"the Gemma server reported: {str(event['error'])[:300]}")
        choices = event.get("choices") or [{}]
        yield (choices[0].get("delta") or {}).get("content") or "", event.get("usage")


class GemmaClient:
    """Requests to one running server, over one kept-alive connection."""

    def __init__(self, base: str, settings: Settings | None = None, backend: str | None = None) -> None:
        self.settings = settings or load_settings()
        self.backend = backend or self.settings.backend
        self.base = base
        self.grammar = grammar()
        self.instructions = INTENT_SYSTEM_PROMPT
        self.last_usage: dict | None = None  # the server's usage block of the last answer, when it sent one
        self.last_call_s: float | None = None  # the last request: seconds until its call was complete
        self.last_total_s: float | None = None  # the last request: seconds until reading its answer ended
        self._http = requests.Session()

    def close(self) -> None:
        self._http.close()

    def _constraint(self) -> dict:
        """The request fields that hold the answer to its shape, and turn thinking off, per backend."""
        if self.backend == "igpu":
            return {"grammar": self.grammar, "chat_template_kwargs": {"enable_thinking": False}}
        # FastFlowLM: thinking on would spoil the pinned instructions for every later request (probe of
        # 2026-11-23), so the field is always sent.
        return {"model": self.settings.npu["model"], "think": False, "response_format": response_format()}

    def _complete(self, user_content, transcript: str, on_call: OnCall | None, words: bool) -> ParsedCommand:
        """One request, its answer read as it is written.

        ``on_call`` gets the call the moment it is complete. With ``words`` false, or when
        ``on_call`` returns False, the reading ends there and the connection is closed: for a
        caller that has the sentence already, or that drops the call. The server is then free
        for the next request sooner (measured on FastFlowLM, 2026-11-23: the request ends 1.3 s
        earlier and the next one's call comes 0.4 s later than usual). An answer that ends
        before its call is complete is ``unknown``.
        """
        request = self.settings.request
        payload = {
            "messages": [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": user_content},
            ],
            "temperature": float(request["temperature"]),
            "max_tokens": int(request["max_tokens"]),  # the command JSON plus the short "heard" phrase
            "stream": True,
            **self._constraint(),
        }
        self.last_usage = self.last_call_s = self.last_total_s = None
        text, call = "", None
        began = time.perf_counter()
        with self._http.post(
            f"{self.base}/v1/chat/completions", json=payload, stream=True, timeout=float(request["timeout_s"])
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(f"the Gemma server answered {response.status_code}: {response.text[:300]}")
            # chunk_size=1: a line is handed on when its last byte arrives. The library's default
            # reads in blocks of 512 bytes and would hold the end of the call back until the
            # server has written that much more.
            for piece, usage in answer_pieces(response.iter_lines(chunk_size=1)):
                self.last_usage = usage or self.last_usage
                text += piece
                if call is None:
                    call = call_from_partial(text)
                    if call is not None:
                        self.last_call_s = time.perf_counter() - began
                        # on_call answers False when it has no use for the words (a call it drops).
                        wanted = on_call(call, self.last_call_s) if on_call is not None else None
                        if not words or wanted is False:
                            break  # leaving the block closes the connection; the rest is not read
        self.last_total_s = time.perf_counter() - began
        if call is None:
            return ParsedCommand(Intent.UNKNOWN, transcript=transcript)
        return ParsedCommand(call.intent, call.params, heard_from_partial(text) or transcript)

    def parse(self, text: str, on_call: OnCall | None = None, words: bool = True) -> ParsedCommand:
        """The sentence as typed text. ``words=False`` returns as soon as the call is complete, without
        Gemma's words for what it read (the sentence itself stands in)."""
        if not text or not text.strip():
            return ParsedCommand(Intent.UNKNOWN, transcript=text or "")
        return self._complete(text.strip(), text, on_call, words)

    def warm_up(self) -> float:
        """One request whose answer is thrown away, so the first real one does not pay for the
        server reading the instructions into the model (measured on FastFlowLM, 2026-11-23: the
        first request's call after 4.65 s, every later one's after about 1.95 s). The seconds it took."""
        self.parse("hello", words=False)
        return float(self.last_total_s)

    def parse_audio(self, wav: bytes, on_call: OnCall | None = None) -> ParsedCommand:
        """One clip, the bytes of a WAV file, as the user message."""
        encoded = base64.b64encode(wav).decode("ascii")
        content = [
            {"type": "text", "text": AUDIO_INSTRUCTION},
            {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}},
        ]
        return self._complete(content, "", on_call, True)
