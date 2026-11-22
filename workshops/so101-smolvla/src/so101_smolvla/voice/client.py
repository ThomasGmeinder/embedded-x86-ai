"""One request to the Gemma server: typed text, or a 16 kHz clip, to a :class:`ParsedCommand`.

The two requests are the ones measured in SPEC.md: a chat completion held to
the grammar of ``intent.gbnf``, with the instructions of ``intents.py`` as the
system message and the sentence, or the clip as ``input_audio``, as the user
message.
"""

from __future__ import annotations

import base64
import hashlib
import json

import requests

from so101_smolvla.voice.intents import INTENT_SYSTEM_PROMPT, Intent, ParsedCommand
from so101_smolvla.voice.settings import Settings, grammar, load_settings

AUDIO_INSTRUCTION = "Classify the spoken robot instruction in this audio."


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class GemmaClient:
    """Requests to one running server, over one kept-alive connection."""

    def __init__(self, base: str, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.base = base
        self.grammar = grammar()
        self.instructions = INTENT_SYSTEM_PROMPT
        self._http = requests.Session()

    def close(self) -> None:
        self._http.close()

    def _complete(self, user_content, transcript: str) -> ParsedCommand:
        server = self.settings.server
        payload = {
            "messages": [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": user_content},
            ],
            "temperature": float(server["temperature"]),
            "max_tokens": int(server["max_tokens"]),  # the command JSON plus the short "heard" phrase
            "grammar": self.grammar,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = self._http.post(
            f"{self.base}/v1/chat/completions", json=payload, timeout=float(server["request_timeout_s"])
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        try:
            data = json.loads(content)
            command = data["command"]
        except (json.JSONDecodeError, KeyError):
            return ParsedCommand(Intent.UNKNOWN, transcript=transcript)
        params: dict[str, str] = {}
        for key in ("object", "place"):
            value = str(data.get(key, "") or "").strip()
            if value:
                params[key] = value
        heard = str(data.get("heard", "") or "").strip() or transcript
        return ParsedCommand(Intent.from_string(command), params=params, transcript=heard)

    def parse(self, text: str) -> ParsedCommand:
        """The sentence as typed text."""
        if not text or not text.strip():
            return ParsedCommand(Intent.UNKNOWN, transcript=text or "")
        return self._complete(text.strip(), text)

    def parse_audio(self, wav: bytes) -> ParsedCommand:
        """One clip, the bytes of a WAV file, as the user message."""
        encoded = base64.b64encode(wav).decode("ascii")
        content = [
            {"type": "text", "text": AUDIO_INSTRUCTION},
            {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}},
        ]
        return self._complete(content, "")
