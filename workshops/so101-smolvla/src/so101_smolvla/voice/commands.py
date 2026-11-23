"""Gemma 4 asked for the fields of a typed or spoken sentence: what the three-cube test uses.

``VoiceCommands`` starts its own Gemma server on the runner's port of
``configs/voice.toml`` (so a server the user runs on the default port is left
alone), on the backend of the settings or the one given, asks it through
:class:`GemmaClient`, and stops it when closed. What comes back for a sentence
are Gemma's fields as it gave them: ``command``, ``object``, ``place`` and
``heard``. Which cube they name is decided elsewhere
(``cube_commands.choose_cube``).
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from so101_smolvla.voice.client import GemmaClient, sha256
from so101_smolvla.voice.intents import schema_text
from so101_smolvla.voice.server import GemmaServer
from so101_smolvla.voice.settings import Settings, check_ready, flm_model, flm_version, load_settings
from so101_smolvla.voice.speech import SyntheticVoice

ROUTE = "gemma4-e4b-it voice client, typed and spoken"
WHERE = {"igpu": "llama-server on the integrated graphics", "npu": "FastFlowLM on the NPU"}
TYPED = "typed"
SPOKEN = "synthetic voice"


@dataclass(frozen=True)
class CommandFields:
    """What Gemma made of one sentence, and how the sentence reached it."""

    sentence: str
    how: str  # TYPED or SPOKEN
    command: str
    object: str
    place: str
    heard: str
    seconds: float  # from the request to the end of the answer, the words heard included
    spoken_s: float | None = None  # the length of the spoken sentence
    call_seconds: float | None = None  # from the request to the complete call, before the words heard

    def as_record(self) -> dict:
        return asdict(self)


def voice_ready(backend: str | None = None) -> tuple[bool, str]:
    """Whether the voice path can run here on the backend: (True, "ready") or (False, what is missing)."""
    missing = check_ready(backend=backend)
    return (True, "ready") if not missing else (False, "; ".join(missing))


def server_version(settings: Settings, backend: str) -> str:
    """What names the server's build: llama-server's version line with its commit, or FastFlowLM's version."""
    if backend == "npu":
        return f"FastFlowLM {flm_version(settings)}"
    ran = subprocess.run([str(settings.server_bin), "--version"], capture_output=True, text=True)
    lines = (ran.stdout + ran.stderr).strip().splitlines()
    return next((line for line in lines if "commit" in line), lines[0] if lines else "")


class VoiceCommands:
    """Gemma on its own server, started here and stopped by ``close``."""

    def __init__(
        self, log_dir: Path, port: int | None = None, settings: Settings | None = None, backend: str | None = None
    ) -> None:
        self.settings = settings or load_settings()
        self.backend = backend or self.settings.backend
        port = self.settings.runner_port if port is None else port
        self.server = GemmaServer(Path(log_dir), port=port, settings=self.settings, backend=self.backend).start()
        self.client = GemmaClient(self.server.base, self.settings, backend=self.backend)
        self._voice: SyntheticVoice | None = None
        self.provenance = {
            "route": f"{ROUTE}, {WHERE[self.backend]}",
            "backend": self.backend,
            "base": self.server.base,
            "settings": self.server.recorded_settings | {"window_s": self.settings.listening["window_s"]},
            "server_version": server_version(self.settings, self.backend),
            **self._model_provenance(),
            "instructions_sha256": sha256(self.client.instructions),
            "server_log": str(self.server.log_path),
            "synthetic_voice": SyntheticVoice(self.settings).recorded(),
        }

    def _model_provenance(self) -> dict:
        """The model and the constraint on the answer, as the backend has them."""
        if self.backend == "igpu":
            return {
                "model_gguf": self.settings.igpu["model"],
                "mmproj": self.settings.igpu["mmproj"],
                "grammar_sha256": sha256(self.client.grammar),
            }
        entry = flm_model(self.settings)
        return {
            "model_tag": entry["model"],
            "model_name": entry["url"].rstrip("/").rsplit("/", 1)[-1],
            "quantization": entry["details"]["quantization_level"],
            "context_length": self.settings.npu["context"] or entry["default_context_length"],
            "schema_sha256": sha256(schema_text()),
        }

    def _fields(self, command, sentence: str, how: str, seconds: float, spoken_s=None) -> CommandFields:
        return CommandFields(
            sentence=sentence, how=how, command=command.intent.value,
            object=command.params.get("object", ""), place=command.params.get("place", ""),
            heard=command.transcript, seconds=seconds, spoken_s=spoken_s,
            call_seconds=self.client.last_call_s,
        )

    def parse(self, sentence: str) -> CommandFields:
        """The sentence as typed text."""
        began = time.perf_counter()
        command = self.client.parse(sentence)
        return self._fields(command, sentence, TYPED, time.perf_counter() - began)

    def parse_spoken(self, sentence: str) -> CommandFields:
        """The sentence said by the synthetic voice, as one clip."""
        if self._voice is None:
            self._voice = SyntheticVoice(self.settings)
        wav, spoken_s = self._voice.clip(sentence)
        began = time.perf_counter()
        command = self.client.parse_audio(wav)
        return self._fields(command, sentence, SPOKEN, time.perf_counter() - began, spoken_s)

    def ask(self, sentence: str, spoken: bool = False) -> CommandFields:
        return self.parse_spoken(sentence) if spoken else self.parse(sentence)

    def close(self) -> None:
        self.client.close()
        self.server.close()
