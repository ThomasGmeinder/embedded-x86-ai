"""The settings of the voice path and where its files are: the one reader of ``configs/voice.json``.

No model file lives in the workshop. The two Gemma files are looked up in the
Hugging Face cache and never downloaded here (``scripts/setup_voice.sh`` does
that); the Silero speech-detection model is the file its own package ships;
``llama-server`` is the build of ``scripts/setup_voice.sh`` under
``third_party/``. A missing piece is a :class:`SetupFailure` that names it.
"""

from __future__ import annotations

import ctypes
import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the workshop
SETTINGS_FILE = ROOT / "configs" / "voice.json"
GRAMMAR_FILE = Path(__file__).with_name("intent.gbnf")
SETUP = "scripts/setup_voice.sh"


class SetupFailure(RuntimeError):
    """Something the voice path needs is not on this machine; the message names it."""

    def __init__(self, what: str) -> None:
        super().__init__(f"SETUP FAILURE: {what}")


@dataclass(frozen=True)
class Settings:
    """The five sections of ``configs/voice.json``, without their ``_``-prefixed notes."""

    llama_cpp: dict
    gemma: dict
    server: dict
    listening: dict
    synthetic_voice: dict

    @property
    def server_bin(self) -> Path:
        return ROOT / self.llama_cpp["server_bin"]

    @property
    def runner_port(self) -> int:
        """Where the three-cube runner and the tests start their own server."""
        return int(self.server["runner_port"])


def _without_notes(section: dict) -> dict:
    return {key: value for key, value in section.items() if not key.startswith("_")}


def load_settings(path: Path = SETTINGS_FILE) -> Settings:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Settings(
        **{name: _without_notes(data[name]) for name in Settings.__dataclass_fields__}
    )


def gemma_file(settings: Settings, which: str) -> Path:
    """The path of ``model_file`` or ``mmproj_file`` in the Hugging Face cache, or a SetupFailure."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    name = settings.gemma[which]
    try:
        return Path(hf_hub_download(settings.gemma["repo"], name, local_files_only=True))
    except LocalEntryNotFoundError:
        raise SetupFailure(
            f"{name} of {settings.gemma['repo']} is not in the Hugging Face cache; run {SETUP}"
        ) from None


def silero_model() -> Path:
    """The Silero speech-detection model, the file the silero-vad package ships."""
    try:
        path = Path(str(resources.files("silero_vad") / "data" / "silero_vad.onnx"))
    except ModuleNotFoundError:
        raise SetupFailure(f"the silero-vad package is not installed in .venv; run {SETUP}") from None
    if not path.is_file():
        raise SetupFailure(f"the silero-vad package has no model at {path}")
    return path


def grammar() -> str:
    return GRAMMAR_FILE.read_text(encoding="utf-8")


def check_ready(settings: Settings | None = None) -> list[str]:
    """What is missing for the voice path, each as a sentence; empty when everything is there."""
    settings = settings or load_settings()
    missing = []
    if not os.access(settings.server_bin, os.X_OK):
        missing.append(f"no llama-server at {settings.server_bin}; run {SETUP}")
    for which in ("model_file", "mmproj_file"):
        try:
            gemma_file(settings, which)
        except (SetupFailure, ImportError) as exc:
            missing.append(str(exc))
    try:
        silero_model()
    except SetupFailure as exc:
        missing.append(str(exc))
    library = settings.synthetic_voice["library"]
    try:
        ctypes.CDLL(library)
    except OSError:
        missing.append(f"the synthetic voice's library {library} is not installed (apt install espeak-ng)")
    return missing
