"""Which model does what and on which processor: the one Python reader of ``configs/models.toml``.

Three sections, one per model of the use case:

- ``command``: Gemma 4 E4B. ``backend`` chooses ``command.npu`` (FastFlowLM on
  the NPU) or ``command.igpu`` (``llama-server`` on the integrated graphics).
- ``outlines``: SAM 3, the reference's checkout and files.
- ``motion``: SmolVLA, the checkpoint.

``device`` of ``outlines`` and ``motion`` states where they run, the
integrated graphics; neither can run elsewhere today, so another value is
refused here instead of being ignored.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the workshop
MODELS_FILE = ROOT / "configs" / "models.toml"
COMMAND_BACKENDS = ("igpu", "npu")
GRAPHICS = "igpu"
ON_THE_GRAPHICS = ("outlines", "motion")


class SetupFailure(RuntimeError):
    """Something a model needs is not on this machine or not in the settings; the message names it."""

    def __init__(self, what: str) -> None:
        super().__init__(f"SETUP FAILURE: {what}")


def load_models(path: Path = MODELS_FILE) -> dict:
    """The three sections of the file, checked; what the file may leave out reads as None."""
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    for section in ("command", *ON_THE_GRAPHICS):
        if section not in data:
            raise SetupFailure(f"{path} has no section {section}")
    command = data["command"]
    backend = command.get("backend")
    if backend not in COMMAND_BACKENDS:
        raise SetupFailure(
            f"{path} names the backend {backend!r}; it must be one of {', '.join(COMMAND_BACKENDS)}"
        )
    for name in COMMAND_BACKENDS:
        if name not in command:
            raise SetupFailure(f"{path} has no section command.{name}")
    for section in ON_THE_GRAPHICS:
        device = data[section].get("device")
        if device != GRAPHICS:
            raise SetupFailure(
                f"{path} gives {section}.device as {device!r}; it runs on the integrated graphics only ({GRAPHICS!r})"
            )
    # TOML has no null: the NPU's context, left out of the file, is the model tag's own.
    command["npu"].setdefault("context", None)
    return data


def motion_checkpoint() -> str:
    """The SmolVLA checkpoint of the runs, a Hugging Face repository."""
    return str(load_models()["motion"]["checkpoint"])
