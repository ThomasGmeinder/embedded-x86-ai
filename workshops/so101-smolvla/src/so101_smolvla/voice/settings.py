"""The settings of the voice path and where its files are: the one reader of ``configs/voice.toml``.

Which Gemma runs, and where, is not in that file: it is the ``command`` section of
``configs/models.toml``, read through ``so101_smolvla.models``. Its ``backend`` key
chooses one of two backends: ``igpu`` is ``llama-server`` on the integrated graphics,
``npu`` is FastFlowLM (``flm``) on the NPU. Each has its own section, ``command.igpu``
and ``command.npu``, with the same names for the same things (``model``, ``context``,
``ready_timeout_s``). ``configs/voice.toml`` holds what both share (``server``,
``request``), the listening session and the synthetic voice.

No model file lives in the workshop. For ``igpu`` the two Gemma files are
looked up in the Hugging Face cache and never downloaded here
(``scripts/setup_voice.sh`` does that) and ``llama-server`` is the build of that
script under ``third_party/``. For ``npu`` the model is the one ``flm pull`` put in
FastFlowLM's own store, found through ``flm list``. The Silero speech-detection
model is the file its own package ships. A missing piece is a
:class:`SetupFailure` that names it.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from so101_smolvla.models import COMMAND_BACKENDS as BACKENDS
from so101_smolvla.models import MODELS_FILE, ROOT, SetupFailure, load_models

SETTINGS_FILE = ROOT / "configs" / "voice.toml"
SETUP = "scripts/setup_voice.sh"
NPU_DEVICE = Path("/dev/accel/accel0")
FLM_INSTALL = "https://github.com/ROCm/FastFlowLM/releases"


@dataclass(frozen=True)
class Settings:
    """The voice path's settings: the backend and its two sections from ``configs/models.toml``, the rest from ``configs/voice.toml``."""

    backend: str  # models.toml, command.backend
    backends: dict  # models.toml, command.npu and command.igpu, by their names
    server: dict  # where the server listens: host, port, runner_port
    request: dict  # one chat completion: temperature, max_tokens, timeout_s
    listening: dict
    synthetic_voice: dict

    @property
    def npu(self) -> dict:
        """FastFlowLM on the NPU: its model tag, program, minimum version, power mode."""
        return self.backends["npu"]

    @property
    def igpu(self) -> dict:
        """llama-server on the integrated graphics: its model files and server options."""
        return self.backends["igpu"]

    @property
    def build(self) -> dict:
        """The llama.cpp build of the iGPU backend, and the graphics override it runs under."""
        return self.igpu["build"]

    @property
    def server_bin(self) -> Path:
        return ROOT / self.build["server_bin"]

    @property
    def runner_port(self) -> int:
        """Where the three-cube runner and the tests start their own server."""
        return int(self.server["runner_port"])


def load_settings(path: Path = SETTINGS_FILE, models_path: Path = MODELS_FILE) -> Settings:
    """The voice path's own file, and the ``command`` section of the models file (checked by ``load_models``)."""
    command = load_models(models_path)["command"]
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    settings = Settings(
        backend=command["backend"], backends={name: command[name] for name in BACKENDS}, **data
    )
    # TOML has no null: the microphone, left out of the file, is the system's default one.
    settings.listening.setdefault("mic_device", None)
    return settings


# ---- the iGPU backend: llama-server and the two Gemma files -------------------------------------

def gemma_file(settings: Settings, which: str) -> Path:
    """The path of ``model`` or ``mmproj`` of the iGPU backend in the Hugging Face cache, or a SetupFailure."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    name, repo = settings.igpu[which], settings.igpu["repo"]
    try:
        return Path(hf_hub_download(repo, name, local_files_only=True))
    except LocalEntryNotFoundError:
        raise SetupFailure(f"{name} of {repo} is not in the Hugging Face cache; run {SETUP}") from None


# ---- the NPU backend: FastFlowLM and its model store --------------------------------------------

def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.strip().split("."))


def flm_binary(settings: Settings) -> Path:
    """The ``flm`` program of the settings, on the PATH, or a SetupFailure."""
    found = shutil.which(settings.npu["binary"])
    if found is None:
        raise SetupFailure(
            f"no {settings.npu['binary']} on the PATH; install FastFlowLM {settings.npu['min_version']} or later "
            f"from {FLM_INSTALL} (the Ubuntu 24.04 .deb), then run {SETUP}"
        )
    return Path(found)


def _flm_json(settings: Settings, *arguments: str) -> dict:
    ran = subprocess.run([str(flm_binary(settings)), *arguments, "-j"], capture_output=True, text=True)
    try:
        return json.loads(ran.stdout)
    except json.JSONDecodeError:
        raise SetupFailure(f"flm {' '.join(arguments)} -j gave no JSON: {(ran.stdout + ran.stderr).strip()[:200]}") from None


def flm_version(settings: Settings) -> str:
    """The version ``flm version -j`` reports, at or above the settings' minimum, or a SetupFailure."""
    version = str(_flm_json(settings, "version")["version"])
    minimum = settings.npu["min_version"]
    if _version_tuple(version) < _version_tuple(minimum):
        raise SetupFailure(f"FastFlowLM {version} is installed; {minimum} or later is needed ({FLM_INSTALL})")
    return version


def flm_model(settings: Settings) -> dict:
    """The entry of ``flm list -j`` for the NPU backend's model tag, installed, or a SetupFailure."""
    tag = settings.npu["model"]
    entries = [entry for entry in _flm_json(settings, "list")["models"] if entry.get("model") == tag]
    if not entries:
        raise SetupFailure(f"FastFlowLM knows no model {tag!r}; see flm list")
    entry = entries[0]
    if not entry.get("installed"):
        raise SetupFailure(f"the FastFlowLM model {tag} is not downloaded; run flm pull {tag} or {SETUP}")
    return entry


def npu_device() -> Path:
    """The NPU's device node, or a SetupFailure naming the driver."""
    if not NPU_DEVICE.exists():
        raise SetupFailure(f"no NPU device {NPU_DEVICE}: the amdxdna driver is not loaded or this machine has no NPU")
    return NPU_DEVICE


# ---- both ------------------------------------------------------------------------------------------

def silero_model() -> Path:
    """The Silero speech-detection model, the file the silero-vad package ships."""
    try:
        path = Path(str(resources.files("silero_vad") / "data" / "silero_vad.onnx"))
    except ModuleNotFoundError:
        raise SetupFailure(f"the silero-vad package is not installed in .venv; run {SETUP}") from None
    if not path.is_file():
        raise SetupFailure(f"the silero-vad package has no model at {path}")
    return path


def check_ready(settings: Settings | None = None, backend: str | None = None) -> list[str]:
    """What is missing for the voice path on the backend, each as a sentence; empty when everything is there.

    ``backend`` overrides the settings' choice, for a check of the other one."""
    settings = settings or load_settings()
    backend = backend or settings.backend
    if backend not in BACKENDS:
        return [f"unknown backend {backend!r}; one of {', '.join(BACKENDS)}"]
    missing = []
    if backend == "igpu":
        if not os.access(settings.server_bin, os.X_OK):
            missing.append(f"no llama-server at {settings.server_bin}; run {SETUP}")
        for which in ("model", "mmproj"):
            try:
                gemma_file(settings, which)
            except (SetupFailure, ImportError) as exc:
                missing.append(str(exc))
    else:
        for check in (lambda: flm_version(settings), lambda: flm_model(settings), npu_device):
            try:
                check()
            except SetupFailure as exc:
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
