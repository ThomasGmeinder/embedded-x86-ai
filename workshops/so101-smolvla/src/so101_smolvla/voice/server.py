"""``llama-server`` with Gemma 4 E4B: started here, with its settings, and stopped here.

The graphics override of ``configs/voice.json`` goes into the environment of the
server process only. The caller's process never sees it: SmolVLA's PyTorch runs
natively on this machine's graphics and must not.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import requests

from so101_smolvla.voice.settings import Settings, SetupFailure, gemma_file, load_settings

# What llama-server logs once its model is loaded and it serves requests.
READY_LINE = "listening on http://"
# The settings of one server, as a result's provenance records them.
RECORDED = (
    "port", "n_gpu_layers", "ctx_size", "parallel", "temperature", "no_warmup",
    "hsa_override_gfx_version",
)


def server_answers(host: str, port: int, timeout_s: float = 0.5) -> bool:
    """Whether something answers at the server's health address."""
    try:
        return requests.get(f"http://{host}:{port}/health", timeout=timeout_s).status_code == 200
    except requests.RequestException:
        return False


class GemmaServer:
    """One ``llama-server`` process; ``start`` returns when it listens, ``close`` ends it."""

    def __init__(self, log_dir: Path, *, port: int | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        used = dict(self.settings.server)
        if port is not None:
            used["port"] = int(port)
        self.used = used
        self.host: str = used["host"]
        self.port: int = int(used["port"])
        self.base = f"http://{self.host}:{self.port}"
        self.log_path = Path(log_dir) / "llama-server.log"
        self._process: subprocess.Popen | None = None
        self._log = None

    def answers(self) -> bool:
        return server_answers(self.host, self.port)

    @property
    def recorded_settings(self) -> dict:
        return {key: self.used.get(key) for key in RECORDED}

    def command(self) -> list[str]:
        """The server's command line, from the settings and the model files' places."""
        server_bin = self.settings.server_bin
        if not os.access(server_bin, os.X_OK):
            raise SetupFailure(f"no llama-server at {server_bin}; run scripts/setup_voice.sh")
        command = [
            str(server_bin),
            "--model", str(gemma_file(self.settings, "model_file")),
            "--mmproj", str(gemma_file(self.settings, "mmproj_file")),
            "--host", self.host,
            "--port", str(self.port),
            "--n-gpu-layers", str(self.used["n_gpu_layers"]),
            "--ctx-size", str(self.used["ctx_size"]),
            "--parallel", str(self.used["parallel"]),
        ]
        if self.used.get("no_warmup"):
            command.append("--no-warmup")
        return command

    def start(self) -> "GemmaServer":
        """Start the server and return when it listens. Refuses a port something already answers on."""
        if self.answers():
            raise SetupFailure(f"something already answers at {self.base}; its settings are unknown")
        command = self.command()
        environment = os.environ.copy()
        environment["HSA_OVERRIDE_GFX_VERSION"] = str(self.used["hsa_override_gfx_version"])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "w", encoding="utf-8")
        self._process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment,
            text=True, errors="replace",
        )
        ready = threading.Event()
        process, log = self._process, self._log

        def relay() -> None:
            """The server's output to its log; its readiness line, or its exit, sets ``ready``."""
            for line in process.stdout:
                log.write(line)
                log.flush()
                if READY_LINE in line:
                    ready.set()
            ready.set()

        threading.Thread(target=relay, daemon=True, name="llama-server-log").start()
        try:
            # One bounded wait for the server's own readiness line, not a retry loop.
            timeout_s = float(self.used["ready_timeout_s"])
            if not ready.wait(timeout_s):
                raise TimeoutError(
                    f"llama-server did not log {READY_LINE!r} within {timeout_s:.0f} s (log: {self.log_path})"
                )
            if process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with status {process.wait()} while starting. Last log lines:\n"
                    f"{self._log_tail()}\n(full log: {self.log_path})"
                )
            if not self.answers():
                raise RuntimeError(f"llama-server logged readiness but {self.base}/health does not answer")
        except BaseException:
            self.close()
            raise
        return self

    def _log_tail(self, lines: int = 20) -> str:
        try:
            return "".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-lines:])
        except OSError:
            return "(no server log)"

    def close(self) -> None:
        """End the server process, if this object started one."""
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None
        if self._log is not None and not self._log.closed:
            self._log.close()
