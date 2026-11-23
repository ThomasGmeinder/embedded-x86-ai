"""The Gemma 4 E4B server, on the backend of the settings: started here, with its settings, and stopped here.

``igpu`` is ``llama-server`` on the integrated graphics. The graphics override of
``configs/models.toml`` goes into the environment of that server process only.
The caller's process never sees it: SmolVLA's PyTorch runs natively on this
machine's graphics and must not.

``npu`` is FastFlowLM's ``flm serve`` on the NPU. It needs no environment of
its own; the graphics stays with SmolVLA and SAM 3.

Either way ``start`` returns when the server has logged its readiness line
(one bounded wait, no retry loop) and answers at its health address.
"""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import requests

from so101_smolvla.voice.settings import Settings, SetupFailure, flm_binary, gemma_file, load_settings


@dataclass(frozen=True)
class Runtime:
    """What differs between the two servers: the one place for it."""

    name: str
    ready_line: str  # what the server logs once its model is loaded and it serves requests
    health_path: str  # a GET that answers 200 while it serves
    log_name: str
    recorded: tuple[str, ...]  # the keys of the backend's own section a result's provenance records


RUNTIMES = {
    "igpu": Runtime(
        "llama-server", "listening on http://", "/health", "llama-server.log",
        ("context", "n_gpu_layers", "parallel", "no_warmup"),
    ),
    "npu": Runtime(
        "flm serve", "WebServer started on port", "/v1/models", "flm-server.log",
        ("model", "pmode", "context"),
    ),
}


def server_answers(host: str, port: int, timeout_s: float = 0.5, path: str = "/health") -> bool:
    """Whether something answers 200 at the server's health address."""
    try:
        return requests.get(f"http://{host}:{port}{path}", timeout=timeout_s).status_code == 200
    except requests.RequestException:
        return False


class GemmaServer:
    """One server process; ``start`` returns when it listens, ``close`` ends it."""

    def __init__(
        self, log_dir: Path, *, port: int | None = None, settings: Settings | None = None, backend: str | None = None
    ) -> None:
        self.settings = settings or load_settings()
        self.backend = backend or self.settings.backend
        if self.backend not in RUNTIMES:
            raise SetupFailure(f"unknown backend {self.backend!r}; one of {', '.join(RUNTIMES)}")
        self.runtime = RUNTIMES[self.backend]
        self.options: dict = self.settings.backends[self.backend]  # the backend's own section
        self.host: str = self.settings.server["host"]
        self.port: int = int(self.settings.server["port"] if port is None else port)
        self.base = f"http://{self.host}:{self.port}"
        self.log_path = Path(log_dir) / self.runtime.log_name
        self._process: subprocess.Popen | None = None
        self._log = None

    def answers(self) -> bool:
        return server_answers(self.host, self.port, path=self.runtime.health_path)

    @property
    def recorded_settings(self) -> dict:
        """What a result's provenance records: the backend, the port, the temperature, and the backend's own."""
        recorded = {
            "backend": self.backend,
            "port": self.port,
            "temperature": self.settings.request["temperature"],
            **{key: self.options[key] for key in self.runtime.recorded},
        }
        if self.backend == "igpu":
            recorded["hsa_override_gfx_version"] = self.settings.build["hsa_override_gfx_version"]
        return recorded

    def command(self) -> list[str]:
        """The server's command line, from the settings and the model's place."""
        if self.backend == "igpu":
            server_bin = self.settings.server_bin
            if not os.access(server_bin, os.X_OK):
                raise SetupFailure(f"no llama-server at {server_bin}; run scripts/setup_voice.sh")
            command = [
                str(server_bin),
                "--model", str(gemma_file(self.settings, "model")),
                "--mmproj", str(gemma_file(self.settings, "mmproj")),
                "--host", self.host,
                "--port", str(self.port),
                "--n-gpu-layers", str(self.options["n_gpu_layers"]),
                "--ctx-size", str(self.options["context"]),
                "--parallel", str(self.options["parallel"]),
            ]
            if self.options["no_warmup"]:
                command.append("--no-warmup")
            return command
        command = [
            str(flm_binary(self.settings)), "serve", str(self.options["model"]),
            "--host", self.host,
            "--port", str(self.port),
            "--pmode", str(self.options["pmode"]),
        ]
        if self.options["context"] is not None:
            command += ["-c", str(int(self.options["context"]))]
        return command

    def environment(self) -> dict:
        """The server process's environment: the caller's, plus the graphics override for llama-server only."""
        environment = os.environ.copy()
        if self.backend == "igpu":
            environment["HSA_OVERRIDE_GFX_VERSION"] = str(self.settings.build["hsa_override_gfx_version"])
        return environment

    def start(self) -> "GemmaServer":
        """Start the server and return when it listens. Refuses a port something already answers on."""
        if self.answers():
            raise SetupFailure(f"something already answers at {self.base}; its settings are unknown")
        command = self.command()
        name = self.runtime.name
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "w", encoding="utf-8")
        self._process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=self.environment(),
            text=True, errors="replace",
        )
        ready = threading.Event()
        process, log, ready_line = self._process, self._log, self.runtime.ready_line

        def relay() -> None:
            """The server's output to its log; its readiness line, or its exit, sets ``ready``."""
            for line in process.stdout:
                log.write(line)
                log.flush()
                if ready_line in line:
                    ready.set()
            ready.set()

        threading.Thread(target=relay, daemon=True, name=f"{self.runtime.log_name}-relay").start()
        try:
            # One bounded wait for the server's own readiness line, not a retry loop.
            timeout_s = float(self.options["ready_timeout_s"])
            if not ready.wait(timeout_s):
                raise TimeoutError(
                    f"{name} did not log {ready_line!r} within {timeout_s:.0f} s (log: {self.log_path})"
                )
            if process.poll() is not None:
                raise RuntimeError(
                    f"{name} exited with status {process.wait()} while starting. Last log lines:\n"
                    f"{self._log_tail()}\n(full log: {self.log_path})"
                )
            if not self.answers():
                raise RuntimeError(
                    f"{name} logged readiness but {self.base}{self.runtime.health_path} does not answer"
                )
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
