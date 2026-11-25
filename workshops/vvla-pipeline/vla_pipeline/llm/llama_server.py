# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""One ``llama-server`` process for a model section of the config: found running, or started and stopped here.

Both voice models run in ``llama-server`` (llama.cpp built with ``-DGGML_HIP=ON``
by ``bootstrap.sh``) on the iGPU: Gemma 4 (:mod:`vla_pipeline.llm.llama_intent`)
and the decision model d1-Omni-2B (:mod:`vla_pipeline.llm.d1_intent`). What they
share is here: the server's command line, its environment, the wait for its
readiness line, and its end. What differs is each parser's ``_server_args``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading

import requests

from vla_pipeline.utils.config import resolve

logger = logging.getLogger(__name__)

# What llama-server logs once its model is loaded and it serves requests.
READY_LINE = "listening on http://"
# The longest a request may go without the server sending anything, in seconds.
REQUEST_TIMEOUT_S = 30


class LlamaServer:
    """A llama-server for the config section ``section``, and the HTTP session that asks it."""

    def __init__(self, cfg: dict, *, section: str):
        l = cfg[section]
        self.section = section
        self.host: str = l["host"]
        self.port: int = int(l["port"])
        self.base = f"http://{self.host}:{self.port}"
        self._cfg = l
        self._log_dir = cfg.get("system", {}).get("log_dir", "logs")
        self._proc: subprocess.Popen | None = None
        self._http = requests.Session()  # keep-alive: saves ~5-20 ms per command

        if not self._server_alive():
            if l.get("autostart", True):
                self._spawn_server()
            else:
                raise ConnectionError(
                    f"llama-server not reachable at {self.base} and autostart is disabled."
                )

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _server_alive(self, timeout: float = 0.5) -> bool:
        """Return True if llama-server responds healthy at /health."""
        try:
            r = self._http.get(f"{self.base}/health", timeout=timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _server_args(self) -> list[str]:
        """The command-line arguments that are this model's own, after the shared ones."""
        return []

    def _spawn_server(self, ready_timeout_s: float = 120.0) -> None:
        """Launch llama-server as a subprocess and block until it logs that it
        is listening (:data:`READY_LINE`), or exits."""
        server_bin = resolve(self._cfg["server_bin"])
        model = resolve(self._cfg["model_gguf"])
        if not server_bin.exists():
            raise FileNotFoundError(
                f"{server_bin} not found - run ./bootstrap.sh (llama.cpp build step)."
            )
        if not model.exists():
            raise FileNotFoundError(
                f"{model} not found - run scripts/ensure_voice.sh (it downloads the voice model)."
            )
        cmd = [
            str(server_bin),
            "--model",
            str(model),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--n-gpu-layers",
            str(self._cfg["n_gpu_layers"]),
        ]
        if self._cfg.get("mmproj"):
            mmproj = resolve(self._cfg["mmproj"])
            if not mmproj.exists():
                raise FileNotFoundError(
                    f"{mmproj} not found - the model's audio projector has not finished downloading."
                )
            cmd += ["--mmproj", str(mmproj)]
        cmd += self._server_args()

        # Deterministic GPU init on Strix Halo (gfx1151): the ROCm runtime can
        # segfault on the first kernel dispatch unless the iGPU is presented as
        # its nearest supported relative. Allow overriding/clearing via config.
        env = os.environ.copy()
        gfx = self._cfg.get("hsa_override_gfx_version", "11.5.1")
        if gfx:
            env.setdefault("HSA_OVERRIDE_GFX_VERSION", str(gfx))
        # Optional compute-unit fence (e.g. "0:0-31" leaves 8 of 40 CUs to the
        # graphics queue). Off by default: measure with so101_sim.bench first.
        if self._cfg.get("cu_mask"):
            env["HSA_CU_MASK"] = str(self._cfg["cu_mask"])

        logger.info("Starting llama-server (ROCm): %s", " ".join(cmd))
        log_dir = resolve(self._log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._server_log_path = log_dir / "llama-server.log"
        self._server_log = open(self._server_log_path, "w")
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, errors="replace",
        )
        ready = threading.Event()

        def relay() -> None:
            """The server's output to its log; its readiness line, or its exit, sets ``ready``."""
            for line in self._proc.stdout:
                self._server_log.write(line)
                self._server_log.flush()
                if READY_LINE in line:
                    ready.set()
            ready.set()

        threading.Thread(target=relay, daemon=True, name="llama-server-log").start()
        try:
            if not ready.wait(ready_timeout_s):
                raise TimeoutError(
                    f"llama-server did not log {READY_LINE!r} within {ready_timeout_s:.0f} s "
                    f"(log: {self._server_log_path})"
                )
            if self._proc.poll() is not None:
                tail = self._read_server_log_tail()
                raise RuntimeError(
                    f"llama-server exited with code {self._proc.wait()} during "
                    f"startup. Last log lines:\n{tail}\n"
                    f"(full log: {self._server_log_path})"
                )
            if not self._server_alive():
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
