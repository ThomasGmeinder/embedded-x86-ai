"""ROCm-side client for the isolated CPU SAPIEN policy environment."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from so101_smolvla.policy_env import PolicyObservation, PolicyStep
from so101_smolvla.sapien_config import (
    ACTION_SHAPE,
    PROTOCOL_VERSION,
    SAPIEN_COMMIT,
    SAPIEN_PYTHON,
    SAPIEN_SOURCE,
    task_spec,
)
from so101_smolvla.sapien_protocol import MAX_REQUEST_BYTES, Response, read_response

ROOT = Path(__file__).resolve().parents[2]


class SapienPolicyEnvironment:
    """Complete ``So101SimEnv`` owned by a persistent CPU subprocess."""

    backend = "sapien"

    def __init__(
        self, horizon: int = 300, task_mode: str = "pick-place"
    ) -> None:
        if not SAPIEN_PYTHON.is_file():
            raise FileNotFoundError(
                f"SAPIEN Python is missing: {SAPIEN_PYTHON}; run scripts/setup_sapien_env.sh"
            )
        if not SAPIEN_SOURCE.is_dir():
            raise FileNotFoundError(
                f"SAPIEN source is missing: {SAPIEN_SOURCE}; run scripts/setup_sapien_env.sh"
            )
        actual_commit = subprocess.check_output(
            ["git", "-C", str(SAPIEN_SOURCE), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if actual_commit != SAPIEN_COMMIT:
            raise RuntimeError(
                f"SAPIEN source is {actual_commit}, expected pinned {SAPIEN_COMMIT}"
            )
        if horizon < 1:
            raise ValueError(f"invalid horizon {horizon}")
        spec = task_spec(task_mode)
        self.task_mode = task_mode
        self.task_text = spec.instruction
        self.facts: dict[str, object] = {}

        env = os.environ.copy()
        source_path = os.pathsep.join((str(ROOT / "src"), str(SAPIEN_SOURCE)))
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            source_path if not existing else source_path + os.pathsep + existing
        )
        self._process = subprocess.Popen(
            [
                str(SAPIEN_PYTHON),
                "-m",
                "so101_smolvla.sapien_env_worker",
                "--serve",
            ],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        self._request_id = 0
        self._closed = False
        try:
            hello = self._request(
                {
                    "operation": "hello",
                    "protocol_version": PROTOCOL_VERSION,
                    "horizon": horizon,
                    "task_mode": task_mode,
                }
            )
        except Exception:
            self._terminate()
            raise
        if hello.metadata["protocol_version"] != PROTOCOL_VERSION:
            self.close()
            raise RuntimeError("SAPIEN worker returned the wrong protocol version")
        if hello.metadata["simulator_commit"] != SAPIEN_COMMIT:
            self.close()
            raise RuntimeError("SAPIEN worker returned the wrong simulator commit")
        self.provenance = {
            "backend": self.backend,
            "task_mode": task_mode,
            "task_text": hello.metadata["task_text"],
            "protocol_version": PROTOCOL_VERSION,
            "simulator_commit": hello.metadata["simulator_commit"],
            "environment_arguments": hello.metadata["environment_arguments"],
            "robot_appearance": hello.metadata["robot_appearance"],
        }

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SAPIEN environment is closed")
        code = self._process.poll()
        if code is not None:
            raise RuntimeError(f"SAPIEN worker exited with status {code}")

    def _request(self, document: dict) -> Response:
        self._ensure_open()
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("SAPIEN worker has no protocol pipes")
        self._request_id += 1
        request = {"request_id": self._request_id, **document}
        payload = json.dumps(request, separators=(",", ":"), allow_nan=False).encode()
        if len(payload) + 1 > MAX_REQUEST_BYTES:
            raise ValueError(f"SAPIEN request is too large: {len(payload) + 1} bytes")
        self._process.stdin.write(payload + b"\n")
        self._process.stdin.flush()
        return read_response(self._process.stdout, self._request_id)

    @staticmethod
    def _observation(response: Response) -> PolicyObservation:
        if response.top is None or response.wrist is None:
            raise RuntimeError("SAPIEN observation is missing camera images")
        state = np.asarray(response.metadata["state"], dtype=np.float64)
        if state.shape != ACTION_SHAPE or not np.isfinite(state).all():
            raise RuntimeError(f"SAPIEN returned invalid state {state}")
        return PolicyObservation(
            state=state,
            top=response.top,
            wrist=response.wrist,
            top_outline=response.top_outline,
            wrist_outline=response.wrist_outline,
        )

    def reset(
        self,
        environment_seed: int,
        *,
        stack_replay: dict[str, object] | None = None,
    ) -> PolicyObservation:
        request = {
            "operation": "reset",
            "environment_seed": int(environment_seed),
        }
        if stack_replay is not None:
            request["stack_replay"] = stack_replay
        response = self._request(request)
        self.facts = dict(response.metadata["facts"])
        return self._observation(response)

    def step(self, action: np.ndarray) -> PolicyStep:
        target = np.asarray(action, dtype=np.float64)
        if target.shape != ACTION_SHAPE or not np.isfinite(target).all():
            raise ValueError(f"invalid action {target}")
        response = self._request({"operation": "step", "action": target.tolist()})
        observation = self._observation(response)
        metadata = response.metadata
        self.facts = dict(metadata["facts"])
        return PolicyStep(
            observation=observation,
            commanded=np.asarray(metadata["commanded"], dtype=np.float64),
            achieved=observation.state,
            reward=float(metadata["reward"]),
            terminated=bool(metadata["terminated"]),
            truncated=bool(metadata["truncated"]),
            success=bool(metadata["success"]),
            facts=self.facts,
        )

    def set_command(self, source: str, target: str) -> dict[str, object]:
        response = self._request(
            {
                "operation": "set_command",
                "source": source,
                "target": target,
            }
        )
        self.facts = dict(response.metadata["facts"])
        return self.facts

    def set_target(self, cube: str) -> dict[str, object]:
        """Name the cube the next command moves; the scene's facts afterwards."""
        response = self._request({"operation": "set_target", "cube": cube})
        self.facts = dict(response.metadata["facts"])
        return self.facts

    def scene_without_cubes(self) -> tuple[np.ndarray, np.ndarray]:
        """The top and wrist picture of the scene as it stands, cubes and robot not drawn."""
        response = self._request({"operation": "scene_without_cubes"})
        if response.top is None or response.wrist is None:
            raise RuntimeError("SAPIEN worker sent no picture of the scene without cubes")
        return response.top, response.wrist

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._request({"operation": "close"})
            if self._process.stdin is not None:
                self._process.stdin.close()
            code = self._process.wait()
            if code != 0:
                raise RuntimeError(f"SAPIEN worker exited with status {code}")
        except Exception:
            self._terminate()
            raise
        finally:
            self._closed = True

    def _terminate(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
        self._process.wait()
        self._closed = True
