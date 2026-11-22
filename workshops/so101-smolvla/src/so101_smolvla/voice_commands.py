"""Gemma 4 asked for the fields of a typed or spoken sentence, from this workshop's environment.

The voice client (voice/, a copy of branch gemma4-e4b-it_voice_intent with
this workshop's changes) runs in the voice workshop's Python, not in this
one. So Gemma is asked through a worker process (voice_worker.py), started
through scripts/run_voice_env.sh, which gives it that workshop's environment
the way the voice notebook's kernel gets it. The worker starts the Gemma
server on PORT and stops it when closed.

What comes back for a sentence are the client's fields as Gemma gave them:
``command``, ``object``, ``place`` and ``heard``. Which cube they name is
decided elsewhere (cube_commands.choose_cube).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass
from multiprocessing import Pipe
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "run_voice_env.sh"
WORKER = Path(__file__).with_name("voice_worker.py")
# Not the client's own port (voice/config/pipeline.yaml): a server the user
# runs there with settings of their own is left alone.
PORT = 8083


@dataclass(frozen=True)
class CommandFields:
    """What Gemma made of one sentence, and how the sentence reached it."""

    sentence: str
    how: str  # "typed" or "synthetic voice"
    command: str
    object: str
    place: str
    heard: str
    seconds: float
    spoken_s: float | None = None  # the length of the spoken sentence

    def as_record(self) -> dict:
        return asdict(self)


def voice_workshop_ready() -> tuple[bool, str]:
    """Whether the worker can run: the voice workshop's Python, its scripts and the model files."""
    if not (LAUNCHER.is_file() and os.access(LAUNCHER, os.X_OK)):
        return False, f"missing launcher {LAUNCHER}"
    ran = subprocess.run(
        [str(LAUNCHER), "python", str(WORKER), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return ran.returncode == 0, (ran.stdout + ran.stderr).strip()


class VoiceCommands:
    """Gemma in its worker process, with the server it starts."""

    def __init__(self, log_dir: Path, port: int = PORT) -> None:
        request_end, self._requests = Pipe(duplex=False)
        self._replies, reply_end = Pipe(duplex=False)
        env = os.environ.copy()
        env["VOICE_PORT"] = str(port)
        env["VOICE_LOG_DIR"] = str(Path(log_dir).resolve())
        # A missing voice workshop or model file ends the launcher or the
        # worker with a message that names it.
        self._process = subprocess.Popen(
            [
                str(LAUNCHER),
                "python",
                str(WORKER),
                str(request_end.fileno()),
                str(reply_end.fileno()),
            ],
            cwd=ROOT,
            env=env,
            pass_fds=(request_end.fileno(), reply_end.fileno()),
        )
        request_end.close()
        reply_end.close()
        self._closed = False
        try:
            hello = self._reply()
        except Exception:
            self._terminate()
            raise
        self.provenance = {key: value for key, value in hello.items() if key != "ready"}

    def _reply(self) -> dict:
        try:
            reply = self._replies.recv()
        except EOFError:
            code = self._process.wait()
            self._closed = True
            raise RuntimeError(f"voice worker exited with status {code}") from None
        if "error" in reply:
            raise RuntimeError(f"voice worker: {reply['error']}")
        return reply

    def _ask(self, request: dict) -> dict:
        if self._closed:
            raise RuntimeError("voice worker is closed")
        try:
            self._requests.send(request)
        except BrokenPipeError:
            pass
        return self._reply()

    def parse(self, sentence: str) -> CommandFields:
        """The sentence as typed text."""
        return CommandFields(**self._ask({"operation": "parse", "text": sentence}))

    def parse_spoken(self, sentence: str) -> CommandFields:
        """The sentence said by the synthetic voice, as one clip."""
        return CommandFields(**self._ask({"operation": "speak", "text": sentence}))

    def ask(self, sentence: str, spoken: bool = False) -> CommandFields:
        return self.parse_spoken(sentence) if spoken else self.parse(sentence)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._ask({"operation": "close"})
            code = self._process.wait()
            if code != 0:
                raise RuntimeError(f"voice worker exited with status {code}")
        except Exception:
            self._terminate()
            raise
        finally:
            self._closed = True
            self._requests.close()
            self._replies.close()

    def _terminate(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
        self._process.wait()
        self._closed = True
