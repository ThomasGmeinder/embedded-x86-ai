"""Gemma 4 E4B (the voice client in voice/) in a process of its own: a sentence in, the command's fields out.

usage: scripts/run_voice_env.sh python src/so101_smolvla/voice_worker.py <descriptor to read requests from> <descriptor to write replies to>
       scripts/run_voice_env.sh python src/so101_smolvla/voice_worker.py --check

Started by so101_smolvla.voice_commands.VoiceCommands through the voice
workshop's environment launcher, which puts this process into that
workshop's Python and environment, the ones the voice notebook's kernel runs
in. This file imports nothing of the workshop's package: it is not installed
there. The client's files are voice/ (notebook_tools, vla_pipeline), put first
on sys.path here.

Environment: VOICE_PORT, the port to start the Gemma server on (not the
client's own, where the user may run a server of their own); VOICE_LOG_DIR,
where the server's log goes. The server is started by notebook_tools'
start_server, as the notebook starts it, and stopped on ``close``.

The requests:

  parse    {"text": sentence}: the sentence typed, through the client's parse()
  speak    {"text": sentence}: the sentence in the synthetic voice (espeak-ng),
           in a quiet clip of the voice session's window, through parse_audio()
  close    stop the server and end the process

A reply holds the fields Gemma gave (command, object, place, heard) and the
seconds the request took. A request that fails is answered with
``{"error": ...}``. ``--check`` only loads the client's settings and says
what is missing, for a test that decides whether it can run.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import traceback
from multiprocessing.connection import Connection
from pathlib import Path

VOICE = Path(__file__).resolve().parents[2] / "voice"
sys.path.insert(0, str(VOICE))

import notebook_tools as tools  # noqa: E402
from vla_pipeline.llm.intents import INTENT_SYSTEM_PROMPT  # noqa: E402

ROUTE = "gemma4-e4b-it voice client, typed and spoken"
TYPED = "typed"
SPOKEN = "synthetic voice"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fields(command, sentence: str, how: str, seconds: float) -> dict:
    return {
        "sentence": sentence,
        "how": how,
        "command": command.intent.value,
        "object": command.params.get("object", ""),
        "place": command.params.get("place", ""),
        "heard": command.transcript,
        "seconds": seconds,
    }


def serve(requests: Connection, replies: Connection) -> None:
    port = int(os.environ["VOICE_PORT"])
    log_dir = os.environ["VOICE_LOG_DIR"]
    parser, base, section = tools.start_server(log_dir, {"port": port, **tools.SERVER_CHANGES}, [])
    try:
        cfg = tools.load_voice_config()
        rate, window_s = int(cfg["audio"]["sample_rate"]), float(section["window_s"])
        espeak = None
        replies.send(
            {
                "ready": True,
                "route": ROUTE,
                "base": base,
                "settings": {
                    key: section.get(key)
                    for key in (
                        "port", "n_gpu_layers", "ctx_size", "parallel", "temperature",
                        "no_warmup", "hsa_override_gfx_version", "window_s", "extra_args",
                    )
                },
                "gfx_override_in_environment": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
                "model_gguf": os.path.basename(section["model_gguf"]),
                "mmproj": os.path.basename(section["mmproj"]),
                "branch_commit": tools.BRANCH_COMMIT,
                "grammar_sha256": _sha256(parser.grammar),
                "instructions_sha256": _sha256(INTENT_SYSTEM_PROMPT),
                "server_log": str(Path(log_dir) / "llama-server.log"),
                "synthetic_voice": {
                    "library": tools.ESPEAK_LIBRARY,
                    "voice": tools.ESPEAK_VOICE.decode(),
                    "words_per_minute": tools.WORDS_PER_MINUTE,
                    "peak": tools.PEAK,
                    "lead_s": tools.LEAD_S,
                    "sample_rate": rate,
                    "clip_s": window_s,
                },
            }
        )
        while True:
            try:
                request = requests.recv()
            except EOFError:
                break
            try:
                operation = request["operation"]
                if operation == "close":
                    replies.send({"closed": True})
                    break
                sentence = str(request["text"])
                if operation == "parse":
                    began = time.perf_counter()
                    command = parser.parse(sentence)
                    reply = _fields(command, sentence, TYPED, time.perf_counter() - began)
                elif operation == "speak":
                    if espeak is None:
                        espeak = tools.open_espeak()
                    sound = tools.speak(espeak[0], espeak[1], sentence, rate)
                    wav = tools.clip(sound, rate, window_s)
                    began = time.perf_counter()
                    command = parser.parse_audio(wav)
                    reply = _fields(command, sentence, SPOKEN, time.perf_counter() - began)
                    reply["spoken_s"] = len(sound) / rate
                else:
                    raise ValueError(f"unknown operation {operation!r}")
            except Exception:
                reply = {"error": traceback.format_exc()}
            replies.send(reply)
    finally:
        parser.close()


def check() -> int:
    """Whether the voice workshop's files the client needs are there: 0, or 1 with the reason."""
    try:
        tools.load_voice_config()
    except RuntimeError as exc:
        print(exc)
        return 1
    print("ready")
    return 0


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments == ["--check"]:
        raise SystemExit(check())
    if len(arguments) != 2:
        raise SystemExit(__doc__)
    serve(
        Connection(int(arguments[0]), writable=False),
        Connection(int(arguments[1]), readable=False),
    )


if __name__ == "__main__":
    main()
