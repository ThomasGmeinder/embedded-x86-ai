# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""d1-Omni-2B intent parsing via llama.cpp running on ROCm: the decision model beside Gemma 4.

Liquid AI's d1-Omni-2B writes no text. It answers a named question about a
sentence or a clip by picking one of the answers it is offered, in one pass of
the model. ``llama-server`` serves it at ``POST /v1/systemone``. So a sentence
becomes a :class:`ParsedCommand` in one request or two:

1. which command (:func:`vla_pipeline.llm.intents.command_question`; the model
   gives a probability for each, and :func:`~vla_pipeline.llm.intents.chosen_command`
   takes a pick when its probability reaches ``min_pick_probability``), and
2. for ``pick_place`` only, which object, out of the names the detector knows
   (:func:`vla_pipeline.llm.intents.object_question`).

Each question is a pass of its own, and its time grows with the answers offered
(measured on a Ryzen AI Max+ 395, 2026-11-24: the command 0.07 s, the object
out of about 100 names 0.13 s), so the object is asked only when it is needed.

The interface is :class:`~vla_pipeline.llm.llama_intent.LlamaIntentParser`'s, so
the speech session takes either: ``on_call`` receives the command when the
answers are in. There are no words heard: the transcript of a clip is empty.

Independent test
----------------
    python -m vla_pipeline.llm.d1_intent --text "hey can you pick up the ball"
    python -m vla_pipeline.llm.d1_intent --selftest
"""

from __future__ import annotations

import argparse
import base64
import logging
import time

from vla_pipeline.llm.intents import (
    SELFTEST_CASES,
    Intent,
    ParsedCommand,
    chosen_command,
    command_from_answers,
    command_question,
    most_probable_thing,
    object_question,
)
from vla_pipeline.llm.llama_intent import OnCall
from vla_pipeline.llm.llama_server import REQUEST_TIMEOUT_S, LlamaServer
from vla_pipeline.utils.config import load_config

logger = logging.getLogger(__name__)

SECTION = "d1-omni-2b"


def known_object_names() -> list[str]:
    """Every name the detector's ``resolve_class`` knows: its classes, and the words said for them."""
    from vla_pipeline.vision.yolo_detect_npu import COCO_NAMES, SYNONYMS

    return list(dict.fromkeys([*COCO_NAMES, *SYNONYMS]))


def object_as_the_detector_names_it(name: str) -> str:
    """The detector's class for a chosen name ("foot ball" is "sports ball"); the name itself
    when the detector has no class for it ("pencil"), so the robot can say what it does not know."""
    from vla_pipeline.vision.yolo_detect_npu import resolve_class

    return resolve_class(name) or name


class D1IntentParser(LlamaServer):
    """d1-Omni-2B on a llama.cpp server (:class:`LlamaServer`): sentences and clips into :class:`ParsedCommand`."""

    def __init__(self, cfg: dict, *, section: str = SECTION):
        l = cfg[section]
        pick = l.get("min_pick_probability")
        self.min_pick_probability = None if pick is None else float(pick)
        self.min_confidence = float(l.get("min_confidence", 0.0))
        self.min_object_confidence = float(l.get("min_object_confidence", 0.0))
        self._command = {"command": command_question()}
        self._object = {"object": object_question(known_object_names())}
        self.last_call_s: float | None = None  # the last request: seconds until its command was decided
        self.last_total_s: float | None = None  # the same: a decision has no words that follow it
        super().__init__(cfg, section=section)  # finds the server running, or starts it
        try:
            self._require_a_decision_model()
        except BaseException:
            self.close()
            raise

    def _server_args(self) -> list[str]:
        """The model's own server arguments: a question is read in one batch, which must hold the clip."""
        batch = str(int(self._cfg.get("batch_size", 8192)))
        return ["-b", batch, "-ub", batch, *list(self._cfg.get("extra_args", []) or [])]

    def _require_a_decision_model(self) -> None:
        """Refuse a server that answers at this address with another kind of model."""
        r = self._http.get(f"{self.base}/v1/models", timeout=REQUEST_TIMEOUT_S)
        r.raise_for_status()
        served = (r.json().get("data") or [{}])[0]
        if "decisions" not in ((served.get("architecture") or {}).get("output_modalities") or []):
            raise RuntimeError(
                f"the llama-server at {self.base} serves {served.get('id')!r}, which is not a decision model; "
                f"{self.section} needs d1-Omni-2B (scripts/ensure_voice.sh) and a llama-server that knows it."
            )

    # ------------------------------------------------------------------
    # Intent parsing
    # ------------------------------------------------------------------

    def _ask(self, state, files: list[str] | None, questions: dict) -> dict:
        """One request: the model's answers to ``questions`` about the sentence or the clip."""
        payload = {"state": state, "questions": questions}
        if files:
            payload["files"] = files
        r = self._http.post(f"{self.base}/v1/systemone", json=payload, timeout=REQUEST_TIMEOUT_S)
        if r.status_code >= 400:
            logger.error("llama-server rejected the request: %s", r.text[:500])
        r.raise_for_status()
        return r.json()["answers"]

    def _decide(self, state, files: list[str] | None, transcript: str, on_call: OnCall | None) -> ParsedCommand:
        """The command for one sentence or clip; ``on_call`` gets it when it is decided."""
        self.last_call_s = self.last_total_s = None
        t0 = time.perf_counter()
        limits = {"min_pick_probability": self.min_pick_probability, "min_confidence": self.min_confidence}
        command = self._ask(state, files, self._command)["command"]
        chosen = None
        if chosen_command(command, **limits) is Intent.PICK_PLACE:
            # The object as the detector names it, by the probabilities of all its names together.
            chosen = most_probable_thing(self._ask(state, files, self._object)["object"], object_as_the_detector_names_it)
        cmd = command_from_answers(command, chosen, min_object_confidence=self.min_object_confidence, **limits)
        cmd.transcript = transcript
        call_s = self.last_call_s = self.last_total_s = time.perf_counter() - t0
        logger.info(
            "Intent %r %s ← %r (decided after %.0f ms; the model's choice %r, confidence %.2f, pick probability %.2f%s)",
            cmd.intent.value,
            cmd.params or "",
            transcript,
            call_s * 1e3,
            command.get("choice"),
            float(command.get("confidence", 0.0)),
            float((command.get("probabilities") or {}).get(Intent.PICK_PLACE.value, 0.0)),
            f", object {chosen.get('choice')!r} {float(chosen.get('confidence', 0.0)):.2f}" if chosen else "",
        )
        if on_call is not None:
            on_call(cmd, call_s)
        return cmd

    def parse(self, transcript: str, on_call: OnCall | None = None) -> ParsedCommand:
        """Map one typed sentence to a ParsedCommand."""
        if not transcript or not transcript.strip():
            return ParsedCommand(Intent.UNKNOWN, transcript=transcript or "")
        return self._decide(transcript.strip(), None, transcript, on_call)

    def parse_audio(self, wav: bytes, on_call: OnCall | None = None) -> ParsedCommand:
        """Map one WAV clip to a ParsedCommand. The model writes no words: the transcript is empty."""
        clip = "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")
        return self._decide(None, [clip], "", on_call)

    def warm_up(self) -> float:
        """One request whose answer is thrown away; the seconds it took."""
        self.parse("hello")
        return float(self.last_total_s)


# =============================================================================
# Standalone component test
# =============================================================================


def main() -> None:
    """CLI entry point: parse one utterance, or run the selftest suite."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="d1-Omni-2B intent parser (llama.cpp + ROCm) - component test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--text", default=None, help="single utterance to parse")
    parser.add_argument("--selftest", action="store_true", help="run the built-in utterance suite")
    args = parser.parse_args()

    llm = D1IntentParser(load_config(args.config))
    try:
        if args.selftest:
            passed = 0
            for text, expected, obj in SELFTEST_CASES:
                got = llm.parse(text)
                # The object is right when it is the detector's class for the word said.
                ok = got.intent == expected and (
                    obj is None or got.object_name == object_as_the_detector_names_it(obj))
                passed += ok
                print(
                    f"  [{'PASS' if ok else 'FAIL'}] {text!r} -> {got.intent.value} "
                    f"{got.params} (expected {expected.value} {obj or ''}) {llm.last_call_s * 1e3:.0f} ms"
                )
            print(f"\n{passed}/{len(SELFTEST_CASES)} cases passed")
            raise SystemExit(0 if passed == len(SELFTEST_CASES) else 1)
        cmd = llm.parse(args.text or "pick up the ball")
        print(cmd.intent.value, cmd.params)
    finally:
        llm.close()


if __name__ == "__main__":
    main()
