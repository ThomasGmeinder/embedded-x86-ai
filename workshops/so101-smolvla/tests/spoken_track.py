"""A speaker who does not stop talking, as a microphone delivers it: for the listening test.

The sound is made as this machine's quiet built-in microphone delivers it: samples around +1028,
speech at a tenth of full scale. The synthetic voice of the workshop speaks
(``so101_smolvla.voice.speech.SyntheticVoice``: espeak-ng, each sentence in a process of its own, so
the same sentence is the same sound every time). From ~/GitHub/voice_intent (``synthetic_voice.py`` at
its commit 4161915).
"""

from __future__ import annotations

import time

import numpy as np

from so101_smolvla.voice.speech import SyntheticVoice

OFFSET = 1028  # the constant this machine's built-in microphone adds to its samples
SPEECH_PEAK = 0.1  # of full scale: quiet speech


def monologue(voice: SyntheticVoice, sentences, rate: int, gap_s: float = 0.25, lead_s: float = 1.0, tail_s: float = 8.0):
    """One speaker saying the sentences one after the other with ``gap_s`` between them, shorter
    than any end-of-speech silence; then ``tail_s`` of quiet. Returns (the microphone's bytes,
    the second at which each sentence's last word ends)."""
    pieces, ends, at = [np.zeros(int(lead_s * rate))], [], int(lead_s * rate)
    for sentence in sentences:
        sound = voice.say(sentence, rate).astype(np.float64) * (SPEECH_PEAK / voice.peak)
        pieces += [sound, np.zeros(int(gap_s * rate))]
        at += len(sound)
        ends.append(at / rate)
        at += int(gap_s * rate)
    track = np.concatenate([*pieces, np.zeros(int(tail_s * rate))])
    return np.clip(np.round(track + OFFSET), -32768, 32767).astype("<i2").tobytes(), ends


def stand_in_microphone(pcm: bytes, rate: int, on_end):
    """A class that stands in for ``sounddevice.RawInputStream``: ``pcm`` in the frames asked for,
    at the pace of the sample clock, as a device delivers it. ``on_end()`` is called once, when
    the track's last frame has been handed out; after it the microphone delivers its offset
    alone. The class's ``began`` is the monotonic time of the first frame."""

    class Microphone:
        began = None

        def __init__(self, **settings) -> None:
            if settings["samplerate"] != rate:
                raise ValueError(f"the stand-in microphone delivers {rate} Hz")
            self.position, self.ended = 0, False

        def __enter__(self):
            Microphone.began = time.monotonic()
            return self

        def __exit__(self, *_) -> bool:
            return False

        def read(self, samples: int):
            self.position += samples
            rest = Microphone.began + self.position / rate - time.monotonic()
            if rest > 0:
                time.sleep(rest)  # the device's own clock: a read returns when its samples have been captured
            chunk = pcm[(self.position - samples) * 2:self.position * 2]
            if len(chunk) < samples * 2:
                chunk += np.full(samples - len(chunk) // 2, OFFSET, "<i2").tobytes()
                if not self.ended:
                    self.ended = True
                    on_end()
            return chunk, False

    return Microphone
