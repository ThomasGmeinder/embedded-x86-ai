# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Small harness to measure recording + ASR durations for fixed vs VAD capture.

Usage:
    python -m utils.latency_harness

This script will:
- Initialize the FasterWhisperASREngine (if available)
- Record a fixed 4s clip and transcribe it, printing timings
- Record using the VAD recorder and transcribe it, printing timings

Note: requires `sounddevice` and `soundfile` installed.
"""

import time
import os
import sys

# Make imports work whether the script is run as a module or as a script
root = os.path.dirname(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

try:
    from utils.asr import FasterWhisperASREngine
except Exception:
    try:
        from asr import FasterWhisperASREngine
    except Exception:
        raise


def measure_fixed(engine: FasterWhisperASREngine):
    print("\n--- Fixed 4s recording ---")
    t0 = time.time()
    wav = engine._record_temp_wav(4.0)
    t_rec = time.time()
    t_asr_start = t_rec
    text = engine.transcribe_file(wav)
    t_asr_end = time.time()
    print(f"Record time: {t_rec - t0:.3f}s, ASR time: {t_asr_end - t_asr_start:.3f}s")
    print("Transcription:", text)


def measure_vad(engine: FasterWhisperASREngine):
    print("\n--- VAD recording (max 4s) ---")
    t0 = time.time()
    wav = engine._record_temp_wav_vad(max_duration=4.0)
    t_rec = time.time()
    t_asr_start = t_rec
    text = engine.transcribe_file(wav)
    t_asr_end = time.time()
    print(f"Record time: {t_rec - t0:.3f}s, ASR time: {t_asr_end - t_asr_start:.3f}s")
    print("Transcription:", text)


def main():
    try:
        engine = FasterWhisperASREngine(model_name="small")
    except Exception as e:
        print("Cannot initialize ASR engine:", e)
        return

    # Warm-up (optional)
    print("Warming up ASR model...")
    try:
        # quick empty pass to make sure model loaded
        _ = engine.model
    except Exception:
        pass

    measure_fixed(engine)
    measure_vad(engine)


if __name__ == "__main__":
    main()
