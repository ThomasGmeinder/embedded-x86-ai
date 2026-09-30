# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

#!/usr/bin/env python3
"""
Test script for Edge-TTS voices - find working cute voices for the robot
"""

import asyncio
import tempfile
import os


def try_voice(voice_name, text="Hello! I'm a cute robot!", description=""):
    """Try a single Edge-TTS voice (utility helper, not a pytest test)."""
    print(f"🎵 Testing {voice_name} - {description}")

    try:

        async def speak_test():
            # Lazy imports so pytest can import this module without optional deps.
            import edge_tts
            import sounddevice as sd
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            communicate = edge_tts.Communicate(text, voice_name)
            await communicate.save(tmp_path)

            # Read and play
            data, sr = sf.read(tmp_path, dtype="float32")
            audio = data

            sd.play(audio, samplerate=sr)
            sd.wait()

            os.remove(tmp_path)
            return True

        asyncio.run(speak_test())
        print(f"  ✅ {voice_name} WORKS!")
        return True

    except Exception as e:
        print(f"  ❌ {voice_name} FAILED: {str(e)[:50]}...")
        return False


def main():
    """Test various cute/youthful voices for Edge-TTS"""
    print("🎭 Testing Edge-TTS Voices for Cute Robot")
    print("=" * 60)

    # Test the working Chinese voice first
    working_voices = []
    try_voice(
        "zh-CN-XiaoxiaoNeural", "你好！我是可爱的机器人！", "Default Chinese (working)"
    )
    working_voices.append(("zh-CN-XiaoxiaoNeural", "Default Chinese"))

    # Test cute/youthful voices from the full Edge-TTS list (prioritizing cartoon/cute ones)
    cute_voice_candidates = [
        # CARTOON VOICES - Perfect for cute robots!
        ("en-US-AnaNeural", "CARTOON - Cute, adorable"),
        ("zh-CN-XiaoyiNeural", "CARTOON - Lively, cute Chinese"),
        ("zh-CN-YunxiaNeural", "CARTOON - Cute male Chinese"),
        ("zh-CN-liaoning-XiaobeiNeural", "DIALECT - Humorous, cute Chinese"),
        # Youthful & Childlike English Voices
        ("en-US-AriaNeural", "Youthful, friendly"),
        ("en-US-JennyNeural", "Friendly, youthful"),
        ("en-GB-LibbyNeural", "British, youthful"),
        ("en-CA-ClaraNeural", "Canadian, clear, youthful"),
        ("en-AU-NatashaNeural", "Australian, friendly"),
        ("en-NZ-MollyNeural", "New Zealand, youthful"),
        ("en-IE-EmilyNeural", "Irish, friendly"),
        # Playful & Energetic
        ("en-US-BrianNeural", "Approachable, casual, sincere"),
        ("en-US-EmmaNeural", "Cheerful, clear, conversational"),
        ("en-US-AvaNeural", "Expressive, caring, pleasant"),
        ("en-GB-SoniaNeural", "British, young"),
        ("en-AU-CarlyNeural", "Australian, cheerful"),
        # Gentle & Sweet
        ("en-IN-NeerjaNeural", "Indian, clear, gentle"),
        ("en-GB-MaisieNeural", "British, warm"),
        ("en-US-MichelleNeural", "Friendly, pleasant"),
        ("en-US-AriaRUS", "Young, cute"),
        ("en-US-ZiraRUS", "Clear, gentle"),
        # More Cute Options
        ("en-US-BenjaminNeural", "Young male, friendly"),
        ("en-CA-LiamNeural", "Canadian boy, cute"),
        ("en-GB-RyanNeural", "British, cheerful"),
        ("en-US-GuyNeural", "Passionate, engaging"),
        ("en-US-RogerNeural", "Lively, enthusiastic"),
        # Unique Regional Voices
        ("en-PH-RosaNeural", "Filipino, friendly"),
        ("en-SG-LunaNeural", "Singapore, gentle"),
        ("en-KE-AsiliaNeural", "Kenyan, warm"),
        ("en-NG-EzinneNeural", "Nigerian, friendly"),
    ]

    print("\n🧸 Testing Cute/Youthful English Voices:")
    print("-" * 50)

    for voice_name, description in cute_voice_candidates:
        if try_voice(voice_name, "Hello! I'm a cute robot assistant!", description):
            working_voices.append((voice_name, description))

    # Test some more voices if user wants
    print("\n📊 Summary of Working Voices:")
    print("=" * 50)
    for voice, desc in working_voices:
        print(f"✅ {voice} - {desc}")

    print(f"\n🎯 Found {len(working_voices)} working voices")

    # Suggest configuration
    if len(working_voices) > 1:
        print("\n💡 Suggested Cute Voice Configuration:")
        print("emotion_voices = {")
        emotions = ["positive", "negative", "question", "activity", "neutral"]
        for i, emotion in enumerate(emotions):
            voice_idx = min(i, len(working_voices) - 1)
            voice_name = working_voices[voice_idx][0]
            print(f'    "{emotion}": "{voice_name}",')
        print("}")


if __name__ == "__main__":
    main()
