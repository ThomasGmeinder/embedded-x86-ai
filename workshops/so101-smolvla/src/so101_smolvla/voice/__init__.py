"""The Gemma 4 voice path of this workshop: a sentence, typed or spoken, to the fields of a command.

- ``settings``: the one reader of ``configs/voice.json`` and of where the model files are.
- ``intents``: the commands, the instructions to the model, and what one answer holds.
- ``server``: ``llama-server`` with Gemma 4 E4B, started and stopped here.
- ``client``: one request to that server, typed text or a 16 kHz clip, grammar-held.
- ``speech``: a sentence in the synthetic voice of espeak-ng, inside a quiet clip.
- ``microphone``: the live session, Silero speech detection and rolling windows.
- ``commands``: ``VoiceCommands``, what the three-cube test and its tests use.
- ``examples``: the sentences of the tool-call test and what each must give.

The text of the instructions, the grammar, the two requests and the listening
parameters are those measured in SPEC.md; only where the code lives and how it
is run changed (SPEC.md section 5.1).
"""
