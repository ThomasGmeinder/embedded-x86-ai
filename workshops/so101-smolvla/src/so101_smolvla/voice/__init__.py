"""The Gemma 4 voice path of this workshop: a sentence, typed or spoken, to the fields of a command.

- ``settings``: the one reader of ``configs/voice.toml``; the backend Gemma runs on comes from
  ``configs/models.toml`` through ``so101_smolvla.models``; and where the model files are.
- ``intents``: the commands, the instructions to the model, the schema of one answer, and what
  one answer holds.
- ``server``: the Gemma 4 E4B server, ``llama-server`` on the integrated graphics or FastFlowLM
  on the NPU, started and stopped here.
- ``client``: one request to that server, typed text or a 16 kHz clip, held to the grammar
  (llama-server) or the schema (FastFlowLM).
- ``speech``: a sentence in the synthetic voice of espeak-ng, inside a quiet clip.
- ``microphone``: the live session, Silero speech detection and rolling windows.
- ``commands``: ``VoiceCommands``, what the three-cube test and its tests use.
- ``examples``: the sentences of the tool-call test and what each must give.

The text of the instructions, the grammar, the two requests and the listening
parameters are those measured in SPEC.md; where the code lives, how it is run
and which backend runs the model are this workshop's (SPEC.md section 5.1).
"""
