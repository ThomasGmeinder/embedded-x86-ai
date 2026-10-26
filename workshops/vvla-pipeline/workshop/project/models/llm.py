# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""TODOs 1.5-1.6: Llama-3.2-3B on the iGPU, grammar-constrained.

The model runs in ``llama-server`` (llama.cpp on ROCm) and is queried over
its OpenAI-compatible HTTP API. A GBNF grammar (``config/intent.gbnf``)
constrains generation so the model can only emit
``{"command": "<name>", "object": "<words>"}``. The pipeline's
:class:`~vla_pipeline.llm.llama_intent.LlamaIntentParser` owns the server
process; these are the launch recipe and one request.

Reference: ``vla_pipeline/llm/llama_intent.py`` (same names and parameters).
Notebook: ``notebooks/01_ai_models/03_igpu_llama.ipynb``
Self-test: ``python -m vvla_workshop.selftest_models``
"""

from __future__ import annotations

import json
import logging
import os

import requests

from vla_pipeline.llm.intents import INTENT_SYSTEM_PROMPT, Intent, ParsedCommand
from vla_pipeline.utils.config import resolve

logger = logging.getLogger(__name__)


def llama_server_command(cfg: dict):
    """Build the ``llama-server`` launch: ``(cmd list, env dict)``.

    Reads ``llm:`` from the config. The command must include the model, host,
    port, ``--n-gpu-layers``, ``--ctx-size``, ``--parallel``, and
    ``--no-warmup`` when configured; the environment must carry
    ``HSA_OVERRIDE_GFX_VERSION`` when the config sets one (don't override a
    value the user already exported).
    """
    l = cfg["llm"]
    server_bin = resolve(l["server_bin"])
    model = resolve(l["model_gguf"])
    if not server_bin.exists():
        raise FileNotFoundError(f"{server_bin} not found - run ./bootstrap.sh (llama.cpp build step).")
    if not model.exists():
        raise FileNotFoundError(f"{model} not found - run ./bootstrap.sh (model download step).")
    # >>> TODO 1.5: llama_server_command - notebooks/01_ai_models/03_igpu_llama.ipynb
    raise NotImplementedError(
        "TODO 1.5 (llama_server_command): implement me - see notebooks/01_ai_models/03_igpu_llama.ipynb")
    # <<< TODO 1.5


def parse_intent(transcript: str, base_url: str, grammar: str, http=None, temperature: float = 0.0) -> ParsedCommand:
    """One transcript → one :class:`ParsedCommand`, grammar-constrained.

    POST an OpenAI-style chat completion to ``{base_url}/v1/chat/completions``
    with the system prompt (:data:`INTENT_SYSTEM_PROMPT`), the transcript as
    the user message, ``temperature`` (0 = deterministic routing), a small
    ``max_tokens``, and - the crucial part - the GBNF ``grammar``, which is
    what makes the output JSON *guaranteed parseable*. An empty transcript
    maps to UNKNOWN without any HTTP round trip.
    """
    http = http or requests
    # >>> TODO 1.6: parse_intent - notebooks/01_ai_models/03_igpu_llama.ipynb
    raise NotImplementedError(
        "TODO 1.6 (parse_intent): implement me - see notebooks/01_ai_models/03_igpu_llama.ipynb")
    # <<< TODO 1.6
