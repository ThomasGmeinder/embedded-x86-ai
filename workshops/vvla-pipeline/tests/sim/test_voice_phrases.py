# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Stop and home phrases, and the rolling window that feeds Gemma.

No microphone and no model download. "stop robot" is the only stop phrase.
"go home" is the only home phrase.
"""

from __future__ import annotations

import pytest

from vla_pipeline.audio.gemma_listen import FRAME_MS, CommandFilter, HopWindows
from vla_pipeline.llm.intents import command_from_json, keyword_parse
from vla_pipeline.main import is_stop
