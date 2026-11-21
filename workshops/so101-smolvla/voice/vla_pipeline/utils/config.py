# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Configuration loading shared by every pipeline component.

Every component file accepts ``--config`` and resolves paths through this
module so each one can be run standalone with the same YAML.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels above this file (vla_pipeline/utils/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "pipeline.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the pipeline YAML config (defaults to ``config/pipeline.yaml``)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_like: str | os.PathLike) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(path_like)
    return p if p.is_absolute() else REPO_ROOT / p


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create cache/log/model directories referenced by the config."""
    for key in ("models_dir", "cache_dir", "log_dir"):
        resolve(cfg["system"][key]).mkdir(parents=True, exist_ok=True)
