# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Workshop config loading (PROVIDED).

The workshop project carries its own configuration (``config/workshop.yaml``)
with the same values the real pipeline uses. Every path in the YAML resolves
relative to the PROJECT root - the ``project/`` or ``solution/`` tree that is
active on ``sys.path`` or as the working directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml


def _find_project_root() -> Path:
    """The active tree (project/ or solution/): the first place on sys.path -
    or the working directory - that carries ``config/workshop.yaml``.

    ``common/`` is shared between the two trees, so the package's location
    can't tell them apart; whichever tree was put on ``sys.path`` (the
    notebooks) or is the working directory (``python -m ...``, ``app.py``)
    is the one whose config, models, and cache apply.
    """
    for cand in (Path.cwd(), *map(Path, sys.path)):
        try:
            if (cand / "config" / "workshop.yaml").is_file():
                return cand.resolve()
        except OSError:
            continue
    raise RuntimeError(
        "cannot locate the project root - no config/workshop.yaml found on "
        "sys.path or in the working directory"
    )


PROJECT_ROOT = _find_project_root()
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "workshop.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the workshop YAML config (defaults to ``config/workshop.yaml``)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_like: str | os.PathLike) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path_like)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create cache/log/model directories referenced by the config."""
    for key in ("models_dir", "cache_dir", "log_dir"):
        resolve(cfg["system"][key]).mkdir(parents=True, exist_ok=True)
