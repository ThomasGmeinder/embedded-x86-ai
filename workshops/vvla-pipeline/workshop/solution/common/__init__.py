# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""PROVIDED helpers for the Ryzen AI VVLA workshop.

The implementation is shared: this package points at the single copy in
``workshop/common`` (generic plumbing - image codecs, motion math, message
conversion, harness reporting - identical for ``project/`` and
``solution/``). Your work lives in ``models/``, ``ros/``, and
``integration/``.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[2] / "common")]
