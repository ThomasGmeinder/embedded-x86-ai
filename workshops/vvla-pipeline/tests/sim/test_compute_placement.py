# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The digital twin notebook's compute placement table follows the config files.

``common.compute_placement.placement`` builds it from ``voice_intent`` and the
voice model's own section in the pipeline's config, and from the two YOLO
models' ``device`` in the workshop's config. The notebook holds no table of
its own.
"""

from __future__ import annotations

import copy

import pytest

from notebook_harness import NOTEBOOKS, PIPE_ROOT

pytestmark = [pytest.mark.sim]


def _cells(table) -> list[list[str]]:
    """The table's Markdown rows as lists of cell texts, without the header's two lines."""
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in str(table).splitlines()[2:]]


def test_the_table_names_the_voice_model_the_config_selects(workshop_cfg, pipeline_cfg):
    from common.compute_placement import placement
    from vla_pipeline.llm.voice import VOICE_INTENTS

    for intent in VOICE_INTENTS:
        chosen = {**copy.deepcopy(pipeline_cfg), "voice_intent": intent}
        rows = _cells(placement(workshop_cfg, chosen))
        voice = rows[1]  # after Silero, under the same task
        assert voice[1] == chosen[intent]["name"] and voice[2] == chosen[intent]["does"], (intent, voice)
        assert voice[3] == "iGPU", voice
        other = [chosen[name]["name"] for name in VOICE_INTENTS if name != intent]
        assert not any(name in str(placement(workshop_cfg, chosen)) for name in other), (intent, other)


def test_a_voice_model_without_gpu_layers_is_listed_on_the_cpu(workshop_cfg, pipeline_cfg):
    from common.compute_placement import placement

    chosen = copy.deepcopy(pipeline_cfg)
    chosen[chosen["voice_intent"]]["n_gpu_layers"] = 0
    assert _cells(placement(workshop_cfg, chosen))[1][3] == "CPU"


@pytest.mark.parametrize("model, row", [("yolo_pose", 2), ("yolo_detect", 4)])
def test_a_yolo_model_is_listed_where_its_device_setting_puts_it(workshop_cfg, pipeline_cfg, model, row):
    from common.compute_placement import placement

    cfg = copy.deepcopy(workshop_cfg)
    for device, runs_on, runtime in (("npu", "NPU", "VitisAI EP"), ("cpu", "CPU", "CPU EP")):
        cfg[model]["device"] = device
        cells = _cells(placement(cfg, pipeline_cfg))[row]
        assert cells[3] == runs_on and runtime in cells[4], (model, device, cells)


def test_a_task_is_named_once_and_the_default_reads_the_pipelines_own_config(workshop_cfg):
    from common.compute_placement import placement
    from vla_pipeline.utils.config import load_config

    on_disk = load_config()  # the file itself: the fixture's voice model can be a test option's
    rows = _cells(placement(workshop_cfg))  # no pipeline config given: config/pipeline.yaml
    assert [row[0] for row in rows] == [
        "**Voice control**", "", "**Gestures**", "", "**Object localization**", "**Digital twin**"]
    assert rows[1][1] == on_disk[on_disk["voice_intent"]]["name"]


def test_the_notebook_builds_its_table_and_holds_none_of_its_own():
    import nbformat

    nb = nbformat.read(NOTEBOOKS / "03_integration" / "02_digital_twin.ipynb", as_version=4)
    code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")
    assert "placement(cfg)" in code, "the notebook does not build the compute placement table"
    assert "<table" not in text and "| Runs on |" not in text, "the notebook holds a hand-written placement table"


def test_the_readme_names_the_default_voice_model_in_its_table():
    """The README's table is written by hand: its voice row must name the model the committed
    config selects."""
    from vla_pipeline.utils.config import load_config

    on_disk = load_config()
    name = on_disk[on_disk["voice_intent"]]["name"]
    table = (PIPE_ROOT / "README.md").read_text().split("## Compute placement", 1)[1].split("</table>", 1)[0]
    assert name in table, f"the README's compute placement table does not name {name}"
