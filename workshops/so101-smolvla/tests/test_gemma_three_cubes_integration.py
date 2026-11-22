"""One real layout commanded by sentences: Gemma 4, the real SAM 3 and the real scene, the arm held.

Needs the voice path (llama-server and the Gemma files, scripts/setup_voice.sh),
the SAM 3 reference's environment and model files, and the SAPIEN backend;
skipped while one is not set up.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from so101_smolvla import sapien_rollout, three_cube_rollout
from so101_smolvla.cube_commands import TEMPLATES, call_text
from so101_smolvla.sam3_outlines import CAMERAS
from so101_smolvla.sapien_config import SAPIEN_PYTHON, SAPIEN_SOURCE, THREE_CUBE_TASK
from so101_smolvla.three_cube_rollout import COMMANDS_FROM, OUTLINE_SOURCES, draw_order
from so101_smolvla.voice.commands import voice_ready

WORKSHOP = Path(__file__).resolve().parents[1]
SETTINGS = json.loads((WORKSHOP / "configs" / "sam3_reference.json").read_text())
SAM3_READY = (WORKSHOP / ".venv-sam3" / "bin" / "python").is_file() and (
    Path(os.path.expanduser(SETTINGS["compiled_dir"])) / "ARTIFACT_MANIFEST.json"
).is_file()
VOICE_READY, VOICE_REASON = voice_ready()

pytestmark = [
    pytest.mark.skipif(
        not SAPIEN_PYTHON.is_file() or not (SAPIEN_SOURCE / ".git").is_dir(),
        reason="run scripts/setup_sapien_env.sh to install the real SAPIEN backend",
    ),
    pytest.mark.skipif(
        not SAM3_READY,
        reason="run scripts/setup_sam3_env.sh and build the reference's model files",
    ),
    pytest.mark.skipif(not VOICE_READY, reason=f"voice: {VOICE_REASON}"),
]


def test_real_layout_commanded_by_typed_sentences_through_gemma_sam3_and_the_scene(
    tmp_path, monkeypatch, capsys
):
    policy = SimpleNamespace(config=SimpleNamespace(n_action_steps=2), reset=lambda: None)
    shown = []

    def hold_the_arm(policy, pre, post, observation, task_text):
        shown.append(observation)
        return observation.state

    monkeypatch.setattr(
        three_cube_rollout, "_load_policy", lambda: (tmp_path / "revision", policy, None, None)
    )
    monkeypatch.setattr(three_cube_rollout, "_action", hold_the_arm)
    destination = tmp_path / "batch"
    sapien_rollout.main(
        [
            "--task", THREE_CUBE_TASK,
            "--pictures", "others-painted",
            "--outlines", "sam3",
            "--sentences", "built-in",
            "--steps", "3",
            "--environment-seed-base", "7",
            "--output", str(destination),
        ]
    )
    out = capsys.readouterr().out
    assert "succeeded 0 of 1" in out
    print(out)

    rows = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    assert len(rows) == 1 and len(shown) == 9
    row = rows[0]
    expected = list(draw_order(7))
    assert row["order"] == expected
    assert row["order_given_by"] == COMMANDS_FROM["typed"] == "typed sentences through Gemma 4"
    assert row["outline_source"] == OUTLINE_SOURCES["sam3"]
    assert row["voice"]["base"].endswith(":8083")
    assert (destination / "llama-server.log").is_file()
    # The three sentences of layout 0 of a batch from seed 7, every one of
    # them read as the cube it names.
    assert [entry["sentence"] for entry in row["sentences"]] == [
        TEMPLATES[position].format(colour=cube) for position, cube in enumerate(expected)
    ]
    assert [entry["as_expected"] for entry in row["sentences"]] == [True, True, True]
    assert all(entry["gemma"]["how"] == "typed" for entry in row["sentences"])
    assert all(entry["gemma"]["command"] == "pick_place" for entry in row["sentences"])
    # The video's captions and the picks carry Gemma's own call.
    calls = [
        call_text(entry["gemma"]["command"], entry["gemma"]["object"], entry["gemma"]["place"])
        for entry in row["sentences"]
    ]
    assert [pick["command"] for pick in row["picks"]] == calls
    assert [plan["command"] for plan in row["plans"]] == [call for call in calls for _ in range(2)]
    for pick in row["picks"]:
        assert pick["cube"] == pick["expected_cube"] and pick["refusal"] is None
        # The two cubes the sentence does not name are painted by SAM 3's outlines.
        assert pick["painted_pixels"]["top_mean"] > 1000
        assert pick["painted_pixels"]["wrist_mean"] > 1500
    for plan in row["plans"]:
        for camera in CAMERAS:
            assert plan["sam3"][camera]["seconds"] > 0.0
    complete = json.loads((destination / "COMPLETE.json").read_text(encoding="utf-8"))
    assert complete["commands"] == COMMANDS_FROM["typed"] and complete["outlines"] == "sam3"
    report = (destination / "REPORT.md").read_text(encoding="utf-8")
    assert "Commands from: typed sentences through Gemma 4." in report
    assert "Sentences through Gemma 4: 3; 3 of 3 with a known cube gave that cube; 0 refused" in report
    assert f"| 7 | {row['sentences'][0]['sentence']} | typed |" in report
    stored = np.load(destination / "seed-00007" / "policy-observations.npz")
    assert np.array_equal(stored["top"][0], shown[0].top)
