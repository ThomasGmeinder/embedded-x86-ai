"""The launcher of the fast SAM 3 reference, scripts/run_sam3_reference.sh.

It runs Python of .venv-sam3 inside the reference's checkout with the environment
the reference's container launcher gives its commands. These tests run the real
launcher; none of them uses the graphics processor.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

WORKSHOP = Path(__file__).resolve().parents[1]
LAUNCHER = WORKSHOP / "scripts" / "run_sam3_reference.sh"
SETTINGS_FILE = WORKSHOP / "configs" / "sam3_reference.json"
SETTINGS = json.loads(SETTINGS_FILE.read_text())
CHECKOUT = Path(os.path.expanduser(SETTINGS["checkout"]))

needs_reference = pytest.mark.skipif(
    not (WORKSHOP / ".venv-sam3" / "bin" / "python").exists() or not (CHECKOUT / ".git").is_dir(),
    reason="run scripts/setup_sam3_env.sh with the reference checked out to install the fast SAM 3 environment",
)

# What a launched command reports about itself.
REPORT = """
import json, os
from tracker import rocm_env
from tracker.ort_plugin_ep import stack_errors
names = (
    "PYTHONPATH", "SAM3_ONNX_DIR", "SAM3_SOURCE_COMMIT", "SAM3_SOURCE_DIRTY",
    "TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "SAM3_DOCKER_IMAGE_REF",
    "SAM3_DOCKER_IMAGE_ID", "HSA_OVERRIDE_GFX_VERSION", "MIGRAPHX_GPU_HIP_FLAGS",
    "SAM3_EC_POWER_MODE", "MIOPEN_FIND_MODE", "SAM3_MODEL_DIR",
)
print("REPORT " + json.dumps({
    "environment": {name: os.environ.get(name) for name in names},
    "directory": os.getcwd(),
    "reference_graphics_override": rocm_env.HSA_OVERRIDE_GFX_VERSION,
    "reference_compiler_flags": rocm_env.MIGRAPHX_GPU_HIP_FLAGS,
    "stack_errors": stack_errors(),
}))
"""


def launch(*arguments: str, **environment: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(LAUNCHER), *arguments],
        capture_output=True,
        text=True,
        env={**os.environ, **environment},
        timeout=120,
    )


def settings_copy(tmp_path: Path, **changed: str) -> str:
    path = tmp_path / "sam3_reference.json"
    path.write_text(json.dumps({**SETTINGS, **changed}))
    return str(path)


def test_print_gives_a_setting_with_the_home_directory_expanded():
    result = launch("--print", "checkout")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(CHECKOUT)


@needs_reference
def test_a_launched_command_gets_the_environment_of_the_reference():
    # The calling shell's own graphics override must not reach the command.
    result = launch("-c", REPORT, HSA_OVERRIDE_GFX_VERSION="11.0.0")
    assert result.returncode == 0, result.stderr
    print(result.stdout)
    line = next(line for line in result.stdout.splitlines() if line.startswith("REPORT "))
    report = json.loads(line.removeprefix("REPORT "))
    environment = report["environment"]

    assert report["directory"] == str(CHECKOUT)
    assert environment["PYTHONPATH"] == str(CHECKOUT)
    assert environment["SAM3_ONNX_DIR"] == os.path.expanduser(SETTINGS["compiled_dir"])
    assert environment["SAM3_MODEL_DIR"] == os.path.expanduser(SETTINGS["model_dir"])
    assert environment["SAM3_SOURCE_COMMIT"] == SETTINGS["commit"]
    assert environment["SAM3_SOURCE_DIRTY"] in {"0", "1"}
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["HSA_OVERRIDE_GFX_VERSION"] == report["reference_graphics_override"]
    assert environment["HSA_OVERRIDE_GFX_VERSION"] != "11.0.0"
    assert environment["MIGRAPHX_GPU_HIP_FLAGS"] == report["reference_compiler_flags"]
    # Without it the first PyTorch convolution fails on this machine (see the launcher).
    assert environment["MIOPEN_FIND_MODE"] == "FAST"
    # No container image exists; the package list of .venv-sam3 stands in for it.
    kind, _, checksum = environment["SAM3_DOCKER_IMAGE_ID"].partition(":")
    assert kind == "venv-sha256"
    assert len(checksum) == 64 and set(checksum) <= set("0123456789abcdef")
    assert "no Docker" in environment["SAM3_DOCKER_IMAGE_REF"]
    # The power mode is the machine's own, never an assertion of the launcher.
    profile = Path("/sys/firmware/acpi/platform_profile")
    if profile.exists():
        assert (environment["SAM3_EC_POWER_MODE"] == "performance") == (
            profile.read_text().strip() == "performance"
        )
    # The reference's own check of MIGraphX, ONNX Runtime and the plug-in versions.
    assert report["stack_errors"] == []


@needs_reference
def test_a_missing_checkout_is_named(tmp_path):
    missing = tmp_path / "no-such-checkout"
    result = launch(
        "-c", "print('ran')",
        SAM3_REFERENCE_SETTINGS=settings_copy(tmp_path, checkout=str(missing)),
    )
    assert result.returncode == 1
    assert "ran" not in result.stdout
    assert f"SAM 3 reference checkout is missing: {missing}" in result.stderr


@needs_reference
def test_a_checkout_at_another_commit_is_refused(tmp_path):
    other = "0" * 40
    result = launch(
        "-c", "print('ran')",
        SAM3_REFERENCE_SETTINGS=settings_copy(tmp_path, commit=other),
    )
    assert result.returncode == 1
    assert "ran" not in result.stdout
    assert f"is not at commit {other}" in result.stderr
