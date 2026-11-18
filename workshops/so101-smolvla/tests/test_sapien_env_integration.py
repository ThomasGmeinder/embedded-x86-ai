from pathlib import Path

import numpy as np
import pytest

from so101_smolvla.sapien_config import IMAGE_SHAPE, SAPIEN_PYTHON, SAPIEN_SOURCE
from so101_smolvla.sapien_env_client import SapienPolicyEnvironment


pytestmark = pytest.mark.skipif(
    not SAPIEN_PYTHON.is_file() or not (SAPIEN_SOURCE / ".git").is_dir(),
    reason="run scripts/setup_sapien_env.sh to install the real SAPIEN backend",
)


def test_real_sapien_reset_step_and_clean_shutdown():
    env = SapienPolicyEnvironment(horizon=2)
    try:
        first = env.reset(environment_seed=0)
        outcome = env.step(first.state)
        assert first.state.shape == (6,)
        assert first.top.shape == IMAGE_SHAPE
        assert first.wrist.shape == IMAGE_SHAPE
        assert outcome.observation.top.shape == IMAGE_SHAPE
        assert outcome.observation.wrist.shape == IMAGE_SHAPE
        assert outcome.achieved.shape == (6,)
        assert np.isfinite(outcome.achieved).all()
        assert Path(SAPIEN_SOURCE).is_dir()
        assert env.provenance["backend"] == "sapien"
    finally:
        env.close()
    env.close()
