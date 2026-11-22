import numpy as np
import pytest

from so101_smolvla.reachable_workspace import draw_reachable_layout, is_reachable_xy

# Corner distances of the footprints of a 20 mm and a 40 mm cube.
SMALL, LARGE = np.sqrt(2.0) * 0.010, np.sqrt(2.0) * 0.020


@pytest.mark.parametrize(
    ("clearances", "minimum_distance"),
    (([SMALL, LARGE], 0.040), ([LARGE, LARGE, LARGE], 0.060)),
)
def test_every_drawn_layout_is_reachable_and_keeps_the_minimum_distance(
    clearances, minimum_distance
):
    margins = np.array(clearances)
    for seed in range(200):
        points = draw_reachable_layout(
            np.random.default_rng(seed), margins, minimum_distance
        )
        assert np.array_equal(
            points,
            draw_reachable_layout(
                np.random.default_rng(seed), margins, minimum_distance
            ),
        )
        for index, (point, margin) in enumerate(zip(points, margins, strict=True)):
            assert is_reachable_xy(point, margin)
            for other in range(index + 1, len(points)):
                assert np.linalg.norm(point - points[other]) >= minimum_distance


def test_a_seed_keeps_its_first_candidate_until_the_distance_rules_it_out():
    margins = np.array([SMALL, LARGE])
    for seed in range(50):
        first = draw_reachable_layout(np.random.default_rng(seed), margins, 0.0)
        apart = float(np.linalg.norm(first[0] - first[1]))
        assert np.array_equal(
            first,
            draw_reachable_layout(np.random.default_rng(seed), margins, apart - 1e-9),
        )
        try:
            later = draw_reachable_layout(
                np.random.default_rng(seed), margins, apart + 1e-6
            )
        except RuntimeError:
            continue
        assert not np.array_equal(first, later)
        assert np.linalg.norm(later[0] - later[1]) >= apart + 1e-6


def test_a_condition_of_the_scene_narrows_the_choice_without_changing_the_draw():
    margins = np.array([LARGE, LARGE, LARGE])
    for seed in range(50):
        plain = draw_reachable_layout(np.random.default_rng(seed), margins, 0.060)
        # A condition every candidate meets leaves the layout as it was.
        assert np.array_equal(
            plain,
            draw_reachable_layout(
                np.random.default_rng(seed),
                margins,
                0.060,
                also_valid=lambda candidates: np.ones(len(candidates), dtype=bool),
            ),
        )

        # A condition the plain layout fails gives a later candidate that
        # meets it: the first object in the nearer half of the workspace.
        def near_half(candidates: np.ndarray) -> np.ndarray:
            return candidates[:, 0, 0] < 0.27

        narrowed = draw_reachable_layout(
            np.random.default_rng(seed), margins, 0.060, also_valid=near_half
        )
        assert narrowed[0, 0] < 0.27
        for index, point in enumerate(narrowed):
            assert is_reachable_xy(point, LARGE)
            for other in range(index + 1, len(narrowed)):
                assert np.linalg.norm(point - narrowed[other]) >= 0.060
        if plain[0, 0] < 0.27:
            assert np.array_equal(plain, narrowed)
        else:
            assert not np.array_equal(plain, narrowed)


def test_a_condition_no_candidate_meets_fails_and_a_malformed_one_is_rejected():
    margins = np.array([SMALL, LARGE])
    with pytest.raises(RuntimeError, match="no non-overlapping layout"):
        draw_reachable_layout(
            np.random.default_rng(0),
            margins,
            0.040,
            also_valid=lambda candidates: np.zeros(len(candidates), dtype=bool),
        )
    with pytest.raises(ValueError, match="the scene's condition returned"):
        draw_reachable_layout(
            np.random.default_rng(0),
            margins,
            0.040,
            also_valid=lambda candidates: np.ones(candidates.shape[:2], dtype=bool),
        )
    with pytest.raises(ValueError, match="the scene's condition returned"):
        draw_reachable_layout(
            np.random.default_rng(0),
            margins,
            0.040,
            also_valid=lambda candidates: np.ones(len(candidates)),
        )


def test_a_layout_that_cannot_keep_the_distance_fails_instead_of_placing_it_anyway():
    # Two cubes cannot stand half a metre apart in the 200 mm workspace.
    with pytest.raises(RuntimeError, match="no non-overlapping layout"):
        draw_reachable_layout(np.random.default_rng(0), np.array([SMALL, LARGE]), 0.5)
    with pytest.raises(ValueError, match="invalid clearances"):
        draw_reachable_layout(np.random.default_rng(0), np.array([-0.01]), 0.040)
    with pytest.raises(ValueError, match="invalid placement limits"):
        draw_reachable_layout(np.random.default_rng(0), np.array([SMALL]), -0.001)
    with pytest.raises(ValueError, match="invalid placement limits"):
        draw_reachable_layout(np.random.default_rng(0), np.array([SMALL]), 0.040, 0)
