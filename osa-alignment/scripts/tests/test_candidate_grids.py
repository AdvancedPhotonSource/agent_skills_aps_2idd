"""Dense BO candidate coverage, spacing, and filtering."""
import numpy as np
import pytest

from test_analytical_alignment import Controller, SyntheticBackend, observation, settings
from align_osa import candidate_grid


@pytest.mark.parametrize('stage, expected_count, spacing', [(1, 31**2, 1.), (2, 101**2, .1)])
def test_default_grid_covers_domain_and_is_independent_of_rng(tmp_path, stage, expected_count, spacing):
    backend = SyntheticBackend([0, 0])
    del backend.candidate_positions
    c = Controller(settings(), backend, tmp_path / 'log')
    get_pool = (lambda: c.stage1_candidates(initial=True)) if stage == 1 else (
        lambda: c.candidates({'position': [0, 0]}, initial=True))
    first = get_pool()
    c.rng.random(1000)
    second = get_pool()
    np.testing.assert_array_equal(first, second)
    assert len(first) == expected_count
    extent = 15 if stage == 1 else 5
    for dimension in range(2):
        axis = np.unique(first[:, dimension])
        np.testing.assert_allclose(np.diff(axis), spacing, atol=1e-12)
        np.testing.assert_array_equal(axis[[0, -1]], [-extent, extent])


def test_nondivisible_grid_includes_edges_without_overshoot():
    grid = candidate_grid([-.35, .12], [.7, .37], .4)
    np.testing.assert_allclose(np.unique(grid[:, 0]), [-.35, .05, .45, .7])
    np.testing.assert_allclose(np.unique(grid[:, 1]), [.12, .37])
    assert grid.shape == (8, 2)
    assert np.all(grid >= [-.35, .12]) and np.all(grid <= [.7, .37])


@pytest.mark.parametrize('stage', [1, 2])
def test_configured_spacing_and_clipped_box(tmp_path, stage):
    backend = SyntheticBackend([14.8, -14.9])
    del backend.candidate_positions
    c = Controller(settings(stage_1_grid_spacing=2., stage_2_grid_spacing=.3), backend, tmp_path / 'log')
    if stage == 1:
        pool = c.stage1_candidates(initial=True)
        assert len(pool) == 16**2
        np.testing.assert_allclose(np.diff(np.unique(pool[:, 0])), 2.)
    else:
        pool = c.candidates({'position': [14.8, -14.9]}, initial=True)
        for dimension, lo, hi in [(0, 9.8, 15.), (1, -15., -9.9)]:
            axis = np.unique(pool[:, dimension])
            np.testing.assert_allclose(axis[[0, -1]], [lo, hi])
            np.testing.assert_allclose(np.diff(axis)[:-1], .3, atol=1e-12)
            assert 0 < axis[-1] - axis[-2] <= .3 + 1e-12
        assert np.all(np.abs(pool) <= 15)


def test_dense_grid_preserves_rejected_regions_and_finds_small_allowed_region(tmp_path):
    backend = SyntheticBackend([0, 0])
    del backend.candidate_positions
    c = Controller(settings(stage_2_bo_halfwidth=[1, 1]), backend, tmp_path / 'log')
    rejected = observation([0, 0], invalid=True)
    c.history = [rejected]
    c.length_scales = np.array([1.4, 1.4])
    pool = c.candidates(rejected, initial=True)
    # Only the four corners survive this exclusion; no random proposals can miss them.
    np.testing.assert_allclose(pool, [[-1, -1], [-1, 1], [1, -1], [1, 1]])
    c.length_scales[:] = 2.
    assert len(c.candidates(rejected, initial=True)) == 0


@pytest.mark.parametrize('stage', [1, 2])
@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_grid_spacing_requires_positive_finite_values(stage, value):
    with pytest.raises(ValueError, match='Grid spacings must be finite and positive'):
        settings(**{f'stage_{stage}_grid_spacing': value}).validate()


def test_legacy_candidate_count_does_not_limit_grid(tmp_path):
    backend = SyntheticBackend([0, 0])
    del backend.candidate_positions
    c = Controller(settings(bo_candidates=4), backend, tmp_path / 'log')
    assert len(c.stage1_candidates(initial=True)) == 961
    assert len(c.candidates({'position': [0, 0]}, initial=True)) == 10201
