"""Regular initialization completes both grids before posterior decisions."""
import numpy as np
import pytest

from test_analytical_alignment import Controller, SyntheticBackend, events, observation, settings


def grid_settings(**kwargs):
    return settings(**dict(dict(bo_initialization="grid", stage_1_max_acquisitions=20,
                                max_acquisitions=40, stage_2_target_m2=None), **kwargs))


def test_grid_initialization_completes_despite_low_m1_and_exclusions(tmp_path, monkeypatch):
    backend = SyntheticBackend([12, 12])
    del backend.candidate_positions

    def measured(position):
        obs = observation(position)
        # Stage 1 already meets entry. Stage 2 grid locations fail eligibility.
        obs["metrics"]["primary_metric"] = .001 if backend.acquisitions <= 18 else .2
        return obs

    c = Controller(grid_settings(stage_2_target_m2=1000), backend, tmp_path / "log", measured)
    # Also exercise a Stage 2 box clipped by the global boundaries.
    target = np.array([14., -14.])
    monkeypatch.setattr(c, "posterior_mean_center", lambda gp, center=None: target.copy())
    result = c.run()
    assert result["status"] != "blocked"
    stage1 = [o for o in c.history if o["stage"] == "stage1_init"]
    stage2 = [o for o in c.history if o["stage"] == "stage_2_init"]
    np.testing.assert_allclose([o["position"] for o in stage1], c.initialization_grid())
    assert len(stage2) == 17
    np.testing.assert_allclose(stage2[0]["position"], target)
    np.testing.assert_allclose([o["position"] for o in stage2[1:]], c.initialization_grid(target))
    assert all(not o["bo_eligible"] for o in stage2[1:])
    transition = next(e for e in events(c) if e["event"] == "stage_transition")
    assert transition["stage1_acquisitions"] == 17
    assert all(e["stage1_acquisitions"] >= 17 for e in events(c) if e["event"] == "stage_2_entry_check")
    boxes = [e for e in events(c) if e["event"] == "search_box"]
    assert all(e["centering"] == "initialization_anchor" for e in boxes)
    assert sum(o["stage"] == "final" for o in c.history) == 2


def test_box_adaptation_waits_for_grid_then_uses_eligible_gp(tmp_path, monkeypatch):
    backend = SyntheticBackend([12, 12])
    del backend.candidate_positions

    def measured(position):
        obs = observation(position)
        obs["metrics"]["primary_metric"] = .001
        return obs

    c = Controller(grid_settings(), backend, tmp_path / "log", measured)
    adaptations = []

    def center(gp, center=None):
        if center is not None:
            adaptations.append(c.acquisitions)
        return np.array([0., 0.])

    monkeypatch.setattr(c, "posterior_mean_center", center)
    result = c.run()
    assert result["status"] != "blocked"
    assert adaptations and min(adaptations) == 34
    assert sum(o["stage"] == "stage_2_init" for o in c.history) == 17
    assert result["acquisitions"] <= 40


@pytest.mark.parametrize("kwargs, message", [
    ({"bo_initialization": "other"}, "bo_initialization"),
    ({"stage_1_max_acquisitions": 16}, "baseline plus 16"),
    ({"max_acquisitions": 38}, "Stage 2 entry plus 16"),
])
def test_invalid_initialization_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        grid_settings(**kwargs).validate()


def test_grid_ignores_random_initial_sample_count():
    grid_settings(bo_initial_samples=100).validate()


def test_grid_rejects_backend_missing_required_locations(tmp_path):
    backend = SyntheticBackend([0, 0])
    c = Controller(grid_settings(), backend, tmp_path / "log")
    with pytest.raises(ValueError, match="all 16 locations"):
        c.initialization_grid([0, 0])


def test_budget_entry_and_grid_without_eligible_m2(tmp_path, monkeypatch):
    backend = SyntheticBackend([12, 12])
    del backend.candidate_positions

    def measured(position):
        obs = observation(position)
        obs["metrics"]["primary_metric"] = .03
        return obs

    c = Controller(grid_settings(), backend, tmp_path / "log", measured)
    monkeypatch.setattr(c, "posterior_mean_center", lambda gp, center=None: np.zeros(2))
    result = c.run()
    transition = next(e for e in events(c) if e["event"] == "stage_transition")
    assert transition["reason"] == "stage 1 budget exhausted"
    assert transition["stage1_acquisitions"] == 20
    assert sum(o["stage"] == "stage_2_init" for o in c.history) == 17
    assert result["status"] == "unconfirmed"
    assert result["reason"].startswith("grid initialization completed without eligible M2 observations")
    assert result["acquisitions"] == 39
