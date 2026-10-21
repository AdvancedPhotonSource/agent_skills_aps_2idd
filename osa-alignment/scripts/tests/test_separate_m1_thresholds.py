"""Entrance predictions and observed M2 eligibility can use different M1 limits."""
import numpy as np
import pytest

from test_analytical_alignment import Controller, SyntheticBackend, observation, settings, events
from align_osa import feasible, fit_objective


def test_default_eligibility_threshold_preserves_existing_configs():
    s = settings(stage_2_entrance_m1=.012)
    assert s.m2_eligibility_threshold == .012
    assert feasible(observation([0, 0]), s)


def test_separate_threshold_controls_training_and_exclusions(tmp_path):
    s = settings(stage_2_entrance_m1=.002, stage_2_eligibility_m1=.012)
    good = observation([0, 0])
    bad = observation([5, 0])
    gp, _ = fit_objective([good, bad], s)
    np.testing.assert_array_equal(gp.X_train_, [[0, 0]])
    backend = SyntheticBackend([0, 0])
    backend.candidate_positions = np.array([[.5, 0], [4.5, 0]])
    c = Controller(s, backend, tmp_path / 'log')
    c.history = [good, bad]
    c.length_scales[:] = 1
    np.testing.assert_array_equal(c.candidates(good, initial=True), [[.5, 0]])


def test_lower_entrance_does_not_trigger_early_or_reject_eligible_m2(tmp_path):
    def measure(p):
        obs = observation(p)
        obs['metrics']['primary_metric'] = .01
        return obs

    c = Controller(settings(stage_2_entrance_m1=.002, stage_2_eligibility_m1=.012,
                            bo_initial_samples=2, stage_1_max_acquisitions=3, max_acquisitions=8),
                   SyntheticBackend([0, 0]), tmp_path / 'log', measure)
    result = c.run()
    transition = next(e for e in events(c) if e['event'] == 'stage_transition')
    assert transition['reason'] == 'stage 1 budget exhausted'
    assert transition['stage1_acquisitions'] == 3
    assert transition['posterior_mean_m1'] > .002
    assert all(o['bo_eligible'] and o['bo_rejection_reason'] is None for o in c.history)
    assert result['status'] == 'metric_target_reached'
    assert result['final_readback'] == [0, 0]


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan')])
def test_invalid_eligibility_threshold(value):
    with pytest.raises(ValueError, match='stage_2_eligibility_m1'):
        settings(stage_2_eligibility_m1=value).validate()
