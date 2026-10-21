"""Stage 2 entry follows the global M1 posterior after full initialization."""
import numpy as np
import pytest

from test_analytical_alignment import Controller, SyntheticBackend, events, observation, settings
import align_osa


@pytest.mark.parametrize('predicted_m1, expected_count', [(.01, 3), (.03, 4)])
def test_entry_waits_for_initialization_and_uses_posterior(tmp_path, monkeypatch, predicted_m1, expected_count):
    backend = SyntheticBackend([12, 12])
    target = np.array([1.25, -.75])
    c = Controller(settings(bo_initial_samples=3, stage_1_max_acquisitions=4,
                            max_acquisitions=9, stage_2_target_m2=1000),
                   backend, tmp_path / 'log', observation)
    fitted_counts = []

    class Posterior:
        kernel_ = type('Kernel', (), {'length_scale': np.array([2.5, 2.5])})()

        def __init__(self, history):
            self.X_train_ = np.array([o['position'] for o in history])

        def predict(self, points, return_std=False):
            mean = predicted_m1 + .001 * np.sum((np.asarray(points)-target)**2, axis=1)
            return (mean, np.ones(len(mean))*.01) if return_std else mean

    real_fit = align_osa.fit_objective

    def fit(history, config, metric='secondary_metric'):
        if metric == 'primary_metric':
            fitted_counts.append(len(history))
            return Posterior(history), False
        return real_fit(history, config, metric)

    def measured(p):
        obs = observation(p)
        # An anomalously low first point must not cause early entry, even when
        # its observed score is lower than the minimum predicted by the model.
        obs['metrics']['primary_metric'] = .001 if backend.acquisitions == 1 else .015
        return obs

    c.measurement = measured
    del backend.candidate_positions
    monkeypatch.setattr(align_osa, 'fit_objective', fit)
    result = c.run()
    assert result['status'] == 'metric_target_reached'
    transition = next(e for e in events(c) if e['event'] == 'stage_transition')
    assert transition['stage1_acquisitions'] == expected_count
    assert transition['posterior_mean_m1'] == pytest.approx(predicted_m1)
    assert transition['reason'] == ('M1 posterior mean threshold reached' if predicted_m1 <= .02
                                     else 'stage 1 budget exhausted')
    assert [o['stage'] for o in c.history[:3]] == ['baseline', 'stage1_init', 'stage1_init']
    entry = c.history[expected_count]
    assert entry['stage'] == 'stage_2_init'
    np.testing.assert_allclose(entry['position'], target, atol=1e-5)
    np.testing.assert_allclose(transition['position'], target, atol=1e-5)
    checks = [e for e in events(c) if e['event'] == 'stage_2_entry_check']
    assert [e['stage1_acquisitions'] for e in checks] == list(range(3, expected_count+1))
    assert fitted_counts == list(range(1, expected_count+1))
    assert len([o for o in c.history if o['stage'] == 'final']) == 2


def test_global_minimum_ignores_local_box_and_m1_exclusions(tmp_path):
    backend = SyntheticBackend([12, 12])
    del backend.candidate_positions
    c = Controller(settings(stage_2_bo_halfwidth=[1, 1]), backend, tmp_path / 'log')
    c.history = [observation([-9, -8], invalid=True)]

    class Posterior:
        def predict(self, points):
            return np.sum((np.asarray(points)-[-9.3, -8.2])**2, axis=1)

    np.testing.assert_allclose(c.posterior_mean_center(Posterior()), [-9.3, -8.2], atol=1e-5)


def test_global_discrete_minimum_can_revisit_measured_position(tmp_path):
    backend = SyntheticBackend([12, 12])
    backend.candidate_positions = np.array([[12, 12], [-9, -8], [0, 0], [16, 16]])
    c = Controller(settings(), backend, tmp_path / 'log')
    c.history = [observation([-9, -8], invalid=True)]

    class Posterior:
        def predict(self, points):
            return np.sum((np.asarray(points)-[-9.3, -8.2])**2, axis=1)

    np.testing.assert_array_equal(c.posterior_mean_center(Posterior()), [-9, -8])


def test_stage1_budget_must_fit_initialization():
    with pytest.raises(ValueError, match='accommodate bo_initial_samples'):
        settings(bo_initial_samples=4, stage_1_max_acquisitions=3).validate()
