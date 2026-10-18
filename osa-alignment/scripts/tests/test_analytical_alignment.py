"""Synthetic evaluation and BO control tests; no instrument or recorded data needed."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_io import measure
from align_osa import Controller, Replay, Settings, fit_objective, expected_improvement, feasible
from evaluate_alignment import Config


def observation(position, *, invalid=False):
    p = np.asarray(position)
    g = .01 + .001 * np.sum(p ** 2)
    return dict(image="synthetic", position=p.tolist(), metrics=dict(
        valid=not invalid, reasons=["synthetic missing boundary"] if invalid else [],
        primary_metric=g,
        secondary_metric=float("inf") if invalid else 10. + np.sum(p ** 2),
        geometry_score=g, image_shape=[40, 40], inner_coverage=.8, outer_coverage=.8,
        fit=dict(center_x=20., center_y=20., angle_degrees=0., outer_axes=[8., 8.])))


class SyntheticBackend:
    units = "um"

    def __init__(self, start):
        self.p = np.array(start, float)
        self.moves = []
        self.acquisitions = 0
        self.candidate_positions = np.array([(x, y) for x in range(-15, 16) for y in range(-15, 16)])

    def read_position(self):
        return self.p.copy()

    def move_absolute(self, target):
        self.moves.append((self.p.copy(), np.array(target)))
        self.p = np.array(target)

    def acquire_image(self):
        self.acquisitions += 1
        return self.p.copy()


def settings(**options):
    options.setdefault("stage1_max_acquisitions", min(3, options.get("max_acquisitions", 40) - 3))
    return Settings(**dict(dict(units="um", bounds=[[-15, 15], [-15, 15]], max_step=5.,
                                min_step=1., target_m2=10., max_acquisitions=40), **options))


def test_gp_uses_only_eligible_objectives():
    observations = [observation([0, 0]), observation([1, 0]), observation([10, 10], invalid=True)]
    gp, fitted = fit_objective(observations, settings())
    assert len(gp.X_train_) == 2 and not fitted
    scores = expected_improvement(gp, np.array([[0, 1], [9, 10]]), 10., settings())
    assert np.isfinite(scores).all()
    assert np.all(scores >= 0)


def test_gp_noise_tracks_eligible_observation_stages():
    observations = []
    stages = ['baseline', 'stage1', 'stage1_init', 'bo_init', 'bo', 'final', None]
    for stage in stages:
        obs = observation([0, 0])
        if stage is not None:
            obs['stage'] = stage
        observations.append(obs)
    rejected = observation([1, 0], invalid=True)
    rejected['stage'] = 'stage1'
    observations.insert(1, rejected)
    gp, _ = fit_objective(observations, settings())
    np.testing.assert_allclose(gp.alpha, [.0225, .0225, .0225, .0025, .0025, .0025, .0025])


def test_stage1_noise_reduces_influence_on_gp_prediction():
    stage1 = observation([0, 0])
    stage1.update(stage='stage1')
    stage1['metrics']['secondary_metric'] = 2.
    fine = observation([0, 0])
    fine.update(stage='bo')
    fine['metrics']['secondary_metric'] = 0.
    equal, _ = fit_objective([stage1, fine], settings(gp_stage1_noise_multiplier=1.))
    weighted, _ = fit_objective([stage1, fine], settings())
    np.testing.assert_allclose(equal.alpha, [.0025, .0025])
    np.testing.assert_allclose(equal.predict([[0, 0]]), [1.])
    assert 0. < weighted.predict([[0, 0]])[0] < equal.predict([[0, 0]])[0]


def test_readback_fault_stops_without_recovery_motion(tmp_path):
    class Faulty(SyntheticBackend):
        def move_absolute(self, target):
            super().move_absolute(np.array(target) + [1, 0])
    backend = Faulty([10, 0])
    result = Controller(settings(), backend, tmp_path / "log",
                        lambda p: observation(p)).run()
    assert result["status"] == "blocked"
    assert "readback" in result["reason"]
    assert len(backend.moves) == 1
    assert backend.acquisitions == 1


def test_replay_is_exact_discrete_and_cycles_repeats():
    backend = Replay(dict(units="um", samples=[dict(position=[0, 0], image="a"),
                                               dict(position=[0, 0], image="b")]), [0, 0])
    assert [backend.acquire_image() for _ in range(3)] == ["a", "b", "a"]
    with pytest.raises(ValueError, match="exactly recorded"):
        backend.move_absolute([.1, 0])


@pytest.mark.parametrize("options", [dict(max_step=0), dict(bo_stage_entrance_m1=0),
                                   dict(bounds=[[15, -15], [-15, 15]]), dict(stage1_max_acquisitions=0),
                                   dict(max_acquisitions=3)])
def test_unsafe_settings_rejected(options):
    with pytest.raises(ValueError):
        settings(**options).validate()


def test_no_feasible_point_is_not_success(tmp_path):
    backend = SyntheticBackend([0, 0])
    result = Controller(settings(max_acquisitions=9), backend, tmp_path / "log",
                        lambda p: observation(p, invalid=True)).run()
    assert result["status"] == "unconfirmed"
    assert backend.acquisitions <= 9


def test_failed_acquisition_counts_and_stops(tmp_path):
    class Faulty(SyntheticBackend):
        def acquire_image(self):
            super().acquire_image()
            raise RuntimeError("detector fault")
    backend = Faulty([0, 0])
    result = Controller(settings(), backend, tmp_path / "log").run()
    assert result["status"] == "blocked"
    assert result["acquisitions"] == backend.acquisitions == 1
    assert backend.moves == []


def test_final_target_must_survive_confirmation(tmp_path):
    backend = SyntheticBackend([0, 0])
    def measured(p):
        result = observation(p)
        if backend.acquisitions > 1:
            result["metrics"]["secondary_metric"] = 20.
        return result
    result = Controller(settings(), backend, tmp_path / "log", measured).run()
    assert result["status"] == "best_found"
    assert result["reason"] == "M2 target not sustained during final confirmation"


def test_missing_boundaries_do_not_train_objective_gp():
    missing = observation([10, 10], invalid=True)
    missing["metrics"]["geometry_score"] = float("inf")
    missing["metrics"]["fit"] = None
    gp, fitted = fit_objective([observation([0, 0]), missing], settings())
    assert len(gp.X_train_) == 1 and not fitted
    scores = expected_improvement(gp, np.array([[1, 0], [9, 10]]), 10., settings())
    assert np.isfinite(scores).all()
    assert np.all(scores >= 0)


def test_continuous_candidates_respect_symmetric_box_and_step_limit(tmp_path):
    backend = SyntheticBackend([5, 5])
    del backend.candidate_positions
    c = Controller(settings(bo_halfwidth=[2, 3]), backend, tmp_path / "log",
                   lambda p: observation(p))
    incumbent = observation([5, 5])
    c.history.append(incumbent)
    pool = c.candidates(incumbent)
    delta = pool - [5, 5]
    assert np.all(np.abs(delta) <= [2, 3])
    assert np.all(np.linalg.norm(delta, axis=1) <= 5)
    assert np.all(delta.min(axis=0) < 0) and np.all(delta.max(axis=0) > 0)


@pytest.mark.parametrize('fixed_axes', [False, True])
def test_measurement_and_controller_accept_translated_resized_images(tmp_path, fixed_axes):
    y, x = np.indices((200, 220))
    dx, dy = x - 110., y - 100.
    ring = ((dx / 65.) ** 2 + (dy / 55.) ** 2 <= 1) & ((dx / 28.) ** 2 + (dy / 24.) ** 2 >= 1)
    original = np.where(ring, 100. + .2 * dx, 0.).astype(np.float32)
    shifted = np.pad(original, ((40, 80), (70, 50)))
    paths = [tmp_path / 'original.tiff', tmp_path / 'shifted.tiff']
    for path, image in zip(paths, [original, shifted]):
        Image.fromarray(image).save(path)
    evaluator = dict(inner_axes=(28, 24), outer_axes=(65, 55)) if fixed_axes else {}
    before, after = [measure(path, config=Config(**evaluator)) for path in paths]
    assert before['metrics']['valid'] and after['metrics']['valid']
    np.testing.assert_allclose(before['metrics']['primary_metric'], after['metrics']['primary_metric'], atol=.001)
    assert before['metrics']['image_shape'] != after['metrics']['image_shape']

    class Images(SyntheticBackend):
        def acquire_image(self):
            super().acquire_image()
            return paths[(self.acquisitions - 1) % 2]

    backend = Images([0, 0])
    c = Controller(settings(evaluator=evaluator, bo_stage_entrance_m1=.05,
                                target_m2=1e9), backend, tmp_path / 'trajectory')
    result = c.run()
    assert result['status'] == 'metric_target_reached'
    assert len({tuple(o['metrics']['image_shape']) for o in c.history}) == 2


def test_worse_eligible_m2_changes_gp_but_rejected_metrics_do_not():
    good = observation([0, 0])
    worse = observation([1, 0])
    high_m1 = observation([8, 0])
    missing_m2 = observation([0, 1])
    missing_m2['metrics']['secondary_metric'] = float('inf')
    gp_before, _ = fit_objective([good], settings())
    gp_after, _ = fit_objective([good, worse, high_m1, missing_m2], settings())
    np.testing.assert_array_equal(gp_after.X_train_, [[0, 0], [1, 0]])
    assert gp_after.predict([[1, 0]])[0] > gp_before.predict([[1, 0]])[0]
    observations = [good, worse, observation([0, 2])]
    gp_fitted, fitted = fit_objective(observations, settings())
    assert fitted
    assert np.all(gp_fitted.kernel_.length_scale >= .5)
    assert np.all(gp_fitted.kernel_.length_scale <= 10.)
    assert not np.allclose(gp_fitted.kernel_.length_scale, [2.5, 2.5])


def test_initialization_counts_eligible_random_measurements_and_reuses_stage1(tmp_path, monkeypatch):
    backend = SyntheticBackend([0, 0])
    def measured(p):
        obs = observation(p, invalid=backend.acquisitions == 2)
        if obs['metrics']['valid']:
            obs['metrics']['primary_metric'] = .01
        return obs
    c = Controller(settings(target_m2=None, max_acquisitions=11),
                   backend, tmp_path / 'log', measured)
    points = iter([[1, 0], [-1, 0], [-2, 0], [-2, 1], [-1, 1], [0, 1], [0, 2], [1, 2]])
    monkeypatch.setattr(c, 'candidates', lambda incumbent, **kwargs: np.array([next(points)], float))
    result = c.run()
    initial = [o for o in c.history if o['stage'] == 'bo_init']
    assert len(initial) == 6
    assert sum(o['bo_eligible'] for o in initial) == 5
    assert len({tuple(o['position']) for o in initial}) == 6
    assert initial[0]['bo_rejection_reason'] == 'invalid fit'
    assert [o['stage'] for o in c.history] == ['baseline'] + ['bo_init'] * 6 + ['bo'] * 2 + ['final'] * 2
    assert result['final_readback'] == [0, 0]
    models = [e for e in map(json.loads, (tmp_path / 'log').read_text().splitlines()) if e['event'] == 'bo_model']
    assert models[-1]['training_acquisitions'] == [1, 3, 4, 5, 6, 7, 8]
    assert models[-1]['exclusion_acquisitions'] == [2]
    assert result['acquisitions'] == 11


@pytest.mark.parametrize('invalid', [False, True])
def test_initialization_budget_exhaustion_reserves_final_confirmations(tmp_path, invalid):
    backend = SyntheticBackend([0, 0])
    def measured(p):
        obs = observation(p, invalid=invalid and backend.acquisitions > 2 and np.any(p))
        if obs['metrics']['valid']:
            obs['metrics']['primary_metric'] = .01
        return obs
    c = Controller(settings(target_m2=None, bo_initial_samples=20, max_acquisitions=8),
                   backend, tmp_path / 'log', measured)
    result = c.run()
    assert result['reason'] == 'random initialization budget exhausted'
    assert result['acquisitions'] == 8
    assert result['status'] == 'best_found'
    assert [o['stage'] for o in c.history[-2:]] == ['final', 'final']
    assert not any(o['stage'] in ('restore', 'bo') for o in c.history)


def test_exclusions_use_current_anisotropic_scales_and_actual_readback(tmp_path):
    backend = SyntheticBackend([4, 0])
    c = Controller(settings(target_m2=None, max_step=3, bo_halfwidth=[4, 3]),
                   backend, tmp_path / 'log')
    best = observation([0, 0])
    high = observation([2, 0])
    high['metrics']['primary_metric'] = .03
    invalid = observation([3, 2], invalid=True)
    c.history = [best, high, invalid]
    c.length_scales = np.array([2., 1.])
    pool = c.candidates(best)
    assert len(pool)
    assert np.all(np.linalg.norm(pool - [4, 0], axis=1) <= 3)
    assert np.all(np.abs(pool) <= [4, 3])
    for rejected in [high, invalid]:
        assert np.all(np.sum(((pool - rejected['position']) / [2., 1.]) ** 2, axis=1) > 1)
    assert not any(np.array_equal(p, [2, 1]) for p in pool)  # On exclusion boundary.
    c.length_scales = np.array([.5, .5])
    smaller = c.candidates(best)
    assert len(smaller) > len(pool)
    assert any(np.array_equal(p, [2, 1]) for p in smaller)


def test_exhausted_exclusions_stop_initialization_and_confirm_best(tmp_path):
    backend = SyntheticBackend([0, 0])
    backend.candidate_positions = np.array([[1, 0]])
    def measured(p):
        obs = observation(p)
        if np.any(p):
            obs['metrics']['primary_metric'] = .03
        return obs
    c = Controller(settings(target_m2=None), backend, tmp_path / 'log', measured)
    result = c.run()
    assert 'no sampleable candidates' in result['reason']
    assert 'random initialization incomplete' in result['reason']
    assert result['status'] == 'best_found'
    assert c.history[1]['bo_rejection_reason'].startswith('M1 exceeds')
    assert result['acquisitions'] == 4
    assert result['final_readback'] == [0, 0]


@pytest.mark.parametrize('replay', [False, True])
def test_final_return_uses_bounded_transits_without_extra_images(tmp_path, replay):
    backend = (Replay(dict(units='um', samples=[dict(position=[x, 0], image=[x, 0])
                                               for x in [0, 3, 6]]), [0, 0])
               if replay else SyntheticBackend([0, 0]))
    if not replay:
        backend.candidate_positions = np.array([[3, 0], [6, 0]])
    def measured(p):
        obs = observation(p)
        obs['metrics']['primary_metric'] = .01
        return obs
    c = Controller(settings(target_m2=None, max_step=3, bo_halfwidth=[8, 8],
                   bo_initial_samples=2, max_acquisitions=6, seed=1), backend, tmp_path / 'log', measured)
    result = c.run()
    assert result['status'] == 'best_found'
    assert result['acquisitions'] == 5
    assert result['final_readback'] == [0, 0]
    assert [o['position'] for o in c.history] == [[0, 0], [3, 0], [6, 0], [0, 0], [0, 0]]
    events = list(map(json.loads, (tmp_path / 'log').read_text().splitlines()))
    transits = [e for e in events if e['event'] == 'plan' and e['stage'] == 'transit']
    assert len(transits) >= 2
    assert all(np.linalg.norm(np.array(e['target']) - e['previous_readback']) <= 3 for e in transits)


@pytest.mark.parametrize('replay', [False, True])
def test_final_return_uses_lowest_m2_after_bo_started(tmp_path, replay):
    backend = (Replay(dict(units='um', samples=[dict(position=[x, 0], image=[x, 0])
                                               for x in [0, 3, 6]]), [0, 0])
               if replay else SyntheticBackend([0, 0]))
    if not replay:
        backend.candidate_positions = np.array([[3, 0], [6, 0]])
    def measured(p):
        obs = observation(p)
        obs['metrics']['primary_metric'] = {0: .01, 3: .015, 6: .005}[p[0]]
        obs['metrics']['secondary_metric'] = {0: 10., 3: 5., 6: 20.}[p[0]]
        return obs
    c = Controller(settings(target_m2=None, max_step=3, bo_halfwidth=[8, 8],
                   bo_initial_samples=2, max_acquisitions=6), backend, tmp_path / 'log', measured)
    result = c.run()
    assert result['status'] == 'best_found'
    assert result['acquisitions'] == 5
    assert result['final_readback'] == [3, 0]
    assert result['best_observed']['position'] == [3, 0]
    assert [o['position'] for o in c.history] == [[0, 0], [6, 0], [3, 0], [3, 0], [3, 0]]


@pytest.mark.parametrize('options', [dict(bo_initial_samples=-1), dict(bo_initial_samples=1.5),
    dict(gp_length_scale_bounds=[0, 10]), dict(gp_length_scale_bounds=[3, 2]),
    dict(gp_length_scale_bounds=[.1, float('inf')]), dict(gp_length_scale=[.1, 2]),
    dict(gp_stage1_noise_multiplier=.5), dict(gp_stage1_noise_multiplier=float('inf')),
    dict(gp_stage1_noise_multiplier=float('nan'))])
def test_bo_settings_validation(options):
    with pytest.raises(ValueError):
        settings(**options).validate()


def test_zero_random_samples_enters_ei_directly(tmp_path):
    c = Controller(settings(target_m2=None, bo_initial_samples=0, max_acquisitions=6),
                   SyntheticBackend([0, 0]), tmp_path / 'log', lambda p: observation(p))
    c.run()
    assert c.history[1]['stage'] == 'bo'
    assert not any(o['stage'] == 'bo_init' for o in c.history)


@pytest.mark.parametrize('metric', ['primary_metric', 'secondary_metric'])
@pytest.mark.parametrize('value', [float('inf'), float('-inf'), float('nan')])
def test_nonfinite_metrics_never_train_gp(metric, value):
    bad = observation([1, 0])
    bad['metrics'][metric] = value
    assert not feasible(bad, settings())
    gp, _ = fit_objective([observation([0, 0]), bad], settings())
    np.testing.assert_array_equal(gp.X_train_, [[0, 0]])


def test_final_transit_readback_fault_stops_without_confirmation(tmp_path):
    class FaultyTransit(SyntheticBackend):
        def move_absolute(self, target):
            super().move_absolute(np.array(target) + ([1, 0] if self.acquisitions >= 3 else [0, 0]))
    backend = FaultyTransit([0, 0])
    backend.candidate_positions = np.array([[3, 0], [6, 0]])
    def measured(p):
        obs = observation(p)
        obs['metrics']['primary_metric'] = .01
        return obs
    c = Controller(settings(target_m2=None, max_step=3, bo_halfwidth=[8, 8],
                   bo_initial_samples=2, max_acquisitions=6), backend, tmp_path / 'log', measured)
    result = c.run()
    assert result['status'] == 'blocked'
    assert 'readback' in result['reason']
    assert result['acquisitions'] == backend.acquisitions == 3
    assert len(backend.moves) == 3
    assert not any(o['stage'] == 'final' for o in c.history)


def events(controller):
    return [json.loads(line) for line in controller.log_path.read_text().splitlines()]


def test_m1_gp_uses_failed_fits_and_high_scores():
    observations = [observation([0, 0]), observation([4, 0], invalid=True), observation([10, 0])]
    gp, fitted = fit_objective(observations, settings(), 'primary_metric')
    assert fitted
    np.testing.assert_array_equal(gp.X_train_, [[0, 0], [4, 0], [10, 0]])
    np.testing.assert_allclose(gp.y_train_ * gp._y_train_std + gp._y_train_mean,
                               [o['metrics']['primary_metric'] for o in observations])
    np.testing.assert_allclose(gp.alpha, settings().gp_noise ** 2)


@pytest.mark.parametrize('invalid', [False, True])
def test_m1_threshold_enters_stage2_without_entry_confirmations(tmp_path, invalid):
    backend = SyntheticBackend([0, 0])
    c = Controller(settings(target_m2=None), backend, tmp_path / 'log',
                   lambda p: observation(p, invalid=invalid))
    c.run()
    transitions = [e for e in events(c) if e['event'] == 'stage_transition']
    assert len(transitions) == 1
    assert transitions[0]['reason'] == 'M1 threshold reached'
    assert transitions[0]['stage1_acquisitions'] == 1
    assert not any(o['stage'] in ('stage1', 'stage1_init', 'confirm') for o in c.history)
    assert c.history[1]['stage'] == 'bo_init'


def test_stage1_budget_enters_stage2_without_eligible_m2(tmp_path, monkeypatch):
    backend = SyntheticBackend([0, 0])

    def measured(p):
        obs = observation(p, invalid=backend.acquisitions <= 2)
        obs['metrics']['primary_metric'] = .1 if backend.acquisitions <= 2 else .01
        return obs

    c = Controller(settings(stage1_max_acquisitions=2, max_acquisitions=8,
                            bo_initial_samples=0, target_m2=None),
                   backend, tmp_path / 'log', measured)
    monkeypatch.setattr(c, 'stage1_candidates', lambda **kwargs: np.array([[10., 0.]]))
    pools = iter([np.array([[5., 0.]]), np.empty((0, 2))])
    monkeypatch.setattr(c, 'candidates', lambda incumbent, **kwargs: next(pools))
    result = c.run()
    assert result['status'] == 'best_found'
    assert [o['stage'] for o in c.history] == ['baseline', 'stage1', 'bo_init', 'final', 'final']
    transition = next(e for e in events(c) if e['event'] == 'stage_transition')
    assert transition['reason'] == 'stage 1 budget exhausted'
    assert transition['stage1_acquisitions'] == 2
    models = [e for e in events(c) if e['event'] == 'bo_model']
    assert [e['objective'] for e in models] == ['primary_metric', 'secondary_metric']
    assert models[0]['training_acquisitions'] == [1]
    assert models[1]['training_acquisitions'] == [3]
    assert all(np.linalg.norm(after - before) <= 5 + 1e-9 for before, after in backend.moves)
    assert result['final_readback'] == [5., 0.]


@pytest.mark.parametrize('continuous', [False, True])
def test_stage1_candidates_cover_full_bounds_without_m1_exclusions(tmp_path, continuous):
    backend = SyntheticBackend([0, 0])
    if continuous:
        del backend.candidate_positions
    c = Controller(settings(bo_halfwidth=[1, 1], max_step=2), backend, tmp_path / 'log')
    c.history = [observation([0, 0], invalid=True), observation([10, 10], invalid=True)]
    pool = c.stage1_candidates()
    assert len(pool)
    assert np.all(pool >= -15) and np.all(pool <= 15)
    assert np.any(np.linalg.norm(pool, axis=1) > c.s.max_step)
    assert np.all(pool.min(axis=0) < -10) and np.all(pool.max(axis=0) > 10)
    # Invalid Stage 1 scores do not exclude their surrounding region.
    assert np.any(np.linalg.norm(pool - [10, 10], axis=1) < 2.5)
    assert np.all(np.linalg.norm(pool, axis=1) >= c.s.min_step)


@pytest.mark.parametrize('replay', [False, True])
def test_stage1_global_search_optimizes_m1_and_uses_bounded_transits(tmp_path, replay):
    points = [[x, 0] for x in range(-12, 13, 3)]
    backend = (Replay(dict(units='um', samples=[dict(position=p, image=p) for p in points]), [12, 0])
               if replay else SyntheticBackend([12, 0]))
    if not replay:
        backend.candidate_positions = np.array(points)
    c = Controller(settings(max_step=3, stage1_max_acquisitions=10, max_acquisitions=16,
                            bo_initial_samples=2, bo_stage_entrance_m1=.011, target_m2=10),
                   backend, tmp_path / 'log', observation)
    result = c.run()
    assert result['status'] == 'metric_target_reached'
    assert result['final_readback'] == [0, 0]
    stage1 = [o for o in c.history if o['stage'] in ('stage1', 'stage1_init')]
    assert stage1 and min(o['metrics']['primary_metric'] for o in stage1) == .01
    assert c.acquisitions <= 16
    moves = [e for e in events(c) if e['event'] == 'plan']
    assert all(np.linalg.norm(np.array(e['target']) - e['previous_readback']) <= 3 + 1e-9
               for e in moves if e['stage'] not in ('stage1_init', 'bo_init'))
    assert any(e['stage'] == 'transit' for e in moves)
    assert not any(o['stage'] == 'restore' for o in c.history)


def test_budget_entry_with_no_eligible_m2_returns_lowest_m1(tmp_path, monkeypatch):
    backend = SyntheticBackend([0, 0])

    def measured(p):
        obs = observation(p, invalid=True)
        obs['metrics']['primary_metric'] = .05 if p[0] else .1
        return obs

    c = Controller(settings(stage1_max_acquisitions=2, max_acquisitions=6, target_m2=None),
                   backend, tmp_path / 'log', measured)
    monkeypatch.setattr(c, 'stage1_candidates', lambda **kwargs: np.array([[10., 0.]]))
    monkeypatch.setattr(c, 'candidates', lambda incumbent, **kwargs: np.empty((0, 2)))
    result = c.run()
    assert result['status'] == 'unconfirmed'
    assert result['final_readback'] == result['best_observed']['position'] == [10., 0.]
    assert [o['stage'] for o in c.history[-2:]] == ['final', 'final']
    assert any(e['event'] == 'stage_transition' for e in events(c))


def test_depleted_stage1_domain_uses_budget_then_transitions(tmp_path):
    backend = SyntheticBackend([0, 0])
    backend.candidate_positions = np.array([[0, 0]])

    def measured(p):
        obs = observation(p, invalid=True)
        obs['metrics']['primary_metric'] = .1
        return obs

    c = Controller(settings(stage1_max_acquisitions=3, max_acquisitions=7), backend, tmp_path / 'log', measured)
    result = c.run()
    assert result['status'] == 'unconfirmed'
    transition = next(e for e in events(c) if e['event'] == 'stage_transition')
    assert transition['reason'] == 'stage 1 budget exhausted'
    assert transition['stage1_acquisitions'] == 3
    assert backend.acquisitions == 5


@pytest.mark.parametrize('value', [0, -1, 1.5, 6, float('inf')])
def test_stage1_budget_validation_preserves_stage2_and_confirmations(value):
    with pytest.raises(ValueError):
        settings(max_acquisitions=8, stage1_max_acquisitions=value).validate()


def test_bundled_config_needs_no_calibration():
    runtime = json.loads((Path(__file__).resolve().parents[1] / 'alignment.json').read_text())
    Settings(**runtime).validate()
    assert 'eta' not in runtime
    assert 'centroid_percentile' not in runtime


def ring_image():
    y, x = np.indices((100, 100))
    r = np.hypot(x - 50, y - 50)
    return np.where((r >= 12) & (r <= 30), 100., 0.)


def test_nonconvergent_optimizer_returns_finite_m1_from_returned_geometry(monkeypatch):
    import evaluate_alignment as module
    original = module.least_squares
    fits = []

    def limited(*args, **kwargs):
        kwargs['max_nfev'] = 1
        fit = original(*args, **kwargs)
        fits.append(fit)
        return fit

    monkeypatch.setattr(module, 'least_squares', limited)
    m = module.evaluate(ring_image(), Config(inner_axes=(12, 12), outer_axes=(30, 30)))
    assert not m['valid']
    assert 'Optimizer did not converge' in m['reasons']
    assert np.isfinite(m['primary_metric'])
    expected = ((m['inner_rms_px'] + m['outer_rms_px']) / (2 * 30)
                + .5 * (1 - (m['inner_coverage'] + m['outer_coverage']) / 2))
    assert m['primary_metric'] == pytest.approx(expected)
    assert m['primary_metric'] == m['geometry_score']
    assert any(np.allclose(fit.x[:2], [m['fit']['center_x'], m['fit']['center_y']]) for fit in fits)
    assert np.isinf(m['secondary_metric'])


@pytest.mark.parametrize('fixed', [False, True])
@pytest.mark.parametrize('image_kind', ['blank', 'no_seeds', 'no_edges', 'partial_ring'])
def test_failed_boundary_fits_always_produce_finite_m1(monkeypatch, fixed, image_kind):
    import evaluate_alignment as module
    image = ring_image()
    if image_kind == 'blank':
        image[:] = 0
    elif image_kind == 'no_seeds':
        monkeypatch.setattr(module, '_seeds', lambda *args: [])
    elif image_kind == 'no_edges':
        monkeypatch.setattr(module, 'canny', lambda image, **kwargs: np.zeros_like(image, dtype=bool))
    else:
        image[:50] = 0
    config = Config(inner_axes=(12, 12), outer_axes=(30, 30)) if fixed else Config()
    m = module.evaluate(image, config)
    assert np.isfinite(m['primary_metric'])
    assert m['primary_metric'] == m['geometry_score']
    assert m['fit'] is not None
    if image_kind in ('blank', 'no_edges', 'partial_ring'):
        assert not m['valid']
        assert m['reasons']
        assert np.isinf(m['secondary_metric'])
    if image_kind in ('blank', 'no_edges'):
        assert m['primary_metric'] > .5


@pytest.mark.parametrize('fixed', [False, True])
def test_smallest_supported_image_still_has_finite_m1(fixed):
    from evaluate_alignment import evaluate
    config = Config(inner_axes=(.1, .1), outer_axes=(100, 100)) if fixed else Config()
    m = evaluate(np.zeros((16, 16)), config)
    assert np.isfinite(m['primary_metric'])
    assert not m['valid']


@pytest.mark.parametrize('stage', [1, 2])
def test_random_initialization_ignores_step_filters_but_keeps_search_domain(tmp_path, stage):
    backend = SyntheticBackend([0, 0])
    backend.candidate_positions = np.array([[0, 0], [.1, 0], [.4, 0], [12, 0], [14, 0], [16, 0]])
    c = Controller(settings(min_step=1, max_step=2, bo_halfwidth=[12, 12]),
                   backend, tmp_path / 'log')
    baseline = observation([0, 0])
    c.history = [baseline]
    if stage == 1:
        initial = c.stage1_candidates(initial=True)
        later = c.stage1_candidates()
        np.testing.assert_array_equal(initial, [[.1, 0], [.4, 0], [12, 0], [14, 0]])
        np.testing.assert_array_equal(later, [[12, 0], [14, 0]])
    else:
        initial = c.candidates(baseline, initial=True)
        later = c.candidates(baseline)
        np.testing.assert_array_equal(initial, [[.1, 0], [.4, 0], [12, 0]])
        assert len(later) == 0


@pytest.mark.parametrize('stage', [1, 2])
def test_random_acquisitions_move_directly_to_near_and_far_targets(tmp_path, stage):
    backend = SyntheticBackend([0, 0])
    backend.candidate_positions = np.array([[.25, 0], [12, 0]])

    def measured(p):
        obs = observation(p)
        obs['metrics']['primary_metric'] = .1 if stage == 1 else .01
        return obs

    c = Controller(settings(min_step=1, max_step=2, bo_halfwidth=[15, 15],
                            bo_initial_samples=3 if stage == 1 else 2,
                            stage1_max_acquisitions=3, max_acquisitions=7, target_m2=None),
                   backend, tmp_path / 'log', measured)
    result = c.run()
    assert result['status'] == ('unconfirmed' if stage == 1 else 'best_found')
    initial = [o for o in c.history if o['stage'] == ('stage1_init' if stage == 1 else 'bo_init')]
    assert len(initial) == 2
    np.testing.assert_array_equal([o['position'] for o in initial], [[12, 0], [.25, 0]])
    assert backend.acquisitions == 5
    np.testing.assert_array_equal([after for _, after in backend.moves[:2]], [[12, 0], [.25, 0]])
    # Only the later return to the final position uses transit moves.
    plans = [e for e in events(c) if e['event'] == 'plan']
    assert [e['stage'] for e in plans[:3]] == ['baseline'] + [initial[0]['stage']] * 2
    assert all('Move directly' in e['text'] for e in plans[1:3])
    assert all(np.linalg.norm(np.array(e['target']) - e['previous_readback']) <= 2 + 1e-9
               for e in plans if e['stage'] not in ('stage1_init', 'bo_init'))


def test_stage2_random_replay_allows_direct_moves_to_distant_recorded_locations(tmp_path):
    backend = Replay(dict(units='um', samples=[dict(position=[x, 0], image=[x, 0])
                                               for x in [0, .25, 3, 6, 12]]), [0, 0])
    c = Controller(settings(max_step=3, bo_halfwidth=[15, 15]), backend, tmp_path / 'log')
    baseline = observation([0, 0])
    c.history = [baseline]
    pool = c.candidates(baseline, initial=True)
    np.testing.assert_array_equal(pool, [[.25, 0], [3, 0], [6, 0], [12, 0]])


@pytest.mark.parametrize('stage', ['stage1_init', 'bo_init'])
def test_initial_moves_still_enforce_global_bounds_and_readbacks(tmp_path, stage):
    backend = SyntheticBackend([0, 0])
    c = Controller(settings(max_step=2), backend, tmp_path / 'log', observation)
    with pytest.raises(RuntimeError, match='bounds'):
        c.acquire([16, 0], 'Outside global bounds', stage)
    assert backend.moves == []
    assert backend.acquisitions == 0
    # The physical step limit still applies outside initialization.
    with pytest.raises(RuntimeError, match='maximum displacement'):
        c.acquire([12, 0], 'Normal BO', 'bo')
    assert backend.moves == []

    class Faulty(SyntheticBackend):
        def move_absolute(self, target):
            super().move_absolute(np.asarray(target) + [1, 0])

    faulty = Faulty([0, 0])
    c = Controller(settings(max_step=2), faulty, tmp_path / 'fault', observation)
    with pytest.raises(RuntimeError, match='readback'):
        c.acquire([12, 0], 'Direct initialization', stage)
    assert len(faulty.moves) == 1
    assert faulty.acquisitions == 0


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize('fraction', [0., .5, 1.])
def test_readback_bounds_allow_configured_tolerance(tmp_path, axis, side, fraction):
    point = np.zeros(2)
    point[axis] = side * (15 + fraction * .01)
    c = Controller(settings(readback_tolerance=.01), SyntheticBackend(point), tmp_path / 'log')
    np.testing.assert_array_equal(c.position(), point)


@pytest.mark.parametrize('point', [[15.011, 0], [-15.011, 0], [0, 15.011], [0, -15.011],
                                   [float('nan'), 0], [0, float('inf')], [0], [0, 0, 0]])
def test_invalid_readbacks_still_stop_before_motion(tmp_path, point):
    backend = SyntheticBackend(point)
    c = Controller(settings(readback_tolerance=.01), backend, tmp_path / 'log')
    with pytest.raises(RuntimeError, match='Readback'):
        c.position()
    assert backend.moves == []


def test_commanded_targets_stay_inside_exact_bounds_with_readback_tolerance(tmp_path):
    backend = SyntheticBackend([15, 0])
    c = Controller(settings(readback_tolerance=.01), backend, tmp_path / 'log', observation)
    with pytest.raises(RuntimeError, match='bounds'):
        c.acquire([15.005, 0], 'Outside exact command bounds', 'stage1_init')
    assert backend.moves == []
    assert backend.acquisitions == 0


@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize('eligible', [False, True])
def test_tolerated_boundary_readbacks_survive_baseline_reimages_and_final_return(tmp_path, side, eligible):
    class BoundaryReadback(SyntheticBackend):
        def read_position(self):
            return super().read_position() + side * np.array([.0005, .0005])

    point = [side * 15, side * 15]
    backend = BoundaryReadback(point)
    backend.candidate_positions = np.array([point])

    def measured(p):
        obs = observation(p)
        obs['metrics']['primary_metric'] = .01 if eligible else .1
        return obs

    c = Controller(settings(stage1_max_acquisitions=3, bo_initial_samples=0,
                            max_acquisitions=7, target_m2=1000 if eligible else None),
                   backend, tmp_path / 'log', measured)
    result = c.run()
    assert result['status'] == ('metric_target_reached' if eligible else 'unconfirmed')
    assert [o['stage'] for o in c.history[-2:]] == ['final', 'final']
    if not eligible:
        assert len([o for o in c.history if o['stage'] == 'stage1']) == 2
    for obs in c.history:
        assert obs['target'] == point
        np.testing.assert_array_equal(obs['position'], np.array(point) + side * .0005)
    assert all(np.all(np.abs(after) <= 15) for _, after in backend.moves)
    np.testing.assert_array_equal(result['final_readback'], np.array(point) + side * .0005)


@pytest.mark.parametrize('options', [dict(min_step=-1), dict(min_step=6, max_step=5),
                                    dict(min_step=float('nan')), dict(min_step=float('inf')),
                                    dict(max_step=0), dict(max_step=-1)])
def test_invalid_step_settings_explain_the_requirement(options):
    with pytest.raises(ValueError, match='0 <= min_step <= max_step'):
        settings(**options).validate()


@pytest.mark.parametrize('stage', [1, 2])
@pytest.mark.parametrize('initial', [False, True])
def test_zero_min_step_allows_close_samples_but_excludes_measured_positions(tmp_path, stage, initial):
    backend = SyntheticBackend([0, 0])
    backend.candidate_positions = np.array([[0, 0], [.00001, 0], [1, 0], [10, 0]])
    c = Controller(settings(min_step=0, max_step=5, bo_halfwidth=[15, 15]),
                   backend, tmp_path / 'log')
    baseline = observation([0, 0])
    c.history = [baseline, observation([1, 0])]
    pool = c.stage1_candidates(initial=initial) if stage == 1 else c.candidates(baseline, initial=initial)
    expected = [[.00001, 0], [10, 0]] if stage == 1 or initial else [[.00001, 0]]
    np.testing.assert_array_equal(pool, expected)
