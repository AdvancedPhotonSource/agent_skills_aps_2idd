"""Synthetic calibration and control tests; no instrument or recorded data needed."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibrate_alignment import centroid_offset, fit_response, measure
from align_osa import Controller, Replay, Settings, fit_objective, expected_improvement, feasible
from evaluate_alignment import Config


def calibration():
    return dict(eta=1., jacobian=[[1., 0.], [0., 1.]], reference_error=[0., 0.])


def observation(position, *, invalid=False):
    p = np.asarray(position)
    g = .01 + .001 * np.sum(p ** 2)
    return dict(image="synthetic", error=p.tolist(), position=p.tolist(), metrics=dict(
        valid=not invalid, reasons=["synthetic missing boundary"] if invalid else [],
        primary_metric=float("inf") if invalid else g,
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
    return Settings(**dict(dict(units="um", bounds=[[-15, 15], [-15, 15]], max_step=5.,
                                min_step=1., target_m2=10., max_acquisitions=40), **options))


def test_calibration_gain_and_flipped_coupled_axes():
    p = np.array([(x, y) for x in [-4, 0, 4] for y in [-4, 0, 4] if x or y])
    j = np.array([[-2., .3], [.7, 1.5]])
    fit = fit_response(p, p @ j.T, [.25, .5, 1., 2.], 100.)
    assert fit["eta"] == 1.
    assert fit["diagnostics"]["held_out_toward_reference_fraction"] == 1.
    assert np.allclose(fit["jacobian"], j)
    assert np.allclose(np.linalg.solve(fit["jacobian"], p[0] @ j.T), p[0])


def test_calibration_rejects_degenerate_response():
    p = np.array([[-2, 0], [-1, 0], [1, 0], [2, 0]])
    with pytest.raises(ValueError, match="both axes"):
        fit_response(p, p, [1.], 5.)


def test_centroid_uses_outer_ellipse_and_background():
    image = np.full((40, 40), 10.)
    image[20, 20] = 20.
    image[20, 22] = 20.
    image[0, 0] = 10000.  # Exterior scattering must not determine the direction.
    offset = centroid_offset(image, observation([0, 0])["metrics"])
    assert np.allclose(offset, [1, 0])


def test_vector_corrections_then_bo_and_confirmed_target(tmp_path):
    backend = SyntheticBackend([12, 8])
    controller = Controller(calibration(), settings(eta=.5), backend, tmp_path / "log.jsonl",
                            lambda p, c: observation(p))
    result = controller.run()
    assert result["status"] == "metric_target_reached"
    assert result["final_readback"] == [0, 0]
    coarse = [o for o in controller.history if o["stage"] == "coarse"]
    assert np.all(np.array(coarse[0]["position"]) < [12, 8])
    assert np.all(np.array(coarse[1]["position"]) < coarse[0]["position"])
    assert any(o["stage"] == "bo" for o in controller.history)
    assert backend.acquisitions == result["acquisitions"] <= 40
    for before, after in backend.moves:
        assert np.linalg.norm(after - before) <= 5 + 1e-9
        assert np.all(np.abs(after) <= 15)
    events = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    for event in events:
        if event["event"] == "observation":
            assert all(event.get(key) for key in ["description", "trend", "next_plan"])


def test_rejected_trial_restores_before_further_search(tmp_path):
    backend = SyntheticBackend([10, 0])
    c = calibration()
    c["jacobian"] = [[-1., 0.], [0., 1.]]
    controller = Controller(c, settings(max_acquisitions=9), backend, tmp_path / "log",
                            lambda p, c: observation(p))
    result = controller.run()
    assert controller.history[1]["position"] == [15, 0]
    assert controller.history[2]["stage"] == "restore"
    assert controller.history[2]["position"] == [10, 0]
    assert result["acquisitions"] <= 9
    assert [o["stage"] for o in controller.history[-2:]] == ["final", "final"]


def test_gp_uses_only_eligible_objectives():
    observations = [observation([0, 0]), observation([1, 0]), observation([10, 10], invalid=True)]
    gp, fitted = fit_objective(observations, settings())
    assert len(gp.X_train_) == 2 and not fitted
    scores = expected_improvement(gp, np.array([[0, 1], [9, 10]]), 10., settings())
    assert np.isfinite(scores).all()
    assert np.all(scores >= 0)


def test_readback_fault_stops_without_recovery_motion(tmp_path):
    class Faulty(SyntheticBackend):
        def move_absolute(self, target):
            super().move_absolute(np.array(target) + [1, 0])
    backend = Faulty([10, 0])
    result = Controller(calibration(), settings(), backend, tmp_path / "log",
                        lambda p, c: observation(p)).run()
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
                                   dict(bounds=[[15, -15], [-15, 15]]), dict(eta=0),
                                   dict(max_acquisitions=3)])
def test_unsafe_settings_rejected(options):
    with pytest.raises(ValueError):
        settings(**options).validate(calibration())


def test_no_feasible_point_is_not_success(tmp_path):
    backend = SyntheticBackend([0, 0])
    result = Controller(calibration(), settings(max_acquisitions=9), backend, tmp_path / "log",
                        lambda p, c: observation(p, invalid=True)).run()
    assert result["status"] == "unconfirmed"
    assert backend.acquisitions <= 9


def test_failed_acquisition_counts_and_stops(tmp_path):
    class Faulty(SyntheticBackend):
        def acquire_image(self):
            super().acquire_image()
            raise RuntimeError("detector fault")
    backend = Faulty([0, 0])
    result = Controller(calibration(), settings(), backend, tmp_path / "log").run()
    assert result["status"] == "blocked"
    assert result["acquisitions"] == backend.acquisitions == 1
    assert backend.moves == []


def test_final_target_must_survive_confirmation(tmp_path):
    backend = SyntheticBackend([0, 0])
    def measured(p, c):
        result = observation(p)
        if backend.acquisitions > 1:
            result["metrics"]["secondary_metric"] = 20.
        return result
    result = Controller(calibration(), settings(), backend, tmp_path / "log", measured).run()
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
    c = Controller(calibration(), settings(bo_halfwidth=[2, 3]), backend, tmp_path / "log",
                   lambda p, cal: observation(p))
    incumbent = observation([5, 5])
    c.history.append(incumbent)
    pool = c.candidates(incumbent)
    delta = pool - [5, 5]
    assert np.all(np.abs(delta) <= [2, 3])
    assert np.all(np.linalg.norm(delta, axis=1) <= 5)
    assert np.all(delta.min(axis=0) < 0) and np.all(delta.max(axis=0) > 0)


def test_direction_solves_jacobian_and_uses_runtime_thresholds(tmp_path):
    cal = calibration()
    j = np.array([[-2., .3], [.7, 1.5]])
    cal['jacobian'] = j.tolist()
    obs = observation([4, 2])
    obs['error'] = (j @ [4, 2]).tolist()
    c = Controller(cal, settings(eta=.5), SyntheticBackend([4, 2]), tmp_path / 'normal')
    np.testing.assert_allclose(c.direction(obs), [-2., -1.])
    c = Controller(cal, settings(min_vector=100), SyntheticBackend([4, 2]), tmp_path / 'fallback')
    assert c.direction(obs) is None


@pytest.mark.parametrize('eta,length', [(10., 5.), (.01, 1.)])
def test_vector_step_limits_preserve_direction(tmp_path, eta, length):
    c = Controller(calibration(), settings(eta=eta), SyntheticBackend([3, 4]), tmp_path / 'log')
    correction = c.direction(observation([3, 4]))
    np.testing.assert_allclose(correction, np.array([-3., -4.]) * length / 5)


def test_continuous_coarse_step_preserves_vector_at_bounds(tmp_path):
    backend = SyntheticBackend([14, 10])
    del backend.candidate_positions
    c = Controller(calibration(), settings(), backend, tmp_path / 'log')
    # Shorten both components by the same factor at the x bound.
    np.testing.assert_allclose(c.coarse_trial(observation([14, 10]), np.array([3., 4.])),
                               [15., 10. + 4. / 3.])
    assert c.coarse_trial(observation([15, 10]), np.array([3., 4.])) is None


def test_untrusted_centroid_uses_2d_probe(tmp_path):
    backend = SyntheticBackend([-10, -10])
    del backend.candidate_positions
    def measured(p, cal):
        obs = observation(p)
        obs['error'] = None
        return obs
    c = Controller(calibration(), settings(max_acquisitions=7), backend, tmp_path / 'log', measured)
    c.run()
    first = c.history[1]
    assert first['stage'] == 'coarse'
    np.testing.assert_allclose(np.array(first['position']) + 10, [5 / np.sqrt(2)] * 2)


def test_rejected_vector_is_restored_reversed_and_shortened(tmp_path):
    backend = SyntheticBackend([0, 0])
    del backend.candidate_positions
    def measured(p, cal):
        obs = observation(p)
        obs['error'] = [3., 4.]
        distance = np.linalg.norm(p)
        score = .2 if distance == 0 else (.1 if distance < 2 else .3)
        obs['metrics']['geometry_score'] = obs['metrics']['primary_metric'] = score
        return obs
    c = Controller(calibration(), settings(max_acquisitions=13), backend, tmp_path / 'log', measured)
    c.run()
    coarse = [o for o in c.history if o['stage'] == 'coarse']
    np.testing.assert_allclose([o['position'] for o in coarse[:5]],
                               [[-3, -4], [3, 4], [-1.5, -2], [1.5, 2], [-.75, -1]])
    assert all(o['position'] == [0., 0.] for o in c.history[:9] if o['stage'] == 'restore')


@pytest.mark.parametrize('field,value', [('min_vector', -1), ('min_coverage', 1.1),
                                         ('centroid_percentile', 100), ('min_vector', float('nan'))])
def test_runtime_confidence_settings_validated(field, value):
    with pytest.raises(ValueError):
        settings(**{field: value}).validate(calibration())


@pytest.mark.parametrize('error', [[0], [0, 0, 0], [float('nan'), 0]])
def test_invalid_reference_error_rejected(error):
    cal = dict(calibration(), reference_error=error)
    with pytest.raises(ValueError, match='reference_error'):
        settings().validate(cal)


def test_runtime_limits_are_independent_of_calibration():
    settings(bounds=[[-100, 100], [-100, 100]], max_step=20).validate(calibration())


def test_direction_independent_of_absolute_fitted_center(tmp_path):
    c = Controller(calibration(), settings(), SyntheticBackend([4, 2]), tmp_path / 'log')
    obs = observation([4, 2])
    expected = c.direction(obs)
    obs['metrics']['fit'].update(center_x=2500., center_y=1800.)
    obs['metrics']['image_shape'] = [3000, 4000]
    np.testing.assert_array_equal(c.direction(obs), expected)


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
    cal = dict(calibration(), reference_error=[.2, -.1])
    before, after = [measure(path, cal, config=Config(**evaluator)) for path in paths]
    assert before['metrics']['valid'] and after['metrics']['valid']
    np.testing.assert_allclose(before['error'], after['error'], atol=.05)
    assert before['metrics']['image_shape'] != after['metrics']['image_shape']

    class Images(SyntheticBackend):
        def acquire_image(self):
            super().acquire_image()
            return paths[(self.acquisitions - 1) % 2]

    backend = Images([0, 0])
    c = Controller(cal, settings(evaluator=evaluator, bo_stage_entrance_m1=.05,
                                target_m2=1e9), backend, tmp_path / 'trajectory')
    result = c.run()
    assert result['status'] == 'metric_target_reached'
    assert len({tuple(o['metrics']['image_shape']) for o in c.history}) == 2


def test_calibration_cli_outputs_independently_usable_files(tmp_path, monkeypatch):
    import calibrate_alignment as module
    positions = [(x, y) for x in [-4, 0, 4] for y in [-4, 0, 4]]
    paths = {str(tmp_path / f'{i}.tiff'): p for i, p in enumerate(positions)}
    j = np.array([[-2., .3], [.7, 1.5]])
    offset = np.array([.2, -.3])
    def measured(path, cal, **kwargs):
        obs = observation(paths[str(path)])
        obs['image'] = str(path)
        obs['error'] = (j @ paths[str(path)] + offset).tolist()
        return obs
    reference_metrics = observation([0, 0])['metrics']
    reference_metrics['fit']['inner_axes'] = [3., 3.]
    monkeypatch.setattr(module, 'read_image', lambda path: np.zeros((40, 40)))
    monkeypatch.setattr(module, 'evaluate', lambda image, config: reference_metrics)
    monkeypatch.setattr(module, 'measure', measured)
    manifest = tmp_path / 'images.json'
    manifest.write_text(json.dumps({'units': 'um', 'samples': [
        {'position': p, 'image': path} for path, p in paths.items()]}))
    output = tmp_path / 'calibration.json'
    monkeypatch.setattr(sys, 'argv', ['calibrate_alignment.py', str(manifest),
        '--reference-position', '0', '0', '--max-step', '4', '--output', str(output),
        '--eta-grid', '.25', '.5', '1', '--min-vector', '2', '--min-coverage', '.7'])
    module.main()
    cal = json.loads(output.read_text())
    runtime = json.loads((tmp_path / 'calibration.alignment.json').read_text())
    report = json.loads((tmp_path / 'calibration.report.json').read_text())
    assert np.allclose(cal['jacobian'], j)
    assert np.allclose(cal['reference_error'], offset)
    assert set(cal) == {'eta', 'jacobian', 'reference_error'}
    assert report['diagnostics']['held_out_positions'] == 8
    assert runtime['min_vector'] == 2 and runtime['min_coverage'] == .7
    assert runtime['max_step'] == 4 and 'evaluator' not in runtime
    controller = Controller(cal, Settings(**runtime), SyntheticBackend([4, 0]),
                            tmp_path / 'trajectory', lambda p, c: observation(p))
    assert controller.direction(dict(observation([4, 0]), error=(j @ [4, 0]).tolist()))[0] < 0


def test_bundled_calibration_and_config_are_compatible():
    root = Path(__file__).resolve().parents[1]
    cal = json.loads((root / 'calibration.json').read_text())
    runtime = json.loads((root / 'calibration.alignment.json').read_text())
    Settings(**runtime).validate(cal)
    assert cal['eta'] == 1.5


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


def test_initialization_counts_eligible_random_measurements_and_reuses_coarse(tmp_path, monkeypatch):
    backend = SyntheticBackend([0, 0])
    def measured(p, cal):
        obs = observation(p, invalid=backend.acquisitions == 3)
        if obs['metrics']['valid']:
            obs['metrics']['primary_metric'] = .01
        return obs
    c = Controller(calibration(), settings(target_m2=None, max_acquisitions=11),
                   backend, tmp_path / 'log', measured)
    points = iter([[1, 0], [-1, 0], [-2, 0], [-2, 1], [-1, 1], [0, 1], [0, 2]])
    monkeypatch.setattr(c, 'candidates', lambda incumbent: np.array([next(points)], float))
    result = c.run()
    initial = [o for o in c.history if o['stage'] == 'bo_init']
    assert len(initial) == 6
    assert sum(o['bo_eligible'] for o in initial) == 5
    assert len({tuple(o['position']) for o in initial}) == 6
    assert initial[0]['bo_rejection_reason'] == 'invalid fit'
    assert [o['stage'] for o in c.history] == ['baseline', 'confirm'] + ['bo_init'] * 6 + ['bo'] + ['final'] * 2
    assert result['final_readback'] == [0, 0]
    models = [e for e in map(json.loads, (tmp_path / 'log').read_text().splitlines()) if e['event'] == 'bo_model']
    assert models[-1]['training_acquisitions'] == [1, 2, 4, 5, 6, 7, 8]
    assert models[-1]['exclusion_acquisitions'] == [3]
    assert result['acquisitions'] == 11


@pytest.mark.parametrize('invalid', [False, True])
def test_initialization_budget_exhaustion_reserves_final_confirmations(tmp_path, invalid):
    backend = SyntheticBackend([0, 0])
    def measured(p, cal):
        obs = observation(p, invalid=invalid and backend.acquisitions > 2 and np.any(p))
        if obs['metrics']['valid']:
            obs['metrics']['primary_metric'] = .01
        return obs
    c = Controller(calibration(), settings(target_m2=None, bo_initial_samples=20, max_acquisitions=8),
                   backend, tmp_path / 'log', measured)
    result = c.run()
    assert result['reason'] == 'random initialization budget exhausted'
    assert result['acquisitions'] == 8
    assert result['status'] == 'best_found'
    assert [o['stage'] for o in c.history[-2:]] == ['final', 'final']
    assert not any(o['stage'] in ('restore', 'bo') for o in c.history)


def test_exclusions_use_current_anisotropic_scales_and_actual_readback(tmp_path):
    backend = SyntheticBackend([4, 0])
    c = Controller(calibration(), settings(target_m2=None, max_step=3, bo_halfwidth=[4, 3]),
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
    def measured(p, cal):
        obs = observation(p)
        if np.any(p):
            obs['metrics']['primary_metric'] = .03
        return obs
    c = Controller(calibration(), settings(target_m2=None), backend, tmp_path / 'log', measured)
    result = c.run()
    assert 'no sampleable candidates' in result['reason']
    assert 'random initialization incomplete' in result['reason']
    assert result['status'] == 'best_found'
    assert c.history[2]['bo_rejection_reason'].startswith('M1 exceeds')
    assert result['acquisitions'] == 5
    assert result['final_readback'] == [0, 0]


@pytest.mark.parametrize('replay', [False, True])
def test_final_return_uses_bounded_transits_without_extra_images(tmp_path, replay):
    backend = (Replay(dict(units='um', samples=[dict(position=[x, 0], image=[x, 0])
                                               for x in [0, 3, 6]]), [0, 0])
               if replay else SyntheticBackend([0, 0]))
    if not replay:
        backend.candidate_positions = np.array([[3, 0], [6, 0]])
    def measured(p, cal):
        obs = observation(p)
        obs['metrics']['primary_metric'] = .01
        return obs
    c = Controller(calibration(), settings(target_m2=None, max_step=3, bo_halfwidth=[8, 8],
                   bo_initial_samples=2, max_acquisitions=6), backend, tmp_path / 'log', measured)
    result = c.run()
    assert result['status'] == 'best_found'
    assert result['acquisitions'] == 6
    assert result['final_readback'] == [0, 0]
    assert [o['position'] for o in c.history] == [[0, 0], [0, 0], [3, 0], [6, 0], [0, 0], [0, 0]]
    events = list(map(json.loads, (tmp_path / 'log').read_text().splitlines()))
    transits = [e for e in events if e['event'] == 'plan' and e['stage'] == 'transit']
    assert len(transits) >= 2
    assert all(np.linalg.norm(np.array(e['target']) - e['previous_readback']) <= 3 for e in transits)


@pytest.mark.parametrize('options', [dict(bo_initial_samples=-1), dict(bo_initial_samples=1.5),
    dict(gp_length_scale_bounds=[0, 10]), dict(gp_length_scale_bounds=[3, 2]),
    dict(gp_length_scale_bounds=[.1, float('inf')]), dict(gp_length_scale=[.1, 2])])
def test_bo_settings_validation(options):
    with pytest.raises(ValueError):
        settings(**options).validate(calibration())


def test_zero_random_samples_enters_ei_directly(tmp_path):
    c = Controller(calibration(), settings(target_m2=None, bo_initial_samples=0, max_acquisitions=6),
                   SyntheticBackend([0, 0]), tmp_path / 'log', lambda p, cal: observation(p))
    c.run()
    assert c.history[2]['stage'] == 'bo'
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
            super().move_absolute(np.array(target) + ([1, 0] if self.acquisitions >= 4 else [0, 0]))
    backend = FaultyTransit([0, 0])
    backend.candidate_positions = np.array([[3, 0], [6, 0]])
    def measured(p, cal):
        obs = observation(p)
        obs['metrics']['primary_metric'] = .01
        return obs
    c = Controller(calibration(), settings(target_m2=None, max_step=3, bo_halfwidth=[8, 8],
                   bo_initial_samples=2, max_acquisitions=6), backend, tmp_path / 'log', measured)
    result = c.run()
    assert result['status'] == 'blocked'
    assert 'readback' in result['reason']
    assert result['acquisitions'] == backend.acquisitions == 4
    assert len(backend.moves) == 3
    assert not any(o['stage'] == 'final' for o in c.history)
