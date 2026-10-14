"""Synthetic calibration and control tests; no instrument or recorded data needed."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibrate_alignment import centroid_offset, fit_response, measure
from align_osa import Controller, Replay, Settings, constrained_ei
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


def test_gp_handles_infinite_metrics_and_both_validity_classes():
    observations = [observation([0, 0]), observation([1, 0]), observation([10, 10], invalid=True)]
    scores = constrained_ei(observations, np.array([[0, 1], [9, 10]]), settings())
    assert np.isfinite(scores).all()
    assert scores[0] > scores[1] >= 0


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


@pytest.mark.parametrize("options", [dict(max_step=0), dict(enter_m1=.04),
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


def test_all_missing_boundaries_do_not_train_geometry_gp():
    missing = observation([10, 10], invalid=True)
    missing["metrics"]["geometry_score"] = float("inf")
    missing["metrics"]["fit"] = None
    scores = constrained_ei([observation([0, 0]), missing], np.array([[1, 0], [9, 10]]), settings())
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
    c = Controller(cal, settings(evaluator=evaluator, enter_m1=.05, exit_m1=.06,
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
