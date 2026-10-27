"""Bounded quadratic minimization and sequential Stage 1 refitting."""
import numpy as np
import pytest

from test_analytical_alignment import Controller, SyntheticBackend, events, observation, settings
from test_grid_initialization import grid_settings
from align_osa import Settings, fit_quadratic


def samples(function):
    result = []
    for x in np.linspace(-15, 15, 4):
        for y in np.linspace(-15, 15, 4):
            obs = observation([x, y])
            obs['metrics']['primary_metric'] = function(x, y)
            result.append(obs)
    return result


@pytest.mark.parametrize('function, expected', [
    (lambda x, y: (x-2)**2 + (x-2)*(y+3) + 2*(y+3)**2, [2, -3]),
    (lambda x, y: (x-30)**2 + (y-2)**2, [15, 2]),
    (lambda x, y: -(x-1)**2 + (y-2)**2, [-15, 2]),
    (lambda x, y: -x*x-y*y+2*x+3*y, [-15, -15]),
    (lambda x, y: 2*x+3*y, [-15, -15]),
    (lambda x, y: (x+y-3)**2, None),
])
def test_global_bounded_minimum(function, expected):
    coefficients, target, value = fit_quadratic(samples(function), settings())
    assert np.all(np.abs(target) <= 15)
    if expected is not None:
        np.testing.assert_allclose(target, expected, atol=1e-8)
    else:
        assert target.sum() == pytest.approx(3)
    assert value == pytest.approx(function(*target), abs=1e-9)
    grid = np.linspace(-15, 15, 101)
    assert value <= min(function(x,y) for x in grid for y in grid) + 1e-9


def test_finite_m1_includes_invalid_geometry_and_repeated_measurements():
    data = samples(lambda x,y: x*x+2*y*y)
    data[0]['metrics']['valid'] = False
    repeated = observation([2, 3])
    repeated['metrics']['primary_metric'] = -10
    data.extend([repeated, repeated])
    ignored = observation([5, 7])
    ignored['metrics']['primary_metric'] = float('inf')
    c, _, _ = fit_quadratic(data+[ignored], settings())
    points = np.array([o['position'] for o in data])/15
    x,y = points.T
    expected = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x, y, x*x, x*y, y*y]),
                              [o['metrics']['primary_metric'] for o in data], rcond=None)[0]
    np.testing.assert_allclose(c, expected)


def test_degenerate_data_is_reported():
    with pytest.raises(ValueError, match='independent finite'):
        fit_quadratic([observation([0,0])]*16, settings())


def test_refits_after_each_measurement_and_enters_at_final_minimum(tmp_path, monkeypatch):
    backend = SyntheticBackend([12, 12])
    del backend.candidate_positions

    def measured(position):
        obs = observation(position)
        x,y = position
        # A smooth non-quadratic surface makes the fitted minimum change as data arrive.
        obs['metrics']['primary_metric'] = .001 + .000001*((x-2)**2+2*(y+3)**2) + .0000001*x**3
        return obs

    c = Controller(Settings(units='um', bounds=[[-15, 15], [-15, 15]], max_step=5,
                            stage_2_target_m2=1000), backend, tmp_path/'log', measured)
    # Stage 1 must not use a GP minimum; Stage 2 exits at its target after its grid.
    monkeypatch.setattr(c, 'posterior_mean_center', lambda *args: pytest.fail('Unexpected GP minimum'))
    result = c.run()
    assert result['status'] != 'blocked'
    models = [e for e in events(c) if e['event']=='quadratic_model']
    assert [len(e['training_acquisitions']) for e in models] == [17,18,19,20]
    assert all(e['training_acquisitions'] == list(range(1,n+1)) for e,n in zip(models,[17,18,19,20]))
    for event, obs in zip(models, c.history[17:21]):
        np.testing.assert_allclose(obs['position'], event['position'])
    assert np.linalg.norm(np.array(models[0]['position'])-models[-1]['position']) > 1e-5
    assert [o['stage'] for o in c.history[17:21]] == ['stage1']*3+['stage_2_init']
    checks = [e for e in events(c) if e['event']=='stage_2_entry_check']
    assert all(not e['threshold_met'] for e in checks)
    transition = next(e for e in events(c) if e['event']=='stage_transition')
    assert transition['stage1_acquisitions']==20
    assert transition['reason']=='stage 1 budget exhausted'
    assert sum(o['stage']=='stage_2_init' for o in c.history)==17
    assert not any(e['event']=='bo_model' and e['stage']=='stage1' for e in events(c))


@pytest.mark.parametrize('kwargs', [{'stage_1_optimizer':'invalid'},
                                  {'stage_1_optimizer':'quadratic', 'bo_initialization':'random'}])
def test_invalid_mode(kwargs):
    with pytest.raises(ValueError):
        grid_settings(**kwargs).validate()
