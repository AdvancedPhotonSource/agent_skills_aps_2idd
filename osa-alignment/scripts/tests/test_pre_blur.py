"""Pre-blur applies to geometry while photometry retains original intensities."""
import json
import sys

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter

from test_analytical_alignment import Config, ring_image
import evaluate_alignment as module


@pytest.mark.parametrize('sigma', [0., 1.])
def test_blur_precedes_normalization_edges_and_seeds(monkeypatch, sigma):
    image = np.random.default_rng(7).poisson(10 * ring_image()).astype(float) / 10 + 2
    original = image.copy()
    fit_image = gaussian_filter(image, sigma) if sigma else image
    lo, hi = np.percentile(fit_image, [10, 99.5])
    expected = np.clip((fit_image - lo) / (hi - lo), 0, 1)
    edge_inputs, seed_inputs = [], []
    canny, seeds = module.canny, module._seeds

    def edges(a, **kwargs):
        edge_inputs.append((a.copy(), kwargs['sigma']))
        return canny(a, **kwargs)

    def starts(a, config):
        seed_inputs.append(a.copy())
        return seeds(a, config)

    monkeypatch.setattr(module, 'canny', edges)
    monkeypatch.setattr(module, '_seeds', starts)
    result = module.evaluate(image, Config(inner_axes=(12, 12), outer_axes=(30, 30), pre_blur_sigma=sigma))
    np.testing.assert_array_equal(edge_inputs[0][0], expected)
    assert edge_inputs[0][1] == 2.
    np.testing.assert_array_equal(seed_inputs[0], expected)
    np.testing.assert_array_equal(image, original)
    assert result['config']['pre_blur_sigma'] == sigma


@pytest.mark.parametrize('fixed', [False, True])
def test_photometry_uses_original_image_with_blurred_geometry(fixed):
    image = np.random.default_rng(7).poisson(10 * ring_image()).astype(float) / 10 + 2
    axes = dict(inner_axes=(12, 12), outer_axes=(30, 30)) if fixed else {}
    config = Config(**axes, pre_blur_sigma=1, background=2, intensity_normalizer=5)
    result = module.evaluate(image, config)
    assert result['valid'], result['reasons']
    fit = result['fit']
    y, x = np.indices(image.shape)
    dx, dy = x-fit['center_x'], y-fit['center_y']
    angle = np.deg2rad(fit['angle_degrees'])
    u = np.cos(angle)*dx + np.sin(angle)*dy
    v = -np.sin(angle)*dx + np.cos(angle)*dy
    a,b = fit['inner_axes']
    hole = (u/a)**2 + (v/b)**2 < 1
    expected = float((image-2)[hole].sum())
    assert result['hole_integral'] == expected
    assert result['secondary_metric'] == expected / 5
    assert abs(float((gaussian_filter(image, 1)-2)[hole].sum())-expected) > 1


@pytest.mark.parametrize('sigma', [-1, np.nan, np.inf])
def test_invalid_pre_blur(sigma):
    with pytest.raises(ValueError, match='pre_blur_sigma'):
        Config(pre_blur_sigma=sigma).validate()


def test_cli_pre_blur_option(tmp_path, monkeypatch, capsys):
    image = tmp_path/'ring.tiff'
    Image.fromarray(ring_image().astype(np.float32)).save(image)
    monkeypatch.setattr(sys, 'argv', ['evaluate_alignment.py', str(image), '--pre-blur-sigma', '1'])
    module.main()
    assert json.loads(capsys.readouterr().out)['config']['pre_blur_sigma'] == 1
