#!/usr/bin/env python3
"""Fit concentric OSA boundaries and measure geometry and central leakage.

Coordinates and semiaxes are in pixels; x increases rightward and y
downward. Ellipse A-axis orientation is clockwise from +x, modulo 180 degrees.
Run with `uv run --project <scripts-dir> <scripts-dir>/evaluate_alignment.py --help`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks, resize


@dataclass(frozen=True)
class Config:
    inner_axes: tuple[float, float] | None = None
    outer_axes: tuple[float, float] | None = None
    min_eccentricity: float | None = None
    sigma: float = 2.0
    edge_low: float = 0.04
    edge_high: float = 0.10
    coverage_weight: float = 0.5
    edge_tolerance: float | None = None
    min_coverage: float = 0.65
    max_gap_degrees: float = 90.0
    background: float = 0.0
    intensity_normalizer: float = 1.0
    pre_blur_sigma: float = 0.0

    def validate(self):
        if self.min_eccentricity is not None:
            if not np.isfinite(self.min_eccentricity) or not 0 <= self.min_eccentricity < 1:
                raise ValueError("min_eccentricity must be finite and in [0, 1)")
        if (self.inner_axes is None) != (self.outer_axes is None):
            raise ValueError("Supply both inner_axes and outer_axes, or neither")
        if self.inner_axes is not None:
            if (len(self.inner_axes) != 2 or len(self.outer_axes) != 2
                or not np.isfinite([*self.inner_axes, *self.outer_axes]).all()
                or not all(0 < i < o for i, o in zip(self.inner_axes, self.outer_axes))):
                raise ValueError("Supply two positive semiaxes per ellipse, with each inner axis < outer axis")
            if self.min_eccentricity is not None:
                for axes in (self.inner_axes, self.outer_axes):
                    ratio = min(axes) / max(axes)
                    if ratio > np.sqrt(1 - self.min_eccentricity**2):
                        raise ValueError("Fixed axes do not meet min_eccentricity")
        if not np.isfinite(self.pre_blur_sigma) or self.pre_blur_sigma < 0:
            raise ValueError("pre_blur_sigma must be nonnegative and finite")
        if not (np.isfinite(self.sigma) and self.sigma > 0):
            raise ValueError("sigma must be positive and finite")
        if not 0 < self.edge_low < self.edge_high <= 1:
            raise ValueError("Require 0 < edge_low < edge_high <= 1")
        if not 0 <= self.min_coverage <= 1 or not 0 < self.max_gap_degrees <= 360:
            raise ValueError("Invalid coverage or angular-gap threshold")
        if not np.isfinite(self.coverage_weight) or self.coverage_weight < 0:
            raise ValueError("coverage_weight must be nonnegative and finite")
        if self.edge_tolerance is not None and not (
            np.isfinite(self.edge_tolerance) and self.edge_tolerance > 0
        ):
            raise ValueError("edge_tolerance must be positive and finite")
        if not np.isfinite(self.background):
            raise ValueError("background must be finite")
        if not np.isfinite(self.intensity_normalizer) or self.intensity_normalizer <= 0:
            raise ValueError("intensity_normalizer must be positive and finite")


def grayscale(array: np.ndarray) -> np.ndarray:
    """Preserve intensity units; RGB uses luminance weights without gamma decoding."""
    a = np.asarray(array, dtype=float)
    if a.ndim == 3 and a.shape[-1] == 3:
        a = a @ np.array([0.2126, 0.7152, 0.0722])
    if a.ndim != 2 or min(a.shape) < 16 or not np.isfinite(a).all():
        raise ValueError("Expected a finite 2-D grayscale or RGB image, at least 16 x 16")
    return a


def read_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Select a single image from a multi-frame file first")
        return grayscale(np.asarray(image))


def _largest_gap(supported: np.ndarray) -> float:
    missing = ~supported
    if missing.all():
        return 360.0
    longest = current = 0
    for value in np.tile(missing, 2):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return min(longest, len(missing)) * 360.0 / len(missing)


def _seeds(normalized: np.ndarray, config: Config) -> list[tuple[float, float, float]]:
    """Coarse circular Hough peaks initialize center and outer radius."""
    scale = max(1, int(np.ceil(max(normalized.shape) / 360)))
    shape = tuple(int(np.ceil(n / scale)) for n in normalized.shape)
    small = resize(normalized, shape, preserve_range=True, anti_aliasing=True)
    edges = canny(small, sigma=max(1, config.sigma / scale),
                  low_threshold=config.edge_low, high_threshold=config.edge_high)
    outer_scale = (float(np.sqrt(np.prod(config.outer_axes)))
                   if config.outer_axes is not None else None)
    if outer_scale is None:
        # Broad initialization prior: outer radius is 15--35% of the short side.
        radii = np.arange(max(3, int(min(shape) * 0.15)), int(min(shape) * 0.35) + 1, 2)
    else:
        radii = np.array([max(2, round(outer_scale / scale))])
    hough = hough_circle(edges, radii)
    peaks, xs, ys, rs = hough_circle_peaks(
        hough, radii, min_xdistance=max(3, int(radii[0] / 3)),
        min_ydistance=max(3, int(radii[0] / 3)), total_num_peaks=4,
    )
    seeds = [(float(x * scale), float(y * scale),
              float(outer_scale or r * scale))
             for strength, x, y, r in zip(peaks, xs, ys, rs) if strength > 0]
    # A bright annulus centroid supplies a complementary start when Hough peaks
    # favor internal texture or displaced arcs rather than the whole envelope.
    labels, count = ndi.label(normalized > 0.5)
    if seeds and count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        cy, cx = ndi.center_of_mass(labels == sizes.argmax())
        seeds.append((float(cx), float(cy), seeds[0][2]))
    return seeds


def boundary_points(fit: dict, boundary: str, angles: np.ndarray):
    """Sample at true polar angles about the shared center (not ellipse phase)."""
    a, b = fit[f"{boundary}_axes"]
    phase = angles - np.deg2rad(fit["angle_degrees"])
    radius = 1 / np.sqrt((np.cos(phase) / a) ** 2 + (np.sin(phase) / b) ** 2)
    return (fit["center_x"] + radius * np.cos(angles),
            fit["center_y"] + radius * np.sin(angles))


def evaluate(image: np.ndarray, config: Config | None = None) -> dict:
    """Return scores plus fit diagnostics. M1 stays finite even when fit validation fails.

    M1 = mean boundary RMS distance / outer equal-area radius + coverage_weight *
         mean unsupported angular fraction. M2 = background-subtracted hole sum
         / intensity_normalizer. Ellipses share a center and orientation. Coverage uses 360 true polar angles, one degree apart.
    Failed fits retain the best candidate geometry for M1; M2 is infinite when invalid.
    """
    config = config or Config()
    config.validate()
    max_axis_ratio = (None if config.min_eccentricity is None else
                      float(np.sqrt(1 - config.min_eccentricity**2)))
    image = grayscale(image)
    height, width = image.shape
    result = dict(valid=False, reasons=[], config=asdict(config),
                  image_shape=list(image.shape), fit=None,
                  secondary_metric=float("inf"),
                  geometry_score=float("inf"), hole_integral=None)
    fit_image = ndi.gaussian_filter(image, config.pre_blur_sigma) if config.pre_blur_sigma else image
    floor, ceiling = np.percentile(fit_image, [10, 99.5])
    base_reasons = ["No usable intensity contrast"] if ceiling <= floor else []
    normalized = (np.zeros_like(image) if ceiling <= floor else
                  np.clip((fit_image - floor) / (ceiling - floor), 0, 1))
    edges = canny(normalized, sigma=config.sigma,
                  low_threshold=config.edge_low, high_threshold=config.edge_high)
    smooth = ndi.gaussian_filter(normalized, config.sigma)
    gy, gx = np.gradient(smooth)
    yy, xx = np.indices(image.shape)
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)

    def distance_maps(cx, cy, geometry=None):
        # Inner edges brighten outward; outer edges darken outward. Suppress
        # tangential texture while retaining up to 60 degrees of normal error.
        maps = []
        for boundary, sign in (("inner", 1), ("outer", -1)):
            nx, ny = xx - cx, yy - cy
            if geometry is not None:
                a, b = geometry[f"{boundary}_axes"]
                theta = np.deg2rad(geometry["angle_degrees"])
                c, s = np.cos(theta), np.sin(theta)
                u, v = (c * nx + s * ny) / a**2, (-s * nx + c * ny) / b**2
                nx, ny = c * u - s * v, s * u + c * v
            alignment = (nx * gx + ny * gy) / np.maximum(
                np.hypot(nx, ny) * np.hypot(gx, gy), 1e-12)
            mask = edges & (sign * alignment > 0.5)
            # No supported edges still supplies a finite residual to the optimizer.
            maps.append(ndi.distance_transform_edt(~mask) if mask.any() else
                        np.full(image.shape, np.hypot(height, width)))
        return maps

    candidates = []
    # Moments give an orientation seed; the edge fit determines the final axes.
    seed_angle, seed_ratio = 0.0, 1.0
    moment_seed = None
    labels, count = ndi.label(normalized > 0.5)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        sy, sx = np.nonzero(labels == sizes.argmax())
        if len(sx) > 2:
            values, vectors = np.linalg.eigh(np.cov(np.array([sx, sy])))
            seed_angle = float(np.arctan2(vectors[1, -1], vectors[0, -1]) % np.pi)
            seed_ratio = float(np.clip(np.sqrt(values[-1] / max(values[0], 1)), 1, 2))
            # Area plus second moments estimates the outer equal-area
            # radius of an annulus without mistaking local curvature for it.
            moment_radius = np.sqrt(2 * np.sqrt(max(np.prod(values), 0)) + len(sx) / (2 * np.pi))
            if config.outer_axes is None:
                moment_seed = (float(sx.mean()), float(sy.mean()), float(moment_radius))
    seeds = _seeds(normalized, config)
    if moment_seed is not None:
        seeds.append(moment_seed)
    if not seeds:
        radius = (float(np.sqrt(np.prod(config.outer_axes)))
                  if config.outer_axes is not None else min(image.shape) * 0.25)
        seeds = [((width - 1) / 2, (height - 1) / 2, radius)]
    for cx, cy, radius in seeds:
        maps = distance_maps(cx, cy)

        def unpack(parameters):
            x, y = map(float, parameters[:2])
            if config.inner_axes is not None:
                inner_axes, outer_axes = config.inner_axes, config.outer_axes
                theta = parameters[2]
            elif max_axis_ratio is not None and config.min_eccentricity > 0:
                fraction_a, fraction_b, a, axis_ratio, theta = parameters[2:]
                b = a * axis_ratio
                inner_a = fraction_a * a
                inner_b = fraction_b * min(0.70 * b, max_axis_ratio * inner_a)
                inner_axes, outer_axes = (inner_a, inner_b), (a, b)
            else:
                fraction_a, fraction_b, a, b, theta = parameters[2:]
                inner_axes, outer_axes = (fraction_a * a, fraction_b * b), (a, b)
            return dict(model="ellipse", center_x=x, center_y=y,
                        inner_axes=list(map(float, inner_axes)), outer_axes=list(map(float, outer_axes)),
                        angle_degrees=float(np.rad2deg(theta) % 180), axes_fixed=config.inner_axes is not None)

        def residuals(parameters):
            geometry = unpack(parameters)
            points = [boundary_points(geometry, boundary, angles) for boundary in ("inner", "outer")]
            return np.concatenate([
                ndi.map_coordinates(distance, [y, x],
                                    order=1, mode="constant", cval=radius)
                for distance, (x, y) in zip(maps, points)
            ])

        lower = [max(0, cx - radius * 0.3), max(0, cy - radius * 0.3)]
        upper = [min(width - 1, cx + radius * 0.3), min(height - 1, cy + radius * 0.3)]
        if config.inner_axes is not None:
            initial = [cx, cy, seed_angle]
            lower += [-np.pi]
            upper += [2 * np.pi]
        else:
            trial_radii = np.arange(max(1, radius * 0.12), max(2, radius * 0.65), 1)
            costs = [np.mean(ndi.map_coordinates(
                maps[0], [cy + r * sin, cx + r * cos], order=1,
                mode="constant", cval=radius)) for r in trial_radii]
            inner = trial_radii[np.argmin(costs)]
            # Fractions enforce nesting for every angle throughout fitting.
            fraction = float(np.clip(inner / radius, 0.11, 0.69))
            if max_axis_ratio is not None and config.min_eccentricity > 0:
                # Parameterize both minor axes so the bound holds at every
                # optimizer step, including nonconvergent fits.
                a = radius * np.sqrt(seed_ratio)
                axis_ratio = min(1 / seed_ratio, max_axis_ratio)
                b = a * axis_ratio
                inner_cap = min(0.70 * b, max_axis_ratio * fraction * a)
                fraction_b = min(0.99, fraction * b / inner_cap)
                initial = [cx, cy, fraction, fraction_b, a, axis_ratio, seed_angle]
                lower += [0.10, 0.01, radius * 0.55, min(0.001, max_axis_ratio / 2), -np.pi]
                upper += [0.70, 1.0, radius * 1.65, max_axis_ratio, 2 * np.pi]
            else:
                initial = [cx, cy, fraction, fraction,
                           radius * np.sqrt(seed_ratio), radius / np.sqrt(seed_ratio), seed_angle]
                lower += [0.10, 0.10, radius * 0.55, radius * 0.55, -np.pi]
                upper += [0.70, 0.70, radius * 1.65, radius * 1.65, 2 * np.pi]
        fit = least_squares(residuals, initial, bounds=(lower, upper),
                            diff_step=1e-3, max_nfev=150)
        # Update edge-normal classification at the fitted center and refine.
        # Final metrics must depend on the final geometry, not on its seed.
        maps = distance_maps(*fit.x[:2], unpack(fit.x))
        fit = least_squares(residuals, fit.x, bounds=(lower, upper),
                            diff_step=1e-3, max_nfev=150)
        maps = distance_maps(*fit.x[:2], unpack(fit.x))
        geometry = unpack(fit.x)
        x, y = geometry["center_x"], geometry["center_y"]
        a, b = geometry["outer_axes"]
        theta = np.deg2rad(geometry["angle_degrees"])
        outer = float(np.sqrt(a * b))
        extent_x = np.hypot(a * np.cos(theta), b * np.sin(theta))
        extent_y = np.hypot(a * np.sin(theta), b * np.cos(theta))
        geometry["outer_equivalent_radius"] = outer
        geometry["inner_equivalent_radius"] = float(np.sqrt(np.prod(geometry["inner_axes"])))
        distances = residuals(fit.x).reshape(2, -1)
        tolerance = config.edge_tolerance or max(2.0, 0.04 * outer)
        coverage = (distances <= tolerance).mean(axis=1)
        rms = np.sqrt(np.mean(distances ** 2, axis=1))
        score = float(rms.mean() / outer + config.coverage_weight * (1 - coverage.mean()))
        gaps = [_largest_gap(d <= tolerance) for d in distances]
        reasons = list(base_reasons)
        if not fit.success:
            reasons.append("Optimizer did not converge")
        if min(x - extent_x, y - extent_y, width - 1 - x - extent_x, height - 1 - y - extent_y) < 0:
            reasons.append("Outer boundary extends beyond image")
        if coverage.min() < config.min_coverage:
            reasons.append("Insufficient angular boundary coverage")
        if max(gaps) > config.max_gap_degrees:
            reasons.append("Boundary gap exceeds angular limit")
        candidates.append(dict(
            valid=not reasons, reasons=reasons, geometry_score=score,
            fit=geometry,
            inner_rms_px=float(rms[0]), outer_rms_px=float(rms[1]),
            inner_coverage=float(coverage[0]), outer_coverage=float(coverage[1]),
            inner_max_gap_degrees=gaps[0], outer_max_gap_degrees=gaps[1],
            edge_tolerance_px=tolerance,
        ))
    # Prefer an adequately supported model over a spurious low-residual arc.
    result.update(min(candidates, key=lambda c: (not c["valid"], c["geometry_score"])))
    result["primary_metric"] = result["geometry_score"]
    fit = result["fit"]
    dx, dy = xx - fit["center_x"], yy - fit["center_y"]
    theta = np.deg2rad(fit["angle_degrees"])
    u, v = np.cos(theta) * dx + np.sin(theta) * dy, -np.sin(theta) * dx + np.cos(theta) * dy
    ia, ib = fit["inner_axes"]
    oa, ob = fit["outer_axes"]
    hole = (u / ia)**2 + (v / ib)**2 < 1
    inside_outer = (u / oa)**2 + (v / ob)**2 <= 1
    annulus = inside_outer & ~hole
    if not hole.any() or not annulus.any():
        result["valid"] = False
        result["reasons"].append("Hole or annulus contains no pixel centers")
        return result
    corrected = image - config.background
    integral = float(corrected[hole].sum())
    ring_sum = float(corrected[annulus].sum())
    result.update(hole_integral=integral, hole_area_px=int(hole.sum()),
                  hole_mean=float(corrected[hole].mean()), annulus_integral=ring_sum,
                  hole_to_annulus_ratio=integral / ring_sum if ring_sum > 0 else None)
    if ring_sum <= 0 or corrected[annulus].mean() <= corrected[hole].mean():
        result["valid"] = False
        result["reasons"].append("Annulus is not brighter than the hole above background")
    if result["valid"]:
        result["secondary_metric"] = integral / config.intensity_normalizer
    return result


def json_safe(value):
    """Use standards-compliant JSON while retaining explicit infinite scores."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, help="JSON destination; default stdout")
    parser.add_argument("--inner-axes", type=float, nargs=2, metavar=("A", "B"), help="Fixed inner ellipse semiaxes in pixels")
    parser.add_argument("--outer-axes", type=float, nargs=2, metavar=("A", "B"), help="Fixed outer ellipse semiaxes in pixels; requires inner axes")
    parser.add_argument("--min-eccentricity", type=float, help="Minimum c/a for both fitted ellipses; 0 allows circles")
    parser.add_argument("--sigma", type=float, default=2.0, help="Canny smoothing sigma in pixels")
    parser.add_argument("--pre-blur-sigma", type=float, default=0.0,
                        help="Additional Gaussian blur before normalization and geometry fitting; 0 disables; M2 uses original intensities")
    parser.add_argument("--edge-low", type=float, default=0.04)
    parser.add_argument("--edge-high", type=float, default=0.10)
    parser.add_argument("--edge-tolerance", type=float, help="Support distance in pixels; default max(2, 0.04 * outer equal-area radius)")
    parser.add_argument("--coverage-weight", type=float, default=0.5)
    parser.add_argument("--min-coverage", type=float, default=0.65)
    parser.add_argument("--max-gap-degrees", type=float, default=90.0)
    parser.add_argument("--background", type=float, default=0.0, help="Scalar detector background in input intensity units")
    parser.add_argument("--intensity-normalizer", type=float, default=1.0, help="Positive exposure/incident-flux normalization divisor")
    args = parser.parse_args()
    config = Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__})
    try:
        result = evaluate(read_image(args.image), config)
    except ValueError as error:
        parser.error(str(error))
    payload = json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
