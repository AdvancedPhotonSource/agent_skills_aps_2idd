#!/usr/bin/env python3
"""Calibrate image-centroid error against recorded transverse motor positions."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from evaluate_alignment import Config, evaluate, json_safe, read_image


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def load_manifest(path):
    """Load positions and image paths; positions use the manifest's explicit units."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if not isinstance(data.get("units"), str) or not data["units"].strip():
        raise ValueError("Manifest must specify motor units")
    if not data.get("samples"):
        raise ValueError("Manifest samples must not be empty")
    for row in data["samples"]:
        p = np.asarray(row["position"], dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError("Each position must contain two finite numbers")
        row["position"] = p.tolist()
        row["image"] = str((path.parent / row["image"]).resolve())
    return data


def centroid_offset(image, result, percentile=10.0):
    """Return CoM minus ellipse center using positive signal inside the outer ellipse."""
    fit = result.get("fit")
    if fit is None:
        return None
    y, x = np.indices(image.shape)
    dx, dy = x - fit["center_x"], y - fit["center_y"]
    angle = np.deg2rad(fit["angle_degrees"])
    u = dx * np.cos(angle) + dy * np.sin(angle)
    v = -dx * np.sin(angle) + dy * np.cos(angle)
    a, b = fit["outer_axes"]
    weights = np.maximum(image - np.percentile(image, percentile), 0)
    weights *= (u / a) ** 2 + (v / b) ** 2 <= 1
    total = weights.sum()
    if total <= 0:
        return None
    return np.array([(weights * dx).sum(), (weights * dy).sum()]) / total


def measure(path, calibration, *, config=None, centroid_percentile=10.0):
    image = read_image(path)
    result = evaluate(image, Config() if config is None else config)
    offset = centroid_offset(image, result, centroid_percentile)
    error = None if offset is None else offset - np.asarray(calibration["reference_error"])
    appearance = {}
    if result.get("fit") is not None:
        fit = result["fit"]
        y, x = np.indices(image.shape)
        dx, dy = x - fit["center_x"], y - fit["center_y"]
        angle = np.deg2rad(fit["angle_degrees"])
        u = dx * np.cos(angle) + dy * np.sin(angle)
        v = -dx * np.sin(angle) + dy * np.cos(angle)
        inner, outer = fit["inner_axes"], fit["outer_axes"]
        hole = (u / inner[0]) ** 2 + (v / inner[1]) ** 2 <= 1
        annulus = ((u / outer[0]) ** 2 + (v / outer[1]) ** 2 <= 1) & ~hole
        sectors = {"left": (dx < 0) & (abs(dx) >= abs(dy)),
                   "right": (dx >= 0) & (abs(dx) >= abs(dy)),
                   "top": (dy < 0) & (abs(dy) > abs(dx)),
                   "bottom": (dy >= 0) & (abs(dy) > abs(dx))}
        appearance = {"ring_sector_means": {
            name: float(image[mask & annulus].mean()) if (mask & annulus).any() else None
            for name, mask in sectors.items()},
            "hole_peak": float(image[hole].max()) if hole.any() else None}
    return {"image": str(path), "metrics": result,
            "error": None if error is None else error.tolist(), "appearance": appearance}


def bounded(vector, limit):
    return vector * min(1.0, limit / max(float(np.linalg.norm(vector)), 1e-15))


def fit_response(positions, errors, gains, max_step, max_condition=100.0):
    """Fit e=J delta-p and select eta by leave-one-position-out correction error."""
    p, e = np.asarray(positions, float), np.asarray(errors, float)
    if len(p) < 4 or np.linalg.matrix_rank(p) < 2:
        raise ValueError("Calibration needs at least four usable samples spanning both axes")
    jacobian = np.linalg.lstsq(p, e, rcond=None)[0].T
    condition = float(np.linalg.cond(jacobian))
    if not np.isfinite(condition) or condition > max_condition:
        raise ValueError(f"Motor-to-centroid response is ill-conditioned: {condition:g}")
    corrections, held_positions = [], []
    for position in np.unique(p, axis=0):
        held = np.all(p == position, axis=1)
        if np.linalg.matrix_rank(p[~held]) < 2:
            raise ValueError("Calibration must span both axes after holding out each position")
        j = np.linalg.lstsq(p[~held], e[~held], rcond=None)[0].T
        if np.linalg.cond(j) > max_condition:
            raise ValueError("A held-out response fit is ill-conditioned; collect more samples")
        corrections.append(-np.linalg.solve(j, e[held].mean(axis=0)))
        held_positions.append(position)
    corrections, held_positions = np.array(corrections), np.array(held_positions)
    baseline = float(np.mean(np.sum(held_positions ** 2, axis=1)))
    scores = []
    for gain in sorted(set(gains)):
        steps = np.array([bounded(gain * d, max_step) for d in corrections])
        mse = float(np.mean(np.sum((held_positions + steps) ** 2, axis=1)))
        scores.append({"eta": gain, "held_out_position_mse": mse})
    best = min(scores, key=lambda row: row["held_out_position_mse"])
    if best["held_out_position_mse"] >= baseline:
        raise ValueError("No candidate eta improves held-out motor-position error")
    return {"jacobian": jacobian.tolist(),
            "eta": best["eta"], "diagnostics": {
                "condition_number": condition, "usable_samples": len(p),
                "held_out_positions": len(held_positions),
                "held_out_toward_reference_fraction": float(np.mean(
                    np.sum(corrections * held_positions, axis=1) < 0)),
                "uncorrected_position_mse": baseline, "gain_scores": scores,
                "gain_validation": "Held-out one-step position prediction; not a hardware convergence test."}}


def calibrate(manifest, reference, config, *, gains, max_step, coarse_threshold=0.02,
              centroid_percentile=10.0, min_vector=0.5, min_coverage=0.3,
              max_condition=100.0):
    config.validate()
    parameters = [*reference, *gains, max_step, coarse_threshold, centroid_percentile,
                  min_vector, min_coverage, max_condition]
    if (not np.isfinite(parameters).all() or not gains or min(gains) <= 0
        or max_step <= 0 or coarse_threshold <= 0 or not 0 <= centroid_percentile < 100
        or min_vector < 0 or not 0 <= min_coverage <= 1
        or max_condition < 1):
        raise ValueError("Invalid calibration parameters")
    reference = np.asarray(reference, float)
    refs = [r for r in manifest["samples"] if np.allclose(r["position"], reference, atol=1e-9, rtol=0)]
    if not refs:
        raise ValueError("Reference position must occur in manifest")
    first = evaluate(read_image(refs[0]["image"]), config)
    if not first["valid"]:
        raise ValueError("Reference image must have a valid ellipse fit")
    config = replace(config, inner_axes=tuple(first["fit"]["inner_axes"]),
                     outer_axes=tuple(first["fit"]["outer_axes"]))
    artifact = {"reference_error": [0., 0.]}
    observations = []
    for row in manifest["samples"]:
        obs = measure(row["image"], artifact, config=config, centroid_percentile=centroid_percentile)
        obs["position"] = row["position"]
        observations.append(obs)
        print(f"Calibration {len(observations)}/{len(manifest['samples'])}: "
              f"p={row['position']}, M1={obs['metrics']['primary_metric']:.5g}", flush=True)
    reference_obs = [o for o in observations if np.allclose(o["position"], reference, atol=1e-9, rtol=0)]
    if any(not o["metrics"]["valid"] or o["metrics"]["primary_metric"] > coarse_threshold
           or o["error"] is None for o in reference_obs):
        raise ValueError("All reference images must satisfy the coarse threshold with fixed semiaxes")
    reference_error = np.mean([o["error"] for o in reference_obs], axis=0)
    positions, errors = [], []
    for obs in observations:
        m, e = obs["metrics"], obs["error"]
        if e is None:
            continue
        e = np.asarray(e) - reference_error
        obs["error"] = e.tolist()
        usable = (np.linalg.norm(e) >= min_vector
                  and min(m["inner_coverage"], m["outer_coverage"]) >= min_coverage
                  and float(m["geometry_score"]) >= coarse_threshold
                  and not np.allclose(obs["position"], reference, atol=1e-9, rtol=0))
        obs["used_for_response"] = bool(usable)
        if usable:
            positions.append(np.asarray(obs["position"]) - reference)
            errors.append(e)
    response = fit_response(positions, errors, gains, max_step, max_condition)
    artifact.update(jacobian=response["jacobian"], eta=response["eta"],
                    reference_error=reference_error.tolist())
    report = {"diagnostics": response["diagnostics"], "units": manifest["units"],
              "bounds": np.array([np.min([o["position"] for o in observations], axis=0),
                                  np.max([o["position"] for o in observations], axis=0)]).T.tolist(),
              "evaluator": asdict(config),
              "reference_position": reference.tolist(),
              "reference_m2": float(np.mean([o["metrics"]["secondary_metric"] for o in reference_obs])),
              "calibration_options": {"eta_grid": gains, "max_step": max_step,
                  "coarse_threshold": coarse_threshold, "min_vector": min_vector,
                  "min_coverage": min_coverage, "centroid_percentile": centroid_percentile,
                  "max_condition": max_condition}}
    return artifact, report, observations


def alignment_settings(report):
    """Initial runtime limits and feature settings, adjustable for each alignment run."""
    options = report["calibration_options"]
    return {"units": report["units"], "bounds": report["bounds"],
            "max_step": options["max_step"], "min_step": min(0.5, options["max_step"]),
            **{name: options[name] for name in ("min_vector", "min_coverage", "centroid_percentile")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--reference-position", type=float, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--report", type=Path, help="Diagnostics JSON; default: OUTPUT stem + .report.json")
    parser.add_argument("--alignment-config", type=Path,
                        help="Suggested runtime settings; default: OUTPUT stem + .alignment.json")
    parser.add_argument("--evaluator-config", type=Path, help="JSON object with evaluate_alignment.Config fields")
    parser.add_argument("--max-step", type=float, required=True, help="Motor units; Euclidean displacement limit")
    parser.add_argument("--eta-grid", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1., 1.5, 2.])
    parser.add_argument("--coarse-threshold", type=float, default=0.02)
    parser.add_argument("--centroid-percentile", type=float, default=10.)
    parser.add_argument("--min-vector", type=float, default=0.5)
    parser.add_argument("--min-coverage", type=float, default=0.3)
    parser.add_argument("--max-condition", type=float, default=100.)
    args = parser.parse_args()
    report_path = args.report or args.output.with_suffix(".report.json")
    settings_path = args.alignment_config or args.output.with_suffix(".alignment.json")
    paths = [args.output, report_path, settings_path] + ([args.observations] if args.observations else [])
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("Calibration, report, alignment config, and observations need distinct output paths")
    config = Config(**(json.loads(args.evaluator_config.read_text()) if args.evaluator_config else {}))
    artifact, report, observations = calibrate(load_manifest(args.manifest), args.reference_position, config,
        gains=args.eta_grid, max_step=args.max_step, coarse_threshold=args.coarse_threshold,
        centroid_percentile=args.centroid_percentile, min_vector=args.min_vector,
        min_coverage=args.min_coverage,
        max_condition=args.max_condition)
    write_json(args.output, artifact)
    write_json(report_path, report)
    write_json(settings_path, alignment_settings(report))
    if args.observations:
        write_json(args.observations, observations)
    print(json.dumps(report["diagnostics"], indent=2))
    print(f"Selected eta={artifact['eta']:g}; saved {args.output}, {report_path}, {settings_path}")


if __name__ == "__main__":
    main()
