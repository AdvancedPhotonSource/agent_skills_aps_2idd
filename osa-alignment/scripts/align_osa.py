#!/usr/bin/env python3
"""Align an OSA with calibrated vector corrections and constrained local Bayesian search."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from calibrate_alignment import load_manifest, measure, write_json
from evaluate_alignment import Config, json_safe


@dataclass
class Settings:
    units: str
    bounds: list[list[float]]
    max_step: float
    min_step: float = 0.5
    min_vector: float = 0.5
    min_coverage: float = 0.5
    max_center_shift: float = 40.0
    eta: float | None = None
    enter_m1: float = 0.02
    exit_m1: float = 0.03
    geometry_improvement: float = 0.0001
    m2_improvement: float = 0.0
    max_acquisitions: int = 40
    confirmations: int = 2
    readback_tolerance: float = 0.001
    bo_halfwidth: tuple[float, float] = (5., 5.)
    gp_length_scale: tuple[float, float] = (2.5, 2.5)
    gp_noise: float = 0.05
    ei_xi: float = 0.0
    min_ei: float = 0.0
    bo_candidates: int = 256
    target_m2: float | None = None
    seed: int = 0

    def validate(self, calibration):
        if calibration.get("version") != 2:
            raise ValueError("Calibration version 2 required; regenerate with calibrate_alignment.py")
        bounds = np.asarray(self.bounds, float)
        numbers = [self.max_step, self.min_step, self.enter_m1, self.exit_m1,
                   self.geometry_improvement, self.m2_improvement, self.readback_tolerance,
                   *self.bo_halfwidth, *self.gp_length_scale, self.gp_noise, self.ei_xi, self.min_ei,
                   self.min_vector, self.min_coverage, self.max_center_shift]
        if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1]) or not np.isfinite(numbers).all()
            or self.min_vector < 0 or not 0 <= self.min_coverage <= 1 or self.max_center_shift <= 0
            or not 0 < self.min_step <= self.max_step
            or not 0 < self.enter_m1 < self.exit_m1
            or min(self.bo_halfwidth) <= 0 or len(self.bo_halfwidth) != 2
            or min(self.gp_length_scale) <= 0 or len(self.gp_length_scale) != 2
            or self.gp_noise <= 0 or self.readback_tolerance <= 0
            or min(self.geometry_improvement, self.m2_improvement, self.ei_xi, self.min_ei) < 0):
            raise ValueError("Invalid bounds, thresholds, step sizes, or GP settings")
        if (not all(isinstance(x, int) for x in [self.max_acquisitions, self.confirmations,
                                               self.bo_candidates, self.seed])
            or self.confirmations < 2 or self.max_acquisitions < self.confirmations + 3
            or self.bo_candidates < 4 or self.seed < 0):
            raise ValueError("Require >=2 confirmations, sufficient acquisition budget, and >=4 candidates")
        if self.eta is not None and (not np.isfinite(self.eta) or self.eta <= 0):
            raise ValueError("eta must be positive and finite")
        if self.target_m2 is not None and not np.isfinite(self.target_m2):
            raise ValueError("target_m2 must be finite")
        if self.units != calibration["units"]:
            raise ValueError("Motor units must match calibration")
        trained = np.asarray(calibration["calibration_bounds"])
        if np.any(bounds[:, 0] < trained[:, 0]) or np.any(bounds[:, 1] > trained[:, 1]):
            raise ValueError("Search bounds must lie inside the calibration envelope")
        if self.max_step > calibration["gain_max_step"]:
            raise ValueError("max_step exceeds the calibrated gain's displacement limit")
        Config(**calibration["evaluator"]).validate()
        jacobian = np.asarray(calibration["jacobian"], float)
        if jacobian.shape != (2, 2) or not np.isfinite(jacobian).all() or np.linalg.matrix_rank(jacobian) < 2:
            raise ValueError("Invalid calibrated motor response")
        if not np.isfinite(calibration["eta"]) or calibration["eta"] <= 0:
            raise ValueError("Invalid calibrated gain")


class Replay:
    """Discrete replay of recorded images. Repeated reads are not new measurements."""
    def __init__(self, manifest, start):
        self.units = manifest["units"]
        self.images = {}
        for row in manifest["samples"]:
            self.images.setdefault(tuple(row["position"]), []).append(row["image"])
        self.candidate_positions = np.array(list(self.images), float)
        self.counts = dict.fromkeys(self.images, 0)
        self.move_absolute(start)

    def read_position(self):
        return self.position.copy()

    def move_absolute(self, position):
        if tuple(position) not in self.images:
            raise ValueError("Replay requires an exactly recorded position")
        self.position = np.asarray(position, float)

    def acquire_image(self):
        key = tuple(self.position)
        paths = self.images[key]
        count = self.counts[key]
        self.counts[key] += 1
        return paths[count % len(paths)]


def feasible(observation, settings):
    m = observation["metrics"]
    return bool(m["valid"] and float(m["primary_metric"]) <= settings.enter_m1
                and np.isfinite(float(m["secondary_metric"])))


def geometry(observation):
    return float(observation["metrics"].get("geometry_score", float("inf")))


def constrained_ei(observations, candidates, settings):
    """Independent GP approximations: EI(M2) * P(geometry) * P(valid)."""
    x = np.array([o["position"] for o in observations])
    candidates = np.asarray(candidates)
    kernel = Matern(length_scale=settings.gp_length_scale, length_scale_bounds="fixed", nu=2.5)

    def predict(mask, values):
        gp = GaussianProcessRegressor(kernel=kernel, alpha=settings.gp_noise ** 2,
                                      normalize_y=True, optimizer=None)
        gp.fit(x[mask], np.asarray(values)[mask])
        mu, std = gp.predict(candidates, return_std=True)
        return mu, np.maximum(std, 1e-12)

    good = np.array([feasible(o, settings) for o in observations])
    if not good.any():
        raise ValueError("Expected improvement needs a feasible incumbent")
    m2 = np.array([float(o["metrics"]["secondary_metric"]) for o in observations])
    mu, std = predict(good, m2)
    improvement = m2[good].min() - mu - settings.ei_xi
    z = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    g = np.array([geometry(o) for o in observations])
    finite = np.isfinite(g)
    gm, gs = predict(finite, g)
    p_geometry = norm.cdf((settings.enter_m1 - gm) / gs)
    valid = np.array([bool(o["metrics"]["valid"]) for o in observations])
    if np.unique(valid).size == 2:
        classifier = GaussianProcessClassifier(kernel=kernel, optimizer=None,
                                                random_state=settings.seed)
        classifier.fit(x, valid.astype(int))
        p_valid = classifier.predict_proba(candidates)[:, 1]
    else:
        p_valid = np.full(len(candidates), (valid.sum() + 1) / (len(valid) + 2))
    return np.maximum(ei, 0) * p_geometry * p_valid


class Controller:
    def __init__(self, calibration, settings, backend, log_path, measurement=measure):
        settings.validate(calibration)
        if backend.units != settings.units:
            raise ValueError("Adapter units must match settings")
        self.cal, self.s, self.backend = calibration, settings, backend
        self.measurement, self.log_path = measurement, Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation keeps a previous trajectory from being overwritten.
        with self.log_path.open("x"):
            pass
        self.history, self.current, self.acquisitions = [], None, 0
        self.rng = np.random.default_rng(settings.seed)
        self.replay = isinstance(backend, Replay)
        self.record("start", calibration=calibration, settings=asdict(settings), replay=self.replay)

    def record(self, event, **data):
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **data}
        with self.log_path.open("a") as stream:
            stream.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
        if "text" in data:
            print(data["text"], flush=True)

    def position(self):
        p = np.asarray(self.backend.read_position(), float)
        bounds = np.asarray(self.s.bounds)
        if (p.shape != (2,) or not np.isfinite(p).all()
            or np.any(p < bounds[:, 0]) or np.any(p > bounds[:, 1])):
            raise RuntimeError("Readback is nonfinite or outside allowed bounds")
        return p

    def acquire(self, target, plan, stage):
        if self.acquisitions >= self.s.max_acquisitions:
            raise RuntimeError("Acquisition budget exhausted")
        before = self.position()
        target = np.asarray(target, float)
        bounds = np.asarray(self.s.bounds)
        if (target.shape != (2,) or not np.isfinite(target).all()
            or np.any(target < bounds[:, 0]) or np.any(target > bounds[:, 1])
            or np.linalg.norm(target - before) > self.s.max_step + 1e-9):
            raise RuntimeError("Proposed move exceeds bounds or maximum displacement")
        self.record("plan", text=plan, target=target.tolist(), previous_readback=before.tolist(), stage=stage)
        if np.linalg.norm(target - before) > 1e-12:
            self.backend.move_absolute(target.tolist())
        actual = self.position()
        if np.linalg.norm(actual - target) > self.s.readback_tolerance:
            raise RuntimeError("Motor readback does not match target; no recovery motion attempted")
        self.acquisitions += 1
        image = self.backend.acquire_image()
        after = self.position()
        if np.linalg.norm(after - actual) > self.s.readback_tolerance:
            raise RuntimeError("Motor position changed during acquisition")
        obs = self.measurement(image, self.cal)
        if obs["metrics"]["image_shape"] != self.cal["image_shape"]:
            raise RuntimeError("Image dimensions differ from calibration")
        obs.update(position=after.tolist(), target=target.tolist(), stage=stage,
                   acquisition=self.acquisitions)
        m = obs["metrics"]
        description = (f"Image {obs['acquisition']} at {obs['position']}: fit "
            f"{'valid' if m['valid'] else 'invalid'}; "
            f"inner/outer coverage={m.get('inner_coverage', 0):.3f}/{m.get('outer_coverage', 0):.3f}; "
            f"M1={float(m['primary_metric']):.5g}, hole integral M2={float(m['secondary_metric']):.6g}; "
            f"centroid error={obs['error']}. Reasons: {', '.join(m['reasons']) or 'none'}.")
        if m.get("fit") is not None:
            description += (f" Unsupported arcs reach {m.get('inner_max_gap_degrees', float('nan')):.1f}"
                            f" degrees inside and {m.get('outer_max_gap_degrees', float('nan')):.1f} outside."
                            f" Hole mean={m.get('hole_mean', float('nan')):.5g}.")
        if obs.get("appearance"):
            a = obs["appearance"]
            description += f" Ring sector means={a['ring_sector_means']}; hole peak={a['hole_peak']}."
        if self.current is None:
            trend = "No preceding image."
        else:
            old = self.current["metrics"]
            trend = (f"Previous M1={float(old['primary_metric']):.5g}, "
                     f"M2={float(old['secondary_metric']):.6g}; "
                     f"geometry {'improved' if geometry(obs) < geometry(self.current) else 'unchanged or worse'}.")
        next_plan = "Compare against the incumbent, then follow the centroid correction, restore, confirm, or select a local BO trial."
        self.record("observation", text=f"{description} {trend} Next: {next_plan}",
                    observation=obs, description=description, trend=trend, next_plan=next_plan,
                    remaining=self.s.max_acquisitions - self.acquisitions)
        self.history.append(obs)
        self.current = obs
        return obs

    def candidates(self, incumbent):
        p = np.array(incumbent["position"])
        bounds = np.asarray(self.s.bounds)
        if hasattr(self.backend, "candidate_positions"):
            pool = np.asarray(self.backend.candidate_positions, float)
        else:
            lo = np.maximum(bounds[:, 0], p - self.s.bo_halfwidth)
            hi = np.minimum(bounds[:, 1], p + self.s.bo_halfwidth)
            pool = self.rng.uniform(lo, hi, (self.s.bo_candidates, 2))
        delta = pool - p
        allowed = (np.all((pool >= bounds[:, 0]) & (pool <= bounds[:, 1]), axis=1)
                   & (np.linalg.norm(delta, axis=1) <= self.s.max_step + 1e-9)
                   & (np.linalg.norm(delta, axis=1) >= self.s.min_step - 1e-9))
        allowed &= np.all(np.abs(delta) <= np.asarray(self.s.bo_halfwidth) + 1e-9, axis=1)
        pool = pool[allowed]
        if len(pool):
            seen = np.array([o["position"] for o in self.history])
            pool = pool[np.min(np.linalg.norm(pool[:, None] - seen[None, :], axis=2), axis=1)
                        >= self.s.min_step * 0.5]
        return pool

    def direction(self, observation):
        e, m = observation["error"], observation["metrics"]
        fit = m.get("fit")
        if e is None or fit is None:
            return None
        center = np.array([fit["center_x"], fit["center_y"]])
        if (not np.isfinite(e).all() or np.linalg.norm(e) <= 0
            or np.linalg.norm(e) < self.s.min_vector
            or min(m["inner_coverage"], m["outer_coverage"]) < self.s.min_coverage
            or np.linalg.norm(center - self.cal["reference_center"]) > self.s.max_center_shift):
            return None
        eta = self.cal["eta"] if self.s.eta is None else self.s.eta
        d = -eta * np.linalg.solve(np.asarray(self.cal["jacobian"]), e)
        length = np.linalg.norm(d)
        return d * (np.clip(length, self.s.min_step, self.s.max_step) / length)

    def coarse_trial(self, incumbent, displacement):
        """Shorten the vector at travel bounds without rotating it."""
        p = np.asarray(incumbent["position"], float)
        bounds = np.asarray(self.s.bounds)
        scale = 1.
        for i, value in enumerate(displacement):
            if value > 0:
                scale = min(scale, (bounds[i, 1] - p[i]) / value)
            elif value < 0:
                scale = min(scale, (bounds[i, 0] - p[i]) / value)
        delta = displacement * scale
        target = p + delta
        length = np.linalg.norm(delta)
        if length < self.s.min_step - 1e-9:
            return None
        if hasattr(self.backend, "candidate_positions"):
            # Replay quantizes the vector to available measurements in its forward half-plane.
            pool = np.asarray(self.backend.candidate_positions, float)
            offsets = pool - p
            distances = np.linalg.norm(offsets, axis=1)
            allowed = (np.all((pool >= bounds[:, 0]) & (pool <= bounds[:, 1]), axis=1)
                       & (distances >= self.s.min_step - 1e-9)
                       & (distances <= length + 1e-9) & (offsets @ delta > 0))
            pool = pool[allowed]
            if not len(pool):
                return None
            target = pool[np.argmin(np.linalg.norm(pool - target, axis=1))]
        return target

    def run(self):
        incumbent, reason, status = None, "budget exhausted", "best_found"
        try:
            incumbent = self.acquire(self.position(), "Acquire baseline before choosing a move.", "baseline")
            stage = "coarse"
            failures, ceiling, confirmed = 0, self.s.max_step, 1
            while self.acquisitions < self.s.max_acquisitions - self.s.confirmations - 1:
                good = feasible(incumbent, self.s)
                target_met = good and self.s.target_m2 is not None and float(
                    incumbent["metrics"]["secondary_metric"]) <= self.s.target_m2
                if target_met:
                    reason = "requested metric target reached"
                    break
                if good and stage == "coarse":
                    if confirmed < self.s.confirmations:
                        incumbent = self.acquire(incumbent["position"], "Confirm geometry before local BO.", "confirm")
                        confirmed = confirmed + 1 if feasible(incumbent, self.s) else 0
                        continue
                    stage = "bo"
                if stage == "bo" and (not incumbent["metrics"]["valid"]
                                      or geometry(incumbent) > self.s.exit_m1):
                    stage, confirmed = "coarse", 0
                    failures, ceiling = 0, self.s.max_step
                if stage == "bo":
                    pool = self.candidates(incumbent)
                    if not len(pool):
                        reason = "local candidate set exhausted"
                        break
                    if not any(feasible(o, self.s) for o in self.history):
                        stage = "coarse"
                        continue
                    scores = constrained_ei(self.history, pool, self.s)
                    if scores.max() <= self.s.min_ei:
                        reason = "constrained expected improvement below threshold"
                        break
                    trial = pool[np.argmax(scores)]
                    plan = f"Local BO trial at {trial.tolist()}; constrained EI={scores.max():.5g}."
                else:
                    correction = self.direction(incumbent)
                    if correction is None:
                        probes = np.array([[1., 1.], [-1., -1.], [1., -1.], [-1., 1.]]) / np.sqrt(2)
                        attempts, step = len(probes), ceiling
                        displacement = probes[failures % attempts] * step
                        source = "Centroid confidence insufficient; bounded 2D probe"
                    else:
                        attempts = 2
                        step = min(np.linalg.norm(correction), ceiling)
                        displacement = correction * (step / np.linalg.norm(correction))
                        if failures % attempts:
                            displacement = -displacement
                        source = "Calibrated centroid correction" if failures % attempts == 0 else "Reverse correction probe"
                    trial = self.coarse_trial(incumbent, displacement)
                    plan = f"{source}, displacement={displacement.tolist()}; seek lower geometry error."
                accepted = False
                if trial is not None:
                    obs = self.acquire(trial, plan, stage)
                    accepted = ((feasible(obs, self.s) and (not good or float(obs["metrics"]["secondary_metric"])
                                < float(incumbent["metrics"]["secondary_metric"]) - self.s.m2_improvement))
                                if stage == "bo" else
                                (feasible(obs, self.s) or geometry(obs) < geometry(incumbent) - self.s.geometry_improvement))
                    if accepted:
                        incumbent, confirmed = obs, 1
                        failures, ceiling = 0, self.s.max_step
                        continue
                    incumbent = self.acquire(incumbent["position"], "Trial did not improve the objective; restore and verify incumbent.", "restore")
                    confirmed = 1
                if stage == "coarse":
                    failures += 1
                    if failures >= attempts:
                        failures = 0
                        ceiling = step / 2
                        if ceiling < self.s.min_step:
                            reason = "coarse vector search exhausted at minimum step"
                            break
            # Final repeated acquisitions count toward the same budget.
            confirmations = []
            for _ in range(self.s.confirmations):
                if self.acquisitions >= self.s.max_acquisitions:
                    break
                obs = self.acquire(incumbent["position"], "Reimage final position to check metric stability.", "final")
                confirmations.append(obs)
            stable = len(confirmations) == self.s.confirmations and all(feasible(o, self.s) for o in confirmations)
            if stable and self.s.target_m2 is not None and all(
                float(o["metrics"]["secondary_metric"]) <= self.s.target_m2 for o in confirmations):
                status = "metric_target_reached"
            elif not stable:
                status = "unconfirmed"
                reason += "; final geometry confirmation failed"
            elif reason == "requested metric target reached":
                reason = "M2 target not sustained during final confirmation"
        except Exception as exc:
            status, reason = "blocked", f"{type(exc).__name__}: {exc}"
            self.record("fault", text=reason)
        best = min((o for o in self.history if feasible(o, self.s)),
                   key=lambda o: float(o["metrics"]["secondary_metric"]), default=None)
        if best is None:
            best = min(self.history, key=geometry, default=None)
        try:
            final_position = self.position().tolist()
        except Exception:
            final_position = None
        result = {"status": status, "reason": reason, "replay": self.replay,
                  "acquisitions": self.acquisitions, "best_observed": best,
                  "final_observation": self.current, "final_readback": final_position,
                  "log": str(self.log_path),
                  "qualification": "Metric convergence only; validate local hole defects and hardware behavior independently."}
        self.record("complete", text=f"Stopped: {status}; {reason}.", result=result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="JSON Settings object")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Replay recorded positions without interpolation")
    source.add_argument("--adapter", help="Beamline adapter MODULE:FACTORY; no hardware binding is supplied")
    parser.add_argument("--adapter-config", type=Path)
    parser.add_argument("--start", type=float, nargs=2, help="Replay starting position")
    parser.add_argument("--log", type=Path, required=True, help="New JSONL trajectory file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    settings = Settings(**json.loads(args.config.read_text()))
    settings.validate(calibration)
    if args.manifest:
        if args.start is None:
            parser.error("Replay requires --start X Y")
        backend = Replay(load_manifest(args.manifest), args.start)
    else:
        if args.start is not None:
            parser.error("Live mode reads its starting position from the adapter")
        module, factory = args.adapter.split(":", 1)
        options = json.loads(args.adapter_config.read_text()) if args.adapter_config else {}
        backend = getattr(importlib.import_module(module), factory)(options)
    result = Controller(calibration, settings, backend, args.log).run()
    write_json(args.output, result)
    if result["status"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
