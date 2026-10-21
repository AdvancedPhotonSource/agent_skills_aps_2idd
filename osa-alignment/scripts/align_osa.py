#!/usr/bin/env python3
"""Align an OSA using global M1 and local M2 Bayesian optimization."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from functools import partial
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from alignment_io import load_manifest, measure, write_json
from evaluate_alignment import Config, json_safe


@dataclass
class Settings:
    units: str
    bounds: list[list[float]]
    max_step: float
    min_step: float = 0.5
    evaluator: dict = field(default_factory=dict)
    stage_2_entrance_m1: float = 0.02
    stage_1_max_acquisitions: int = 15
    max_acquisitions: int = 40
    confirmations: int = 2
    readback_tolerance: float = 0.001
    stage_2_bo_halfwidth: tuple[float, float] = (5., 5.)
    gp_length_scale: tuple[float, float] = (2.5, 2.5)
    gp_length_scale_bounds: tuple[float, float] = (0.5, 10.)
    bo_initial_samples: int = 5
    gp_noise: float = 0.05
    stage_2_gp_stage_1_noise_multiplier: float = 3.0
    ei_xi: float = 0.0
    stage_2_min_ei: float = 0.0
    bo_candidates: int = 256
    stage_2_target_m2: float | None = None
    seed: int = 0

    def validate(self):
        bounds = np.asarray(self.bounds, float)
        if (not np.isfinite([self.min_step, self.max_step]).all()
            or self.max_step <= 0 or not 0 <= self.min_step <= self.max_step):
            raise ValueError("Require max_step > 0 and 0 <= min_step <= max_step")
        numbers = [self.max_step, self.min_step, self.stage_2_entrance_m1,
                   self.readback_tolerance,
                   *self.stage_2_bo_halfwidth, *self.gp_length_scale, *self.gp_length_scale_bounds, self.gp_noise, self.ei_xi, self.stage_2_min_ei,
                   self.stage_2_gp_stage_1_noise_multiplier]
        if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1]) or not np.isfinite(numbers).all()
            or not 0 < self.stage_2_entrance_m1
            or min(self.stage_2_bo_halfwidth) <= 0 or len(self.stage_2_bo_halfwidth) != 2
            or min(self.gp_length_scale) <= 0 or len(self.gp_length_scale) != 2
            or len(self.gp_length_scale_bounds) != 2
            or not 0 < self.gp_length_scale_bounds[0] < self.gp_length_scale_bounds[1]
            or any(not self.gp_length_scale_bounds[0] <= v <= self.gp_length_scale_bounds[1]
                   for v in self.gp_length_scale)
            or self.gp_noise <= 0 or self.readback_tolerance <= 0
            or self.stage_2_gp_stage_1_noise_multiplier < 1
            or min(self.ei_xi, self.stage_2_min_ei) < 0):
            raise ValueError("Invalid bounds, thresholds, step sizes, or GP settings")
        if (not all(isinstance(x, int) for x in [self.max_acquisitions, self.confirmations,
                                               self.bo_candidates, self.bo_initial_samples, self.stage_1_max_acquisitions, self.seed])
            or self.confirmations < 2 or self.max_acquisitions < self.confirmations + 3
            or not 1 <= self.stage_1_max_acquisitions < self.max_acquisitions - self.confirmations
            or self.bo_initial_samples < 0 or self.bo_candidates < 4 or self.seed < 0):
            raise ValueError("Require >=2 confirmations, sufficient acquisition budget, and >=4 candidates")
        if self.stage_2_target_m2 is not None and not np.isfinite(self.stage_2_target_m2):
            raise ValueError("stage_2_target_m2 must be finite")
        if self.bo_initial_samples > self.stage_1_max_acquisitions:
            raise ValueError("stage_1_max_acquisitions must accommodate bo_initial_samples")
        Config(**self.evaluator).validate()


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


def eligible_geometry(observation, settings):
    m = observation["metrics"]
    m1 = float(m["primary_metric"])
    return bool(m["valid"] and np.isfinite(m1) and m1 <= settings.stage_2_entrance_m1)


def feasible(observation, settings):
    return bool(eligible_geometry(observation, settings)
                and np.isfinite(float(observation["metrics"]["secondary_metric"])))


def objective_observations(observations, settings, metric):
    if metric == "primary_metric":
        return [o for o in observations if np.isfinite(float(o["metrics"][metric]))]
    return [o for o in observations if feasible(o, settings)]


def fit_objective(observations, settings, metric="secondary_metric"):
    """Model every finite M1, or only eligible M2, including repeated measurements."""
    eligible = objective_observations(observations, settings, metric)
    if not eligible:
        raise ValueError("Expected improvement needs an eligible observation")
    x = np.array([o["position"] for o in eligible])
    y = np.array([float(o["metrics"][metric]) for o in eligible])
    noise = np.array([settings.gp_noise * (
        settings.stage_2_gp_stage_1_noise_multiplier
        if metric == "secondary_metric" and o.get("stage") in {"baseline", "stage1", "stage1_init"}
        else 1.) for o in eligible])
    # Repeated confirmations alone cannot identify spatial length scales.
    fitted = len(np.unique(x, axis=0)) >= 3
    kernel = Matern(length_scale=settings.gp_length_scale,
                    length_scale_bounds=settings.gp_length_scale_bounds, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=noise ** 2,
                                  normalize_y=True,
                                  optimizer="fmin_l_bfgs_b" if fitted else None)
    gp.fit(x, y)
    return gp, fitted


def expected_improvement(gp, candidates, best_value, settings):
    mu, std = gp.predict(np.asarray(candidates), return_std=True)
    std = np.maximum(std, 1e-12)
    improvement = best_value - mu - settings.ei_xi
    z = improvement / std
    return np.maximum(improvement * norm.cdf(z) + std * norm.pdf(z), 0)


class Controller:
    def __init__(self, settings, backend, log_path, measurement=None):
        settings.validate()
        if backend.units != settings.units:
            raise ValueError("Adapter units must match settings")
        self.s, self.backend = settings, backend
        self.measurement = measurement if measurement is not None else partial(
            measure, config=Config(**settings.evaluator))
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation keeps a previous trajectory from being overwritten.
        with self.log_path.open("x"):
            pass
        self.history, self.current, self.acquisitions = [], None, 0
        self.best = None
        self.length_scales = np.asarray(settings.gp_length_scale, float)
        self.rng = np.random.default_rng(settings.seed)
        self.replay = isinstance(backend, Replay)
        self.record("start", settings=asdict(settings), replay=self.replay)

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
            or np.any(p < bounds[:, 0] - self.s.readback_tolerance)
            or np.any(p > bounds[:, 1] + self.s.readback_tolerance)):
            raise RuntimeError("Readback is nonfinite or outside allowed bounds")
        return p

    def move(self, target, plan, stage, *, limit_step=True):
        before = self.position()
        target = np.asarray(target, float)
        bounds = np.asarray(self.s.bounds)
        if (target.shape != (2,) or not np.isfinite(target).all()
            or np.any(target < bounds[:, 0]) or np.any(target > bounds[:, 1])
            or (limit_step and np.linalg.norm(target - before) > self.s.max_step + 1e-9)):
            raise RuntimeError("Proposed move exceeds bounds or maximum displacement")
        self.record("plan", text=plan, target=target.tolist(), previous_readback=before.tolist(), stage=stage)
        if np.linalg.norm(target - before) > 1e-12:
            self.backend.move_absolute(target.tolist())
        actual = self.position()
        if np.linalg.norm(actual - target) > self.s.readback_tolerance:
            raise RuntimeError("Motor readback does not match target; no recovery motion attempted")
        return actual

    def return_to_best(self, observation):
        self.travel_to(observation["target"])

    def best_observed(self, bo_started):
        metric = "secondary_metric" if bo_started else "primary_metric"
        observations = (o for o in self.history
                        if (feasible(o, self.s) if bo_started
                            else np.isfinite(float(o["metrics"][metric]))))
        return min(observations, key=lambda o: float(o["metrics"][metric]), default=None)

    def acquire(self, target, plan, stage):
        if self.acquisitions >= self.s.max_acquisitions:
            raise RuntimeError("Acquisition budget exhausted")
        target = np.asarray(target, float)
        actual = self.move(target, plan, stage,
                           limit_step=stage not in {"stage1_init", "stage_2_init"})
        self.acquisitions += 1
        image = self.backend.acquire_image()
        after = self.position()
        if np.linalg.norm(after - actual) > self.s.readback_tolerance:
            raise RuntimeError("Motor position changed during acquisition")
        obs = self.measurement(image)
        obs.update(position=after.tolist(), target=target.tolist(), stage=stage,
                   acquisition=self.acquisitions)
        m = obs["metrics"]
        description = (f"Image {obs['acquisition']} at {obs['position']}: fit "
            f"{'valid' if m['valid'] else 'invalid'}; "
            f"inner/outer coverage={m.get('inner_coverage', 0):.3f}/{m.get('outer_coverage', 0):.3f}; "
            f"M1={float(m['primary_metric']):.5g}, hole integral M2={float(m['secondary_metric']):.6g}; "
            f"Reasons: {', '.join(m['reasons']) or 'none'}.")
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
                     f"M1 {'improved' if float(m['primary_metric']) < float(old['primary_metric']) else 'unchanged or worse'}.")
        eligible = feasible(obs, self.s)
        if not m["valid"]:
            rejection = "invalid fit"
        elif not np.isfinite(float(m["primary_metric"])) or float(m["primary_metric"]) > self.s.stage_2_entrance_m1:
            rejection = "M1 exceeds BO threshold or is nonfinite"
        elif not np.isfinite(float(m["secondary_metric"])):
            rejection = "nonfinite M2"
        else:
            rejection = None
        obs.update(bo_eligible=eligible, bo_rejection_reason=rejection)
        if eligible and (self.best is None or float(m["secondary_metric"]) <
                         float(self.best["metrics"]["secondary_metric"])):
            self.best = obs
        next_plan = "Update measured eligibility and best position; continue the current search stage or confirm the final position."
        self.record("observation", text=f"{description} {trend} Next: {next_plan}",
                    observation=obs, description=description, trend=trend, next_plan=next_plan,
                    remaining=self.s.max_acquisitions - self.acquisitions)
        self.history.append(obs)
        self.current = obs
        return obs

    def candidates(self, incumbent, *, initial=False):
        p = np.array(incumbent["position"])
        bounds = np.asarray(self.s.bounds)
        if hasattr(self.backend, "candidate_positions"):
            pool = np.asarray(self.backend.candidate_positions, float)
        else:
            lo = np.maximum(bounds[:, 0], p - self.s.stage_2_bo_halfwidth)
            hi = np.minimum(bounds[:, 1], p + self.s.stage_2_bo_halfwidth)
            pool = self.rng.uniform(lo, hi, (self.s.bo_candidates, 2))
        allowed = np.all((pool >= bounds[:, 0]) & (pool <= bounds[:, 1]), axis=1)
        if not initial:
            distance = np.linalg.norm(pool - self.position(), axis=1)
            allowed &= ((distance <= self.s.max_step + 1e-9)
                        & (distance >= max(1e-9, self.s.min_step - 1e-9)))
        allowed &= np.all(np.abs(pool - p) <= np.asarray(self.s.stage_2_bo_halfwidth) + 1e-9, axis=1)
        pool = pool[allowed]
        if len(pool) and self.history:
            seen = np.array([o["position"] for o in self.history])
            pool = pool[np.min(np.linalg.norm(pool[:, None] - seen[None, :], axis=2), axis=1)
                        >= (1e-9 if initial else max(1e-9, self.s.min_step * 0.5))]
        rejected = [o["position"] for o in self.history
                    if not eligible_geometry(o, self.s)]
        if len(pool) and rejected:
            offsets = (pool[:, None] - np.asarray(rejected)[None, :]) / self.length_scales
            pool = pool[np.all(np.sum(offsets ** 2, axis=2) > 1., axis=1)]
        return pool

    def posterior_mean_center(self, gp, center=None):
        """Minimize M1 globally, or M2 within the current unexcluded box."""
        global_search = center is None
        bounds = np.asarray(self.s.bounds)
        center = np.clip(self.position(), bounds[:, 0], bounds[:, 1]) if global_search else np.asarray(center, float)
        lo = bounds[:, 0] if global_search else np.maximum(bounds[:, 0], center - self.s.stage_2_bo_halfwidth)
        hi = bounds[:, 1] if global_search else np.minimum(bounds[:, 1], center + self.s.stage_2_bo_halfwidth)
        rejected = np.array([] if global_search else [o['position'] for o in self.history
                             if not eligible_geometry(o, self.s)]).reshape(-1, 2)

        def clearance(points):
            offsets = (np.asarray(points)[..., None, :] - rejected) / self.length_scales
            return np.sum(offsets ** 2, axis=-1) - 1.

        def allowed(points):
            return (np.all((points >= lo) & (points <= hi), axis=1)
                    & np.all(clearance(points) >= -1e-8, axis=1))

        if global_search and hasattr(self.backend, 'candidate_positions'):
            # Recorded/discrete domains must be minimized at available positions.
            pool = np.asarray(self.backend.candidate_positions, float)
            pool = pool[allowed(pool)]
            pool = pool[np.argsort(np.linalg.norm(pool - center, axis=1), kind='stable')]
            return pool[np.argmin(gp.predict(pool))].copy()

        grid = np.stack(np.meshgrid(*[np.linspace(a, b, 17) for a, b in zip(lo, hi)]), axis=-1).reshape(-1, 2)
        measured = np.array([o['position'] for o in self.history]).reshape(-1, 2)
        pool = np.vstack([center, grid, measured])
        pool = pool[allowed(pool)]
        if not len(pool):
            return center.copy()
        means = gp.predict(pool)
        scale = float(np.ptp(means))
        if scale == 0:
            return pool[0].copy()
        offset = float(np.min(means))

        def objective(point):
            return (float(gp.predict(np.asarray(point)[None, :])[0]) - offset) / scale

        constraints = [{'type': 'ineq', 'fun': clearance}] if len(rejected) else []
        refined = []
        for start in pool[np.argsort(means, kind='stable')[:5]]:
            result = minimize(objective, start, method='SLSQP', bounds=list(zip(lo, hi)),
                              constraints=constraints, options={'ftol': 1e-10, 'maxiter': 100})
            point = np.clip(result.x, lo, hi)
            if result.success and allowed(point[None, :])[0]:
                refined.append(point)
        if refined:
            pool = np.vstack([pool, refined])
        return pool[np.argmin(gp.predict(pool))].copy()

    def stage1_candidates(self, *, initial=False):
        """Propose throughout the global bounds without geometry exclusions."""
        bounds = np.asarray(self.s.bounds)
        pool = (np.asarray(self.backend.candidate_positions, float)
                if hasattr(self.backend, "candidate_positions") else
                self.rng.uniform(bounds[:, 0], bounds[:, 1], (self.s.bo_candidates, 2)))
        allowed = np.all((pool >= bounds[:, 0]) & (pool <= bounds[:, 1]), axis=1)
        pool = pool[allowed]
        if len(pool) and self.history:
            seen = np.array([o["position"] for o in self.history])
            pool = pool[np.min(np.linalg.norm(pool[:, None] - seen[None, :], axis=2), axis=1)
                        >= (1e-9 if initial else max(1e-9, self.s.min_step))]
        if not initial and self.replay and len(pool):
            reachable, _ = self.replay_routes()
            pool = pool[[tuple(p) in reachable for p in pool]]
        return pool

    def replay_routes(self):
        """Find recorded waypoints reachable by moves no longer than max_step."""
        bounds = np.asarray(self.s.bounds)
        points = np.asarray(self.backend.candidate_positions, float)
        points = points[np.all((points >= bounds[:, 0]) & (points <= bounds[:, 1]), axis=1)]
        start = tuple(self.position())
        parents, queue = {start: None}, [start]
        for point in queue:
            distances = np.linalg.norm(points - point, axis=1)
            for neighbor in points[distances <= self.s.max_step + 1e-9]:
                key = tuple(neighbor)
                if key not in parents:
                    parents[key] = point
                    queue.append(key)
        return parents.keys(), parents

    def travel_to(self, target):
        """Reach a global BO target using bounded transit moves without images."""
        target = np.asarray(target, float)
        if self.replay:
            _, parents = self.replay_routes()
            key, path = tuple(target), []
            if key not in parents:
                raise RuntimeError("No bounded replay path to target")
            while parents[key] is not None:
                path.append(key)
                key = parents[key]
            for waypoint in reversed(path):
                self.move(waypoint, "Bounded replay transit to BO target.", "transit")
        else:
            distance = np.linalg.norm(target - self.position())
            while distance > self.s.max_step:
                before = self.position()
                waypoint = before + (target - before) * (self.s.max_step / distance)
                actual = self.move(waypoint, "Bounded transit to BO target.", "transit")
                remaining = np.linalg.norm(target - actual)
                if remaining >= distance:
                    raise RuntimeError("Transit made no progress")
                distance = remaining
            self.move(target, "Reach BO target.", "transit")

    def run(self):
        incumbent, reason, status = None, "budget exhausted", "best_found"
        bo_started = False
        try:
            bounds = np.asarray(self.s.bounds)
            baseline_target = np.clip(self.position(), bounds[:, 0], bounds[:, 1])
            incumbent = self.acquire(baseline_target, "Acquire baseline before choosing a move.", "baseline")
            stage, initial_samples = "stage1", 0
            while self.acquisitions < self.s.max_acquisitions - self.s.confirmations:
                primary_best = self.best_observed(False)
                if stage == "stage1":
                    gp, fitted = fit_objective(self.history, self.s, "primary_metric")
                    self.record("bo_model", stage=stage, objective="primary_metric",
                                length_scales=np.asarray(gp.kernel_.length_scale).tolist(),
                                length_scales_fitted=fitted, training_count=len(gp.X_train_),
                                training_acquisitions=[o["acquisition"] for o in
                                    objective_observations(self.history, self.s, "primary_metric")],
                                exclusion_acquisitions=[])
                    if self.acquisitions >= self.s.bo_initial_samples:
                        stage2_center = self.posterior_mean_center(gp)
                        predicted_m1 = float(gp.predict(stage2_center[None, :])[0])
                        threshold_met = predicted_m1 <= self.s.stage_2_entrance_m1
                        budget_met = self.acquisitions >= self.s.stage_1_max_acquisitions
                        self.record("stage_2_entry_check", position=stage2_center.tolist(),
                                    posterior_mean_m1=predicted_m1, threshold_met=threshold_met,
                                    stage1_acquisitions=self.acquisitions)
                        if threshold_met or budget_met:
                            stage, bo_started = "stage_2", True
                            self.record("stage_transition", stage=stage, from_stage="stage1",
                                        reason="M1 posterior mean threshold reached" if threshold_met else "stage 1 budget exhausted",
                                        stage1_acquisitions=self.acquisitions,
                                        position=stage2_center.tolist(), posterior_mean_m1=predicted_m1)
                            self.record("search_box", stage=stage, center=stage2_center.tolist(),
                                        centering="initialization_anchor")
                            acquisition_stage = "stage_2_init" if self.s.bo_initial_samples else "stage_2"
                            if acquisition_stage == "stage_2":
                                self.travel_to(stage2_center)
                            obs = self.acquire(stage2_center, "Measure Stage 2 entry at the M1 posterior mean minimum.",
                                               acquisition_stage)
                            if acquisition_stage == "stage_2_init" and feasible(obs, self.s):
                                initial_samples += 1
                            continue
                    pool = self.stage1_candidates(initial=self.acquisitions < self.s.bo_initial_samples)
                    if not len(pool):
                        # A discrete or very small domain can be fully sampled before the budget.
                        trial = np.asarray(primary_best["target"])
                        plan, acquisition_stage = "Reimage best M1 after exhausting distinct Stage 1 candidates.", "stage1"
                    elif self.acquisitions < self.s.bo_initial_samples:
                        trial = pool[self.rng.integers(len(pool))]
                        plan, acquisition_stage = f"Move directly to random M1 initialization target {trial.tolist()}.", "stage1_init"
                    else:
                        scores = expected_improvement(gp, pool, float(primary_best["metrics"]["primary_metric"]), self.s)
                        trial = pool[np.argmax(scores)]
                        plan, acquisition_stage = f"Global M1 BO trial; EI={scores.max():.5g}.", "stage1"
                    if acquisition_stage != "stage1_init":
                        self.travel_to(trial)
                    self.acquire(trial, plan, acquisition_stage)
                    continue

                # Stage 2 can start on budget exhaustion before any eligible M2 exists.
                incumbent = self.best or primary_best
                if (self.best is not None and self.s.stage_2_target_m2 is not None
                    and float(self.best["metrics"]["secondary_metric"]) <= self.s.stage_2_target_m2):
                    reason = "requested metric target reached"
                    break
                gp = None
                if self.best is not None:
                    gp, fitted = fit_objective(self.history, self.s)
                    self.length_scales = np.asarray(gp.kernel_.length_scale)
                    self.record("bo_model", stage="stage_2", objective="secondary_metric",
                                length_scales=self.length_scales.tolist(),
                                length_scales_fitted=fitted, training_count=len(gp.X_train_),
                                training_acquisitions=[o["acquisition"] for o in self.history if feasible(o, self.s)],
                                initial_samples=initial_samples,
                                exclusion_acquisitions=[o["acquisition"] for o in self.history
                                    if not eligible_geometry(o, self.s)])
                initializing = initial_samples < self.s.bo_initial_samples or gp is None
                if not initializing:
                    stage2_center = self.posterior_mean_center(gp, stage2_center)
                self.record("search_box", stage="stage_2", center=stage2_center.tolist(),
                            centering="initialization_anchor" if initializing else "posterior_mean")
                pool = self.candidates({'position': stage2_center}, initial=initializing)
                if not len(pool):
                    reason = "no sampleable candidates after travel, spacing, and M1 exclusions"
                    if initial_samples < self.s.bo_initial_samples or gp is None:
                        reason += "; random initialization incomplete"
                    break
                if initializing:
                    trial = pool[self.rng.integers(len(pool))]
                    plan = (f"Move directly to random M2 initialization target {trial.tolist()}; "
                            f"{initial_samples + 1}/{self.s.bo_initial_samples} eligible locations.")
                    acquisition_stage = "stage_2_init"
                else:
                    scores = expected_improvement(gp, pool, float(incumbent["metrics"]["secondary_metric"]), self.s)
                    if scores.max() <= self.s.stage_2_min_ei:
                        reason = "expected improvement below threshold"
                        break
                    trial = pool[np.argmax(scores)]
                    plan = f"Local M2 BO trial at {trial.tolist()}; EI={scores.max():.5g}."
                    acquisition_stage = "stage_2"
                obs = self.acquire(trial, plan, acquisition_stage)
                if acquisition_stage == "stage_2_init" and feasible(obs, self.s):
                    initial_samples += 1
            if stage == "stage_2" and initial_samples < self.s.bo_initial_samples and reason == "budget exhausted":
                reason = "random initialization budget exhausted"
            incumbent = self.best_observed(bo_started) or self.best_observed(False) or incumbent
            self.return_to_best(incumbent)
            # Final repeated acquisitions count toward the same budget.
            confirmations = []
            for _ in range(self.s.confirmations):
                if self.acquisitions >= self.s.max_acquisitions:
                    break
                obs = self.acquire(incumbent["target"], "Reimage final position to check metric stability.", "final")
                confirmations.append(obs)
            stable = len(confirmations) == self.s.confirmations and all(feasible(o, self.s) for o in confirmations)
            if stable and self.s.stage_2_target_m2 is not None and all(
                float(o["metrics"]["secondary_metric"]) <= self.s.stage_2_target_m2 for o in confirmations):
                status = "metric_target_reached"
            elif not stable:
                status = "unconfirmed"
                reason += "; final geometry confirmation failed"
            elif reason == "requested metric target reached":
                reason = "M2 target not sustained during final confirmation"
        except Exception as exc:
            status, reason = "blocked", f"{type(exc).__name__}: {exc}"
            self.record("fault", text=reason)
        best = self.best_observed(bo_started) or self.best_observed(False)
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
    parser.add_argument("--config", type=Path, required=True, help="JSON Settings object")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Replay recorded positions without interpolation")
    source.add_argument("--adapter", help="Beamline adapter MODULE:FACTORY, e.g. aps2idd_adapter:create_adapter")
    parser.add_argument("--adapter-config", type=Path)
    parser.add_argument("--start", type=float, nargs=2, help="Replay starting position")
    parser.add_argument("--log", type=Path, required=True, help="New JSONL trajectory file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--webui", action="store_true", help="Serve a read-only live dashboard")
    parser.add_argument("--webui-port", type=int, default=8765, help="Local dashboard port (0 selects a free port)")
    args = parser.parse_args()
    if not 0 <= args.webui_port <= 65535:
        parser.error("--webui-port must be between 0 and 65535")
    settings = Settings(**json.loads(args.config.read_text()))
    settings.validate()
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
    controller = Controller(settings, backend, args.log)
    dashboard = None
    if args.webui:
        from alignment_webui import Dashboard
        dashboard = Dashboard(args.log, port=args.webui_port)
        dashboard.start()
        print(f"OSA web UI: {dashboard.url} (read-only)", flush=True)
    try:
        result = controller.run()
        write_json(args.output, result)
    finally:
        if dashboard is not None:
            dashboard.close()
    if result["status"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
