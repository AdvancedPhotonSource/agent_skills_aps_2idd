# Analytical calibration and alignment

The analytical workflow uses three scripts: `evaluate_alignment.py` supplies the
ellipse metrics, `calibrate_alignment.py` estimates the motor response and gain,
and `align_osa.py` controls a bounded search. Run the commands below from the skill
directory. Install the locked dependencies in the local environment first:

```sh
uv sync --project scripts --locked
```

## Quantitative evaluation

Use [scripts/evaluate_alignment.py](../scripts/evaluate_alignment.py) to measure
concentric boundary geometry and integrated hole intensity. Dependencies are
managed by the local `uv` project in `scripts/`. From the skill directory:

```sh
uv sync --project scripts --locked
uv run --project scripts --locked python scripts/evaluate_alignment.py IMAGE --output metrics.json
uv run --project scripts --locked python scripts/evaluate_alignment.py IMAGE --inner-axes 38 34 --outer-axes 98 94 --output metrics.json
```

The script fits two ellipses sharing a center and orientation, with independently
fitted semiaxes. Supply both `--inner-axes A B` and `--outer-axes A B` to hold the
four semiaxes fixed and fit only the center and common orientation. Omit both
options to fit the semiaxes as well. Fixed semiaxes keep the hole area comparable
across an alignment sequence. The example dimensions are illustrative. These are
semiaxis lengths in pixels, not diameters; corresponding inner semiaxes must be
smaller than the outer ones. `angle_degrees` describes the A-axis clockwise from
image +x toward image +y, modulo 180°. A and B need not be ordered by length.
The fit uses ellipse normals to classify edges and integrates the elliptical
hole. Nearly circular ellipses have poorly determined orientation. Extra shape
freedom can absorb actual misalignment, so inspect fitted aspect ratios and the
image as well as the residual.

The script detects Canny edges, distinguishes inward and outward intensity
transitions, and fits the concentric ellipses. Lower scores are better:

- `primary_metric`: mean inner/outer circumference-to-edge RMS distance divided
  by outer equal-area radius (√(AB) for an ellipse), plus a penalty for unsupported
  angular coverage. Both boundaries receive equal weight, with 360 true polar-angle
  samples per boundary and a default coverage weight λ = 0.5. This measures
  geometry and completeness.
- `secondary_metric`: background-subtracted intensity summed inside the inner
  boundary, optionally divided by `--intensity-normalizer`. Use `--background`
  for a known scalar detector background. The input intensity scale is preserved;
  RGB images use luminance weights. Display-scaled images do not give photon counts.

An invalid fit sets both metrics to infinity (encoded as `"inf"` in JSON) and
provides failure reasons. Candidate boundaries and `geometry_score` may remain for
diagnosis. A valid fit establishes measurable geometry, not successful alignment.
Review boundary RMS errors, angular coverage, hole mean, and hole area alongside
the scores. The support thresholds are adjustable prototype defaults; use
`--help` for options. Texture, saturation, and soft edges can bias the fit.

The controller first establishes feasible geometry using M1, then minimizes M2
within the configured M1 constraint.

## Calibration from recorded images

Collect images at known transverse motor positions spanning both axes around a
known good reference. Keep optics, image orientation, detector settings, exposure,
and incident-flux normalization consistent. Include repeated reference images
when available. Positions must use the same coordinate system and units as the
adapter readbacks. A generic manifest associates images with positions:

```json
{
  "units": "um",
  "samples": [
    {"position": [0, 0], "image": "images/reference.tiff"},
    {"position": [-5, 0], "image": "images/x_minus.tiff"},
    {"position": [5, 0], "image": "images/x_plus.tiff"},
    {"position": [0, -5], "image": "images/y_minus.tiff"},
    {"position": [0, 5], "image": "images/y_plus.tiff"}
  ]
}
```

Image paths resolve relative to the manifest. The offsets and units above are
illustrative. Collect enough displaced images that at least four usable samples
span both axes, including after holding out every distinct position. Samples
below the coarse M1 threshold are excluded from response fitting because the
centroid can cease to predict motor error near the optimum. The example's four
nonreference samples must therefore be sufficiently displaced for the local setup.

```sh
uv run --project scripts --locked python scripts/calibrate_alignment.py calibration-images.json \
  --reference-position 0 0 --max-step 5 --eta-grid 0.25 0.5 0.75 1 1.5 2 \
  --output calibration.json --observations calibration-observations.json
```

Calibration writes three separate files (names follow `--output`):

| File | Contents |
| --- | --- |
| `calibration.json` | Only `eta`, `jacobian`, and `reference_error` |
| `calibration.alignment.json` | Suggested runtime units, bounds, step limits, and direction-confidence thresholds; pass this through `align_osa.py --config` |
| `calibration.report.json` | Gain scores, fit diagnostics, reference position/M2, and calibration measurement/filter settings; not read during alignment |

Use `--alignment-config` and `--report` to choose other output paths. All output
paths must differ. `--observations` optionally saves per-image evaluation results.
Generated alignment settings are a starting point: adjust permitted travel,
minimum useful step, BO settings, and budgets for the instrument or replay grid.

Alignment reads exactly three calibration quantities: `eta`, the forward
`jacobian`, and `reference_error`. The controller solves `J delta = e` at runtime.
`reference_error` is the relative CoM-to-ellipse-center offset, expressed in pixels.
Each image supplies its own fitted center, semiaxes, and dimensions. Translations
of the illumination and changes to the image canvas are supported.

Configure image processing for the current run through `evaluator` (the fields
of `evaluate_alignment.Config`) and `centroid_percentile` in the alignment JSON.
The default evaluator fits the semiaxes independently for each image. To fix
semiaxes for a particular run, explicitly supply both `inner_axes` and
`outer_axes` in `evaluator`. Motor units, travel bounds, and step limits likewise
come from the runtime configuration, with units checked against the adapter.

The bundled [calibration](../scripts/calibration.json),
[alignment settings](../scripts/calibration.alignment.json), and
[report](../scripts/calibration.report.json) keep the fitted response, runtime
choices, and historical calibration diagnostics separate. Additional fields in
older calibration files are ignored. Remove `max_center_shift` from older
runtime configurations; fitting options belong in their `evaluator` object.

The bundled runtime file overrides eta to `0.75`, sets `enter_m1 = 0.012` and
`exit_m1 = 0.02`, and limits total acquisitions to 30. The tighter entry gate
keeps the controller following the centroid correction longer before local BO.
Their earlier simulator tuning used explicitly fixed semiaxes. Choose fitting
options for the current images when assessing convergence and comparing M2.
The parameter table below lists script defaults; explicit runtime settings
override them. New calibration runs emit initial settings for local tuning.

During calibration, the reference is fitted first; its inner and outer semiaxes
are then fixed for the remaining calibration images. Each fit can translate and rotate. To specify semiaxes or
change edge detection, background subtraction, or intensity normalization, pass
`--evaluator-config evaluator.json`, containing fields from the evaluator's
`Config`, for example:

```json
{"inner_axes": [38, 34], "outer_axes": [102, 92], "background": 0, "intensity_normalizer": 1}
```

The feature is

\[
 e=(c_{\mathrm{CoM}}-c_{\mathrm{ellipse}})-e_{\mathrm{reference}}.
\]

The centroid uses positive intensity after subtraction of the image's 10th
percentile, restricted to the fitted outer ellipse. The reference offset is the
average centroid offset over the reference images. This background estimate is
for the centroid; M2 uses the evaluator's separately configured background.
Image x points right and image y points down.

Calibration fits `e = J (p - p_reference)` by least squares. `J` contains axis
signs, scale, rotation, and coupling in pixels per motor unit. The proposed motor
correction is `-eta * inverse(J) @ e`. A scalar eta alone cannot establish camera
to motor signs. Coarse alignment moves both motors along this full vector.
Step limits and travel bounds shorten the vector while preserving its direction.

Each candidate eta is scored by leaving out every image at one motor position,
refitting J on the remaining positions, and predicting the residual distance to
the reference after one bounded correction. The smallest gain wins a tie.
The separate report records this position MSE, response condition number, and fraction
of held-out directions pointing toward the reference. This estimates a useful
initial gain; physical convergence and the search policy require separate
validation. Poor or singular calibration is rejected when no candidate gain
improves held-out distance or a response fit exceeds the condition limit.

Calibration options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--max-step` | required | Maximum Euclidean motor displacement used to score gains |
| `--eta-grid` | 0.25, 0.5, 0.75, 1, 1.5, 2 | Gains tested by held-out prediction |
| `--coarse-threshold` | 0.02 | Minimum diagnostic geometry score for response samples; maximum reference M1 |
| `--centroid-percentile` | 10 | Per-image background percentile for centroid weights |
| `--min-vector` | 0.5 | Minimum reference-corrected centroid error in pixels |
| `--min-coverage` | 0.5 | Required support fraction on each candidate boundary for direction estimation |
| `--max-condition` | 100 | Maximum response-matrix condition number, including held-out fits |

Review calibration diagnostics before use. The saved envelope describes sampled
motor coordinates, not a guaranteed basin of convergence. Recalibrate after a
change in optics, camera orientation, magnification, or intensity conditions.

## Alignment settings

Supply a JSON configuration; unknown keys are rejected. This example illustrates
motor units and limits that must be replaced with appropriate local values:

```json
{
  "units": "um",
  "bounds": [[-15, 15], [-15, 15]],
  "max_step": 5,
  "min_step": 0.5,
  "enter_m1": 0.02,
  "exit_m1": 0.03,
  "max_acquisitions": 40,
  "confirmations": 2,
  "bo_halfwidth": [5, 5],
  "gp_length_scale": [2.5, 2.5],
  "gp_noise": 0.05
}
```

| Field | Default | Meaning |
| --- | --- | --- |
| `units`, `bounds`, `max_step` | required | Adapter units, `[x_min,x_max]` and `[y_min,y_max]`, maximum Euclidean displacement for every move including recovery |
| `min_step` | 0.5 | Smallest trial displacement in motor units |
| `min_vector` | 0.5 | Minimum centroid error in pixels for using its direction |
| `min_coverage` | 0.5 | Minimum support fraction on each candidate boundary for using the centroid direction |
| `evaluator` | `{}` | Current-run evaluator options; omit semiaxes to fit them per image, or supply both to fix them |
| `centroid_percentile` | 10 | Per-image background percentile for centroid weights |
| `eta` | calibration value | Positive gain override for coarse correction |
| `enter_m1`, `exit_m1` | 0.02, 0.03 | Fine-stage entry and coarse-stage recovery thresholds; entry < exit |
| `geometry_improvement` | 0.0001 | Minimum diagnostic geometry decrease for accepting a coarse trial |
| `m2_improvement` | 0 | Minimum M2 decrease for accepting a feasible BO trial |
| `max_acquisitions` | 40 | Total budget including baseline, recovery, and confirmation images |
| `confirmations` | 2 | Consecutive feasible images for BO entry; number of final reimages |
| `readback_tolerance` | 0.001 | Maximum Euclidean readback discrepancy in motor units |
| `bo_halfwidth` | [5, 5] | Half-widths of a symmetric motor search box centered on the incumbent |
| `gp_length_scale` | [2.5, 2.5] | Fixed Matérn-5/2 spatial length scales in motor units |
| `gp_noise` | 0.05 | Observation standard deviation in standardized GP target units; squared for GP alpha |
| `bo_candidates` | 256 | Random candidate count per continuous BO iteration |
| `ei_xi` | 0 | EI improvement margin in M2 units |
| `min_ei` | 0 | Stop if maximum constrained EI is at most this value, in M2 units |
| `target_m2` | null | Optional absolute M2 target under the calibrated intensity conditions |
| `seed` | 0 | Random seed for reproducible candidate generation |

Set bounds and `max_step` for the current instrument. All settings and the calibration are recorded at
the beginning of the trajectory. Thresholds are prototype settings to qualify
with repeated measurements at the local instrument; they are not universal.

### Coarse stage

Use the full correction `-eta * solve(J, e)` when the feature passes the runtime
configuration's vector and boundary-coverage checks. Scale its Euclidean
length into `[min_step, max_step]`, preserving the component ratio. Shorten it
further at travel bounds along the same direction; skip a move if the available
travel is shorter than `min_step`. Recompute the correction after each accepted
image. A rejected trial is followed by a return to the incumbent and a fresh
image, then an opposite-direction probe. If both directions fail, halve the
step ceiling and retry. Stop when refinement would fall below `min_step`.

When the centroid is unavailable or fails the confidence checks, probe the four
diagonal directions `(+x,+y)`, `(-x,-y)`, `(+x,-y)`, `(-x,+y)` at the current
step ceiling, within travel bounds. Halve the ceiling if all four fail. An
accepted trial resets the ceiling to `max_step`; geometry improvement controls
acceptance. Exact-grid replay chooses the nearest available measurement to a
proposed endpoint within its step radius and forward half-plane, so replay can
quantize the direction. Continuous adapters preserve the proposed direction.

If fitting fails, use finite `geometry_score` diagnostics when available. A fully
unmeasurable image supplies no numerical geometry improvement; bounded probes may
recover a measurable donut. Failed readbacks or acquisition exceptions stop the
run and leave recovery to the instrument procedure.

### Fine stage

After consecutive valid images satisfy `M1 <= enter_m1`, minimize M2 using local
Bayesian optimization. Optional fixed ellipse semiaxes are image geometry
parameters configured for the current run, separate from the motor search bounds. At motor position `(5,5)`,
half-widths `(D,D)` define a symmetric box `[5-D,5+D]` on both axes, clipped to
permitted travel and the per-move displacement limit.

The acquisition score is `EI(M2) * P(geometry <= enter_m1) * P(valid fit)`.
Separate independent models approximate these factors:

- A GP regresses finite M2 values from feasible observations only.
- A GP regresses finite diagnostic geometry scores, including invalid candidates.
- A GP classifier learns valid/invalid fits. With a single observed class, a
  smoothed empirical probability substitutes until both classes are available.

No infinity enters GP training. A single feasible M2 observation can start BO;
initial uncertainty then comes from the fixed GP prior. More observed variation
makes its objective scale informative. Confirmatory images contribute observations,
so choose a noise level consistent with repeatability. There is no multi-fidelity
assumption: geometry is a constraint and M2 is the objective.

Accept a fine-stage trial only if it is feasible and improves M2 by the configured
amount, or restores feasibility to an incumbent in the hysteresis band. Rejected
trials remain in the surrogate history; restore and reimage the incumbent. If the
restored image becomes invalid or exceeds `exit_m1`, resume coarse alignment.
Previously visited positions are excluded from new BO trials within half the
minimum step; confirmation and recovery can repeat positions. The local box moves
with the incumbent, and exploration stops on budget, exhausted candidates, or the
EI threshold. A difficult image landscape can still trap this local search.

## Offline replay

```sh
uv run --project scripts --locked python scripts/align_osa.py \
  --calibration calibration.json --config calibration.alignment.json \
  --manifest recorded-images.json --start 10 -10 \
  --log replay-trajectory.jsonl --output replay-result.json
```

Replay chooses exactly recorded positions and evaluates their recorded images.
It has no interpolation. Select `min_step`, `max_step`, and BO widths appropriate
to the recording grid. Repeated images at a position cycle in manifest order;
reusing an image is not an independent confirmation or a fresh noise sample.
The result explicitly identifies replay. Calibration and replay on the same data
are development checks, not held-out evidence of convergence on an instrument.

## Beamline adapter placeholder

Live use requires a facility implementation. No instrument control is bundled.
Create an importable module whose factory receives the adapter configuration and
returns an object with this contract:

```python
class BeamlineAdapter:
    units = "um"

    def read_position(self):
        """Return actual [x, y] motor readbacks."""
        raise NotImplementedError

    def move_absolute(self, position):
        """Command OSA x/y; return only after completion and settling; raise on faults."""
        raise NotImplementedError

    def acquire_image(self):
        """Acquire a fresh image under calibrated settings and return its saved path."""
        raise NotImplementedError


def create_adapter(config):
    return BeamlineAdapter()
```

Launch with `--adapter beamline_adapter:create_adapter --adapter-config beamline.json`
in place of `--manifest` and `--start`. The controller reads its actual starting
position and verifies readbacks before and after each acquisition. The adapter
owns acquisition settings, freshness, motion settling, hardware interlocks, and
finite motion/acquisition timeouts. Keep longitudinal position and zone-plate
settings fixed. Binding this interface to zone-plate motion requires an explicit
local procedure and calibration for those axes.

## Logs and outcomes

Every acquisition logs measured position, commanded target, metrics, centroid
error, image path, a textual observation, a comparison with the preceding image,
and a next-plan statement. Descriptions include boundary coverage, unsupported
arc lengths, hole mean and peak, and left/right/top/bottom ring brightness when
a candidate fit is available. Ring sector means and hole peak use input intensity
units; the evaluator's hole mean uses its background correction. Each concrete
decision is logged before execution.
Logs are JSONL and must use a new path per run. Restart from actual readbacks with
a fresh baseline; automatic continuation of an interrupted run is not provided.

Results distinguish the best historical observation, the final observation, and
final measured position. A motor or acquisition fault gives `blocked` without
blind recovery. Exhausted search with confirmed feasible geometry gives
`best_found`. Unconfirmed or infeasible final geometry gives `unconfirmed`.
`metric_target_reached` requires all final reimages to satisfy both the M1 bound
and explicit `target_m2`. Omitting the M2 target reports the best result without
asserting optimality. This numerical result supports the skill's visual criteria;
a low integrated hole signal alone cannot exclude every localized bright defect.
