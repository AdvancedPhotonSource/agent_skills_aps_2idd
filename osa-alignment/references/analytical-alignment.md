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

The bundled runtime file overrides eta to `0.75`, sets `bo_stage_entrance_m1 = 0.012`,
and limits total acquisitions to 30. The tighter entry gate
keeps the controller following the centroid correction longer before local BO.
Choose fitting options for the current images when assessing convergence and
comparing M2.
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
| `--min-coverage` | 0.3 | Required support fraction on each candidate boundary for direction estimation |
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
  "bo_stage_entrance_m1": 0.02,
  "max_acquisitions": 40,
  "confirmations": 2,
  "bo_halfwidth": [5, 5],
  "bo_initial_samples": 5,
  "gp_length_scale": [2.5, 2.5],
  "gp_length_scale_bounds": [0.5, 10],
  "gp_noise": 0.05
}
```

| Field | Default | Meaning |
| --- | --- | --- |
| `units`, `bounds`, `max_step` | required | Adapter units, `[x_min,x_max]` and `[y_min,y_max]`, maximum Euclidean displacement for every move including recovery |
| `min_step` | 0.5 | Smallest trial displacement in motor units |
| `min_vector` | 0.5 | Minimum centroid error in pixels for using its direction |
| `min_coverage` | 0.3 | Minimum support fraction on each candidate boundary for using the centroid direction |
| `evaluator` | `{}` | Current-run evaluator options; omit semiaxes to fit them per image, or supply both to fix them |
| `centroid_percentile` | 10 | Per-image background percentile for centroid weights |
| `eta` | calibration value | Positive gain override for coarse correction |
| `bo_stage_entrance_m1` | 0.02 | Positive M1 threshold for BO entrance and observation eligibility |
| `geometry_improvement` | 0.0001 | Minimum diagnostic geometry decrease for accepting a coarse trial |
| `max_acquisitions` | 40 | Total budget including baseline, recovery, and confirmation images |
| `confirmations` | 2 | Consecutive feasible images for BO entry; number of final reimages |
| `readback_tolerance` | 0.001 | Maximum Euclidean readback discrepancy in motor units |
| `bo_halfwidth` | [5, 5] | Half-widths of a symmetric motor search box centered on the best eligible observation |
| `bo_initial_samples` | 5 | Eligible distinct random measurements before EI sampling; rejected attempts consume budget; 0 starts EI immediately |
| `gp_length_scale` | [2.5, 2.5] | Initial Matérn-5/2 length scales and fallback before three distinct eligible locations, in motor units |
| `gp_length_scale_bounds` | [0.5, 10] | Positive finite lower/upper bounds shared by both fitted spatial scales, in motor units |
| `gp_noise` | 0.05 | Observation standard deviation in standardized GP target units; squared for GP alpha |
| `gp_coarse_noise_multiplier` | 3 | Multiplier on `gp_noise` for `coarse` observations; finite and at least 1; default gives nine times the variance |
| `bo_candidates` | 256 | Random candidate count per continuous BO iteration |
| `ei_xi` | 0 | EI improvement margin in M2 units |
| `min_ei` | 0 | Stop if maximum EI is at most this value, in M2 units |
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

After consecutive valid images satisfy `M1 <= bo_stage_entrance_m1`, minimize
M2 using local Bayesian optimization. Initialize with `bo_initial_samples`
eligible random measurements at distinct motor locations (default five).
Rejected initialization attempts consume the total acquisition budget but do
not count toward those five. Eligible coarse and confirmation images also train
the GP; they do not replace the requested random measurements. An explicit M2
target can end the search early, subject to final confirmation.

One Matérn-5/2 GP models M2. All valid observations with finite M1 at or below
the threshold and finite M2 enter its training set, including worse M2 values
and repeated confirmations. The acquisition function is expected improvement
`EI(M2)` relative to the lowest eligible observed M2. Targets are standardized,
and each observation receives a diagonal noise variance in standardized units.
For observations labeled `coarse`, the variance is
`(gp_noise * gp_coarse_noise_multiplier)**2`; other stages use `gp_noise**2`.
The default multiplier of 3 reduces coarse measurements' influence with nine
times the variance. Set it to 1 for equal observation noise. After three
distinct eligible locations, fit anisotropic length scales by maximizing marginal
likelihood within `gp_length_scale_bounds`; before then, use `gp_length_scale`
as a configured fallback. Each iteration rebuilds the GP from eligible history
and logs its scales, whether they were fitted, and training acquisition IDs.

Invalid fits and nonfinite or above-threshold M1 create spatial exclusions.
For each rejected position `q`, exclude candidates `p` satisfying
`((p_x-q_x)/ell_x)^2 + ((p_y-q_y)/ell_y)^2 <= 1`.
This ellipse uses the latest GP scales, so its size changes on subsequent fits;
before fitting is supported, it uses the configured fallback scales. Retain all
rejection locations, including coarse observations, in the trajectory. Each
observation records eligibility and a rejection reason, and each model event
lists the acquisitions defining exclusions. Nonfinite M2 alone excludes an
observation from training without creating an M1 neighborhood exclusion.

Both initialization and EI sampling apply the same candidate filters. The
symmetric box with half-widths `bo_halfwidth` moves with the best eligible
observation. Clip it to global travel bounds and filter displacements from the
actual readback to `[min_step, max_step]`. Exclude points within half `min_step`
of any previously measured position, as well as the M1 neighborhoods above.
Continuous candidates are uniform random proposals within the box; exact-grid
replay uses available recorded locations. A depleted filtered candidate pool
ends the search with an explicit reason, including incomplete initialization
when applicable. A finite random pool can miss a small remaining sampleable area.

Keep the physical position at each trial, even when M2 is worse or M1 fails.
Track the best eligible observation separately. Reserve `confirmations` images
from the total budget for final verification. If the initialization budget runs
out first, report that reason and confirm the best eligible observation found.
At termination, return to that best position using bounded transit moves without
acquisitions, then acquire the final confirmations. Exact-grid replay retraces
measured waypoints for this return. A difficult landscape or large exclusions
can still limit this local search.

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

## Live dashboard

Append `--webui` to either an adapter or replay command:

```sh
uv run --project scripts --locked python scripts/align_osa.py \
  --calibration calibration.json --config calibration.alignment.json \
  --adapter beamline_adapter:create_adapter --adapter-config beamline.json \
  --log trajectory.jsonl --output result.json --webui --webui-port 8765
```

Open the printed URL, normally `http://127.0.0.1:8765/`, while alignment runs.
The dashboard shows settings and calibration, the latest acquired image,
acquisition positions, M1 with its BO entry threshold, M2, and observation history
with stages and eligibility. Trajectory axes use equal motor-unit scales.
Nonfinite metrics leave gaps in the charts and remain explicit in the table.
Detector previews use per-image linear min/max scaling; metric calculations
retain the original intensity units. The trajectory's latest marker denotes the
latest acquisition; the completed run reports its final motor readback separately.

The server reads the trajectory and saved images and refreshes once per second.
It exposes read-only endpoints, binds to `127.0.0.1`, and uses the existing Python
dependencies with no external browser libraries. Omit `--webui` to run without
the server. Use `--webui-port 0` for an automatically selected port.

For agent-launched runs, enable the flag, leave the command running, and give the
user the printed URL. For an agent on a remote host, forward the selected port:
`ssh -L 8765:127.0.0.1:8765 USER@HOST`, then open `http://127.0.0.1:8765/` locally.
The script does not launch a browser or wait for user interaction.

The server closes when the command exits, including blocked alignment outcomes.
A self-contained snapshot is saved using the log filename with an `.html` suffix
(for example, `trajectory.html`), preserving any existing file at that path.
Open this snapshot for final results after the server closes; an already-open live
page may show its last received state if the run finishes between refreshes.

## Beamline adapter placeholder

Live use requires a facility adapter, such as the APS 2-ID-D implementation below.
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

### APS 2-ID-D QueueServer adapter

[`APS2IDDAdapter`](../scripts/aps2idd_adapter.py) implements this interface using
the 2-ID-D QueueServer and its existing `acquire_xeye_image` function. Positions
are absolute motor readbacks in micrometers; the adapter converts to/from the
motors' millimeters. Image files must be available at the same absolute path on
the worker and the machine running alignment.

On the QueueServer host, put [`aps2idd_qserver.py`](../scripts/aps2idd_qserver.py)
on the worker's Python import path and add these imports to its startup namespace:

```python
from aps2idd_qserver import read_osa_position, move_osa_absolute
from s2idd_uprobe.qserver.xeye_osa import acquire_xeye_image
```

Load the helpers after beamline startup initializes the `osa` and `savedata`
devices. Allow `move_osa_absolute` in the user's QueueServer plan permissions and
`read_osa_position` / `acquire_xeye_image` in function permissions, then reload the
worker environment and its allowed lists using the beamline's normal procedure.
The move plan validates both motor targets, waits with a finite motor timeout,
and settles before completing. Existing motor limits and beamline interlocks
remain responsible for hardware protection.

Install the optional client dependency and edit
[`aps2idd.adapter.json`](../scripts/aps2idd.adapter.json):

```sh
uv sync --project scripts --locked --group beamline
```

Replace `url` with the actual HTTP gateway URL (`http://HOST:PORT`) or direct
QueueServer control address (`tcp://HOST:PORT`). The placeholder fails validation
before any connection. `user` and `user_group` apply to direct QueueServer requests;
HTTP uses the gateway's authenticated identity. For HTTP API-key authentication,
set `api_key_env` to the name of an environment variable containing the key.

`request_timeout` bounds individual requests, `motion_timeout` bounds motor
statuses, and `acquisition_timeout` is passed to the camera helper.
`operation_timeout` sets the polling deadline and must exceed motion plus settling
and the acquisition timeout; in-flight requests have separate request timeouts.
`settle_time` and `poll_interval` are in seconds, as are all timeouts.

```sh
uv run --project scripts --locked --group beamline python scripts/align_osa.py \
  --calibration calibration.json --config alignment.json \
  --adapter aps2idd_adapter:create_adapter --adapter-config scripts/aps2idd.adapter.json \
  --log trajectory.jsonl --output result.json
```

Use bounds in the actual absolute coordinate frame and a calibration valid for
the current optics, camera orientation, and acquisition settings. Configure the
camera exposure/gain consistently before alignment; the existing helper fetches
JPEG snapshots and stores them as TIFF without setting those camera parameters.

The adapter uses foreground functions and executes each move immediately with
`item_execute`, checking its specific item UID in plan history for successful
completion. This follows the [QueueServer execution API](https://blueskyproject.io/bluesky-queueserver/re_manager_api.html).
Reserve the instrument for alignment and start with an idle worker. A rejected
request, failed/stopped/aborted move, paused plan, or expired task raises an error.
Timeouts and connection loss leave remote execution potentially unresolved;
inspect QueueServer and the motor state before starting another run. Each image
gets a unique filename so repeated confirmations retain separate files.

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
