# Two-stage OSA alignment

The workflow uses `evaluate_alignment.py` for ellipse metrics and
`align_osa.py` for quadratic M1 optimization followed by M2 Bayesian optimization.
Run commands from the skill
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
Use `--min-eccentricity E` (or `evaluator.min_eccentricity` in the controller
configuration) to require both ellipses to satisfy \(c/a\ge E\), with
\(c=\sqrt{a^2-b^2}\), \(a\) the larger semiaxis, and \(0\le E<1\). Omit it to
fit without this bound. Fixed axes must already satisfy the bound.
The fit uses ellipse normals to classify edges and integrates the elliptical
hole. Nearly circular ellipses have poorly determined orientation. Extra shape
freedom can absorb actual misalignment, so inspect fitted aspect ratios and the
image as well as the residual.

Set `--pre-blur-sigma` (or `evaluator.pre_blur_sigma` in the alignment config)
to smooth the grayscale image before percentile normalization and geometry fitting.
It is a nonnegative Gaussian standard deviation in pixels; 0 disables it.
This precedes Canny's own `sigma` smoothing (default 2 pixels). All geometry
initialization and edge fitting use the pre-blurred image; hole and annulus
intensities are measured on the original image using the resulting fitted masks.

The script detects Canny edges, distinguishes inward and outward intensity
transitions, and fits the concentric ellipses. Lower scores are better:

- `primary_metric` (M1): mean inner/outer circumference-to-edge RMS distance divided
  by outer equal-area radius (√(AB) for an ellipse), plus a penalty for unsupported
  angular coverage. Both boundaries receive equal weight, with 360 true polar-angle
  samples per boundary and a default coverage weight λ = 0.5. This measures
  geometry and completeness.
- `secondary_metric` (M2): background-subtracted intensity summed inside the inner
  boundary, optionally divided by `--intensity-normalizer`. Use `--background`
  for a known scalar detector background. The input intensity scale is preserved;
  RGB images use luminance weights. Display-scaled images do not give photon counts.

M1 is always finite for an accepted image input, including failed optimizer
convergence or fit validation. It uses residuals and unsupported angular coverage
from the best available optimizer geometry. If no seed or boundary edge is found,
a centered ellipse seed and finite image-diagonal distances for absent edges keep
the optimization and score defined. The optimizer's returned parameters are used
even when it reaches its evaluation limit. `valid` and `reasons` still report
fit quality, and invalid fits retain infinite M2 (encoded as `"inf"` in JSON).
A finite M1 does not imply a valid fit or successful alignment. Review the image,
RMS errors, coverage, hole mean, and hole area alongside the scores.

No motor response calibration, centroid direction, reference center, Jacobian,
or eta is needed. Configure fitting for the current images in `evaluator`.
The bundled [alignment settings](../scripts/alignment.json) configure the motor
search and acquisition budget. Set both `evaluator.inner_axes` and
`evaluator.outer_axes` to hold semiaxes fixed for the current instrument.

## Alignment settings

Supply a JSON configuration; unknown keys are rejected. This example illustrates
motor units and limits that must be replaced with appropriate local values:

```json
{
  "units": "um",
  "bounds": [[-15, 15], [-15, 15]],
  "max_step": 5,
  "min_step": 0.5,
  "stage_2_entrance_m1": 0.02,
  "stage_1_max_acquisitions": 20,
  "max_acquisitions": 40,
  "confirmations": 2,
  "stage_2_bo_halfwidth": [5, 5],
  "bo_initial_samples": 5,
  "bo_initialization": "grid",
  "stage_1_optimizer": "quadratic",
  "stage_1_grid_spacing": 1.0,
  "stage_2_grid_spacing": 0.1,
  "gp_length_scale": [2.5, 2.5],
  "gp_length_scale_bounds": [0.5, 10],
  "gp_noise": 0.05
}
```

| Field | Default | Meaning |
| --- | --- | --- |
| `units`, `bounds`, `max_step` | required | Adapter units, `[x_min,x_max]` and `[y_min,y_max]`, maximum Euclidean displacement for moves after random initialization, including final transit |
| `min_step` | 0.5 | Smallest trial displacement/spacing after random initialization, in motor units; 0 disables minimum spacing while still excluding repeated positions |
| `evaluator` | `{}` | Current-run evaluator options; omit semiaxes to fit them per image, or supply both to fix them |
| `stage_2_entrance_m1` | 0.02 | Threshold on the minimum M1 posterior mean for Stage 2 entry |
| `stage_2_eligibility_m1` | null | Observed M1 threshold for M2 training, final selection, and neighborhood exclusions; null uses `stage_2_entrance_m1` |
| `stage_1_max_acquisitions` | 20 | Stage 1 acquisition budget, including baseline; grid mode requires at least 17 images and reserves a full Stage 2 grid plus confirmations |
| `max_acquisitions` | 40 | Total budget including baseline and final confirmation images; transit moves take no images |
| `confirmations` | 2 | Number of final reimages; at least two |
| `readback_tolerance` | 0.001 | Maximum Euclidean readback discrepancy and per-axis allowance outside global bounds, in motor units; commanded targets remain within exact bounds |
| `stage_2_bo_halfwidth` | [5, 5] | Half-widths of the motor search box, anchored during initialization and then centered on the M2 GP posterior-mean minimum |
| `bo_initial_samples` | 5 | Stage 1 initial measurements including baseline (must fit its budget), and Stage 2 eligible initialization measurements including entry before EI; 0 starts EI when a training observation exists |
| `bo_initialization` | `grid` | `random` uses `bo_initial_samples` and requires `stage_1_optimizer: "bo"`; `grid` measures a regular 4×4 initialization grid in each stage, plus baseline and Stage 2 entry |
| `stage_1_optimizer` | `quadratic` | `bo` uses the M1 GP and EI; `quadratic` requires grid initialization, refits a full M1 quadratic after each image, and measures its bounded minimum until the Stage 1 budget |
| `gp_length_scale` | [2.5, 2.5] | Initial Matérn-5/2 length scales and fallback before three distinct eligible locations, in motor units |
| `gp_length_scale_bounds` | [0.5, 10] | Positive finite lower/upper bounds shared by both fitted spatial scales, in motor units |
| `gp_noise` | 0.05 | Observation standard deviation in standardized GP target units; squared for GP alpha |
| `stage_2_gp_stage_1_noise_multiplier` | 3 | Multiplier on M2 GP noise for baseline and Stage 1 observations; finite and at least 1; default gives nine times the variance |
| `stage_1_grid_spacing` | 1.0 | Candidate grid spacing along both axes over the global bounds, in motor units (µm for `units: "um"`) |
| `stage_2_grid_spacing` | 0.1 | Candidate grid spacing along both axes within the current clipped local box, in motor units |
| `bo_candidates` | null | Legacy field accepted when loading older configs; candidate counts are determined by the grid spacings |
| `ei_xi` | 0 | EI improvement margin in the current objective's units (M1 or M2) |
| `stage_2_min_ei` | 0 | Stop if maximum EI is at most this value, in M2 units |
| `stage_2_target_m2` | null | Optional absolute M2 target under consistent acquisition conditions |
| `seed` | 0 | Random seed for reproducible initialization selections |

Set bounds and `max_step` for the current instrument. All settings are recorded at
the beginning of the trajectory. Thresholds are prototype settings to qualify
with repeated measurements at the local instrument; they are not universal.

### Stage 1: global M1 optimization

With `stage_1_optimizer: "bo"`, minimize M1 using a Matérn-5/2 Gaussian process and expected improvement over the
full configured `bounds`. `stage_2_bo_halfwidth` and Stage 2 M1 neighborhood exclusions
do not restrict this domain. All finite M1 observations train this GP, including
failed fits and scores above the entry threshold. Use `gp_noise` to tolerate
uncertain scores. With `bo_initialization: "random"`, the first `bo_initial_samples` measurements, counting the
baseline, initialize the search randomly; 0 starts EI after the baseline.
Random initialization ignores `min_step`/`max_step` and moves directly to each
sampled target, avoiding duplicate positions and staying within global bounds.

With `bo_initialization: "grid"`, measure all 16 points of a regular 4×4 grid
spanning the global bounds after the baseline. Coordinates include both endpoints
of each axis and are visited in x-major order. Stage 2 measures its model-minimum
entry point, then a 4×4 grid spanning the clipped entry box. Both grids are completed
regardless of observed M1; M2 eligibility remains active for training and selection.
Grid initialization bypasses M1 neighborhood exclusions and step limits. Subsequent
BO applies the usual exclusions and adapts the box. `bo_initial_samples` is ignored
in this mode. Require `stage_1_max_acquisitions >= 17` and
`max_acquisitions >= stage_1_max_acquisitions + 17 + confirmations`. Recorded
backends must contain every grid location. The grids are measured even when a
location coincides with an earlier baseline, entry, or Stage 1 observation.

In BO mode, complete all initial measurements before checking entry. Then minimize the M1 GP
posterior mean over the global bounds after every acquisition. Enter Stage 2 when
that minimum is at or below `stage_2_entrance_m1` or `stage_1_max_acquisitions` is
exhausted. Measure the posterior minimum as the first Stage 2 point in either case.
The numerical search uses a 17-by-17 grid and measured positions to seed five local
optimizations; discrete replay minimizes over recorded positions. Entry checks log
the predicted M1 and position. Stage 1 does not terminate on `stage_2_min_ei`; when no
distinct candidate remains it reimages the best M1 until a transition condition
is met. Its acquisition budget is part of `max_acquisitions`, which also reserves
`confirmations` final images and at least one Stage 2 acquisition.

With the default `stage_1_optimizer: "quadratic"`, use grid initialization and fit
`M1 = a + b*u + c*v + d*u² + e*u*v + f*v²` by ordinary least squares to all finite
Stage 1 M1 measurements, including the baseline and repeated locations. Coordinates
`u` and `v` map the global bounds to `[-1, 1]` for conditioning. Measure the global
minimum over the bounded rectangle and refit with the new observation. The bounded
minimum is selected from corners, edge minima, and the interior stationary point;
concave or indefinite fits can therefore select an edge or corner. Continuous targets
can revisit earlier positions. After the last Stage 1 acquisition, refit once more
and acquire the Stage 2 entry image at that final minimum. This mode uses the full
Stage 1 budget, with no early M1-threshold transition. A 20-image Stage 1 budget
contains the baseline, 16 grid images, and three quadratic-minimum images.
`quadratic_model` log events store normalized coefficients, training acquisition IDs,
the minimum position, and predicted M1. Stage 2 retains its existing initialization,
eligibility, GP noise, box adaptation, and BO rules.

Candidate grids cover the global bounds in Stage 1 and the clipped local box in
Stage 2. Both spacings must be finite and positive. Grids start at the lower bounds
and include the upper bounds; the last interval can be shorter than the configured
spacing. Evaluate EI at every surviving grid point. Random initialization chooses
uniformly among surviving grid points. Recorded replay uses its available positions.
The posterior-mean minimization used for entry and box centering retains its separate
grid and local optimization. After random initialization,
exclude measured positions within `min_step`. Subsequent BO targets farther
than `max_step` use successive bounded transit moves without images. After
initialization, exact-grid replay uses recorded locations connected by bounded
moves; intermediate recorded waypoints need no acquisition. Motor readbacks are
verified at every move and before/after acquisitions. Faults stop the run.

### Stage 2: local M2 optimization

Minimize M2 in a local box. During random initialization, keep the box centered
on the global M1 posterior-mean minimum used for the entry measurement. The entry
measurement counts toward initialization if eligible, followed by random measurements
at distinct motor locations. After collecting `bo_initial_samples` eligible initialization measurements,
recenter the box on a numerical minimum of the M2 GP posterior mean each iteration. Rejected
attempts consume the total budget. Stage 1 observations can train the
M2 GP but do not count toward Stage 2 random initialization. An explicit M2 target
can end Stage 2 early, subject to final confirmation.

On budget entry without an eligible M2 observation, anchor the initialization
box at the global M1 posterior-mean minimum and sample randomly until initialization is complete
and an eligible M2 is available. Apply the same Stage 2 exclusions throughout.
If the exclusions or travel limits leave
no candidates, report that reason and return to the lowest M1 observation for
final confirmation; infeasible geometry remains unconfirmed.

Find the posterior-mean minimum within the current box, clipped to global bounds
and subject to M1 neighborhood exclusions. A 17-by-17 grid, the current center,
and measured positions seed five constrained local optimizations. The center can
lie between measurements; updating it takes no motor move or image. A flat
posterior retains the center when it remains allowed. Each `search_box` event
records the center and whether it is anchored for initialization or follows the
posterior mean.

One Matérn-5/2 GP models M2. All valid observations with finite M1 at or below
the threshold and finite M2 enter its training set, including worse M2 values
and repeated confirmations. The acquisition function is expected improvement
`EI(M2)` relative to the lowest eligible observed M2. Targets are standardized,
and each observation receives a diagonal noise variance in standardized units.
For `baseline`, `stage1`, and `stage1_init` observations, the M2 GP variance is
`(gp_noise * stage_2_gp_stage_1_noise_multiplier)**2`; other stages use `gp_noise**2`.
The default multiplier of 3 reduces Stage 1 measurements' influence with nine
times the variance. Set it to 1 for equal observation noise. After three
distinct eligible locations, fit anisotropic length scales by maximizing marginal
likelihood within `gp_length_scale_bounds`; before then, use `gp_length_scale`
as a configured fallback. Each iteration rebuilds the GP from eligible history
and logs its scales, whether they were fitted, and training acquisition IDs.

Invalid fits and nonfinite or above-threshold M1 create spatial exclusions.
For each rejected position `q`, exclude candidates `p` satisfying
`((p_x-q_x)/ell_x)^2 + ((p_y-q_y)/ell_y)^2 <= 1`.
This ellipse uses the latest GP scales, so its size changes on subsequent fits;
before the M2 GP can be fitted, it uses the configured fallback scales. Retain all
rejection locations, including Stage 1 observations, in the trajectory. Each
observation records eligibility and a rejection reason, and each model event
lists the acquisitions defining exclusions. Nonfinite M2 alone excludes an
observation from training without creating an M1 neighborhood exclusion.

Initialization and EI sampling both stay within global `bounds`, the local box
with half-widths `stage_2_bo_halfwidth`, and the M1 neighborhood exclusions above. Random
initialization ignores `min_step`/`max_step`, moves directly to each sampled
target, and avoids exact duplicate measurements. Global bounds, readback
verification, the local box, and M1 exclusions still apply. For subsequent EI sampling, filter displacements from the actual
readback to `[min_step, max_step]` and exclude points within half `min_step` of
any previously measured position. The local box follows the posterior-mean minimum after initialization.
Continuous candidates are uniform random proposals within the box; exact-grid
replay uses available recorded locations. A depleted filtered candidate pool
ends the search with an explicit reason, including incomplete initialization
when applicable. A finite random pool can miss a small remaining sampleable area.

Keep the physical position at each trial, even when M2 is worse or M1 fails.
Track the best eligible observation separately. Reserve `confirmations` images
from the total budget for final verification. If the initialization budget runs
out first, report that reason and confirm the best eligible observation found.
At termination, return to that best position using bounded transit moves without
acquisitions, then acquire the final confirmations. Exact-grid replay follows
recorded waypoints connected by bounded moves for this return. A difficult landscape or large exclusions
can still limit this local search.

## Offline replay

```sh
uv run --project scripts --locked python scripts/align_osa.py \
  --config scripts/alignment.json \
  --manifest recorded-images.json --start 10 -10 \
  --log replay-trajectory.jsonl --output replay-result.json
```

A replay manifest uses explicit motor units and absolute readback coordinates;
image paths resolve relative to the manifest:

```json
{
  "units": "um",
  "samples": [
    {"position": [0, 0], "image": "images/reference.tiff"},
    {"position": [3, 0], "image": "images/displaced.tiff"}
  ]
}
```

Replay chooses exactly recorded positions and evaluates their recorded images.
It has no interpolation. Select `min_step`, `max_step`, and BO widths appropriate
to the recording grid. Repeated images at a position cycle in manifest order;
reusing an image is not an independent confirmation or a fresh noise sample.
The result explicitly identifies replay. Replay is a development check; it does not establish convergence on an instrument.

## Live dashboard

Append `--webui` to either an adapter or replay command:

```sh
uv run --project scripts --locked python scripts/align_osa.py \
  --config scripts/alignment.json \
  --adapter beamline_adapter:create_adapter --adapter-config beamline.json \
  --log trajectory.jsonl --output result.json --webui --webui-port 8765
```

Open the printed URL, normally `http://127.0.0.1:8765/`, while alignment runs.
The dashboard shows settings, the latest acquired image,
acquisition positions, M1 with its BO entry threshold, M2, and observation history
with stages, eligibility, fit validity, fitted center, inner/outer semiaxes, angle,
and whether semiaxes were fixed or fitted. Centers and semiaxes use detector
pixels; motor positions use the configured motor units. Failed fits show their
retained candidate parameters, and unavailable parameters appear as dashes. Trajectory axes use equal motor-unit scales.
Nonfinite metrics leave gaps in the charts and remain explicit in the table.
Detector previews use per-image linear min/max scaling; metric calculations
retain the original intensity units. The trajectory's latest marker denotes the
latest acquisition; the completed run reports its final motor readback separately.

The server reads the trajectory and saved images and refreshes once per second.
It exposes read-only endpoints, binds to `127.0.0.1`, and uses the existing Python
dependencies with no external browser libraries. Omit `--webui` to run without
the server. Use `--webui-port 0` for an automatically selected port.

For agent-launched runs, enable the flag, leave the command running, and give the
user the printed URL. If possible, open the URL in a browser.
For an agent on a remote host, forward the selected port:
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
        """Acquire a fresh image under consistent settings and return its saved path."""
        raise NotImplementedError


def create_adapter(config):
    return BeamlineAdapter()
```

Launch with `--adapter beamline_adapter:create_adapter --adapter-config beamline.json`
in place of `--manifest` and `--start`. The controller reads its actual starting
position and verifies readbacks before and after each acquisition. The adapter
owns acquisition settings, freshness, motion settling, hardware interlocks, and
finite motion/acquisition timeouts. 

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
`post_call_sleep` is an optional dwell in seconds applied after every successful
adapter call (read, move, and acquisition); it defaults to `0.0`, which disables it.

```sh
uv run --project scripts --locked --group beamline python scripts/align_osa.py \
  --config scripts/alignment.json \
  --adapter aps2idd_adapter:create_adapter --adapter-config scripts/aps2idd.adapter.json \
  --log trajectory.jsonl --output result.json
```

Use bounds in the actual absolute motor coordinate frame for the intended global
search region. Configure the
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

Observation stages are `baseline`, `stage1_init`, `stage1`, `stage_2_init`,
`stage_2`, and `final`. The `_init` stages identify initialization measurements;
`stage1` identifies M1 refinement by the configured optimizer; `stage_2` identifies
Bayesian optimization of M2.
Stage 2 transition and model records also use `stage_2`.

Every acquisition logs measured position, commanded target, metrics, image path, a textual observation, a comparison with the preceding image,
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
and explicit `stage_2_target_m2`. Omitting the M2 target reports the best result without
asserting optimality. This numerical result supports the skill's visual criteria;
a low integrated hole signal alone cannot exclude every localized bright defect.
