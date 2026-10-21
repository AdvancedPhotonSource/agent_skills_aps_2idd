---
name: osa-alignment
description: Align an order sorting aperture (OSA) in a synchrotron X-ray microscope using ellipse metrics and Bayesian optimization to evaluate illumination and adjust transverse position. Use for automated OSA alignment, and diagnosing asymmetric donut illumination or stray signal in its central hole.
---

# OSA alignment

An order sorting aperture blocks unwanted diffraction orders from the focusing
optics while passing the intended focused order. Alignment of the OSA relative
to the optics should produce a symmetric, complete donut with a clean, dark hole.
The two quantitative objectives are well-defined concentric boundaries and low
integrated intensity inside the hole.

**Use the analytical scripts by default.** They handle image evaluation and the
two-stage Bayesian motor search. Detailed input formats, configuration parameters,
and integration instructions are in [the analytical guide](references/analytical-alignment.md).

## Recognize the target in words

Evaluate the following features together. A dark hole alone is insufficient,
and maximum total brightness is not the objective.

- **Ring shape and continuity:** A good image has a nearly circular, continuous
  annulus. Its inner and outer boundaries are approximately concentric, with
  comparable ring thickness on opposing sides. Look for a missing sector,
  flattened or clipped edge, indentation, crescent, or lopsided outer envelope.
- **Angular balance:** Opposing sectors should have broadly similar brightness
  and extent. A bright half-ring facing a dim, diffuse, or missing half indicates
  misalignment. Fine grain, radial texture, and modest brightness variation are
  present even in the good reference; a perfectly smooth white ring is not required.
- **Central hole:** The center should be a clearly enclosed, approximately round,
  dark region, free of distinct bright spots, spokes, streaks, or tongues of light
  intruding from the ring. “Clean” means no conspicuous stray structure above the
  local dark signal; it does not require every central pixel to be zero.
- **Exterior:** Faint haze, isolated spots, or weak rays beyond the outer boundary
  can be tolerated when the ring is balanced and the hole is clean. Strong
  one-sided spill associated with ring distortion supports a misalignment diagnosis.

Symmetry is assessed about the donut's own center; its location in the detector
frame is not itself the alignment criterion. A cropped ring, detector clipping,
or darkness from loss of illumination cannot establish good alignment.

### Observed examples

These descriptions are sufficient to guide inspection without loading the example
images. PNG links are optional visual aids. Offsets below are labels in micrometers,
relative to the aligned reference, not absolute motor readbacks.

| Offset (x, y), µm | Appearance and interpretation | Optional PNG |
| --- | --- | --- |
| (0, 0) | Good reference: a complete, bright, nearly circular annulus around a round dark hole. The hole is roughly one third of the outer diameter. Ring texture includes fine radial lines and mottling; the outer edge is soft, with a slightly diffuse right side. There is no conspicuous bright spot or spoke inside the hole. The donut sits right of the frame center, with faint exterior haze and a faint isolated mark farther right; these do not invalidate it. | [Good alignment](assets/aligned.png) |
| (−15, 0) | The right side forms a bright C-shaped arc; the left side is weaker and spills outward into broad, roughly horizontal streaks. The central hole remains recognizable but has uneven light structure. The opposed sectors are substantially unbalanced. | [Left-side distortion](assets/x_m15_y_0.png) |
| (+15, 0) | A bright left crescent faces a much dimmer, textured right arc. Radial streaks converge toward an irregular dark center, and the upper ring is poorly defined. This is not a balanced annulus even though a dark central region remains. | [Right-side distortion](assets/x_15_y_0.png) |
| (0, +15) | The lower and side arcs remain bright while the upper sector fades into an upward diffuse plume. Bright spoke-like features intrude into the central hole. Both the angular balance and hole cleanliness are worse than the reference. | [Upper-sector distortion](assets/x_0_y_15.png) |
| (+15, −15) | A conspicuous bright compact spot lies in the central region, with several rays extending toward the upper and left sectors. The surrounding ring is dim and uneven. The internal bright signal makes this unacceptable even if parts of the outer contour look round. | [Contaminated hole](assets/x_15_y_m15.png) |

Small offsets can produce images that resemble the reference closely. Appearance
alone does not establish an exact motor offset; the controller searches the
configured motor bounds directly. Example PNG brightness is qualitative,
not calibrated photon counts.

## Script usage

Run commands from the skill directory using the local `uv` environment:

```sh
uv sync --project scripts --locked
```

### Evaluate an image

[scripts/evaluate_alignment.py](scripts/evaluate_alignment.py) fits concentric
ellipses and returns M1 (boundary error and completeness) and M2 (integrated hole
intensity). Lower values are better. M1 stays finite on failed fits using the best available
optimizer geometry; M2 remains infinite when the fit is invalid.

```sh
uv run --project scripts --locked python scripts/evaluate_alignment.py IMAGE --output metrics.json
```

Supply both `--inner-axes A B` and `--outer-axes A B` to fix semiaxes in pixels;
otherwise they are fitted. Add `--min-eccentricity E` to require both ellipses to
have focal-distance-to-major-semiaxis ratio \(c/a\ge E\), where \(0\le E<1\).
Omit it for an unconstrained fit. See [metric details](references/analytical-alignment.md#quantitative-evaluation)
for normalization, fit diagnostics, and options. 
**Supply the best guess of the semiaxes whenever possible as they are important for accurate fitting**.

#### Definitions and interpretation of the M1 and M2 metrics

**M1 (`primary_metric`) measures boundary geometry and completeness.** The inner
and outer ellipses share a center and orientation. For a valid fit,

\[
M_1 = \frac{d_{\mathrm{inner}} + d_{\mathrm{outer}}}{2\sqrt{a_o b_o}}
      + \lambda\left(1-\frac{C_{\mathrm{inner}}+C_{\mathrm{outer}}}{2}\right).
\]

Here, each \(d\) is the RMS distance in pixels from a fitted boundary to its
corresponding detected edges; \(a_o,b_o\) are the outer semiaxes. Each \(C\) is the
fraction of boundary samples supported by an edge within the configured distance
tolerance. The evaluator samples 360 polar angles per boundary and defaults to
\(\lambda=0.5\). M1 is dimensionless: lower values indicate closer boundary matches
and fewer unsupported arcs. With complete coverage, M1 = 0.02 means the mean
boundary RMS distance is 2% of the outer equal-area radius. With incomplete
coverage, the score also includes the missing-arc penalty. M1 does not directly
measure angular brightness balance or contamination inside the hole.

**M2 (`secondary_metric`) measures integrated signal inside the fitted hole.**

\[
M_2 = \frac{1}{S}\sum_{p\,\mathrm{inside\ inner\ ellipse}} (I_p-B),
\]

where \(B\) is the configured scalar background and \(S\) is the intensity
normalizer (defaults: 0 and 1). This is a pixel sum, not mean intensity or a
hole-to-ring ratio. Lower M2 means less residual central signal under comparable
conditions. Its scale depends on image intensity units, exposure, normalization,
and hole area; it has no universal acceptable cutoff. Fixed semiaxes and consistent
acquisition settings make comparisons meaningful. A low sum can still conceal
a localized bright spot.

### Run alignment

[scripts/align_osa.py](scripts/align_osa.py) aligns the OSA in two stages.
Stage 1 uses Bayesian optimization to minimize M1 across the full `bounds` in
the alignment config, including finite scores from failed fits. It completes all
initial measurements, then enters Stage 2 when the global minimum of the M1 GP
posterior mean meets `stage_2_entrance_m1` or `stage_1_max_acquisitions` is used up.
The first Stage 2 measurement and initial search-box center are at that minimum.
The baseline counts toward the Stage 1 budget. Stage 2 uses random
initialization and local Bayesian optimization to minimize M2, retaining its
M1 eligibility constraint and neighborhood exclusions. The final position is the
best eligible observation, verified by repeated images. If none is eligible,
return to the lowest observed M1 and report an unconfirmed result.

No calibration file, centroid vector, reference center, Jacobian, or eta is used.
The bundled [alignment settings](scripts/alignment.json) allocate 15 of the 40
acquisitions to Stage 1, use `M1 <= 0.02` for Stage 2 entry and eligibility, and
reserve two final confirmation images from the total budget. Set `bounds` for
the intended global search region and motor coordinate frame. Random
initialization in both stages ignores `min_step`/`max_step`, moving directly to
each sampled target within global bounds. Stage 2 retains its local box and M1
exclusions. Its center stays fixed during random initialization, then follows the
minimum of the M2 GP posterior mean. Subsequent moves respect `max_step`; distant Stage 1 BO targets and
final returns use bounded transit moves without acquisitions.
Set `bo_initial_samples` for both stages (Stage 1 includes the baseline; Stage 2
counts eligible initialization measurements, including its entry measurement).
The Stage 1 budget must accommodate `bo_initial_samples`.
Set `stage_1_grid_spacing` and `stage_2_grid_spacing` for the acquisition candidate
grids (defaults: 1 and 0.1 in motor units, respectively). Stage 1 covers the global
bounds; Stage 2 covers its clipped local box. EI is evaluated at every surviving
grid point, and initialization samples randomly from that grid. Recorded replay
uses its available positions.
Configure image fitting through `evaluator`. Supplying both
`inner_axes` and `outer_axes` is strongly recommended for accurate fitting.

```sh
uv run --project scripts --locked python scripts/align_osa.py \
  --config scripts/alignment.json \
  --adapter beamline_adapter:create_adapter --adapter-config beamline.json \
  --log trajectory.jsonl --output result.json
```

For APS 2-ID-D, use the [QueueServer adapter](references/analytical-alignment.md#aps-2-id-d-queueserver-adapter)
with the server URL in `scripts/aps2idd.adapter.json` and the documented worker
helpers loaded. Other facilities can implement the
[adapter interface](references/analytical-alignment.md#beamline-adapter-placeholder).
For recorded images, use [offline replay](references/analytical-alignment.md#offline-replay).

Add `--webui` to show a read-only live dashboard with the run settings, latest
image, search trajectory, and M1/M2 histories. The script prints its local URL
(default `http://127.0.0.1:8765/`); `--webui-port PORT` selects another port,
and `0` selects an available port. Agents can enable this flag and share the URL
while keeping the alignment process running. The dashboard refreshes every second
and requires no browser on the agent's host. At exit, a self-contained HTML
snapshot is saved beside the trajectory log. See
[dashboard access and lifecycle](references/analytical-alignment.md#live-dashboard).

[Alignment settings](references/analytical-alignment.md#alignment-settings) configure
motor bounds, step limits, thresholds, GP parameters, and acquisition budgets.
The script handles readback verification, trial recovery, confirmation, and
trajectory logging. Its result distinguishes the best observation from the actual
final position. See [output statuses](references/analytical-alignment.md#logs-and-outcomes).
`metric_target_reached` means the configured numerical criteria were met; the
image criteria above also account for localized bright defects in the hole.

## Manual fallback

Use [the manual alignment guide](references/manual-alignment.md) only when the
scripts cannot proceed or complete alignment, or the user explicitly requests
manual search. Read its operating and logging instructions when selecting that
fallback.
