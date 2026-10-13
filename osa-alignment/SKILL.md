---
name: osa-alignment
description: Align an order sorting aperture (OSA) in a synchrotron X-ray microscope using calibrated analytical scripts to evaluate illumination and adjust transverse position. Use for OSA calibration, automated alignment, and diagnosing asymmetric donut illumination or stray signal in its central hole.
---

# OSA alignment

An order sorting aperture blocks unwanted diffraction orders from the focusing
optics while passing the intended focused order. Alignment of the OSA relative
to the optics should produce a symmetric, complete donut with a clean, dark hole.
The two quantitative objectives are well-defined concentric boundaries and low
integrated intensity inside the hole.

**Use the analytical scripts by default.** They handle calibration, image evaluation,
and the iterative motor search. Detailed input formats, configuration parameters,
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
alone does not establish an exact motor offset; camera orientation and the motor
response are determined by local calibration. Example PNG brightness is qualitative,
not calibrated photon counts.

## Script usage

Run commands from the skill directory using the local `uv` environment:

```sh
uv sync --project scripts --locked
```

### Evaluate an image

[scripts/evaluate_alignment.py](scripts/evaluate_alignment.py) fits concentric
ellipses and returns M1 (boundary error and completeness) and M2 (integrated hole
intensity). Lower values are better; invalid fits return infinite metrics.

```sh
uv run --project scripts --locked python scripts/evaluate_alignment.py IMAGE --output metrics.json
```

Supply both `--inner-axes A B` and `--outer-axes A B` to fix semiaxes in pixels;
otherwise they are fitted. See [metric details](references/analytical-alignment.md#quantitative-evaluation)
for normalization, fit diagnostics, and options.

#### Definitions and interpretation of M1 and M2

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

**Interpret M1 first, then M2.** The controller requires a valid fit and
`M1 <= enter_m1` (default 0.02) before optimizing M2 within that constraint.
A lower M2 cannot compensate for an incomplete or badly fitted donut. Qualify
thresholds and meaningful differences using a good reference and repeat images.
Invalid fits return both metrics as `"inf"` in JSON; this means the measurements
are unusable, not that the physical intensity is infinite. A finite diagnostic
`geometry_score` may still be available, but it does not establish fit validity.

### Calibrate

[scripts/calibrate_alignment.py](scripts/calibrate_alignment.py) uses recorded
images at known motor positions around a good reference to estimate the reference
centroid offset, motor response, and step gain eta. It saves these with the fixed
ellipse semiaxes in a calibration JSON file. Reuse an existing calibration when
its optics, camera orientation, units, and intensity conditions still apply.

```sh
uv run --project scripts --locked python scripts/calibrate_alignment.py calibration-images.json \
  --reference-position 0 0 --max-step 5 --output calibration.json
```

The reference coordinates and maximum step above are illustrative values in the
manifest's motor units. See [calibration inputs and options](references/analytical-alignment.md#calibration-from-recorded-images).

### Run alignment

[scripts/align_osa.py](scripts/align_osa.py) performs calibrated coarse axis passes,
then constrained local Bayesian optimization of M2 once M1 meets its threshold.
The calibration file is required. Eta defaults to its calibrated value and can
be overridden in the alignment configuration.

```sh
uv run --project scripts --locked python scripts/align_osa.py \
  --calibration calibration.json --config alignment.json \
  --adapter beamline_adapter:create_adapter --adapter-config beamline.json \
  --log trajectory.jsonl --output result.json
```

Live operation requires a facility implementation of the
[adapter interface](references/analytical-alignment.md#beamline-adapter-placeholder);
motion and acquisition remain placeholders until it is supplied. For recorded
images, use [offline replay](references/analytical-alignment.md#offline-replay).

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
