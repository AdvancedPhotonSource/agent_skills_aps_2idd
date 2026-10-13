---
name: osa-alignment
description: Align an order sorting aperture (OSA) in a synchrotron X-ray microscope by iteratively inspecting illumination images and adjusting transverse position. Use for OSA alignment, diagnosing asymmetric donut illumination or stray signal in its central hole, and choosing the next alignment move from an observation history.
---

# OSA alignment

An order sorting aperture blocks unwanted diffraction orders from the focusing
optics while passing the intended focused order. For this workflow, the target
illumination is a symmetric donut with a clean, dark central hole. Optimize the
OSA's alignment relative to the optics by inspecting an image after each move.

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
  Do not spoil a good ring or hole just to remove faint exterior light.

Assess symmetry about the donut's own center. Its location in the detector frame
is not itself the alignment criterion. Describe directions in displayed-image
coordinates (left/right/up/down), separately from motor x/y directions.

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

Small offsets can produce images that resemble the reference closely. Do not claim
an exact offset from appearance or use example offsets as live motor step sizes.
Camera orientation and the motor-to-image response must be established locally.

## Tool integration placeholders

The entries below are interface placeholders, not callable tools or implemented
hardware control. Bind them to the actual beamline tools before live alignment.
With no bindings, inspect supplied images and propose moves, explicitly marking
them as unexecuted; do not report simulated calls as measurements.

| Placeholder | Required information |
| --- | --- |
| `MOVE_OSA_XY` | Actual OSA motor identifiers, explicit absolute targets or relative displacements, units, permitted travel and maximum step, completion/settling status, and resulting position readbacks. |
| `ADJUST_ZONE_PLATE_POSITION` | Actual zone-plate axes, targets, units, limits, completion status, and readbacks. Reserved for a beamline procedure that explicitly uses zone-plate motion to adjust relative alignment. |
| `COLLECT_ILLUMINATION_IMAGE` | Acquisition settings, a fresh image and saved path or acquisition ID, acquisition time, and contemporaneous OSA/zone-plate readbacks. |

OSA and zone-plate motors are distinct. Default to OSA transverse x/y motion with
the optics fixed. If a local procedure instead moves the zone plate, establish
that mapping and record the controlled device before starting; do not transfer
OSA move signs or step sizes to zone-plate axes. Keep longitudinal position and
other optics settings fixed during the transverse search.

## Establish the starting state

Use available session and beamline information to establish the controlled device,
axis units, current readbacks, allowed x/y search range, maximum per-move displacement,
initial trial step, and smallest useful step. Obtain missing motion constraints
before moving. Use a finite move/time budget from the session or beamline procedure;
if none is given, state a default of at most 20 trial moves for this pass.

Acquire a baseline image at the current position, or inspect a supplied baseline
whose acquisition conditions and position are known. Keep exposure, gain, image
orientation, and display scaling consistent across comparisons. Check that the
whole donut is visible and its central signal can be assessed. Darkness from loss
of illumination, detector clipping, or a cropped ring cannot establish good alignment.
Treat the example PNGs' displayed brightness as qualitative evidence, not calibrated
photon counts.

## Keep a textual trajectory

**After every image, write its description, the trend compared with the previous
image, and the next plan in text before taking another action.** Include a
description even when the image looks unchanged. Images may be pruned from context;
“looks better” or an image link alone cannot preserve the trajectory.

Append each observation to the session's established durable text log, choosing a
session-appropriate location if one has not yet been established. Include the same
observation/trend/plan in the user-facing update. If file persistence is unavailable,
keep a compact cumulative text summary in the conversation. Use this record format:

```text
Iteration / acquisition ID / image path / time:
Controlled device; measured x, y and units; fixed optics/acquisition settings:
Last executed move: previous readback -> current readback, axis, signed delta:
Observation: ring outline and continuity; thickness/brightness by sector;
             hole shape and any internal spots/rays; exterior spill; image quality.
Trend vs previous: better/worse/unchanged/uncertain, naming the changed features.
Comparison with best: which criteria improve or regress; confidence and reason.
Best verified position and image ID; current step sizes; trials remaining:
Next plan: axis, signed displacement and target (or repeat/restore/stop),
           expected visible change, and why the trajectory supports it.
```

For the baseline, state that no preceding image exists. Distinguish commanded
targets from measured positions and proposed moves from completed moves. Preserve
failed trials as well as improvements. Keep an updated compact summary of the
current and best positions, active axis, tested directions, reversals, step sizes,
and unresolved uncertainty. On resuming after context loss, read this summary and verify current
readbacks before moving; reacquire if the physical state or image association is uncertain.

## Iterate: inspect, choose, move, measure

Tune one axis at a time: hold y fixed while finding the best image along x, then
hold x at that best position while tuning y. Either axis may be tuned first.
Alternate these axis-tuning passes, taking multiple steps along the active axis
as needed. Switch after finding its best position, rather than after each trial.
Single-step alternation is appropriate only when the image is already close to
the target and only small refinements remain.

1. **Judge the current image.** Use the target features above. Prioritize a clean
   hole and balanced, complete ring together; exterior cleanup is secondary. If
   these already hold, proceed to confirmation without unnecessary exploratory moves.
2. **Choose a bounded trial.** Use the recorded response to previous moves. With
   no established response, make a small permitted trial on one axis while holding
   the other fixed. State the predicted improvement as a hypothesis. Image-side
   brightness alone does not determine the sign of a motor correction.
3. **Execute and verify.** Check the target against the permitted range and step
   limit, move the selected motor, wait for completion/settling, and verify the
   readback. Then collect a fresh image. Never pair a pre-move image with a post-move
   position. Record the full observation, comparison, and plan before continuing.
4. **Follow improvement.** Continue on the active axis in a direction while it gives credible gains
   in ring balance and hole cleanliness. If one improves while the other worsens,
   record the tradeoff; do not promote the result to “good” until both are acceptable.
   When the change is ambiguous, repeat at the same position with comparable settings
   before attributing it to motor motion. If two repeat acquisitions still leave
   the comparison unresolved, stop and report the uncertainty rather than retrying
   indefinitely or choosing a direction from noise.
5. **Recover from deterioration.** If a trial clearly worsens the image, return to
   the best verified position within the established motion limits and reimage to
   confirm recovery. On the same axis, try the opposite direction if untested, or reduce the step
   (for example, halve it) when both directions at that scale fail to improve it.
   Do not continue into a visibly deteriorating regime solely to complete a scan.
6. **Finish the axis pass, then switch.** Continue trials and step refinement on
   the active axis until neighboring trials on both sides of its best position
   give no reproducible improvement at the current useful step size. Return to
   that best position and hold it fixed while tuning the other axis in the same
   way. Repeat passes because changes on one axis can shift the best position on
   the other. Reduce step sizes as alignment improves; near the target, single-step
   alternation may be used for fine adjustments. Use the tested trajectory,
   rather than a memorized example offset, to decide the next move.

If acquisition settings must change, record the change and acquire a comparable
baseline before judging a motor trend. Quantitative image measures may support
inspection when suitable raw data are available, but total intensity or a single
symmetry score must not override visible internal stray signal or a missing sector.

### Quantitative evaluation

Use [scripts/evaluate_alignment.py](scripts/evaluate_alignment.py) to measure
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

Prefer a meaningful primary improvement; use the secondary metric to distinguish
primary scores within a tolerance established from repeated unchanged-position
images. Require a meaningful secondary difference as well. Record scores and fit
validity in the textual trajectory, retaining the visual checks for angular
brightness balance and localized hole contamination.

## Completion and stopping

Declare good alignment when the ring is continuous and approximately concentric,
opposing sectors are reasonably balanced, and the hole is clean of conspicuous
stray features. Confirm with another fresh image at the selected position under
the same settings. Both images must satisfy the criteria; report faint tolerated
exterior features explicitly. Stop once this condition is met rather than seeking
pixel-perfect uniformity.

Stop the search without claiming success if the move/time budget is exhausted,
completed passes on both axes at the smallest useful step yield no reproducible
improvement while defects remain,
or further trials would exceed the allowed range. Restore and reimage the best
verified position when feasible within the established limits. On a motor fault,
unexpected readback, or loss of usable illumination, suspend exploratory moves and
resolve the uncertain state before any recovery motion.

Report the outcome as **aligned**, **best found but still imperfect**, or **blocked**.
Give the final measured position and units, controlled device, confirming image
IDs, visual evidence, remaining defects, and trajectory-log path. Distinguish the
best observed position from the actual final position if restoration was unavailable.
