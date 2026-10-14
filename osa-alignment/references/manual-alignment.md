# Manual OSA alignment fallback

Use this workflow only when the analytical scripts cannot complete the task or
the user explicitly requests manual search. Record why this mode was selected.
Carry forward the last measured state, best observed position, image history,
and remaining motion/acquisition budget from any script run. An unresolved motor
or acquisition fault must be resolved before continuing in either mode.

Use the shared [image criteria and examples](../SKILL.md#recognize-the-target-in-words)
and the tool integration requirements below.
The evaluator can still supply metrics when the alignment controller is unavailable.

## APS 2-ID-D adapter

Use [`APS2IDDAdapter`](../scripts/aps2idd_adapter.py) for manual OSA alignment at
2-ID-D. Follow the [QueueServer setup instructions](analytical-alignment.md#aps-2-id-d-queueserver-adapter)
to load the worker helpers, configure permissions, and replace the server URL
placeholder in [`aps2idd.adapter.json`](../scripts/aps2idd.adapter.json).
Start with an idle worker and reserve the instrument for alignment. The acquired
TIFF paths must be accessible from the machine running the manual session.

From the skill directory, start an interactive Python session:

```sh
uv run --project scripts --locked --group beamline python
```

Initialize the adapter in that session:

```python
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path("scripts").resolve()))
from aps2idd_adapter import APS2IDDAdapter

adapter = APS2IDDAdapter(json.loads(Path("scripts/aps2idd.adapter.json").read_text()))
position_um = adapter.read_position()
```

| Operation | Manual-session use |
| --- | --- |
| `adapter.read_position()` | Return actual `[x, y]` OSA coordinates in micrometers. |
| `adapter.move_absolute([x_um, y_um])` | Command absolute coordinates and wait for successful completion and settling. Read the position again to verify the target. |
| `adapter.acquire_image()` | Acquire a fresh X-eye image and return its unique saved TIFF path. Read position before and after acquisition to associate the image with a verified position. |
| `adapter.close()` | Release the client when the manual session ends. |

The adapter converts between micrometers and the motors' millimeters. For a
relative trial, add the signed displacement to a fresh readback of the active
axis and retain the other axis's readback as its target. The manual workflow owns
search bounds, maximum step, readback tolerance, budgets, and trajectory logging;
calling the adapter directly does not apply the alignment controller's checks.
Configure exposure and gain consistently through the beamline's camera controls.

If the adapter raises an error or times out, resolve the QueueServer and motor
state before another move or acquisition; remote execution may still be active.
With the server unavailable, use supplied images or recorded-image replay and
record proposed motions as unexecuted. Other facilities must supply an adapter
implementing the [same interface](analytical-alignment.md#beamline-adapter-placeholder).

OSA and zone-plate motors are distinct. Default to OSA transverse x/y motion with
the optics fixed. If a local procedure instead moves the zone plate, establish
that mapping and record the controlled device before starting; do not transfer
OSA move signs or step sizes to zone-plate axes. Keep longitudinal position and
other optics settings fixed during the transverse search.

## Establish the starting state

Use available session and beamline information to establish the controlled device,
axis units, current readbacks, allowed x/y search range, maximum per-move displacement,
readback tolerance, initial trial step, and smallest useful step. Obtain missing motion constraints
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

1. **Judge the current image.** Use the target features in [SKILL.md](../SKILL.md#recognize-the-target-in-words). Prioritize a clean
   hole and balanced, complete ring together; exterior cleanup is secondary. If
   these already hold, proceed to confirmation without unnecessary exploratory moves.
2. **Choose a bounded trial.** Use the recorded response to previous moves. With
   no established response, make a small permitted trial on one axis while holding
   the other fixed. State the predicted improvement as a hypothesis. Image-side
   brightness alone does not determine the sign of a motor correction.
3. **Execute and verify.** Use `adapter.read_position()` to check the absolute
   target against the permitted range and maximum displacement from the current
   position. Call `adapter.move_absolute(target_um)`, then `adapter.read_position()`
   and verify its Euclidean discrepancy from the target is within the established
   tolerance. Call `adapter.acquire_image()` and read the position again; require
   the before/after acquisition discrepancy to remain within that tolerance.
   Apply the same acquisition checks to baseline, repeat, and confirmation images.
   Record the image path, acquisition time, full observation, comparison, and plan
   before continuing. Resolve an unexpected readback before taking further action.
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

## Quantitative support

Use [the ellipse evaluator](../scripts/evaluate_alignment.py) when available;
its [metric definitions](analytical-alignment.md#quantitative-evaluation) also apply here.
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

Stop without claiming success when the remaining budget is exhausted, completed
passes on both axes at the smallest useful step yield no reproducible improvement
while defects remain, or further trials would exceed the allowed range. Preserve an unsuccessful script result when
considering the manual fallback; switching modes does not reset the budget.
Restore and reimage the best verified position when feasible within the established
limits. On a motor fault, unexpected readback, or loss of usable illumination,
suspend exploratory moves and resolve the uncertain state before any recovery
motion.

Report the outcome as **aligned**, **best found but still imperfect**, or **blocked**.
Give the final measured position and units, controlled device, confirming image
IDs, visual evidence, remaining defects, and trajectory-log path. Distinguish the
best observed position from the actual final position if restoration was unavailable.
