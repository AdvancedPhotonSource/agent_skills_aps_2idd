---
name: beamline-scan-health-monitor
description: Evaluate and continuously monitor APS 2-ID-D beamline and in-progress scan health, detect anomalies (no/low ring beam, hung sample axes, paused scan, hung detectors), and drive response actions such as detector recovery and user notification.
---

## Overview

This skill judges whether the APS 2-ID-D beamline and any in-progress scan are
healthy, and decides what to do when they are not.

The health evaluation runs **server-side**. The
`aps2idd_control.evaluate_snapshot` MCP tool fetches the device snapshot from
QueueServer, evaluates it, and returns **only the verdict** — you do not parse
the raw device/PV JSON. You read the verdict and act on it. (If you ever need the
underlying PVs, `aps2idd_control.get_global_health_snapshot` returns the full
snapshot with the same verdict attached under `evaluation`.)

The deterministic anomaly criteria live in
`control_suite_mcp_aps_2idd/health.py` (in the MCP server repo) — that module is
the source of truth. This skill is about *calling the tool, interpreting the
result, and responding*.

Use this skill when the user asks you to check beamline/scan health once, to keep
watching it during an experiment, or to react when a scan stalls. For the generic
"sleep, re-check, repeat" loop mechanics, compose this skill with the
`monitor-status` skill: this skill defines *the monitoring action and the
response actions*; `monitor-status` defines *the loop*.

## What this skill needs

### 1. The health verdict (the observable)

Call **`aps2idd_control.evaluate_snapshot`**. It returns just the verdict:

```json
{
  "overall": "error",
  "devices": {"ring": "ok", "sample": "ok", "scanrecord_fly": "ok", "xrf": "error", "sis3820": "error", "...": "..."},
  "anomalies": [
    {"kind": "detector_hung", "device": "xrf", "severity": "error", "message": "Detector 'xrf' looks hung: ..."},
    {"kind": "detector_hung", "device": "sis3820", "severity": "error", "message": "Detector 'sis3820' looks hung: ..."}
  ],
  "ring": {"current": 200.3, "mode": "Delivered Beam"}
}
```

Read it directly — there is no raw snapshot to parse:

- `overall` — worst severity: `ok` / `warning` / `error`.
- `devices` — per-device severity.
- `anomalies` — the actionable list; each item has `kind`, `severity`,
  `message`, and `device` or `axes`.
- `ring` — `{current, mode}` for quick reference.

Use `aps2idd_control.get_global_health_snapshot` instead only when you need the
underlying PV values (it returns the full device/PV snapshot with this same
verdict under its `evaluation` key).

If `aps2idd_control.evaluate_snapshot` is not available (e.g. an older MCP server
build), **stop and tell the user** the server needs updating/restarting.

### 2. Response actions (what to do on an anomaly)

Confirm with the user, before an automated run, what you should do for each
anomaly kind. Sensible defaults:

- `detector_hung` — attempt detector recovery ("unhang") via the
  `aps2idd_control.recover_detector` MCP tool, then notify the user. For 2-ID-D
  fly scans, recover **both** `sis3820` and `xrf` (call `recover_detector` once
  per device), not just the named one: the scaler can be the actual blocker
  holding `WAIT:DETCTRS` even when the area detector is the one flagged. Detector
  recovery is a real beamline operation; if the tool is unavailable, notify the
  user instead of attempting an ad-hoc fix.
- `sample_hung` — do **not** auto-correct sample motion. Notify the user and, if
  asked, pause the scan. Moving stages blindly can damage hardware or the sample.
- `no_beam` / `low_ring_current` — notify the user; the scan usually pauses
  itself. Do not take corrective beamline action.
- `scan_paused` — informational; report it. A scan can be legitimately paused.

### 3. Loop parameters (only when continuously monitoring)

If the user wants ongoing monitoring rather than a one-shot check, get the
**interval** between checks and the **number of checks** (default: indefinite),
then follow the `monitor-status` skill for the sleep/re-check loop.

## Procedure

1. If the user wants continuous monitoring, confirm the response actions above
   and the interval/count, then follow `monitor-status` for the loop mechanics.
2. Call `aps2idd_control.evaluate_snapshot`.
3. Read `overall` and `anomalies`. For each anomaly, perform the agreed response
   action for its `kind`. Report clearly to the user what was found and what you
   did, labeling severities.
4. If this is a one-shot check, summarize and stop. If continuous, enter the
   sleep phase (per `monitor-status`) and return to step 2 when you wake up.
5. Repeat until the user's check count is reached, the user stops you, or an
   anomaly's agreed action is "stop and wait for instructions."

## Anomaly criteria (implemented in `control_suite_mcp_aps_2idd/health.py`)

Summarized so you can explain them; the code is the source of truth:

- **no_beam** (`error`): ring `current` < 10 **and** operating mode contains
  "NO BEAM".
- **low_ring_current** (`warning`): ring `current` < 100 mA.
- **sample_hung** (`error`): for an axis (x/y/z/theta), the axis is **not**
  moving (`{axis}.motor_is_moving` is false) yet its `user_setpoint` and
  `user_readback` differ by more than the position tolerance (default `0.1`).
- **scan_paused** (`warning`): a scan record (`scanrecord_fly` /
  `scanrecord_step`) has its `pause_signal` set or a `wait` PV >= 1.
- **detector_hung** (`error`): a 2-ID-D detector (`xrf`, `tmm1`, `sis3820`) is
  acquiring while a scan record sits in `WAIT:DETCTRS` but the scan has made no
  progress for longer than `max(30 s, 3 x inner_points x dwell)` (dwell from
  `fscanh_dwell`). Scan-progress staleness is used rather than a per-detector
  capture timestamp because the scaler has no capture timestamp and the area
  detector's only updates once per fly line. `tmm1` participates automatically
  if it is registered and added to the snapshot.
- Per-device connectivity (fallback): `ok` if all PVs connected and none
  errored; `warning` if some connected; `error` if none connected.

Thresholds (`detector_timeout_factor`, `sample_position_tolerance`) are tunable
in `health.py` via `configure_health(...)`; they are not currently exposed
through the MCP tool.

## Hints

- The evaluation is computed fresh on every call; just re-call the tool each loop
  iteration. Do not cache verdicts across iterations.
- If the user's instructions contradict this document, follow the user. For
  example, the user may want you to only *report* anomalies and never take
  corrective action — in that case skip all response actions and just notify.
- Detector recovery and any stage motion are real beamline operations with
  physical consequences. Never invent a recovery procedure here; only invoke an
  existing, sanctioned beamline tool, and otherwise defer to the user.
