"""Import these OSA helpers into the 2-ID-D QueueServer worker startup namespace."""
import math


def read_osa_position():
    """Return actual OSA [x, y] motor coordinates in millimeters."""
    from apsbits.core.instrument_init import oregistry

    osa = oregistry["osa"]
    return [float(osa.x.position), float(osa.y.position)]


def move_osa_absolute(x_mm: float, y_mm: float, timeout: float = 30., settle_time: float = .2):
    """Move OSA x/y in millimeters, wait for completion, then settle."""
    import bluesky.plan_stubs as bps
    from apsbits.core.instrument_init import oregistry

    if (not all(math.isfinite(v) for v in (x_mm, y_mm, timeout, settle_time))
        or timeout <= 0 or settle_time < 0):
        raise ValueError("Require finite coordinates, a positive timeout and nonnegative settling time")
    osa = oregistry["osa"]
    # Validate both axes before either starts moving; retain the motor's limit checks.
    osa.x.check_value(x_mm)
    osa.y.check_value(y_mm)
    try:
        yield from bps.mv(osa.x, x_mm, osa.y, y_mm, timeout=timeout)
        if settle_time:
            yield from bps.sleep(settle_time)
    except Exception:
        yield from bps.stop(osa.x)
        yield from bps.stop(osa.y)
        raise
