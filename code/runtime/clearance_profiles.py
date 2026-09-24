"""Bounded acceleration alternatives for leaving a dynamic conflict zone.

These are candidates, never a safety override. The caller must apply its final
map/curvature speed cap and all road, collision and learned-risk checks. The
4 s default matches the current 40-point execution check, not an 8 s guarantee.
"""
import numpy as np


def candidates(path, speed, acceleration, dt=.1, horizon=4.):
    path = np.asarray(path, float)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError('finite XY path required')
    if not np.isfinite([speed, acceleration, dt, horizon]).all() or speed < 0 or dt <= 0 or horizon <= 0:
        raise ValueError('invalid state or time grid')
    h = .01
    stride = round(dt / h)
    steps = round(horizon / h)
    if stride < 1 or not np.isclose(stride*h, dt) or not np.isclose(steps*h, horizon):
        raise ValueError('time grid must be an integer multiple of 10 ms')
    if not -3. <= acceleration <= 2.:
        # An initially infeasible state needs explicit recovery, not silently
        # clamping the measurement or claiming a bound that was already false.
        return []
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-7]
    arc, path = arc[keep], path[keep]
    if len(path) < 2:
        return []
    result = []
    for target in [1., 1.5, 2.]:
        for duration in [.5, 1., 2., 4.]:
            v, a, s = float(speed), float(acceleration), 0.
            positions, velocities, accelerations = [s], [v], [a]
            for step in range(steps):
                desired = target if step*h < duration else 0.
                next_a = a + np.clip(desired-a, -3.5*h, 3.5*h)
                next_v = max(0., v + .5*(a+next_a)*h)
                s += .5*(v+next_v)*h
                v, a = next_v, next_a
                if (step+1) % stride == 0:
                    positions.append(s); velocities.append(v); accelerations.append(a)
            if s > arc[-1]+1e-7:
                continue  # Never clamp geometry and invent an instantaneous stop.
            result.append(dict(
                xy=np.column_stack([np.interp(positions, arc, path[:,k]) for k in (0,1)]),
                station=np.asarray(positions), speed=np.asarray(velocities),
                acceleration=np.asarray(accelerations), target=target,
                duration=duration, timing_source='bounded_clearance', horizon=horizon))
    return result
