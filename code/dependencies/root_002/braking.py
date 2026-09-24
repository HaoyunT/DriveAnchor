"""Current-state braking candidate; no actor futures or learned parameters."""
import numpy as np

VERSION = 'current_state_smooth_brake_v1'


def braking_trajectory(speed, acceleration, curvature=0., dt=.1, points=40):
    """First feasible cubic velocity profile with zero terminal speed/accel.

    The polynomial preserves measured initial longitudinal speed/acceleration.
    Its extrema certify longitudinal acceleration/jerk bounds over the entire
    stop, including beyond the prediction horizon. Return None if the initial
    state or curved-path finite differences cannot satisfy the bounds.
    """
    speed, acceleration, curvature = map(float, (speed, acceleration, curvature))
    if not np.isfinite([speed, acceleration, curvature]).all() or speed < 0 or speed > 25:
        return None, dict(reason='invalid initial state')
    if not -3.5 <= acceleration <= 2.4:
        return None, dict(reason='initial acceleration outside brake-profile bounds')
    if speed < 1e-8 and abs(acceleration) < 1e-8:
        return np.zeros((points, 2)), dict(version=VERSION, stop_seconds=0., stop_distance_m=0., stationary=True)
    # A deterministic one-dimensional feasibility solver, not a model sweep.
    for duration in np.arange(.2, 20.001, .05):
        c1 = acceleration*duration
        c2, c3 = -3*speed-2*c1, 2*speed+c1
        # v(q)=(1-q)^2*(speed+(2*speed+c1)*q).
        if min(speed, 3*speed+c1) < -1e-9:
            continue
        aq = [0., 1.]
        if abs(c3) > 1e-12 and 0 < -c2/(3*c3) < 1:
            aq.append(-c2/(3*c3))
        a = acceleration+(2*c2*np.array(aq)+3*c3*np.array(aq)**2)/duration
        j = (2*c2+6*c3*np.array([0., 1.]))/duration**2
        if a.min() < -3.5-1e-9 or a.max() > 2.4+1e-9 or np.abs(j).max() > 4.+1e-9:
            continue
        q = np.minimum(np.arange(points)*dt/duration, 1.)
        distance = duration*(speed*q+c1*q**2/2+c2*q**3/3+c3*q**4/4)
        if abs(curvature) < 1e-7:
            xy = np.column_stack([distance, np.zeros_like(distance)])
        else:
            xy = np.column_stack([np.sin(curvature*distance)/curvature,
                                  (1-np.cos(curvature*distance))/curvature])
        velocity = np.diff(xy, axis=0)/dt
        acc = np.diff(velocity, axis=0)/dt
        jerk = np.diff(acc, axis=0)/dt
        if (np.linalg.norm(velocity,axis=1).max() > 25.+1e-8 or
            np.linalg.norm(acc,axis=1).max() > 4.05+1e-8 or
            np.linalg.norm(jerk,axis=1).max() > 4.13+1e-8):
            continue
        return xy, dict(version=VERSION, stop_seconds=float(duration),
            stop_distance_m=float(duration*(speed+c1/2+c2/3+c3/4)),
            initial_speed=speed, initial_acceleration=acceleration, curvature=curvature,
            analytic_min_longitudinal_acceleration=float(a.min()),
            analytic_max_longitudinal_acceleration=float(a.max()),
            analytic_max_abs_longitudinal_jerk=float(np.abs(j).max()),
            sampled_max_acceleration=float(np.linalg.norm(acc,axis=1).max()),
            sampled_max_jerk=float(np.linalg.norm(jerk,axis=1).max()))
    return None, dict(reason='no bounded monotone stop profile for current state')
