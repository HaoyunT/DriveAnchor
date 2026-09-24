"""Jerk-limited partial braking candidates; no future observations required.

The production PDM path uses the Python reference implementation directly.
The historical optional C++ brake kernel is intentionally not imported here.
"""
import numpy as np
from brake_profiles_reference import candidates as _reference_candidates


def candidates(path, speed, acceleration, dt=.1, horizon=8.):
    return _reference_candidates(path, speed, acceleration, dt, horizon)
