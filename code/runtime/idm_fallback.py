"""Lazy nuPlan IDMPlanner fallback for frames with no usable learned plan.

The fallback is deliberately lazy: importing the learned planner remains
possible in lightweight parity environments without nuPlan installed.  The
real evaluator initializes this wrapper once with the same PlannerInitialization
and then reuses the stateful IDM planner across control frames.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class IDMPlannerFallback:
    """Stateful adapter around nuPlan's open-source IDMPlanner."""

    def __init__(
        self,
        *,
        target_velocity: float = 10.0,
        min_gap_to_lead_agent: float = 1.0,
        headway_time: float = 1.5,
        accel_max: float = 1.0,
        decel_max: float = 2.0,
        planned_trajectory_samples: int = 40,
        planned_trajectory_sample_interval: float = 0.1,
        occupancy_map_radius: float = 100.0,
    ) -> None:
        self._kwargs = dict(
            target_velocity=target_velocity,
            min_gap_to_lead_agent=min_gap_to_lead_agent,
            headway_time=headway_time,
            accel_max=accel_max,
            decel_max=decel_max,
            planned_trajectory_samples=planned_trajectory_samples,
            planned_trajectory_sample_interval=planned_trajectory_sample_interval,
            occupancy_map_radius=occupancy_map_radius,
        )
        self._planner = None
        self._initialized = False
        self._init_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self._planner is not None and self._initialized

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def initialize(self, initialization) -> bool:
        """Construct and initialize IDM; return False with an explicit reason."""

        try:
            from nuplan.planning.simulation.planner.idm_planner import IDMPlanner

            self._planner = IDMPlanner(**self._kwargs)
            self._planner.initialize(initialization)
            self._initialized = True
            self._init_error = None
            return True
        except Exception as exc:  # evaluator records this; learned path remains intact
            self._planner = None
            self._initialized = False
            self._init_error = f"{type(exc).__name__}: {exc}"
            return False

    def compute_planner_trajectory(self, current_input):
        if not self.available:
            raise RuntimeError(self._init_error or "IDM fallback is not initialized")
        return self._planner.compute_planner_trajectory(current_input)

    @staticmethod
    def _to_local_xy(trajectory, current_input, count: int, dt: float) -> np.ndarray:
        """Convert an IDM trajectory to the model's rear-axle local grid.

        nuPlan planners may return either a trajectory beginning at ``t0`` or
        one beginning at the first future sample.  The selector always uses a
        40-point grid whose first point is the current rear axle, so both forms
        are normalized here.  ``np.interp`` deliberately holds the final IDM
        pose if a runner returns a shorter horizon; that keeps the candidate
        finite and lets PDM score it rather than silently dropping it.
        """

        if not hasattr(trajectory, "get_sampled_trajectory"):
            raise TypeError("IDM trajectory does not expose sampled states")
        states = list(trajectory.get_sampled_trajectory())
        if not states:
            raise ValueError("IDM returned an empty trajectory")

        history = current_input.history
        ego = history.ego_states[-1].rear_axle
        t0 = float(history.ego_states[-1].time_point.time_s)
        times = np.asarray([float(s.time_point.time_s) for s in states], dtype=float)
        world = np.asarray([[float(s.rear_axle.x), float(s.rear_axle.y)] for s in states], dtype=float)
        finite = np.isfinite(times) & np.isfinite(world).all(axis=1)
        times, world = times[finite], world[finite]
        if len(times) == 0:
            raise ValueError("IDM returned no finite states")
        order = np.argsort(times, kind="stable")
        times, world = times[order], world[order]
        unique = np.r_[True, np.diff(times) > 1e-7]
        times, world = times[unique], world[unique]

        # Normalize the origin exactly to the current rear axle.  A future-only
        # IDM trajectory gets an explicit t0 sample; a t0 trajectory is pinned
        # to the same pose to avoid a one-sample coordinate mismatch.
        if times[0] > t0 + 1e-3:
            times = np.r_[t0, times]
            world = np.vstack(([ego.x, ego.y], world))
        else:
            times[0] = t0
            world[0] = [ego.x, ego.y]

        target = t0 + np.arange(int(count), dtype=float) * float(dt)
        c, s = np.cos(float(ego.heading)), np.sin(float(ego.heading))
        local = (world - np.asarray([ego.x, ego.y])) @ np.asarray([[c, -s], [s, c]])
        result = np.column_stack([
            np.interp(target, times, local[:, 0]),
            np.interp(target, times, local[:, 1]),
        ])
        result[0] = 0.0
        if result.shape != (count, 2) or not np.isfinite(result).all():
            raise ValueError("IDM local candidate is not finite on the selector grid")
        return result

    def compute_candidate(self, current_input, *, count: int = 40, dt: float = 0.1):
        """Return the native IDM trajectory and its selector-grid local XY.

        Keeping both objects avoids a second IDM invocation when the final
        planner check elects the same candidate as the hard fallback.
        """

        trajectory = self.compute_planner_trajectory(current_input)
        local_xy = self._to_local_xy(trajectory, current_input, count, dt)
        return trajectory, local_xy
