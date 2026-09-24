"""Full500 fixed-rule selection with prepared, inclusive point predicates.

Only the geometry predicate implementation changes. For a point p,
intersects(region, p) equals contains(region, p) OR touches(region, p),
including boundaries and hole boundaries. No candidate is removed.
"""
import hashlib
import inspect
from pathlib import Path

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

REFERENCE_PLANNER_SHA = '78b036bd670e6013f7b458484244e53b580f2dd8030a3d2e082fc7439d021991'
REFERENCE_RULES_SHA = '60fffe5b17e902caa61d53d83d72ac1add7de88c53bcdee79e113311b9b6f235'
VERSION = 'full500_prepared_point_predicates_v1'


def inclusive_xy(region, x, y):
    shapely.prepare(region)
    return shapely.intersects_xy(region, x, y)


def make_fast_choose(reference_choose):
    """Bind the verified reference's geometry enums and immutable score rules."""
    reference_path = Path(inspect.getsourcefile(reference_choose))
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != REFERENCE_PLANNER_SHA:
        raise ValueError('Reference planner changed')
    if not all(hasattr(shapely, name) for name in ('prepare', 'intersects_xy')):
        raise ValueError('Prepared coordinate predicates require Shapely2')
    env = reference_choose.__globals__
    stable_heading = env['stable_heading']
    rules_path = Path(inspect.getsourcefile(stable_heading))
    if hashlib.sha256(rules_path.read_bytes()).hexdigest() != REFERENCE_RULES_SHA:
        raise ValueError('Reference rules changed')
    RULES = env['RULES']
    constrained_choice = env['constrained_choice']
    local = env['local']
    TrafficLightStatusType = env['TrafficLightStatusType']
    SemanticMapLayer = env['SemanticMapLayer']

    def choose(self, xy, route, tracks, ego, speed, lights, lanes):
        if xy.shape != (500, 40, 2) or not np.isfinite(xy).all():
            raise ValueError('All500 finite original-order trajectories required')
        distance = cKDTree(route).query(xy.reshape(-1, 2))[0].reshape(len(xy), 40)
        velocity = np.diff(xy, axis=1) / RULES.dt
        acceleration = np.diff(velocity, axis=1) / RULES.dt
        score = RULES.route_weight * (distance**2).mean(1)
        score += RULES.speed_match_weight * (np.linalg.norm(velocity[:, 0], axis=-1) - speed)**2 + RULES.acceleration_weight * (acceleration**2).mean((1, 2)) - RULES.endpoint_distance_weight * np.linalg.norm(xy[:, -1], axis=-1)
        heading = stable_heading(xy)
        collision = np.zeros(len(xy), bool)
        red = np.zeros(len(xy), bool)
        centers = xy + RULES.center_offset * np.stack([np.cos(heading), np.sin(heading)], -1)
        for obj in tracks:
            pos = local([[obj.center.x, obj.center.y]], ego)[0]
            vel = local([[ego.x + obj.velocity.x, ego.y + obj.velocity.y]], ego)[0] if hasattr(obj, 'velocity') else np.zeros(2)
            pred = pos + np.arange(40)[:, None] * RULES.dt * vel
            yaw = obj.center.heading - ego.heading
            c, s = np.cos(yaw), np.sin(yaw)
            relative = (centers - pred) @ np.array([[c, -s], [s, c]])
            angle = heading - yaw
            half_l = obj.box.length / 2 + np.abs(np.cos(angle)) * RULES.half_length + np.abs(np.sin(angle)) * RULES.half_width
            half_w = obj.box.width / 2 + np.abs(np.sin(angle)) * RULES.half_length + np.abs(np.cos(angle)) * RULES.half_width
            hit = (np.abs(relative[..., 0]) < half_l) & (np.abs(relative[..., 1]) < half_w)
            collision |= hit.any(1)
        for light in lights:
            if light.status != TrafficLightStatusType.RED:
                continue
            connector = self.map.get_map_object(str(light.lane_connector_id), SemanticMapLayer.LANE_CONNECTOR)
            if connector is None:
                continue
            if connector.polygon.covers(Point(ego.x, ego.y)):
                continue
            polygon = Polygon(local(np.array(connector.polygon.exterior.coords), ego))
            red |= inclusive_xy(polygon, xy[..., 0], xy[..., 1]).any(1)
        forward = np.stack([np.cos(heading), np.sin(heading)], -1)
        side = np.stack([-np.sin(heading), np.cos(heading)], -1)
        corners = np.stack([centers + l * RULES.half_length * forward + w * RULES.half_width * side
                            for l, w in [(1, 1), (1, -1), (-1, -1), (-1, 1)]], axis=2)
        c, s = np.cos(ego.heading), np.sin(ego.heading)
        world = corners @ np.array([[c, s], [-s, c]]) + np.array([ego.x, ego.y])
        region = unary_union([lane.polygon for lane in lanes])
        inside = inclusive_xy(region, world[..., 0], world[..., 1])
        outside = ~inside.all((1, 2))
        vbad = (np.linalg.norm(velocity, axis=-1) > RULES.speed_limit).any(1)
        abad = (np.linalg.norm(acceleration, axis=-1) > RULES.acceleration_limit).any(1)
        violations = np.column_stack([collision, outside, red, vbad, abad])
        index, diagnostic = constrained_choice(-score, violations)
        diagnostic['constraint_names'] = ['predicted_collision', 'lane_footprint_exit', 'current_red_entry', 'speed', 'acceleration']
        diagnostic['selected_violations'] = violations[index].tolist()
        diagnostic['rejected_by_constraint'] = violations.sum(0).tolist()
        self.selection_diagnostic = diagnostic
        return index

    return choose
