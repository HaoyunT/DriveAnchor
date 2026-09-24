"""nuPlan route-boundary adapter for EF v3 (no future trajectory input).

Shares the planner's deterministic route-constrained lane traversal. Unlike the
old constant-width ribbon, samples the actual left/right map boundaries and
insets them by ego half-width, following the walle feature builder. Junction
multi-lane expansion and lane-change entry metadata are not available here;
we do not fabricate either flag.
"""
import numpy as np
from shapely.geometry import Point
from rebase.ef_geometry import BranchCorridor


def local(points, ego):
    c, s = np.cos(ego.heading), np.sin(ego.heading)
    return (np.asarray(points) - [ego.x, ego.y]) @ np.array([[c, -s], [s, c]])


def route_corridor(lanes, route_ids, state, half_vehicle_width, rear_margin=5.):
    ego = state.rear_axle
    route = set(map(str, route_ids))
    eligible = [lane for lane in lanes if str(lane.get_roadblock_id()) in route]
    if not eligible:
        raise ValueError('No route lane; record excluded, no synthetic polygon fallback')
    origin = Point(ego.x, ego.y)

    def rank(lane):
        line = lane.baseline_path.linestring
        d = line.project(origin)
        a, b = line.interpolate(max(0., d - .5)), line.interpolate(min(line.length, d + .5))
        h = np.arctan2(b.y - a.y, b.x - a.x)
        return line.distance(origin) + 2 * (1 - np.cos(h - ego.heading))

    lane = min(eligible, key=rank)
    # Keep the full lane topology, but allow a controlled 0.5 m boundary
    # relaxation for the EF reference-point corridor. The exact vehicle
    # footprint remains checked downstream by the three-circle selector.
    boundary_relaxation = 0.5
    corridor_half_width = max(0.0, float(half_vehicle_width) - boundary_relaxation)
    desired = max(30., 4 * state.dynamic_car_state.speed)
    pieces = []; seen = set(); distance = 0.
    for i in range(12):
        if lane.id in seen:
            break
        seen.add(lane.id)
        line = lane.baseline_path.linestring
        begin = max(0., line.project(origin) - rear_margin) if i == 0 else 0.
        length = line.length - begin
        if length > 1e-3:
            pieces.append((distance, distance + length, lane, begin))
            distance += length
        if distance >= desired + rear_margin:
            break
        next_lanes = [candidate for candidate in lane.outgoing_edges
                      if str(candidate.get_roadblock_id()) in route and candidate.id not in seen]
        if not next_lanes:
            break
        def successor_score(candidate):
            a = lane.baseline_path.linestring.interpolate(max(0., lane.baseline_path.linestring.length - .5))
            b = candidate.baseline_path.linestring.interpolate(min(.5, candidate.baseline_path.linestring.length))
            v0 = np.array([a.x, a.y]) - np.array(lane.baseline_path.linestring.interpolate(max(0., lane.baseline_path.linestring.length - 1.)).coords[0][:2])
            v1 = np.array(candidate.baseline_path.linestring.interpolate(min(1., candidate.baseline_path.linestring.length)).coords[0][:2]) - np.array([b.x, b.y])
            n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
            if n0 < 1e-6 or n1 < 1e-6: return 1e6
            heading_delta = abs(np.arctan2(np.cross(v0, v1), np.dot(v0, v1)))
            return heading_delta + .12 * a.distance(b)
        lane = min(next_lanes, key=successor_score)
    if distance < 8. or not pieces:
        raise ValueError('Insufficient native route boundary length')
    stations = np.linspace(0., min(distance, desired + rear_margin), 8)
    left, right, centers = [], [], []
    for station in stations:
        low, high, lane, begin = next((p for p in pieces if p[1] >= station - 1e-6), pieces[-1])
        center_point = lane.baseline_path.linestring.interpolate(begin + station - low)
        center = np.asarray(center_point.coords[0][:2]); centers.append(center)
        pair = []
        for boundary in (lane.left_boundary, lane.right_boundary):
            line = boundary.linestring
            point = np.asarray(line.interpolate(line.project(center_point)).coords[0][:2])
            pair.append(point)
        delta = pair[1] - pair[0]; width = np.linalg.norm(delta)
        if width <= 2 * corridor_half_width + .1:
            raise ValueError('Native lane too narrow after vehicle-width inset')
        left.append(pair[0] + delta / width * corridor_half_width)
        right.append(pair[1] - delta / width * corridor_half_width)
    center_local = local(centers, ego)
    direction = center_local[-1] - center_local[-2]
    angle = float(np.arctan2(direction[1], direction[0]))
    # This is observable route geometry, not an oracle scenario label. Lane
    # changes are deliberately not inferred from turn direction.
    scene = 3 if angle > np.pi / 8 else 4 if angle < -np.pi / 8 else 0
    vertices = np.concatenate([local(left, ego), local(right, ego)[::-1]])
    corridor = BranchCorridor(dict(polygon=vertices.tolist(), scene_type=scene))
    return corridor, dict(route_lane_ids=[p[2].id for p in pieces], length_m=float(stations[-1]),
                          heading_rad=angle, half_vehicle_width_m=float(half_vehicle_width),
                          corridor_half_width_m=float(corridor_half_width), boundary_relaxation_m=boundary_relaxation,
                          scene_source='route heading; no lane-change/straight-junction oracle labels',
                          boundary_source='native map left/right, nearest boundary point to route station',
                          junction_expansion=False)
