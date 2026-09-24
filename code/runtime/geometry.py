"""Observable-map repair for native EF corridor construction.

Keep the legacy result whenever valid and route-continuous. Otherwise use existing route
topology and oriented native boundary arclengths; never jitter vertices, buffer
an invalid polygon, use expert motion, or relax BranchCorridor validation.
"""
import numpy as np
from shapely.geometry import Point
from rebase.ef_geometry import BranchCorridor
from native_geometry import route_corridor as legacy_corridor, local
from nuplan.common.maps.abstract_map_objects import LaneConnector

VERSION = 'native_boundary_committed_lane_change_v5'


def _rank(lane, ego):
    line = lane.baseline_path.linestring
    origin = Point(ego.x, ego.y)
    d = line.project(origin)
    a = line.interpolate(max(0., d-.5))
    b = line.interpolate(min(line.length, d+.5))
    h = np.arctan2(b.y-a.y, b.x-a.x)
    return line.distance(origin)+2*(1-np.cos(h-ego.heading))


def _chains(start, route, origin, rear_margin, target):
    """Bounded route-only traversal, preserving outgoing-edge order."""
    stack = [(start, [], set(), 0.)]
    visited = 0
    while stack and visited < 256:
        lane, pieces, seen, distance = stack.pop()
        visited += 1
        if lane.id in seen:
            continue
        line = lane.baseline_path.linestring
        begin = max(0., line.project(origin)-rear_margin) if not pieces else 0.
        length = line.length-begin
        next_pieces = pieces+[(distance, distance+length, lane, begin)] if length > 1e-3 else pieces
        distance += max(0., length)
        successors = [x for x in lane.outgoing_edges
                      if str(x.get_roadblock_id()) in route and x.id not in seen | {lane.id}]
        if distance >= target or len(seen) >= 11 or not successors:
            if distance >= 8. and next_pieces:
                yield next_pieces, distance
            continue
        for child in reversed(successors):
            stack.append((child, next_pieces, seen | {lane.id}, distance))


def _boundary_at(lane, boundary, fraction):
    line = boundary.linestring
    if line.is_empty or line.length < 1e-6:
        raise ValueError('Empty native lane boundary')
    baseline = lane.baseline_path.linestring
    start, end = Point(baseline.coords[0]), Point(baseline.coords[-1])
    forward = Point(line.coords[0]).distance(start)+Point(line.coords[-1]).distance(end)
    reverse = Point(line.coords[-1]).distance(start)+Point(line.coords[0]).distance(end)
    f = fraction if forward <= reverse else 1-fraction
    return np.asarray(line.interpolate(float(np.clip(f, 0, 1)), normalized=True).coords[0][:2])


def _construct(pieces, distance, state, half_width, target):
    stations = np.linspace(0., min(distance, target), 8)
    left, right, centers = [], [], []
    for station in stations:
        low, high, lane, begin = next((p for p in pieces if p[1] >= station-1e-6), pieces[-1])
        baseline = lane.baseline_path.linestring
        d = np.clip(begin+station-low, 0, baseline.length)
        centers.append(np.asarray(baseline.interpolate(d).coords[0][:2]))
        pair = [_boundary_at(lane, boundary, d/baseline.length)
                for boundary in (lane.left_boundary, lane.right_boundary)]
        delta = pair[1]-pair[0]
        width = np.linalg.norm(delta)
        if width <= 2*half_width+.1:
            raise ValueError('Native lane too narrow after vehicle-width inset')
        left.append(pair[0]+delta/width*half_width)
        right.append(pair[1]-delta/width*half_width)
    centers = local(centers, state.rear_axle)
    direction = centers[-1]-centers[-2]
    angle = float(np.arctan2(direction[1], direction[0]))
    scene = 3 if angle > np.pi/8 else 4 if angle < -np.pi/8 else 0
    vertices = np.concatenate([local(left, state.rear_axle), local(right, state.rear_axle)[::-1]])
    corridor = BranchCorridor(dict(polygon=vertices.tolist(), scene_type=scene))
    route_points = []
    for station in np.arange(0., min(distance, target)+1e-6, .5):
        low, high, lane, begin = next((p for p in pieces if p[1] >= station-1e-6), pieces[-1])
        route_points.append(lane.baseline_path.linestring.interpolate(begin+station-low).coords[0][:2])
    route_points = local(route_points, state.rear_axle)
    route_points = route_points[np.argmin(np.linalg.norm(route_points, axis=1)):]
    return corridor, dict(route_lane_ids=[p[2].id for p in pieces], length_m=float(stations[-1]),
        scoring_route_xy=route_points.tolist(),
        heading_rad=angle, half_vehicle_width_m=float(half_width),
        scene_source='observable route heading',
        boundary_source='oriented native boundary arclength; no polygon repair or synthetic boundary',
        junction_expansion=False)


def route_corridor(lanes, route_ids, state, half_vehicle_width, rear_margin=5., preferred_start_id=None):
    legacy = None
    try:
        corridor, diagnostic = legacy_corridor(lanes, route_ids, state, half_vehicle_width, rear_margin)
        legacy = corridor, dict(diagnostic, repair_version=VERSION, repaired=False)
        reason = 'Insufficient route-connected lookahead on current lane'
    except ValueError as error:
        reason = str(error)
    origin = Point(state.rear_axle.x, state.rear_axle.y)
    route = set(map(str, route_ids))
    eligible = sorted([x for x in lanes if str(x.get_roadblock_id()) in route],
                      key=lambda x: _rank(x, state.rear_axle))
    if not eligible:
        raise ValueError(reason)
    target = max(30., 4*state.dynamic_car_state.speed)+rear_margin
    preferred = next((x for x in eligible if str(x.id)==str(preferred_start_id)),None)
    if preferred is not None and preferred is not eligible[0]:
        for pieces,distance in _chains(preferred,route,origin,rear_margin,target):
            try:
                corridor,diagnostic=_construct(pieces,distance,state,half_vehicle_width,target)
                return corridor,dict(diagnostic,repair_version=VERSION,repaired=True,
                    original_error='Continue previously initiated observable lane change',alternative_start=True)
            except ValueError:
                continue
    primary = list(_chains(eligible[0], route, origin, rear_margin, target))
    if legacy is not None and any(distance >= target for _, distance in primary):
        return legacy
    if legacy is not None and (isinstance(eligible[0], LaneConnector) or any(
            str(x.get_roadblock_id()) in route for x in eligible[0].outgoing_edges)):
        # Do not weave across intersection connectors or change a valid current
        # lane merely because a MORE DISTANT lane lacks route continuation.
        # Prepare only at a current lane with no route-compatible successor.
        return legacy
    remaining = eligible[0].baseline_path.linestring.length-eligible[0].baseline_path.linestring.project(origin)
    if legacy is not None and remaining < 3.*state.dynamic_car_state.speed:
        # Do not begin a lane change with less than a three-second lane segment
        # left at the current speed. Keep an already-valid corridor in that case.
        return legacy
    # A route roadblock can contain lanes leading to DIFFERENT roadblocks.
    # Prepare a lane change before the current lane's route-compatible end.
    # Alternatives stay in the same observable roadblock or current footprint.
    footprint = state.car_footprint.geometry
    starts = [eligible[0]]+[x for x in eligible[1:] if x.polygon.intersects(footprint)
        or str(x.get_roadblock_id()) == str(eligible[0].get_roadblock_id())]
    if legacy is not None:
        starts = [x for x in starts if not isinstance(x, LaneConnector)
            and str(x.get_roadblock_id()) == str(eligible[0].get_roadblock_id())]
    errors = []
    candidates = []
    for start in starts:
        chains = primary if start is eligible[0] else list(_chains(start, route, origin, rear_margin, target))
        for pieces, distance in chains:
            candidates.append((start, pieces, distance))
    # Only change a still-valid legacy corridor if a full lookahead is available.
    # A genuine route end is not a reason to switch lanes arbitrarily.
    if legacy is not None:
        candidates = [c for c in candidates if c[2] >= target]
    candidates.sort(key=lambda c: c[2] < target)
    for start, pieces, distance in candidates:
        try:
            corridor, diagnostic = _construct(pieces, distance, state, half_vehicle_width, target)
            return corridor, dict(diagnostic, repair_version=VERSION, repaired=True,
                original_error=reason, alternative_start=str(start.id) != str(eligible[0].id))
        except ValueError as error:
            errors.append(str(error))
    if legacy is not None:
        return legacy
    raise ValueError('Native corridor repair unavailable: '+reason+'; '+str(errors[:5]))
