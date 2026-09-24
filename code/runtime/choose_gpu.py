from tracker_forecast import risk as tracker_forecast
from tracker_control import estimate as tracker_estimate
from signal_scope import controls_route
from comfort_progress import evaluate_cached, choose_index, VERSION as COMFORT_VERSION
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
        if (xy.ndim!=3 or xy.shape[1:]!=(40,2) or not 1<=len(xy)<=3000) or not np.isfinite(xy).all():
            raise ValueError('Up to3000 finite original-order trajectories required')
        broadcast_count=len(xy) if xy.strides[0]==0 else 1
        if broadcast_count>1:xy=xy[:1]
        cache_key=xy.tobytes()
        if not hasattr(self,'_score_cache'):self._score_cache={}
        if cache_key not in self._score_cache:
            distance = cKDTree(route).query(xy.reshape(-1, 2))[0].reshape(len(xy), 40)
            velocity = np.diff(xy, axis=1) / RULES.dt
            acceleration = np.diff(velocity, axis=1) / RULES.dt
            score = RULES.route_weight * (distance**2).mean(1)
            score += RULES.speed_match_weight * (np.linalg.norm(velocity[:, 0], axis=-1) - speed)**2 + RULES.acceleration_weight * (acceleration**2).mean((1, 2)) - RULES.endpoint_distance_weight * np.linalg.norm(xy[:, -1], axis=-1)
            self._score_cache[cache_key]=score
        score=self._score_cache[cache_key]
        red=np.zeros(len(xy),bool)
        cache_key=xy.tobytes()
        if not hasattr(self,'_invariant_cache'):self._invariant_cache={}
        if cache_key not in self._invariant_cache:
            for light in lights:
                if light.status != TrafficLightStatusType.RED:
                    continue
                connector = self.map.get_map_object(str(light.lane_connector_id), SemanticMapLayer.LANE_CONNECTOR)
                if connector is None or not controls_route(connector,self.route_ids):
                    continue
                if connector.polygon.covers(Point(ego.x, ego.y)):
                    continue
                polygon = Polygon(local(np.array(connector.polygon.exterior.coords), ego))
                red |= inclusive_xy(polygon, xy[..., 0], xy[..., 1]).any(1)
            self._invariant_cache[cache_key]=red
        else:
            red=self._invariant_cache[cache_key]
        outside=np.zeros(len(xy),bool)  # Placeholder only; GPU road provider replaces it.
        ttc_bad, ttc_diag = self.dtpp_prediction_only(xy)
        if broadcast_count>1:
            outside,red,ttc_bad,score=[np.repeat(v,broadcast_count) for v in (outside,red,ttc_bad,score)]
            ttc_diag=dict(ttc_diag,min_ttc_capped_s=ttc_diag['min_ttc_capped_s']*broadcast_count,predicted_collision=ttc_diag['predicted_collision']*broadcast_count,rejected_count=int(ttc_bad.sum()))
        return dict(outside=outside,red=red,ttc_bad=ttc_bad,quality=-score,ttc_diag=ttc_diag)

    return choose
