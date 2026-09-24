"""V6 frame adapters: GPU road/direction and shared CV broad phase.

CPU work is limited to per-frame map/actor preparation and explicit ambiguous
nearest-vertex tie recovery. Occupancy is a broad-phase certificate, not TTC.
Final learned-prediction and time-reserve checks remain in the planner.
"""
import numpy as np
import torch
from shapely.ops import unary_union

from device_features import candidate_features
from device_map_triton import RoadCache, DirectionCache
from device_costmap import build_costmap
from device_obstacles import collision_masks_tensor


def _direction_cache(context):
    if len(context) < 3:
        # Runtime prepare supplies the frame-local third dictionary. Supporting
        # old two-tuples here preserves correctness, without claiming reuse.
        return DirectionCache(context)
    return context[2].setdefault('_v6_direction_holder', {})


@torch.inference_mode()
def direction_tensor(features, xy_cpu, context):
    holder = _direction_cache(context)
    if isinstance(holder, dict):
        if 'provider' not in holder:
            holder['provider'] = DirectionCache(context)
        provider = holder['provider']
    else:
        provider = holder
    # Direction uses the actual vehicle offset from its context, independently
    # of the immutable selector's footprint offset.
    centers = features['xy'] + context[1] * features['direction_unit']
    result = provider.evaluate_tensors(centers, features['direction_unit'],
                                       features['direction_norm'])
    # cKDTree's exact-distance tie order is not argmin vertex order. This is an
    # exceptional reference boundary, not a mandatory whole-pool CPU shadow.
    # nonzero has dynamic-shape synchronization; the compact D2H is reported.
    rows_device = torch.nonzero(result['tie_ambiguous'], as_tuple=True)[0]
    rows = rows_device.cpu().numpy()
    bad = result['bad'] | ~features['finite']
    if len(rows):
        from direction_guard import _bad_uncached
        exact = _bad_uncached(np.asarray(xy_cpu)[rows], context)
        bad = bad.clone()
        bad.index_copy_(0, rows_device,
                        torch.as_tensor(exact, device=bad.device, dtype=torch.bool)
                        | ~features['finite'][rows_device])
    return bad, dict(tie_reference_rows=len(rows), tie_index_d2h_batches=1)


def bad_direction_numpy(xy, context):
    """Existing final-check API, GPU geometry core and explicit result download."""
    xy = np.asarray(xy)
    if xy.strides[0] == 0 and len(xy) > 1:
        return np.repeat(bad_direction_numpy(xy[:1], context), len(xy))
    key = (xy.shape, xy.dtype.str, xy.tobytes())
    cache = context[2].setdefault('_v6_numpy_results', {}) if len(context) > 2 else {}
    if key not in cache:
        device_xy = torch.as_tensor(np.array(xy, copy=True), device='cuda')
        features = candidate_features(device_xy, center_offset=context[1])
        bad, _ = direction_tensor(features, xy, context)
        if len(cache) >= 8:
            cache.clear()
        cache[key] = bad.cpu().numpy()
    return cache[key].copy()


class FrameChecks:
    def __init__(self):
        self.frame_id = None
        self.roads = {}
        self.costmaps = {}
        self.builds = 0

    def begin_frame(self, frame_id):
        if self.frame_id != frame_id:
            self.frame_id = frame_id
            self.roads.clear()
            self.costmaps.clear()
            self.builds = 0

    def road_outside(self, features, lanes, ego, rules):
        polygons = [lane.polygon for lane in lanes]
        key = (tuple(p.wkb for p in polygons), float(ego.x), float(ego.y),
               float(ego.heading), rules.half_length, rules.half_width)
        if key not in self.roads:
            self.roads[key] = RoadCache(
                unary_union(polygons), ego, half_length=rules.half_length,
                half_width=rules.half_width)
        return self.roads[key].outside(features) | ~features['finite']

    @torch.inference_mode()
    def collision(self, features, field, rules):
        """Conservative map blanks skip SAT; every hit keeps exact old SAT.

        Field identity stays alive in the existing frame actor cache and keys
        normal/inflated-lead variants separately. Out-of-map/time queries are
        sent to exact checking rather than being declared safe or colliding.
        """
        key = (id(field), rules.half_length, rules.half_width)
        if key not in self.costmaps:
            self.costmaps[key] = build_costmap(
                field, half_length=rules.half_length, half_width=rules.half_width,
                bounds=(-150., -150., 150., 150.), resolution=1., layer_dt=.5)
            self.builds += 1
        costmap = self.costmaps[key]
        centers = features['box_centers'].double()
        heading = features['stable_heading'].double()
        query = costmap.query(centers)
        invalid = (~torch.isfinite(centers).flatten(1).all(1)
                   | ~torch.isfinite(heading).all(1) | field.invalid)
        needs_exact = (query['hit'] | query['invalid']).any(1) | invalid
        # Device compaction; torch.nonzero may synchronize to size the result.
        indices = torch.nonzero(needs_exact, as_tuple=True)[0]
        exact = collision_masks_tensor(
            centers.index_select(0, indices), heading.index_select(0, indices),
            field, half_length=rules.half_length, half_width=rules.half_width)
        n, t = heading.shape
        static = torch.zeros((n, t), device=centers.device, dtype=torch.bool)
        dynamic = torch.zeros_like(static)
        static.index_copy_(0, indices, exact['static_by_time'])
        dynamic.index_copy_(0, indices, exact['dynamic_by_time'])
        return dict(static_collision=static.any(1), dynamic_collision=dynamic.any(1),
                    static_by_time=static, dynamic_by_time=dynamic, invalid=invalid,
                    normal_rejected=static.any(1) | dynamic.any(1) | invalid,
                    map_candidate_hits=needs_exact.sum(),
                    map_candidate_clear=(~needs_exact).sum(),
                    map_invalid_queries=query['invalid'].sum(),
                    map_builds=self.builds, map_layers=len(costmap.layers),
                    shared_costmap=costmap)
