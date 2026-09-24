"""FM2 planner features: immutable world cache, batched local transform, tensor output.

No NuPlan or torch import is needed until install()/the device backend is used.
The reference history interpolation, map query, ordering and sampling are retained.
"""
from collections import OrderedDict
import numpy as np


SHAPES = OrderedDict([
    ("ego", (1, 1, 40, 8)),
    ("interact_obstacle", (1, 8, 40, 22)),
    ("lane_instance_wise", (1, 100, 5)),
    ("lane_instance_points", (1, 100, 50, 8)),
    ("polygon_instance_wise", (1, 100, 2)),
    ("polygon_instance_points", (1, 100, 20, 2)),
])
SLICES = {}
TOTAL = 0
for _name, _shape in SHAPES.items():
    _size = int(np.prod(_shape))
    SLICES[_name] = slice(TOTAL, TOTAL + _size)
    TOTAL += _size
DYNAMIC_SIZE = SLICES["interact_obstacle"].stop


def views(packed):
    return {name: packed[SLICES[name]].reshape(shape) for name, shape in SHAPES.items()}


def transform_numpy(world, ego):
    """Reference double subtraction/matmul; float32 conversion happens on assignment."""
    c, s = np.cos(ego.heading), np.sin(ego.heading)
    return ((world.reshape(-1, 2) - np.array([ego.x, ego.y])) @
            np.array([[c, -s], [s, c]])).reshape(world.shape)


def reference_boundary_flips(lanes, ego, local):
    """Exact original per-line call shapes and scalar norms for a QA/compatibility gate."""
    flags = np.zeros((len(lanes), 2), bool)
    for i, lane in enumerate(lanes):
        center = local(lane[0], ego)
        for j in range(2):
            boundary = local(lane[j + 1], ego)
            flags[i, j] = np.linalg.norm(boundary[0] - center[0]) > np.linalg.norm(boundary[-1] - center[0])
    return flags


def fill_map_numpy(packed, lane_world, polygon_world, kinds, ego):
    """CPU oracle and explicit fallback. Preserve scalar norm boundary tie decisions."""
    feats = views(packed)
    n, p = len(lane_world), len(polygon_world)
    local = transform_numpy(lane_world, ego)
    center, left, right = local[:, 0], local[:, 1].copy(), local[:, 2].copy()
    flips = np.zeros((n, 2), bool)
    for i in range(n):
        for j, boundary in enumerate((left, right)):
            flip = np.linalg.norm(boundary[i, 0] - center[i, 0]) > np.linalg.norm(boundary[i, -1] - center[i, 0])
            flips[i, j] = flip
            if flip:
                boundary[i] = boundary[i, ::-1].copy()
    li, lp = feats["lane_instance_wise"][0], feats["lane_instance_points"][0]
    li[:n, 0] = 1
    li[:n, 4] = 1
    li[:n, 2] = np.linalg.norm(left - center, axis=2).mean(axis=1)
    li[:n, 3] = np.linalg.norm(right - center, axis=2).mean(axis=1)
    lp[:n, :, :2] = 1
    lp[:n, :, 2:4] = center
    lp[:n, :, 4:6] = left
    lp[:n, :, 6:8] = right
    feats["polygon_instance_wise"][0, :p, 0] = kinds
    feats["polygon_instance_wise"][0, :p, 1] = 1
    feats["polygon_instance_points"][0, :p] = transform_numpy(polygon_world, ego)
    return flips


class WorldCache:
    """Content-keyed static samples; no local coordinates, width or flip is cached."""
    def __init__(self, capacity=4096):
        self.capacity = capacity
        self.map_ref = None
        self.map_name = None
        self.entries = OrderedDict()
        self.generation = 0
        self.layout_key = None

    def check_map(self, map_api):
        name = getattr(map_api, "map_name", None)
        if self.map_ref is not map_api or self.map_name != name:
            self.map_ref, self.map_name = map_api, name
            self.entries.clear()
            self.generation += 1
            self.layout_key = None

    def get(self, key, create):
        if key in self.entries:
            value = self.entries.pop(key)
        else:
            value = np.ascontiguousarray(create(), dtype=np.float64)
            value.setflags(write=False)
        self.entries[key] = value
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return value

    def prepare(self, planner, ego, symbols):
        self.check_map(planner.map)
        L = symbols["SemanticMapLayer"]
        layers = [L.LANE, L.LANE_CONNECTOR, L.INTERSECTION, L.CROSSWALK,
                  L.STOP_LINE, L.WALKWAYS, L.CARPARK_AREA]
        # Preserve installed spatial-index/ROI behavior and original stable ordering.
        near = planner.map.get_proximal_map_objects(symbols["Point2D"](ego.x, ego.y), 100, layers)
        point = symbols["Point"](ego.x, ego.y)
        lanes = list(near[L.LANE]) + list(near[L.LANE_CONNECTOR])
        lanes.sort(key=lambda lane: lane.polygon.distance(point))
        valid, lane_keys, lane_values = [], [], []
        for lane in lanes:
            if len(valid) == 100:
                break
            line = lane.baseline_path.linestring
            if line.is_empty or line.length < .01:
                continue
            lines = (line, lane.left_boundary.linestring, lane.right_boundary.linestring)
            key = ("lane",) + tuple(x.wkb for x in lines)
            value = self.get(key, lambda: np.stack([planner.sample_line_cached(x, 50) for x in lines]))
            valid.append(lane)
            lane_keys.append(key)
            lane_values.append(value)
        if not valid:
            raise ValueError("No native map lanes; refusing fabricated features")
        poly = []
        for layer, kind in [(L.INTERSECTION, 1), (L.CROSSWALK, 3), (L.STOP_LINE, 5),
                            (L.WALKWAYS, 10), (L.CARPARK_AREA, 6)]:
            poly.extend((obj, kind) for obj in near[layer])
        poly.sort(key=lambda pair: pair[0].polygon.distance(point))
        polygon_keys, polygon_values, kinds = [], [], []
        for obj, kind in poly[:100]:
            geometry = obj.polygon
            key = ("polygon", geometry.wkb)
            value = self.get(key, lambda: planner.sample_line_cached(
                geometry.simplify(.1, preserve_topology=True).exterior, 20))
            polygon_keys.append(key)
            polygon_values.append(value)
            kinds.append(kind)
        signature = (tuple(lane_keys), tuple(polygon_keys))
        if self.layout_key != signature:
            self.lane_world = np.stack(lane_values)
            self.polygon_world = np.stack(polygon_values) if polygon_values else np.empty((0, 20, 2), np.float64)
            self.layout_key = signature
        return valid, kinds, signature, lane_values, polygon_values


class DeviceWorld:
    """Bounded GPU banks: newly seen geometries upload together; order changes gather."""
    def __init__(self, torch, device, capacity=4096):
        self.torch, self.device, self.capacity = torch, device, capacity
        self.generation = None
        self.lane_ids, self.polygon_ids = {}, {}
        self.layout_key = None
        self.template_key = None
        self.uploaded_geometries = 0

    def reset(self, generation):
        self.generation = generation
        self.lane_ids.clear()
        self.polygon_ids.clear()
        self.layout_key = None
        self.lane_bank = self.torch.empty((self.capacity, 3, 50, 2), dtype=self.torch.float64, device=self.device)
        self.polygon_bank = self.torch.empty((self.capacity, 20, 2), dtype=self.torch.float64, device=self.device)

    def world(self, cache, signature, lane_values, polygon_values):
        if self.generation != cache.generation:
            self.reset(cache.generation)
        new_lanes = len(set(signature[0]) - self.lane_ids.keys())
        new_polys = len(set(signature[1]) - self.polygon_ids.keys())
        if len(self.lane_ids) + new_lanes > self.capacity or len(self.polygon_ids) + new_polys > self.capacity:
            self.reset(cache.generation)
        for keys, values, mapping, bank in (
            (signature[0], lane_values, self.lane_ids, self.lane_bank),
            (signature[1], polygon_values, self.polygon_ids, self.polygon_bank),
        ):
            added = []
            start = len(mapping)
            for key, value in zip(keys, values):
                if key not in mapping:
                    mapping[key] = len(mapping)
                    added.append(value)
            if added:
                # A single delta transfer per geometry kind, outside the per-lane loop.
                data = self.torch.from_numpy(np.stack(added)).to(self.device)
                bank[start:start + len(added)].copy_(data)
                self.uploaded_geometries += len(added)
        if self.layout_key != signature:
            indices = np.array([self.lane_ids[k] for k in signature[0]] +
                               [self.polygon_ids[k] for k in signature[1]], np.int64)
            indices = self.torch.from_numpy(indices).to(self.device)
            n = len(signature[0])
            self.lanes = self.lane_bank.index_select(0, indices[:n])
            self.polygons = self.polygon_bank.index_select(0, indices[n:])
            self.layout_key = signature
        return self.lanes, self.polygons

    def template(self, kinds, n):
        key = (n, tuple(kinds))
        if self.template_key != key:
            packed = np.zeros(TOTAL, np.float32)
            f = views(packed)
            f["lane_instance_wise"][0, :n, 0] = 1
            f["lane_instance_wise"][0, :n, 4] = 1
            f["lane_instance_points"][0, :n, :, :2] = 1
            f["polygon_instance_wise"][0, :len(kinds), 0] = kinds
            f["polygon_instance_wise"][0, :len(kinds), 1] = 1
            self.packed_template = self.torch.from_numpy(packed).to(self.device)
            self.mask = self.torch.from_numpy(make_mask(n, len(kinds))).to(self.device)
            self.template_key = key
        # Returned features own this clone; the next frame cannot overwrite context.
        return self.packed_template.clone(), self.mask.clone()

    def fill(self, packed, lane_world, polygon_world, ego, reference_flips=None):
        torch = self.torch
        c, s = np.cos(ego.heading), np.sin(ego.heading)
        transform = torch.from_numpy(np.array([ego.x, ego.y, c, -s, s, c], np.float64)).to(self.device)
        origin, rotation = transform[:2], transform[2:].reshape(2, 2)
        local = (lane_world - origin) @ rotation
        center, left, right = local[:, 0], local[:, 1], local[:, 2]
        # Strict > preserves exact ties; reductions remain float64 until feature copy.
        if reference_flips is None:
            left_flip = torch.linalg.vector_norm(left[:, 0] - center[:, 0], dim=-1) > torch.linalg.vector_norm(left[:, -1] - center[:, 0], dim=-1)
            right_flip = torch.linalg.vector_norm(right[:, 0] - center[:, 0], dim=-1) > torch.linalg.vector_norm(right[:, -1] - center[:, 0], dim=-1)
        else:
            flags = torch.from_numpy(reference_flips).to(self.device)
            left_flip, right_flip = flags[:, 0], flags[:, 1]
        left = torch.where(left_flip[:, None, None], left.flip(1), left)
        right = torch.where(right_flip[:, None, None], right.flip(1), right)
        f = views(packed)
        n, p = len(lane_world), len(polygon_world)
        li, lp = f["lane_instance_wise"][0], f["lane_instance_points"][0]
        li[:n, 2] = torch.linalg.vector_norm(left - center, dim=-1).mean(dim=1)
        li[:n, 3] = torch.linalg.vector_norm(right - center, dim=-1).mean(dim=1)
        lp[:n, :, 2:4], lp[:n, :, 4:6], lp[:n, :, 6:8] = center, left, right
        f["polygon_instance_points"][0, :p] = (polygon_world - origin) @ rotation
        return torch.stack((left_flip, right_flip), dim=1)


def make_mask(n, p):
    mask = np.zeros((1, 208), bool)
    mask[:, :8] = True  # Eight ego time tokens, not actor visibility.
    mask[:, 8:8 + n] = True
    mask[:, 108:108 + p] = True
    return mask


def fill_dynamic(planner, history, packed, local):
    """Retain the exact short-history interpolation and actor filtering semantics."""
    states = list(history.ego_states)
    ego, now = states[-1].rear_axle, states[-1].time_point.time_s
    times = np.array([x.time_point.time_s for x in states])
    xy = np.array([[x.rear_axle.x, x.rear_axle.y] for x in states])
    query = now + np.arange(-39, 1) * .1
    past = np.stack([np.interp(query, times, xy[:, i]) for i in range(2)], 1)
    ef = packed[SLICES["ego"]].reshape(SHAPES["ego"])
    agents = packed[SLICES["interact_obstacle"]].reshape(SHAPES["interact_obstacle"])
    ef[0, 0, :, 0] = 1
    ef[0, 0, :, 1:3] = local(past, ego)
    ef[0, 0, :, 3] = states[-1].dynamic_car_state.speed
    ef[0, 0, :, 4] = 1
    ef[0, 0, :, 6:8] = [4.5, 2.]
    tracks = list(history.observations[-1].tracked_objects.tracked_objects)
    moving = [x for x in tracks if hasattr(x, "velocity") and np.hypot(x.velocity.x, x.velocity.y) > .5 and np.hypot(x.center.x - ego.x, x.center.y - ego.y) < 80]
    moving.sort(key=lambda x: np.hypot(x.center.x - ego.x, x.center.y - ego.y))
    if not hasattr(planner, "_actor_frames"):
        planner._actor_frames = OrderedDict()
    frames = []
    for obs in history.observations:
        key = id(obs)
        if key not in planner._actor_frames:
            by_token = {}
            for actor in obs.tracked_objects.tracked_objects:
                by_token.setdefault(actor.track_token, actor)
            planner._actor_frames[key] = (obs, by_token)
            if len(planner._actor_frames) > 64:
                planner._actor_frames.popitem(last=False)
        frames.append(planner._actor_frames[key][1])
    for i, obj in enumerate(moving[:8]):
        seen = []
        for t, frame in zip(times, frames):
            match = frame.get(obj.track_token)
            if match is not None:
                seen.append((t, match.center.x, match.center.y))
        if not seen:
            continue
        a = np.array(seen)
        h = np.stack([np.interp(query, a[:, 0], a[:, j]) for j in (1, 2)], 1)
        agents[0, i, :, :2] = local(h, ego)
        vel = local([[ego.x + obj.velocity.x, ego.y + obj.velocity.y]], ego)[0]
        agents[0, i, :, 2:4] = vel
        agents[0, i, :, 4:7] = [obj.box.width, obj.box.length, obj.box.height]
        agents[0, i, :, 7] = (obj.center.heading - ego.heading + np.pi) % (2 * np.pi) - np.pi
    return ego, tracks


class FeatureBuilder:
    def __init__(self, symbols, device, backend="device", flip_backend="device"):
        if backend not in ("device", "numpy"):
            raise ValueError("backend must be device or numpy")
        if flip_backend not in ("device", "reference"):
            raise ValueError("flip_backend must be device or reference")
        self.symbols, self.device, self.backend = symbols, device, backend
        self.flip_backend = flip_backend
        self.cache = WorldCache()
        self.gpu = None

    def __call__(self, planner, history):
        torch = self.symbols["torch"]
        dynamic = np.zeros(DYNAMIC_SIZE, np.float32)
        ego, tracks = fill_dynamic(planner, history, dynamic, self.symbols["local"])
        valid, kinds, signature, lanes, polygons = self.cache.prepare(planner, ego, self.symbols)
        if self.backend == "numpy":
            packed = np.zeros(TOTAL, np.float32)
            packed[:DYNAMIC_SIZE] = dynamic
            self.last_flips = fill_map_numpy(packed, self.cache.lane_world, self.cache.polygon_world, kinds, ego)
            output = torch.from_numpy(packed).to(self.device)
            mask = torch.from_numpy(make_mask(len(valid), len(kinds))).to(self.device)
        else:
            if self.gpu is None:
                self.gpu = DeviceWorld(torch, self.device)
            world_lanes, world_polygons = self.gpu.world(self.cache, signature, lanes, polygons)
            output, mask = self.gpu.template(kinds, len(valid))
            output[:DYNAMIC_SIZE].copy_(torch.from_numpy(dynamic).to(self.device))
            reference_flips = None
            if self.flip_backend == "reference":
                # Explicit compatibility option, not an automatic CPU shadow.
                reference_flips = reference_boundary_flips(lanes, ego, self.symbols["local"])
            self.last_flips = self.gpu.fill(output, world_lanes, world_polygons, ego, reference_flips)
        return views(output), mask, valid, tracks


def install(planner_class, backend="device", reference_features=None, flip_backend="device"):
    """Patch the actual FM2 class alias. Call after importing FM2Top250Planner.

    Returns the original method for explicit rollback/reference validation. No model
    or DTPP change, no automatic CPU shadow, and no device synchronization is added.
    """
    original = reference_features or getattr(planner_class, "_planner_features_v1_original", planner_class.features)
    symbols = original.__globals__
    for name in ("torch", "local", "SemanticMapLayer", "Point2D", "Point"):
        if name not in symbols:
            raise ValueError("reference features globals missing " + name)

    def features(self, history):
        config = (backend, flip_backend, str(self.device), id(symbols))
        builder = getattr(self, "_planner_features_v1", None)
        if builder is None or getattr(self, "_planner_features_v1_config", None) != config:
            builder = FeatureBuilder(symbols, self.device, backend, flip_backend)
            self._planner_features_v1 = builder
            self._planner_features_v1_config = config
        return builder(self, history)

    planner_class._planner_features_v1_original = original
    planner_class.features = features
    return original


def invalidate(planner):
    """Explicitly discard prepared geometry after an external map revision."""
    planner.__dict__.pop("_planner_features_v1", None)


def restore(planner_class):
    planner_class.features = planner_class._planner_features_v1_original
