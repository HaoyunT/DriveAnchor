"""Instance-local DTPP feature preparation; prediction/weights are untouched.

Install after the application's existing fast_features/indexed_features patches.
The CPU map mode is the numerical reference. The default candidate runs the
map point transform and headings on the encoder device; validate before use.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
from types import MethodType, SimpleNamespace
import time
import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


LAYERS = ("LANE", "ROUTE_LANES", "CROSSWALK")
ELEMENTS = {"LANE": 40, "ROUTE_LANES": 10, "CROSSWALK": 5}
POINTS = {"LANE": 50, "ROUTE_LANES": 50, "CROSSWALK": 30}
OUTPUTS = {"LANE": "map_lanes", "ROUTE_LANES": "route_lanes",
           "CROSSWALK": "map_crosswalks"}


@dataclass
class _Polyline:
    # Retain the owner: id() cannot silently be recycled while cached.
    owner: object
    world: torch.Tensor
    samples: dict = field(default_factory=dict)


class WorldMapCache:
    """Bounded immutable-world-polyline cache; no ego/traffic/ROI caching.

    Only persistent .polylines point lists receive identity caching, matching
    the host tensor_map.py assumption. Generic to_vector() output is transient.
    Call clear() if map geometry is edited in place.
    """

    def __init__(self, adapter, capacity=4096):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.adapter = adapter
        self.capacity = capacity
        self._entries = OrderedDict()
        self.hits = self.misses = 0

    def clear(self):
        self._entries.clear()

    def entries(self, feature):
        result = []
        if not hasattr(feature, "polylines"):
            return [_Polyline(None, torch.tensor(x, dtype=torch.float32))
                    for x in feature.to_vector()]
        for points in feature.polylines:
            key = id(points)
            entry = self._entries.get(key)
            if entry is None or entry.owner is not points:
                array = np.fromiter(
                    (v for point in points for v in (point.x, point.y)),
                    dtype=np.float32, count=2 * len(points)).reshape(-1, 2)
                entry = _Polyline(points, torch.from_numpy(array))
                self._entries[key] = entry
                if len(self._entries) > self.capacity:
                    self._entries.popitem(last=False)
            else:
                self._entries.move_to_end(key)
            result.append(entry)
        return result

    def sample(self, entry, count):
        # Cache output of the *original* interpolation, preserving its rounding.
        key = (count, "linear")
        cached = entry.samples.get(key)
        if cached is None:
            self.misses += 1
            cached = self.adapter.interpolate_points(
                entry.world, count, interpolation="linear").clone()
            entry.samples[key] = cached
        else:
            self.hits += 1
        return cached

    def fixed(self, feature, traffic, anchor, layer):
        entries = self.entries(feature)
        lights = (None if traffic is None else [
            torch.tensor(x, dtype=torch.float32) for x in traffic.to_vector()])
        if lights is not None and len(entries) != len(lights):
            raise ValueError("Size between feature coords and traffic light data inconsistent")
        # Preserve Python's stable ordering and the original float32 norm/min.
        # No approximate GPU ranking: a near-tie can change selected lane slots.
        ordering = sorted(range(len(entries)), key=lambda i: torch.norm(
            entries[i].world - anchor[None, :2], dim=-1).min())
        selected = ordering[:ELEMENTS[layer]]
        shape = (ELEMENTS[layer], POINTS[layer])
        coords = torch.zeros((*shape, 2), dtype=torch.float32)
        avail = torch.zeros(shape, dtype=torch.bool)
        tl = (None if lights is None else torch.zeros(
            (*shape, self.adapter.LaneSegmentTrafficLightData.encoding_dim()),
            dtype=torch.float32))
        for slot, index in enumerate(selected):
            coords[slot] = self.sample(entries[index], POINTS[layer])
            avail[slot] = True
            if tl is not None:
                tl[slot] = lights[index]
        return coords, avail, tl, len(selected)


def batch_polyline_features(coords, avail, traffic=None):
    """Original atan2/fmod sequence, batched; supports CPU and CUDA tensors."""
    delta = coords[:, 1:] - coords[:, :-1]
    heading = torch.atan2(delta[..., 1], delta[..., 0])
    heading = torch.fmod(heading, 2 * torch.pi)
    heading = torch.cat((heading, heading[:, -1:]), dim=1).unsqueeze(-1)
    parts = (coords, heading) if traffic is None else (coords, heading, traffic)
    values = torch.cat(parts, dim=-1)
    return torch.where(avail[:, :1, None], values, torch.zeros_like(values))


def pack_to_device(values, device):
    """One float32 H2D allocation/copy, device views; no device scalar reads."""
    names = tuple(values)
    flat = torch.cat([values[k].float().reshape(-1) for k in names])
    # No explicit stream/device synchronization. PyTorch owns the source copy's
    # lifetime; no reused mutable pinned staging buffer is exposed to callers.
    packed = flat.to(device, non_blocking=True)
    outputs = {}
    offset = 0
    for name in names:
        count = values[name].numel()
        outputs[name] = packed[offset:offset + count].view(values[name].shape)
        offset += count
    return outputs


def map_affine_payload(anchor):
    """Original host math trig + NumPy inverse, once per frame, not per layer.

    The 9 double coefficients are reinterpreted as 18 floats for one H2D pack.
    No float64-to-float32 conversion occurs. Put this at offset zero in the pack
    so Tensor.view(float64) satisfies the dtype alignment requirement.
    """
    x, y, heading = anchor.tolist()  # CPU anchor already rounded to float32.
    cosine, sine = math.cos(heading), math.sin(heading)
    transform = np.array([[cosine, -sine, x], [sine, cosine, y], [0., 0., 1.]],
                         dtype=np.float64)
    inverse = torch.from_numpy(np.linalg.inv(transform)).contiguous()
    return inverse.view(torch.float32).reshape(-1)


def device_map_local(world, affine_payload):
    """One batched FP64 homogeneous matmul; no device-to-host reads or inverse.

    Inputs keep original float32 world-coordinate rounding. FP64 CPU/CUDA
    matmul reduction can differ at last bits; feature/prediction QA is required.
    """
    affine = affine_payload.view(torch.float64).reshape(3, 3)
    homogeneous = torch.nn.functional.pad(world.double(), (0, 1), value=1.)
    return torch.matmul(affine, homogeneous.transpose(0, 1)).transpose(0, 1)[:, :2].float()


class DTPPFeatureBuilder:
    def __init__(self, adapter, cache_capacity=4096, map_mode="cuda"):
        if map_mode not in ("cpu", "cuda"):
            raise ValueError("map_mode must be 'cpu' or 'cuda'")
        self.adapter = adapter
        self.cache = WorldMapCache(adapter, cache_capacity)
        self.map_mode = map_mode
        self.last_stats = {}

    def _agents_and_map(self, inp, initialization):
        begin = time.perf_counter()
        adapter = self.adapter
        h = inp.history
        history = SimpleNamespace(
            ego_state_buffer=list(h.ego_states)[-22:],
            observation_buffer=list(h.observations)[-22:],
            current_state=h.current_state)
        ego = adapter.sampled_past_ego_states_to_tensor(history.ego_state_buffer)
        agents, types = adapter.sampled_tracked_objects_to_tensor_list(
            history.observation_buffer)
        # agent_past_process mutates these tensors; mapping needs world values.
        raw_anchor, raw_current = ego[-1].clone(), agents[-1].clone()
        stamps = adapter.sampled_past_timestamps_to_tensor(
            [state.time_point for state in history.ego_state_buffer])
        extracted = time.perf_counter()
        state = history.current_state[0]
        coords, traffic = adapter.get_neighbor_vector_set_map(
            initialization.map_api, list(LAYERS),
            adapter.Point2D(state.rear_axle.x, state.rear_axle.y), 80,
            initialization.route_roadblock_ids, list(inp.traffic_light_data or []))
        queried = time.perf_counter()
        # Deliberately reuse nearest-20, behind cutoff, padding, type features,
        # least-squares yaw rate and float64 local transform from the exact host.
        ego, neighbors = adapter.agent_past_process(ego, stamps, agents, types, 20)
        agent_done = time.perf_counter()
        anchor = torch.tensor([state.rear_axle.x, state.rear_axle.y,
                               state.rear_axle.heading], dtype=torch.float32)
        fixed = {layer: self.cache.fixed(coords[layer], traffic.get(layer), anchor, layer)
                 for layer in LAYERS}
        self.last_stats = {
            "history_extract_cpu_ms": (extracted - begin) * 1000,
            "map_query_cpu_ms": (queried - extracted) * 1000,
            "agent_process_cpu_ms": (agent_done - queried) * 1000,
            "map_select_sample_cpu_ms": (time.perf_counter() - agent_done) * 1000,
            "map_sample_cache_hits_total": self.cache.hits,
            "map_sample_cache_misses_total": self.cache.misses,
        }
        features = {"ego_agent_past": ego[1:],
                    "neighbor_agents_past": neighbors[:, 1:]}
        return features, fixed, anchor, raw_anchor, raw_current

    def _mapping(self, inp, features, anchor, raw_current):
        adapter = self.adapter
        current = inp.history.observations[-1].tracked_objects.get_tracked_objects_of_types([
            adapter.TrackedObjectType.VEHICLE, adapter.TrackedObjectType.PEDESTRIAN,
            adapter.TrackedObjectType.BICYCLE])
        slot = features["neighbor_agents_past"][:10, -1].numpy()
        present = np.flatnonzero(abs(slot).sum(-1) > 0)
        mapping = {}
        if len(present):
            rel = adapter.convert_absolute_quantities_to_relative(raw_current, anchor, "agent")
            centers = rel[:, [adapter.AgentInternalIndex.x(),
                              adapter.AgentInternalIndex.y()]].numpy()
            cost = np.linalg.norm(slot[present, None, :2] - centers[None], axis=-1)
            rows, columns = linear_sum_assignment(cost)
            assert len(rows) == len(present) and np.max(cost[rows, columns]) < .02
            mapping = {current[k].track_token: int(present[j])
                       for j, k in zip(rows, columns)}
        return mapping, [o for o in current if o.track_token in mapping]

    @torch.inference_mode()
    def prepare(self, predictor, inp, initialization):
        begin = time.perf_counter()
        # Mandatory per-frame reset: predictions depend on the new context.
        predictor._shared_cache = {}
        features, fixed, anchor, raw_anchor, raw_current = self._agents_and_map(inp, initialization)
        predictor.state = inp.history.ego_states[-1]
        mapping_start = time.perf_counter()
        predictor.mapping, predictor.tracks = self._mapping(
            inp, features, raw_anchor, raw_current)
        self.last_stats["mapping_cpu_ms"] = (time.perf_counter() - mapping_start) * 1000
        upload_start = time.perf_counter()
        if self.map_mode == "cpu":
            for layer, (world, avail, traffic, _) in fixed.items():
                local = self.adapter.vector_set_coordinates_to_local_frame(world, avail, anchor)
                features[OUTPUTS[layer]] = batch_polyline_features(local, avail, traffic)
            features = pack_to_device(features, predictor.device)
        else:
            # Agent features and all selected world samples/traffic/anchor share
            # one upload. Map intermediates/results stay on the encoder device.
            if torch.device(predictor.device).type != "cuda":
                raise ValueError("map_mode='cuda' requires a CUDA encoder device")
            # Keep the affine at offset zero for FP64 reinterpretation alignment.
            payload = {"_affine": map_affine_payload(anchor), **features}
            payload["_world"] = torch.cat([fixed[layer][0].reshape(-1, 2)
                                            for layer in LAYERS])
            for layer, (world, _, traffic, _) in fixed.items():
                if traffic is not None:
                    payload["_traffic_" + layer] = traffic
            device = pack_to_device(payload, predictor.device)
            all_local = device_map_local(device["_world"], device["_affine"])
            features = {name: device[name] for name in features}
            offset = 0
            for layer, (_, _, traffic, count) in fixed.items():
                npoints = ELEMENTS[layer] * POINTS[layer]
                local = all_local[offset:offset + npoints].view(ELEMENTS[layer], POINTS[layer], 2)
                offset += npoints
                avail = (torch.arange(ELEMENTS[layer], device=all_local.device) < count)
                avail = avail[:, None].expand(-1, POINTS[layer])
                features[OUTPUTS[layer]] = batch_polyline_features(
                    local, avail, None if traffic is None else device["_traffic_" + layer])
        predictor.features = {name: value.unsqueeze(0) for name, value in features.items()}
        self.last_stats["map_transform_pack_dispatch_ms"] = (time.perf_counter() - upload_start) * 1000
        encoder_start = time.perf_counter()
        predictor.context = predictor.enc(predictor.features)
        # These are CPU wall/dispatch timings, not synchronized GPU durations.
        self.last_stats["encoder_dispatch_ms"] = (time.perf_counter() - encoder_start) * 1000
        self.last_stats["prepare_dispatch_ms"] = (time.perf_counter() - begin) * 1000
        self.last_stats["mode"] = "cached_cpu" if self.map_mode == "cpu" else "device_map"
        predictor._dtpp_feature_stats = dict(self.last_stats)


def install(predictor, *, adapter=None, cache_capacity=4096, map_mode="cuda"):
    """Bind only this predictor instance; existing evaluate() remains untouched.

    Return the builder (stats/cache). Call uninstall(predictor) to restore.
    map_mode='cuda' is the candidate default; map_mode='cpu' is the reference.
    """
    if hasattr(predictor, "_dtpp_original_prepare"):
        raise ValueError("DTPP feature builder is already installed on this predictor")
    if adapter is None:
        import obs_adapter as adapter
    builder = DTPPFeatureBuilder(adapter, cache_capacity, map_mode)
    predictor._dtpp_original_prepare = predictor.prepare
    predictor._dtpp_feature_builder = builder

    def prepare(instance, inp, initialization):
        return builder.prepare(instance, inp, initialization)

    predictor.prepare = MethodType(prepare, predictor)
    return builder


def uninstall(predictor):
    predictor.prepare = predictor._dtpp_original_prepare
    del predictor._dtpp_original_prepare
    del predictor._dtpp_feature_builder
