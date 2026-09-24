"""Connect device agent history to the accepted DTPP map-feature preparation.

The predictor already has dtpp_features installed. Only this instance's prepare
is replaced; encoder, decoder, evaluate and candidate conditioning stay intact.
"""
from types import MethodType
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from dtpp_features.dtpp_prepare import (
    DTPPFeatureBuilder, LAYERS, ELEMENTS, POINTS, OUTPUTS,
    map_affine_payload, pack_to_device, device_map_local,
    batch_polyline_features,
)
from .core import AgentHistoryBuilder


def host_array(value):
    """Metadata is host-side by contract: never silently download a tensor."""
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            raise ValueError("Agent history mapping metadata must stay on CPU")
        return value.detach().numpy()
    return np.asarray(value)


def mapping_from_metadata(adapter, inp, metadata):
    current = inp.history.observations[-1].tracked_objects.get_tracked_objects_of_types([
        adapter.TrackedObjectType.VEHICLE,
        adapter.TrackedObjectType.PEDESTRIAN,
        adapter.TrackedObjectType.BICYCLE,
    ])
    centers = host_array(metadata["current_centers"])
    slots = host_array(metadata["slot_centers"])
    present = np.flatnonzero(host_array(metadata["slot_present"])[:10])
    if centers.shape != (len(current), 2) or slots.shape != (20, 2):
        raise ValueError("Agent history metadata changed current-row/slot shape")
    mapping = {}
    if len(present):
        cost = np.linalg.norm(slots[present, None, :] - centers[None], axis=-1)
        rows, columns = linear_sum_assignment(cost)
        # Keep the existing predictor's exact Hungarian association and gate.
        assert len(rows) == len(present) and np.max(cost[rows, columns]) < .02
        mapping = {current[k].track_token: int(present[j])
                   for j, k in zip(rows, columns)}
    return mapping, [obj for obj in current if obj.track_token in mapping]


class HistoryFeatureBuilder(DTPPFeatureBuilder):
    def __init__(self, previous_builder, device):
        super().__init__(previous_builder.adapter, map_mode="cuda")
        # Reuse only immutable map samples, never ego-local data or predictions.
        self.cache = previous_builder.cache
        self.history_builder = AgentHistoryBuilder(self.adapter, device=device)

    @torch.inference_mode()
    def prepare(self, predictor, inp, initialization):
        started = time.perf_counter()
        adapter = self.adapter
        predictor._shared_cache = {}
        history = inp.history
        states = list(history.ego_states)[-22:]
        observations = list(history.observations)[-22:]
        ego = adapter.sampled_past_ego_states_to_tensor(states)
        agents, types = adapter.sampled_tracked_objects_to_tensor_list(observations)
        stamps = adapter.sampled_past_timestamps_to_tensor([s.time_point for s in states])
        extracted = time.perf_counter()
        state = history.current_state[0]
        coords, traffic = adapter.get_neighbor_vector_set_map(
            initialization.map_api, list(LAYERS),
            adapter.Point2D(state.rear_axle.x, state.rear_axle.y), 80,
            initialization.route_roadblock_ids, list(inp.traffic_light_data or []))
        queried = time.perf_counter()
        plan = self.history_builder.prepare_cpu(ego, stamps, agents, types, num_agents=20)
        planned = time.perf_counter()
        predictor.state = history.ego_states[-1]
        predictor.mapping, predictor.tracks = mapping_from_metadata(
            adapter, inp, plan.metadata_cpu)
        mapped = time.perf_counter()
        anchor = torch.tensor([state.rear_axle.x, state.rear_axle.y,
                               state.rear_axle.heading], dtype=torch.float32)
        fixed = {layer: self.cache.fixed(coords[layer], traffic.get(layer), anchor, layer)
                 for layer in LAYERS}
        sampled = time.perf_counter()

        # The accepted map affine occupies offset zero for its lossless FP64 view.
        # Agent raw values/gather indices join the same float32 bulk upload.
        payload = {"_affine": map_affine_payload(anchor)}
        for name, value in plan.payload.items():
            if value.device.type != "cpu" or value.dtype != torch.float32:
                raise ValueError("History bulk payload must be CPU float32: " + name)
            payload["_history_" + name] = value
        payload["_world"] = torch.cat([fixed[layer][0].reshape(-1, 2) for layer in LAYERS])
        for layer, (_, _, lights, _) in fixed.items():
            if lights is not None:
                payload["_traffic_" + layer] = lights
        device = pack_to_device(payload, predictor.device)
        history_device = {name: device["_history_" + name] for name in plan.payload}
        history_result = self.history_builder.compute_device(plan, device_payload=history_device)
        history_dispatched = time.perf_counter()
        features = {
            "ego_agent_past": history_result["ego_gpu"][1:],
            "neighbor_agents_past": history_result["neighbors_gpu"][:, 1:],
        }
        all_local = device_map_local(device["_world"], device["_affine"])
        offset = 0
        for layer, (_, _, lights, count) in fixed.items():
            npoints = ELEMENTS[layer] * POINTS[layer]
            local = all_local[offset:offset + npoints].view(ELEMENTS[layer], POINTS[layer], 2)
            offset += npoints
            available = (torch.arange(ELEMENTS[layer], device=all_local.device) < count)
            available = available[:, None].expand(-1, POINTS[layer])
            features[OUTPUTS[layer]] = batch_polyline_features(
                local, available, None if lights is None else device["_traffic_" + layer])
        predictor.features = {name: value.unsqueeze(0) for name, value in features.items()}
        feature_dispatched = time.perf_counter()
        predictor.context = predictor.enc(predictor.features)
        completed_dispatch = time.perf_counter()
        self.last_stats = {
            "history_extract_cpu_ms": (extracted - started) * 1000,
            "map_query_cpu_ms": (queried - extracted) * 1000,
            "history_plan_cpu_ms": (planned - queried) * 1000,
            "mapping_cpu_ms": (mapped - planned) * 1000,
            "map_select_sample_cpu_ms": (sampled - mapped) * 1000,
            "history_upload_compute_dispatch_ms": (history_dispatched - sampled) * 1000,
            "map_compute_dispatch_ms": (feature_dispatched - history_dispatched) * 1000,
            "encoder_dispatch_ms": (completed_dispatch - feature_dispatched) * 1000,
            "prepare_dispatch_ms": (completed_dispatch - started) * 1000,
            "main_payload_h2d_bytes": sum(x.numel() * x.element_size() for x in payload.values()),
            "history_plan": dict(plan.stats),
            "mode": "gpu_agent_history_and_map",
            "timing_scope": "host dispatch; not synchronized GPU durations",
        }
        predictor._agent_history_stats = dict(self.last_stats)
        predictor._dtpp_feature_stats = dict(self.last_stats)


def install(predictor):
    """Replace the already-accepted device-map prepare on one predictor only."""
    if hasattr(predictor, "_agent_history_previous_prepare"):
        raise ValueError("Agent history integration is already installed")
    previous_builder = getattr(predictor, "_dtpp_feature_builder", None)
    if previous_builder is None or previous_builder.map_mode != "cuda":
        raise ValueError("Install accepted dtpp_features CUDA preparation first")
    if torch.device(predictor.device).type != "cuda":
        raise ValueError("Agent history integration requires CUDA")
    builder = HistoryFeatureBuilder(previous_builder, predictor.device)
    predictor._agent_history_previous_prepare = predictor.prepare
    predictor._agent_history_previous_builder = previous_builder
    predictor._dtpp_feature_builder = builder
    predictor._agent_history_builder = builder

    def prepare(instance, inp, initialization):
        return builder.prepare(instance, inp, initialization)

    predictor.prepare = MethodType(prepare, predictor)
    return builder


def uninstall(predictor):
    predictor.prepare = predictor._agent_history_previous_prepare
    predictor._dtpp_feature_builder = predictor._agent_history_previous_builder
    for name in ("_agent_history_previous_prepare", "_agent_history_previous_builder",
                 "_agent_history_builder"):
        delattr(predictor, name)
