"""Boundary tests and host-injected real-frame parity for DTPP feature prepare.

This module never loads weights or launches CUDA at import time. Real scenarios
are supplied by the execution harness through validate_sequence().
"""
import argparse
import importlib
import json
from types import SimpleNamespace

import numpy as np
import torch

try:
    from .dtpp_prepare import (ELEMENTS, POINTS, LAYERS, OUTPUTS, WorldMapCache,
                              batch_polyline_features, map_affine_payload,
                              device_map_local, pack_to_device, DTPPFeatureBuilder, install, uninstall)
except ImportError:
    from dtpp_prepare import (ELEMENTS, POINTS, LAYERS, OUTPUTS, WorldMapCache,
                             batch_polyline_features, map_affine_payload,
                             device_map_local, pack_to_device, DTPPFeatureBuilder, install, uninstall)


def snapshot(value):
    """Validation-only downloads; never called by production prepare."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(snapshot(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def compare(reference, candidate, path="value", atol=0., rtol=0.):
    """Return compact tensor error statistics and raise on any failed gate."""
    result = {}
    if torch.is_tensor(reference):
        assert torch.is_tensor(candidate), path
        assert reference.shape == candidate.shape and reference.dtype == candidate.dtype, path
        if reference.is_floating_point():
            assert torch.equal(torch.isfinite(reference), torch.isfinite(candidate)), path
            torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol, equal_nan=True,
                                       msg=lambda msg: f"{path}: {msg}")
            finite = torch.isfinite(reference)
            error = (reference[finite] - candidate[finite]).abs()
            result[path] = {"max_abs": float(error.max()) if error.numel() else 0.,
                            "bitwise_equal": torch.equal(reference, candidate)}
        else:
            assert torch.equal(reference, candidate), path
    elif isinstance(reference, dict):
        assert reference.keys() == candidate.keys(), path
        for key in reference:
            result.update(compare(reference[key], candidate[key], path + "." + str(key), atol, rtol))
    elif isinstance(reference, (list, tuple)):
        assert len(reference) == len(candidate), path
        for index, (left, right) in enumerate(zip(reference, candidate)):
            result.update(compare(left, right, f"{path}[{index}]", atol, rtol))
    elif isinstance(reference, np.ndarray):
        np.testing.assert_allclose(candidate, reference, atol=atol, rtol=rtol, equal_nan=True,
                                   err_msg=path)
    elif isinstance(reference, (float, np.floating)):
        assert np.isclose(reference, candidate, atol=atol, rtol=rtol, equal_nan=True), (path, reference, candidate)
    else:
        assert reference == candidate, (path, reference, candidate)
    return result


class _Feature:
    def __init__(self, arrays):
        self.polylines = [[SimpleNamespace(x=float(x), y=float(y)) for x, y in line]
                          for line in arrays]

    def to_vector(self):
        return [[[p.x, p.y] for p in line] for line in self.polylines]


class _Traffic:
    def __init__(self, rows):
        self.rows = rows

    def to_vector(self):
        return self.rows


def synthetic(adapter, device="cpu", *, feature_atol=1e-5):
    """Compare original host fixed-size/transform/polyline functions.

    Covers stable distance ties, capacity boundary, empty/padded layers, large
    coordinates, degenerate/reversed lines, changed anchor and traffic, cache
    eviction, no stale feature aliases, and lossless affine payload upload.
    """
    rng = np.random.default_rng(9273)
    cache = WorldMapCache(adapter, capacity=4096)
    report = {}
    scenarios = [
        ("empty", [], [0., 0., 0.]),
        ("tie_over_capacity", [np.array([[float(i % 2) * 2 - 1, 0.],
                                        [float(i % 2) * 2 - 1, 2.]]) for i in range(48)],
         [0., 0., 0.]),
        ("degenerate", [np.array([[0., 0.], [0., 0.]]),
                        np.array([[2., 0.], [1., 0.], [0., 0.]]),
                        np.array([[0., 0.], [-1., -0.], [-2., -0.]])], [0., 0., 0.]),
        ("large_world", [rng.normal(size=(i % 8 + 2, 2)) * 20 + [523417., 4140931.]
                         for i in range(45)], [523417.125, 4140931.25, 1.273]),
    ]
    for scenario, arrays, anchor_values in scenarios:
        feature = _Feature(arrays)
        traffic = _Traffic([[i % 2, (i + 1) % 2, 0., 0.] for i in range(len(arrays))])
        anchor = torch.tensor(anchor_values, dtype=torch.float32)
        for repetition in range(2):
            # Repeated call changes the frame; world interpolation alone is reusable.
            current_anchor = anchor + torch.tensor([repetition * .5, repetition * .25,
                                                   repetition * .003], dtype=torch.float32)
            if repetition:
                traffic.rows = [[0., 0., 1., 0.] for _ in arrays]
            for layer in LAYERS:
                tl = traffic if layer == "LANE" else None
                world, avail, lights, count = cache.fixed(feature, tl, current_anchor, layer)
                original_lines = [torch.tensor(line, dtype=torch.float32)
                                  for line in feature.to_vector()]
                original_lights = (None if tl is None else
                                   [torch.tensor(x, dtype=torch.float32) for x in tl.to_vector()])
                ref_world, ref_lights, ref_avail = adapter.convert_feature_layer_to_fixed_size(
                    current_anchor, original_lines, original_lights,
                    ELEMENTS[layer], POINTS[layer],
                    adapter.LaneSegmentTrafficLightData.encoding_dim(), "linear")
                key = f"{scenario}.{repetition}.{layer}"
                compare((ref_world, ref_avail, ref_lights), (world, avail, lights), key + ".fixed")
                original_local = adapter.vector_set_coordinates_to_local_frame(
                    ref_world, ref_avail, current_anchor)
                expected = adapter.polyline_process(original_local, ref_avail, ref_lights)
                cpu_batch = batch_polyline_features(original_local, ref_avail, ref_lights)
                compare(expected, cpu_batch, key + ".cpu_batch")
                # The byte-reinterpreted affine must survive float32 packing unchanged.
                affine = map_affine_payload(current_anchor)
                packed = pack_to_device({"affine": affine, "world": world.reshape(-1, 2)}, device)
                assert torch.equal(affine.view(torch.int32), packed["affine"].cpu().view(torch.int32))
                local = device_map_local(packed["world"], packed["affine"]).view(world.shape)
                actual = batch_polyline_features(local, avail.to(device),
                                                 None if lights is None else lights.to(device)).cpu()
                report.update(compare(expected, actual, key + ".device", feature_atol, 0.))
                assert torch.equal(actual[count:], torch.zeros_like(actual[count:])), key
                # A following call cannot mutate previously returned tensors.
                saved = world.clone()
                cache.fixed(feature, tl, current_anchor + 1., layer)
                assert torch.equal(world, saved), key
    assert cache.hits > 0 and cache.misses > 0
    small = WorldMapCache(adapter, capacity=1)
    old_feature = _Feature([np.array([[1., 2.], [3., 4.]])])
    first = small.fixed(old_feature, None, torch.zeros(3), "ROUTE_LANES")[0]
    small.fixed(_Feature([np.array([[10., 2.], [30., 4.]])]), None,
                torch.zeros(3), "ROUTE_LANES")
    restored = small.fixed(old_feature, None, torch.zeros(3), "ROUTE_LANES")[0]
    assert torch.equal(first, restored)
    # Generic to_vector objects are intentionally not identity-cached.
    generic = _Traffic([[[1., 2.], [3., 4.]]])
    before = small.fixed(generic, None, torch.zeros(3), "ROUTE_LANES")[0]
    generic.rows = [[[8., 9.], [10., 11.]]]
    after = small.fixed(generic, None, torch.zeros(3), "ROUTE_LANES")[0]
    assert not torch.equal(before, after)
    mapping_report = mapping_boundary(adapter)
    return {"passed": True, "device": device, "feature_atol": feature_atol,
            "mapping_boundaries": mapping_report,
            "cache_hits": cache.hits, "cache_misses": cache.misses, "errors": report}


def mapping_boundary(adapter):
    """Use original agent history processing; compare reused vs fresh world rows.

    Includes >20 current actors, the x=-6 cutoff, equal/duplicate centers, sparse
    history, nonzero global token IDs and no-actor output. No nearest rule is
    reimplemented by the candidate.
    """
    from scipy.optimize import linear_sum_assignment
    agent_index, ego_index = adapter.AgentInternalIndex, adapter.EgoInternalIndex
    builder = DTPPFeatureBuilder(adapter, map_mode="cpu")
    summaries = []
    for actor_count in (0, 25):
        raw = torch.zeros(actor_count, agent_index.dim(), dtype=torch.float32)
        if actor_count:
            raw[:, agent_index.track_token()] = torch.arange(actor_count) + 100
            raw[:, agent_index.x()] = torch.arange(actor_count) + 1
            raw[:, agent_index.width()] = 2.
            raw[:, agent_index.length()] = 4.
            raw[0, agent_index.x()] = -6.01
            raw[1, agent_index.x()] = -5.99
            raw[2, agent_index.x()] = 5.99
            raw[3, agent_index.x()] = 6.01
            raw[6, agent_index.x()] = raw[5, agent_index.x()]
        ego = torch.zeros(22, ego_index.dim(), dtype=torch.float32)
        anchor = ego[-1].clone()
        history = [raw.clone() for _ in range(22)]
        if actor_count:
            history[0] = history[0][:-1]  # Padding a newly seen actor is unchanged.
        actor_type = adapter.TrackedObjectType.VEHICLE
        type_history = [[actor_type] * len(frame) for frame in history]
        stamps = adapter.sampled_past_timestamps_to_tensor([
            SimpleNamespace(time_us=i * 100000) for i in range(22)])
        _, neighbors = adapter.agent_past_process(ego, stamps, history, type_history, 20)
        current = [SimpleNamespace(track_token=f"actor-{i}") for i in range(actor_count)]
        objects = SimpleNamespace(get_tracked_objects_of_types=lambda _types: current)
        inp = SimpleNamespace(history=SimpleNamespace(
            observations=[SimpleNamespace(tracked_objects=objects)]))
        features = {"neighbor_agents_past": neighbors[:, 1:]}
        actual, tracks = builder._mapping(inp, features, anchor.clone(), raw.clone())
        # Fresh extract in Base.prepare assigns 0-based IDs instead of history IDs.
        fresh = raw.clone()
        fresh[:, agent_index.track_token()] = torch.arange(actor_count)
        relative = adapter.convert_absolute_quantities_to_relative(fresh, anchor.clone(), "agent")
        slots = neighbors[:10, -1].numpy()
        present = np.flatnonzero(abs(slots).sum(-1) > 0)
        reference = {}
        if len(present):
            centers = relative[:, [agent_index.x(), agent_index.y()]].numpy()
            distances = np.linalg.norm(slots[present, None, :2] - centers[None], axis=-1)
            rows, columns = linear_sum_assignment(distances)
            assert len(rows) == len(present) and np.max(distances[rows, columns]) < .02
            reference = {current[k].track_token: int(present[j]) for j, k in zip(rows, columns)}
        assert actual == reference
        assert [o.track_token for o in tracks] == [o.track_token for o in current if o.track_token in reference]
        assert len(actual) <= 10
        if actor_count:
            assert "actor-0" not in actual, "behind cutoff changed"
        summaries.append({"current_actors": actor_count, "predicted_slots": len(actual)})
    return summaries


def _capture(predictor, inp, initialization, prepare, trajectory_sets):
    # Capture decoder outputs, including the n==1 branch that has no shared cache.
    decoded = []
    hook = predictor.dec.register_forward_hook(lambda _module, _args, output: decoded.append(snapshot(output)))
    try:
        predictor._shared_cache = {"validation_previous_frame": object()}
        prepare(inp, initialization)
        assert not predictor._shared_cache, "prepare reused predictions from a previous frame"
        features, context = snapshot(predictor.features), snapshot(predictor.context)
        mapping = dict(predictor.mapping)
        tracks = [o.track_token for o in predictor.tracks]
        evaluations = []
        for trajectories in trajectory_sets:
            bad, diagnostics = predictor.evaluate(trajectories)
            # The public diagnostics contains elapsed wall time, which is not semantic.
            evaluations.append((snapshot(bad), snapshot({k: v for k, v in diagnostics.items()
                                                         if k != "seconds"})))
        return dict(features=features, context=context, mapping=mapping,
                    tracks=tracks, decoded=decoded, evaluations=evaluations)
    finally:
        hook.remove()


def validate_sequence(predictor, cases, *, adapter=None, map_mode="cuda",
                      feature_atol=1e-5, encoder_atol=1e-5, prediction_atol=1e-5):
    """Run exact R3 prepare and candidate on the same existing predictor weights.

    cases: iterable of (label, input, initialization, trajectory_sets).
    Each trajectory_sets must exercise both n==1 and n>1 evaluate branches,
    normally with both 31 and 81 timestamps. Supply adjacent frames plus changed
    ego/traffic/agent inputs: same-frame repeats alone are insufficient.

    CPU reference mode is required to be bitwise identical. GPU tolerances are
    explicit gates, not silent rounding. Collision/rejection booleans, slot
    identities, actor masks and diagnostic counts always require exact equality.
    Run this before timing; its deliberate downloads/hooks are not production.
    """
    if map_mode == "cpu":
        feature_atol = encoder_atol = prediction_atol = 0.
    if hasattr(predictor, "_dtpp_original_prepare"):
        raise ValueError("validate_sequence expects an unpatched predictor instance")
    builder = install(predictor, adapter=adapter, map_mode=map_mode)
    reports = []
    try:
        for label, inp, initialization, trajectories in cases:
            trajectories = list(trajectories)
            assert any(len(x) == 1 for x in trajectories), "missing single-path conditioning case"
            assert any(len(x) > 1 for x in trajectories), "missing shared-ego conditioning case"
            reference = _capture(predictor, inp, initialization,
                                 predictor._dtpp_original_prepare, trajectories)
            candidate = _capture(predictor, inp, initialization, predictor.prepare, trajectories)
            errors = compare(reference["features"], candidate["features"],
                             str(label) + ".features", feature_atol)
            errors.update(compare(reference["context"], candidate["context"],
                                  str(label) + ".context", encoder_atol))
            errors.update(compare(reference["decoded"], candidate["decoded"],
                                  str(label) + ".decoded", prediction_atol))
            compare(reference["mapping"], candidate["mapping"], str(label) + ".mapping")
            compare(reference["tracks"], candidate["tracks"], str(label) + ".tracks")
            # Small prediction differences must not change downstream safety outputs.
            compare(reference["evaluations"], candidate["evaluations"],
                    str(label) + ".evaluate", prediction_atol)
            reports.append({"label": str(label), "passed": True, "errors": errors,
                            "dispatch_stats": dict(builder.last_stats)})
        assert reports, "at least one case is required"
        return {"passed": True, "map_mode": map_mode, "cases": reports}
    finally:
        uninstall(predictor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bootstrap", action="append", default=[],
                        help="Import module, or module:function() to install exact host patches")
    parser.add_argument("--feature-atol", type=float, default=1e-5)
    parser.add_argument("--output")
    args = parser.parse_args()
    for name in args.bootstrap:
        module_name, _, function = name.partition(":")
        module = importlib.import_module(module_name)
        if function:
            getattr(module, function)()
    adapter = importlib.import_module("obs_adapter")
    report = synthetic(adapter, args.device, feature_atol=args.feature_atol)
    encoded = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
