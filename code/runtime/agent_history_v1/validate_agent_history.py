"""Exact-host actor-history parity. CUDA runs belong to the execution owner.

Importing this module neither loads weights nor starts a CUDA job. The reference
is the live adapter with the application's original patches already installed.
"""
import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import time
from types import FunctionType

import numpy as np
import torch


@dataclass
class Case:
    name: str
    ego: torch.Tensor
    stamps: torch.Tensor
    frames: list
    types: list
    check_mapping: bool = True


def clone_case(case):
    return Case(case.name, case.ego.clone(), case.stamps.clone(),
                [x.clone() for x in case.frames], [list(x) for x in case.types], case.check_mapping)


def cpu(value):
    """Validation-only readback; never called from the candidate."""
    return value.detach().cpu().clone()


def tensor_error(expected, actual):
    actual = cpu(actual)
    assert expected.shape == actual.shape, (expected.shape, actual.shape)
    assert expected.dtype == actual.dtype, (expected.dtype, actual.dtype)
    finite = torch.isfinite(expected)
    assert torch.equal(finite, torch.isfinite(actual)), "finite mask differs"
    error = (expected[finite].double() - actual[finite].double()).abs()
    return {"max_abs": float(error.max()) if error.numel() else 0.,
            "differing_values": int(torch.count_nonzero(expected != actual)),
            "shape": list(expected.shape), "dtype": str(expected.dtype)}


def exact_reference(adapter, case):
    """Trace reference internals using copied globals; do not monkeypatch adapter."""
    fn = adapter.agent_past_process
    trace = {}
    original_pad = fn.__globals__["pad_agent_states"]
    original_pack = fn.__globals__["pack_agents_tensor"]

    def pad(*args, **kwargs):
        result = original_pad(*args, **kwargs)
        trace["padded_world"] = torch.stack([x.clone() for x in result])
        return result

    def pack(states, yaws):
        trace["local_history"] = torch.stack([x.clone() for x in states])
        trace["yaw_rates"] = yaws.clone()
        result = original_pack(states, yaws)
        trace["packed_history"] = result.clone()
        return result

    globals_copy = dict(fn.__globals__, pad_agent_states=pad, pack_agents_tensor=pack)
    reference = FunctionType(fn.__code__, globals_copy, fn.__name__, fn.__defaults__, fn.__closure__)
    inputs = clone_case(case)
    ego, neighbors = reference(inputs.ego, inputs.stamps, inputs.frames, inputs.types, 20)
    trace["ego"], trace["neighbors"] = ego.clone(), neighbors.clone()
    if len(case.frames[-1]):
        packed = trace["packed_history"]
        # Exact reference operation/order: no stable=True/topk and no backfill.
        ordered = list(torch.argsort(torch.norm(packed[-1, :, :2], dim=-1)).numpy())[:20]
        selected = [i for i in ordered if not bool(packed[-1, i, 0] < -6.)]
    else:
        selected = []
    trace["selected_rows"] = torch.tensor(selected, dtype=torch.int64)
    relative = adapter.convert_absolute_quantities_to_relative(case.frames[-1].clone(), case.ego[-1].clone(), "agent")
    I = adapter.AgentInternalIndex
    trace["current_centers"] = relative[:, [I.x(), I.y()]].clone()
    slots = neighbors[:, -1]
    trace["slot_centers"] = slots[:, :2].clone()
    trace["slot_present"] = torch.from_numpy(abs(slots.numpy()).sum(-1) > 0)
    return trace


def mapping(centers, slot_centers, present):
    """Original Hungarian mapping keyed by current row; includes duplicate centers."""
    from scipy.optimize import linear_sum_assignment
    centers, slot_centers, present = np.asarray(centers), np.asarray(slot_centers), np.asarray(present, np.int64)
    if not len(present):
        return {}
    costs = np.linalg.norm(slot_centers[present, None, :] - centers[None, :, :], axis=-1)
    rows, columns = linear_sum_assignment(costs)
    assert len(rows) == len(present) and np.max(costs[rows, columns]) < .02
    return {int(column): int(present[row]) for row, column in zip(rows, columns)}


def make_cases(adapter):
    """22-frame cases with explicit changes that can alter native selection/padding."""
    I, E = adapter.AgentInternalIndex, adapter.EgoInternalIndex
    object_types = [adapter.TrackedObjectType.VEHICLE, adapter.TrackedObjectType.PEDESTRIAN,
                    adapter.TrackedObjectType.BICYCLE]

    def build(name, count, *, origin=(0., 0.), heading=.0, sparse=False, irregular=False,
              special=False, duplicate_past=False, duplicate_current=False):
        T = 22
        intervals = np.array([83000, 117000, 101000, 97000, 103000] * 5)[:T - 1] if irregular else np.full(T - 1, 100000)
        microseconds = np.r_[0, np.cumsum(intervals)] + 1_650_000_000_000_000
        stamps = torch.tensor(microseconds, dtype=torch.int64)
        seconds = (microseconds - microseconds[-1]) * 1e-6
        ego = torch.zeros(T, E.dim(), dtype=torch.float32)
        ego[:, E.x()] = torch.tensor(origin[0] + seconds * .35)
        ego[:, E.y()] = float(origin[1])
        ego[:, E.heading()] = float(heading)
        ego[:, E.vx()], ego[:, E.vy()] = 1.25, -.35
        ego[:, E.ax()], ego[:, E.ay()] = .17, -.21
        frames, type_history = [], []
        for t in range(T):
            rows, types = [], []
            for j in range(count):
                if sparse and t < T - 1 and ((j == 0 and t < T - 1) or (j % 3 == 1 and t % 4 in (1, 2))):
                    continue
                row = torch.zeros(I.dim(), dtype=torch.float32)
                row[I.track_token()] = 1000 + j * 7
                x, y = 2. + j * .35 + seconds[t] * .1, (j % 3 - 1) * .5
                if special:
                    positions = [(-6.01, 0.), (-6., 0.), (-5.99, 0.), (3., 4.), (4., 3.),
                                 (5., 0.), (5., 0.), (0., 5.), (0., -5.), (-3., 4.), (-4., 3.)]
                    if j < len(positions):
                        x, y = positions[j]
                    else:
                        x, y = 6. + (j - 11) * .2, 0.
                c, s = np.cos(heading), np.sin(heading)
                row[I.x()], row[I.y()] = float(origin[0] + x * c - y * s), float(origin[1] + x * s + y * c)
                # Wrap in both directions; rows next to +/-pi exercise tie behavior.
                yaw = ((heading + 3.02 + (1 if j % 2 else -1) * .035 * t + np.pi) % (2 * np.pi)) - np.pi
                if j == 2:
                    yaw = (-np.pi, 0., np.pi)[t % 3]
                row[I.heading()] = float(yaw)
                row[I.vx()], row[I.vy()] = 1. + j * .01, -.4 + j * .005
                row[I.width()], row[I.length()] = 1.8 + j * .01, 3.9 + j * .02
                rows.append(row)
                types.append(object_types[j % 3])
                if duplicate_past and j == 1 and t == 7:
                    duplicate = row.clone()
                    duplicate[I.x()] += 1.
                    duplicate[I.heading()] -= .3
                    rows.append(duplicate)
                    types.append(object_types[j % 3])
            if sparse and t < T - 1:
                # Disappeared actor must be filtered before reverse padding.
                vanished = torch.zeros(I.dim(), dtype=torch.float32)
                vanished[I.track_token()] = 987654
                vanished[I.x()], vanished[I.width()], vanished[I.length()] = .1, 2., 4.
                rows.append(vanished)
                types.append(object_types[0])
            if duplicate_current and t == T - 1 and rows:
                rows.append(rows[0].clone())
                types.append(types[0])
            # Current order stays deterministic, while historical rows vary.
            if t % 2 == 0 and t < T - 1:
                rows, types = rows[::-1], types[::-1]
            frames.append(torch.stack(rows) if rows else torch.empty((0, I.dim()), dtype=torch.float32))
            type_history.append(types)
        return Case(name, ego, stamps, frames, type_history, not duplicate_current)

    cases = [
        build("no_current_agents", 0, sparse=True),
        build("one_agent_scalar_yaw", 1),
        build("mixed_types_dense", 8),
        build("sparse_birth_gap_death_reordered", 8, sparse=True),
        build("ties_cutoff_no_backfill_first10", 25, special=True),
        build("large_world_irregular_yaw_wrap", 25, origin=(523417., 4140931.), heading=1.273, irregular=True, sparse=True),
        build("heading_plus_pi", 8, heading=float(np.pi), irregular=True),
        build("heading_minus_pi", 8, heading=float(-np.pi), irregular=True),
        build("duplicate_past_last_row_wins", 8, duplicate_past=True),
        build("duplicate_current_native_padding", 8, duplicate_current=True),
    ]
    behind = build("all_current_agents_behind", 25)
    for frame in behind.frames:
        frame[:, I.x()] = -10. - torch.arange(len(frame), dtype=torch.float32)
        frame[:, I.y()] = 0.
    cases.append(behind)
    return cases


def _metadata_array(metadata, key):
    value = metadata[key]
    if torch.is_tensor(value):
        assert value.device.type == "cpu", f"{key} metadata unexpectedly on GPU"
        return value.detach().numpy()
    return np.asarray(value)


@torch.inference_mode()
def validate_case(adapter, builder, case, *, atol=0., artifacts=None):
    reference = exact_reference(adapter, case)
    original_inputs = clone_case(case)
    result = builder.build(case.ego, case.stamps, case.frames, case.types, num_agents=20)
    expected_device = torch.device(builder.device)
    assert result["ego_gpu"].device.type == expected_device.type
    assert result["neighbors_gpu"].device.type == expected_device.type
    errors = {"ego": tensor_error(reference["ego"], result["ego_gpu"]),
              "neighbors": tensor_error(reference["neighbors"], result["neighbors_gpu"])}
    # Reference mutates its private copies; the candidate must preserve cached inputs.
    assert torch.equal(case.ego, original_inputs.ego), (case.name, "ego input mutated")
    assert torch.equal(case.stamps, original_inputs.stamps), (case.name, "timestamps mutated")
    assert len(case.frames) == len(original_inputs.frames)
    assert all(torch.equal(a, b) for a, b in zip(case.frames, original_inputs.frames)), (case.name, "history cache mutated")
    assert case.types == original_inputs.types, (case.name, "types mutated")
    metadata = result["metadata_cpu"]
    actual_rows = _metadata_array(metadata, "selected_current_rows")
    actual_centers = _metadata_array(metadata, "current_centers")
    actual_slots = _metadata_array(metadata, "slot_centers")
    actual_present = _metadata_array(metadata, "slot_present")
    assert np.array_equal(actual_rows, reference["selected_rows"].numpy()), (case.name, "selected current rows")
    assert np.array_equal(actual_centers, reference["current_centers"].numpy()), (case.name, "current CPU centers")
    assert np.array_equal(actual_slots, reference["slot_centers"].numpy()), (case.name, "first10 CPU slots")
    assert np.array_equal(actual_present, reference["slot_present"].numpy()), (case.name, "present slots")
    if case.check_mapping:
        assert mapping(actual_centers, actual_slots, np.flatnonzero(actual_present[:10])) == mapping(
            reference["current_centers"], reference["slot_centers"], np.flatnonzero(reference["slot_present"][:10].numpy())), (case.name, "Hungarian slot mapping")
    # Available types/padding and heading branch must not be hidden by atol.
    actual_neighbors = cpu(result["neighbors_gpu"])
    assert torch.equal(reference["neighbors"][..., -3:], actual_neighbors[..., -3:]), (case.name, "type one-hot")
    empty = torch.all(reference["neighbors"] == 0, dim=-1)
    assert torch.equal(empty, torch.all(actual_neighbors == 0, dim=-1)), (case.name, "missing/padded rows")
    # The heading channel is fixed by AgentFeatureIndex, not the raw internal order.
    F = adapter.AgentFeatureIndex
    yaw_error = (reference["neighbors"][..., F.heading()] - actual_neighbors[..., F.heading()]).abs()
    assert not bool((yaw_error > .01).any()), (case.name, "heading branch changed")
    if artifacts:
        destination = Path(artifacts)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save({"reference": reference, "actual_ego": cpu(result["ego_gpu"]),
                    "actual_neighbors": actual_neighbors, "metadata": metadata,
                    "errors": errors}, destination / (case.name + ".pt"))
    assert all(x["max_abs"] <= atol for x in errors.values()), (case.name, errors, "atol", atol)
    previous = (result["ego_gpu"], result["neighbors_gpu"])
    previous_copy = (cpu(result["ego_gpu"]), actual_neighbors)
    second = builder.build(case.ego, case.stamps, case.frames, case.types, num_agents=20)
    assert torch.equal(cpu(previous[0]), previous_copy[0]) and torch.equal(cpu(previous[1]), previous_copy[1]), (case.name, "old output overwritten")
    assert torch.equal(cpu(second["ego_gpu"]), previous_copy[0]) and torch.equal(cpu(second["neighbors_gpu"]), previous_copy[1]), (case.name, "repeat call changed")
    return {"case": case.name, "errors": errors, "selection_mapping_padding": "exact",
            "cached_inputs": "unchanged", "old_outputs": "unchanged", "stats": result["stats"]}


@torch.inference_mode()
def validate_split_payload(adapter, builder, case, *, atol=0.):
    """Exercise prepare_cpu -> a single combined map/history H2D -> compute_device."""
    expected = builder.build(case.ego, case.stamps, case.frames, case.types, num_agents=20)
    plan = builder.prepare_cpu(case.ego, case.stamps, case.frames, case.types, num_agents=20)
    payload = plan.payload
    assert all(torch.is_tensor(v) and v.device.type == "cpu" and v.dtype == torch.float32
               for v in payload.values()), "Plan payload must be CPU float32 tensors"
    # Offset zero belongs to DTPP's 18-float lossless affine; history views may
    # begin at odd offsets and must not rely on aligned dtype reinterpretation.
    combined = {"map_affine": torch.arange(18, dtype=torch.float32),
                "map_padding": torch.zeros(1, dtype=torch.float32)}
    combined.update({"history_" + k: v for k, v in payload.items()})
    packed = torch.cat([v.reshape(-1) for v in combined.values()]).to(builder.device)
    offset, device_payload = 0, {}
    for name, value in combined.items():
        view = packed[offset:offset + value.numel()].reshape(value.shape)
        if name.startswith("history_"):
            device_payload[name[len("history_"):]] = view
        offset += value.numel()
    actual = builder.compute_device(plan, device_payload=device_payload)
    errors = {k: tensor_error(cpu(expected[k]), actual[k]) for k in ("ego_gpu", "neighbors_gpu")}
    assert all(x["max_abs"] <= atol for x in errors.values()), ("split payload", errors)
    return {"single_pack": "passed", "offset": 19, "errors": errors}


@torch.inference_mode()
def exceptional_cases(adapter, builder):
    """Invalid time/nonfinite behavior must use an explicit reference fallback."""
    base = make_cases(adapter)[2]
    cases = []
    duplicate_time = clone_case(base)
    duplicate_time.name = "duplicate_timestamp"
    duplicate_time.stamps[8] = duplicate_time.stamps[7]
    cases.append(duplicate_time)
    reversed_time = clone_case(base)
    reversed_time.name = "decreasing_timestamp"
    reversed_time.stamps[8] = reversed_time.stamps[7] - 1
    cases.append(reversed_time)
    short = clone_case(base)
    short.name = "two_frames"
    short.ego, short.stamps, short.frames, short.types = short.ego[-2:], short.stamps[-2:], short.frames[-2:], short.types[-2:]
    cases.append(short)
    nonfinite = clone_case(base)
    nonfinite.name = "nonfinite_historical_heading"
    nonfinite.frames[7][0, adapter.AgentInternalIndex.heading()] = float("nan")
    cases.append(nonfinite)
    reports = []
    for case in cases:
        original = clone_case(case)
        reference_error = None
        try:
            expected = adapter.agent_past_process(original.ego, original.stamps, original.frames, original.types, 20)
        except Exception as error:
            reference_error = error
        try:
            actual = builder.build(case.ego, case.stamps, case.frames, case.types, num_agents=20)
        except Exception as error:
            assert reference_error is not None and type(error) is type(reference_error), (
                case.name, "exception type changed", type(reference_error).__name__, type(error).__name__)
            reports.append({"case": case.name, "original_exception": type(error).__name__})
            continue
        assert reference_error is None, (case.name, "original exception was suppressed")
        assert actual["stats"].get("cpu_fallback_reason"), (case.name, "unreported CPU fallback")
        torch.testing.assert_close(cpu(actual["ego_gpu"]), expected[0], atol=0., rtol=0., equal_nan=True)
        torch.testing.assert_close(cpu(actual["neighbors_gpu"]), expected[1], atol=0., rtol=0., equal_nan=True)
        reports.append({"case": case.name, "explicit_fallback": actual["stats"]["cpu_fallback_reason"]})
    return reports


def cases_from_histories(adapter, histories):
    cases = []
    for i, history in enumerate(histories):
        states, observations = list(history.ego_states)[-22:], list(history.observations)[-22:]
        ego = adapter.sampled_past_ego_states_to_tensor(states)
        frames, types = adapter.sampled_tracked_objects_to_tensor_list(observations)
        stamps = adapter.sampled_past_timestamps_to_tensor([state.time_point for state in states])
        cases.append(Case(f"native_frame_{i}", ego, stamps, frames, types))
    return cases


@torch.inference_mode()
def validate_sequence(adapter, builder, histories, *, atol=0., artifacts=None):
    """Inject real histories; downstream prepare/encoder/selector QA remains separate."""
    return [validate_case(adapter, builder, c, atol=atol, artifacts=artifacts)
            for c in cases_from_histories(adapter, histories)]


@torch.inference_mode()
def benchmark(adapter, builder, case, *, warmup=3, repeats=10):
    device = torch.device(builder.device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else lambda: None
    def reference():
        c = clone_case(case)
        return adapter.agent_past_process(c.ego, c.stamps, c.frames, c.types, 20)
    result = {}
    for name, fn in (("reference_cpu", reference), ("candidate_complete", lambda: builder.build(
            case.ego, case.stamps, case.frames, case.types, num_agents=20))):
        measurements = []
        for i in range(warmup + repeats):
            sync()
            begin = time.perf_counter()
            output = fn()
            sync()
            if i >= warmup:
                measurements.append((time.perf_counter() - begin) * 1000)
        result[name] = {"p50_ms": float(np.median(measurements)),
                        "p95_ms": float(np.percentile(measurements, 95)), "samples_ms": measurements}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="obs_adapter", help="Already patched exact host adapter module")
    parser.add_argument("--candidate", default="core", help="Module exporting AgentHistoryBuilder")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=0.)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--bootstrap", help="module:function to install the existing baseline patches first")
    parser.add_argument("--context-factory", help="module:function returning a list of native history buffers")
    args = parser.parse_args()
    if args.bootstrap:
        module, name = args.bootstrap.split(":")
        getattr(importlib.import_module(module), name)()
    adapter = importlib.import_module(args.adapter)
    candidate = importlib.import_module(args.candidate)
    builder = candidate.AgentHistoryBuilder(adapter, device=args.device)
    cases = make_cases(adapter)
    report = {"device": args.device, "atol": args.atol, "cases": [
        validate_case(adapter, builder, case, atol=args.atol, artifacts=args.artifacts) for case in cases]}
    report["split_payload"] = validate_split_payload(adapter, builder, cases[5], atol=args.atol)
    report["exceptions"] = exceptional_cases(adapter, builder)
    if args.context_factory:
        module, name = args.context_factory.split(":")
        report["native_sequence"] = validate_sequence(adapter, builder, getattr(importlib.import_module(module), name)(),
                                                       atol=args.atol, artifacts=args.artifacts)
    if args.benchmark:
        report["benchmark"] = benchmark(adapter, builder, cases[5])
    text = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.write_text(text + "\n")
        print(json.dumps({"status": "passed", "cases": len(cases), "output": str(args.output)}))
    else:
        print(text)


if __name__ == "__main__":
    main()
