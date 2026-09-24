"""GPU history candidate with exact CPU identity/selection metadata.

The live adapter is injected; this module never patches original globals. Plans
contain float32 bulk payloads plus one small float64 constants tensor. Normal
execution does not download GPU features or compute local history on the CPU.
"""
from collections import OrderedDict
from dataclasses import dataclass
import math
import time

import numpy as np
import torch
import torch.nn.functional as functional


@dataclass
class AgentHistoryPlan:
    payload: dict
    metadata_cpu: dict
    stats: dict
    constants_cpu: torch.Tensor
    num_agents: int
    time_steps: int
    selected_count: int
    fallback: bool = False


class AgentHistoryBuilder:
    def __init__(self, adapter, device="cuda"):
        self.adapter = adapter
        self.device = torch.device(device)
        self.I = adapter.AgentInternalIndex
        self.E = adapter.EgoInternalIndex
        self.F = adapter.AgentFeatureIndex
        self._coefficient_cache = OrderedDict()

    def _padding_rows(self, frames):
        """Compile exact filter(reverse=True) + pad(reverse=True) to a gather.

        Filter uses float token membership; pad uses int(token) dictionary keys.
        Last duplicate current ID owns the destination row; every frame's last
        duplicate wins. Unassigned duplicate-current rows retain all-zero state.
        Index zero is an explicit zero-state sentinel in the concatenated table.
        """
        current_ids = frames[-1][:, self.I.track_token()].tolist()
        allowed = set(current_ids)
        destinations = {int(token): row for row, token in enumerate(current_ids)}
        offsets, next_offset = [], 1
        for frame in frames:
            offsets.append(next_offset)
            next_offset += len(frame)
        if next_offset >= 2 ** 24:
            raise OverflowError("History row indices exceed exact float32 integer range")
        current = np.zeros(len(current_ids), dtype=np.int64)
        rows = np.zeros((len(frames), len(current_ids)), dtype=np.int64)
        for t in range(len(frames) - 1, -1, -1):
            ids = frames[t][:, self.I.track_token()].tolist()
            for row, token in enumerate(ids):
                if token in allowed:
                    current[destinations[int(token)]] = offsets[t] + row
            rows[t] = current
        table = torch.cat([torch.zeros(1, self.I.dim(), dtype=torch.float32)] + list(frames))
        return table, torch.from_numpy(rows)

    def _coefficients(self, timestamps, time_steps):
        """The original CPU least-squares system, cached only by its exact delta.

        Keep mean(diff(shifted_seconds)), including irregular timestamp behavior.
        No pseudoinverse, alternate cutoff, or endpoint correction is introduced.
        """
        if timestamps.ndim != 1:
            raise ValueError("Unexpected timestamps shape: {}".format(timestamps.shape))
        seconds = (timestamps - int(torch.min(timestamps).item())).double() * 1e-6
        if len(seconds) != time_steps:
            raise ValueError("Yaw and timestamp lengths differ")
        window_length = min(3, len(seconds))
        if not 2 < window_length:
            raise ValueError("2 < {} does not hold!".format(window_length))
        dx = torch.diff(seconds)
        if float(torch.min(dx).item()) <= 0:
            raise RuntimeError("dx is not monotonically increasing!")
        delta = dx.mean()
        key = float(delta.item())
        cached = self._coefficient_cache.get(key)
        if cached is not None:
            self._coefficient_cache.move_to_end(key)
            return cached
        x = torch.arange(-1., 2., dtype=torch.float64)
        order = torch.arange(3).reshape(-1, 1)
        target = torch.zeros(3, dtype=torch.float64)
        target[1] = math.factorial(1) / (delta ** 1)
        # The exact host base_predictor patches CPU lstsq to NumPy, same rcond.
        coefficients = torch.linalg.lstsq(x ** order, target)[0].clone()
        self._coefficient_cache[key] = coefficients
        if len(self._coefficient_cache) > 32:
            self._coefficient_cache.popitem(last=False)
        return coefficients

    def _fallback(self, ego, timestamps, frames, type_history, num_agents, reason):
        """Exceptional inputs use the original semantics, including exceptions.

        This is never the normal finite/int64 trajectory path and is observable
        in stats. If the original rejects the input, its exception propagates.
        """
        begin = time.perf_counter()
        anchor = ego[-1].clone()
        original_ego, neighbors = self.adapter.agent_past_process(
            ego.clone(), timestamps.clone(), [x.clone() for x in frames],
            type_history, num_agents)
        current = self.adapter.convert_absolute_quantities_to_relative(
            frames[-1].clone(), anchor, "agent")
        slots = neighbors[:, -1]
        metadata = {
            "current_centers": current[:, [self.I.x(), self.I.y()]].clone(),
            "slot_centers": slots[:, :2].clone(),
            "slot_present": torch.from_numpy((abs(slots.numpy()).sum(-1) > 0).copy()),
            "selected_current_rows": torch.empty(0, dtype=torch.int64),
        }
        return AgentHistoryPlan(
            payload={"reference_ego": original_ego.float(), "reference_neighbors": neighbors.float()},
            metadata_cpu=metadata,
            stats={"cpu_fallback_reason": reason, "current_frame_override": False,
                   "prepare_cpu_ms": (time.perf_counter() - begin) * 1000},
            constants_cpu=torch.empty(0, dtype=torch.float64), num_agents=num_agents,
            time_steps=len(frames), selected_count=0, fallback=True)

    @torch.inference_mode()
    def prepare_cpu(self, ego_cpu, timestamps_cpu, raw_agents_cpu, types_by_frame, num_agents=20):
        begin = time.perf_counter()
        frames = list(raw_agents_cpu)
        if not frames:
            # Original agent_past_process also requires a current frame.
            raise IndexError("No current agent frame")
        if (ego_cpu.device.type != "cpu" or timestamps_cpu.device.type != "cpu"
                or any(x.device.type != "cpu" for x in frames)):
            raise ValueError("prepare_cpu requires CPU inputs; downloading history is not supported")
        # Keep malformed/unusual provider inputs on the original exceptional path.
        if (ego_cpu.dtype != torch.float32 or ego_cpu.ndim != 2
                or ego_cpu.shape[-1] != self.E.dim()
                or any(x.dtype != torch.float32 or x.ndim != 2 or x.shape[-1] != self.I.dim()
                       for x in frames)):
            return self._fallback(ego_cpu, timestamps_cpu, frames, types_by_frame, num_agents,
                                  "unsupported_shape_or_dtype")
        if timestamps_cpu.dtype != torch.int64 and len(frames[-1]):
            return self._fallback(ego_cpu, timestamps_cpu, frames, types_by_frame, num_agents,
                                  "non_int64_timestamps")
        # Check before converting token values to Python int: the literal
        # reference decides how nonfinite tokens behave after torch.isin.
        if (not np.isfinite(ego_cpu.numpy()).all()
                or any(not np.isfinite(frame.numpy()).all() for frame in frames)):
            return self._fallback(ego_cpu, timestamps_cpu, frames, types_by_frame, num_agents,
                                  "nonfinite_world_input")
        try:
            table, padding_rows = self._padding_rows(frames)
        except OverflowError:
            return self._fallback(ego_cpu, timestamps_cpu, frames, types_by_frame, num_agents,
                                  "index_exceeds_float32_exact_range")
        indexed = time.perf_counter()
        anchor = ego_cpu[-1].clone()
        current_raw = frames[-1]
        padded_current = table[padding_rows[-1]]
        # Exact original CPU current-frame math fixes distance ties/cutoff and
        # gives mapping centers. With ordinary unique IDs this is one conversion.
        current_local = self.adapter.convert_absolute_quantities_to_relative(
            padded_current.clone(), anchor, "agent")
        if torch.equal(padded_current, current_raw):
            raw_local = current_local
        else:
            raw_local = self.adapter.convert_absolute_quantities_to_relative(
                current_raw.clone(), anchor, "agent")
        positions = current_local[:, [self.I.x(), self.I.y()]]
        nearest = torch.argsort(torch.norm(positions, dim=-1)).numpy()[:num_agents]
        # The truncate-before-behind-filter order is deliberate: no backfill.
        selected = [int(i) for i in nearest if not bool(positions[i, 0] < -6.)]
        selected_rows = torch.tensor(selected, dtype=torch.int64)
        count = len(selected)
        slot_centers = torch.zeros(num_agents, 2, dtype=torch.float32)
        slot_centers[:count] = positions[selected_rows]
        one_hot = torch.zeros(count, 3, dtype=torch.float32)
        for slot, row in enumerate(selected):
            kind = types_by_frame[-1][row]
            column = (0 if kind == self.adapter.TrackedObjectType.VEHICLE else
                      1 if kind == self.adapter.TrackedObjectType.PEDESTRIAN else 2)
            one_hot[slot, column] = 1.
        selected_done = time.perf_counter()
        if len(current_raw):
            # Even when no actors survive selection, keep original timestamp
            # validation: it precedes selection in agent_past_process.
            try:
                coefficients = self._coefficients(timestamps_cpu, len(frames))
            except (ValueError, RuntimeError):
                return self._fallback(ego_cpu, timestamps_cpu, frames, types_by_frame, num_agents,
                                      "timestamp_or_derivative_validation")
        else:
            coefficients = torch.zeros(3, dtype=torch.float64)
        anchor_pose = anchor[[self.E.x(), self.E.y(), self.E.heading()]].double()
        cosine, sine = torch.cos(anchor_pose[2]), torch.sin(anchor_pose[2])
        constants = torch.cat((anchor_pose, cosine.reshape(1), sine.reshape(1), coefficients))
        # For finite world states and valid int64 microsecond increments, local
        # heading/unwrap/yaw are finite; each filled row has a 1 in its one-hot.
        # Consequently abs(packed_current).sum > 0 exactly means a filled slot.
        # Nonfinite source inputs went through the literal reference above.
        present = torch.zeros(num_agents, dtype=torch.bool)
        present[:count] = True
        metadata = {"current_centers": raw_local[:, [self.I.x(), self.I.y()]].clone(),
                    "slot_centers": slot_centers, "slot_present": present,
                    "selected_current_rows": selected_rows}
        # Every yaw signal is independent, so gather/transform only the retained
        # K<=num_agents histories after exact current-frame CPU selection.
        payload = {
            "ego_world": ego_cpu.clone(),
            "agent_world": table if count else table[:1],
            "padding_rows": padding_rows[:, selected_rows].float(),
            "current_local_selected": current_local[selected_rows].clone(),
            "one_hot": one_hot,
        }
        stats = {"cpu_fallback_reason": None, "current_frame_override": True,
                 "input_frames": len(frames), "current_actors": len(current_raw),
                 "selected_actors": count, "raw_rows": len(table) - 1,
                 "padding_index_cpu_ms": (indexed - begin) * 1000,
                 "current_selection_cpu_ms": (selected_done - indexed) * 1000,
                 "prepare_cpu_ms": (time.perf_counter() - begin) * 1000,
                 "bulk_payload_bytes": sum(x.numel() * x.element_size() for x in payload.values()),
                 "double_constants_bytes": constants.numel() * constants.element_size()}
        return AgentHistoryPlan(payload, metadata, stats, constants, num_agents,
                                len(frames), count)

    def _local_pose(self, states, constants, x_index, y_index, heading_index):
        origin, cosine, sine = constants[:3], constants[3], constants[4]
        x = states[..., x_index].double() - origin[0]
        y = states[..., y_index].double() - origin[1]
        heading = states[..., heading_index].double() - origin[2]
        return (x * cosine + y * sine, -x * sine + y * cosine,
                torch.atan2(torch.sin(heading), torch.cos(heading)))

    def _yaw_rate(self, local_heading, coefficients):
        # Exact operation order from torch_math.unwrap, on the local headings
        # after the original float32 rounding and before float64 unwrapping.
        angles = local_heading.transpose(0, 1).double()
        # All operands are float64; the Python double is an immediate scalar,
        # avoiding an additional tiny CPU->GPU tensor copy each call.
        pi = math.pi
        dphi = functional.pad(torch.diff(angles, dim=-1), (1, 0))
        dphi_m = ((dphi + pi) % (2. * pi)) - pi
        dphi_m = torch.where((dphi_m == -pi) & (dphi > 0), pi, dphi_m)
        adjustment = dphi_m - dphi
        adjustment = torch.where(dphi.abs() < pi, torch.zeros_like(adjustment), adjustment)
        unwrapped = angles + adjustment.cumsum(dim=-1)
        rate = functional.conv1d(unwrapped.unsqueeze(1), coefficients.reshape(1, 1, 3),
                                 padding="same").reshape(unwrapped.shape)
        # Keep the reference quirk: these endpoint differences are NOT / delta.
        rate[:, 0] = unwrapped[:, 1] - unwrapped[:, 0]
        rate[:, -1] = unwrapped[:, -1] - unwrapped[:, -2]
        return rate.transpose(0, 1)

    @torch.inference_mode()
    def compute_device(self, plan, device_payload=None):
        begin = time.perf_counter()
        if device_payload is None:
            names = tuple(plan.payload)
            packed = torch.cat([plan.payload[name].reshape(-1) for name in names]).to(
                self.device, non_blocking=True)
            device_payload, offset = {}, 0
            for name in names:
                tensor = plan.payload[name]
                count = tensor.numel()
                device_payload[name] = packed[offset:offset + count].view(tensor.shape)
                offset += count
        # Only CPU schema metadata is read. Device scalar values are never read.
        for name, tensor in plan.payload.items():
            value = device_payload[name]
            if value.shape != tensor.shape or value.dtype != torch.float32:
                raise ValueError("Device payload shape/dtype differs for " + name)
        if plan.fallback:
            return {"ego_gpu": device_payload["reference_ego"],
                    "neighbors_gpu": device_payload["reference_neighbors"]}
        device = device_payload["ego_world"].device
        # Separate 64-byte double upload: preserves original CPU trig/lstsq
        # coefficients without opaque reinterpretation inside the float payload.
        constants = plan.constants_cpu.to(device, non_blocking=True)
        ego = device_payload["ego_world"].clone()
        x, y, heading = self._local_pose(ego, constants, self.E.x(), self.E.y(), self.E.heading())
        ego[:, self.E.x()] = x.float()
        ego[:, self.E.y()] = y.float()
        ego[:, self.E.heading()] = heading.float()
        # Original DTPP ego preprocessing leaves velocities/accelerations alone.
        neighbors = torch.zeros(plan.num_agents, plan.time_steps, self.F.dim() + 3,
                                dtype=torch.float32, device=device)
        if plan.selected_count:
            local = device_payload["agent_world"][device_payload["padding_rows"].long()]
            x, y, heading = self._local_pose(local, constants, self.I.x(), self.I.y(), self.I.heading())
            local[..., self.I.x()] = x.float()
            local[..., self.I.y()] = y.float()
            local[..., self.I.heading()] = heading.float()
            # Preserve float32 vector × float64 zero-dimensional scalar promotion
            # of original global_velocity_to_local (do not promote the vectors).
            vx, vy = local[..., self.I.vx()], local[..., self.I.vy()]
            cosine, sine = constants[3], constants[4]
            velocity_x = vx * cosine + vy * sine
            velocity_y = vy * cosine - vx * sine
            local[..., self.I.vx()] = velocity_x
            local[..., self.I.vy()] = velocity_y
            # CPU selected-current metadata and the actual last feature row use
            # the same original rounding. History remains GPU-computed.
            local[-1] = device_payload["current_local_selected"]
            rate = self._yaw_rate(local[..., self.I.heading()], constants[5:8])
            packed = torch.zeros(plan.time_steps, plan.selected_count, self.F.dim(),
                                 dtype=torch.float32, device=device)
            for name in ("x", "y", "heading", "vx", "vy", "width", "length"):
                packed[..., getattr(self.F, name)()] = local[..., getattr(self.I, name)()]
            packed[..., self.F.yaw_rate()] = rate.float()
            neighbors[:plan.selected_count, :, :self.F.dim()] = packed.transpose(0, 1)
            neighbors[:plan.selected_count, :, self.F.dim():] = device_payload["one_hot"][:, None, :]
        plan.stats["compute_device_dispatch_ms"] = (time.perf_counter() - begin) * 1000
        return {"ego_gpu": ego, "neighbors_gpu": neighbors}

    @torch.inference_mode()
    def build(self, ego_cpu, timestamps_cpu, raw_agents_cpu, types_by_frame, num_agents=20):
        plan = self.prepare_cpu(ego_cpu, timestamps_cpu, raw_agents_cpu, types_by_frame, num_agents)
        result = self.compute_device(plan)
        result.update(metadata_cpu=plan.metadata_cpu, stats=plan.stats)
        return result
