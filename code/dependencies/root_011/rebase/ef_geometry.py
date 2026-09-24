"""Ordered 50D EF geometry, following guide_flow_training_v6.0_rebased.

Vertices 0..7 follow the left boundary; 8..15 follow the reversed right
boundary. Edge 7 is the forward exit. Labels use sampled-point transitions,
as in the reference torch classifier; they are not vehicle collision labels.
"""
import numpy as np
import torch
from shapely.geometry import Polygon


class BranchCorridor:
    def __init__(self, spec):
        self.vertices = np.asarray(spec['polygon'], dtype=np.float64)
        if self.vertices.shape != (16, 2) or not np.isfinite(self.vertices).all():
            raise ValueError('EF v3 requires 16 ordered finite boundary points')
        self.scene_type = int(spec.get('scene_type', 0))
        if self.scene_type != spec.get('scene_type', 0) or self.scene_type not in range(5):
            raise ValueError('scene_type must be an integer in 0..4')
        self.enterable = np.asarray(spec.get('enterable', np.ones(16)), dtype=float)
        if self.enterable.shape != (16,) or not np.isin(self.enterable, [0, 1]).all():
            raise ValueError('enterable must contain 16 binary point flags')
        self.expanded = float(spec.get('is_expanded', 0))
        if self.expanded not in (0., 1.):
            raise ValueError('is_expanded must be binary')
        self.required_time = spec.get('required_time_seconds')
        if self.required_time is not None and (not np.isfinite(self.required_time) or self.required_time < 0):
            raise ValueError('required_time_seconds must be nonnegative or None')
        self.polygon = Polygon(self.vertices)
        if not self.polygon.is_valid or self.polygon.area < 1e-6:
            raise ValueError('Invalid ordered EF polygon; do not repair labels silently')
        if (np.linalg.norm(self.vertices - np.roll(self.vertices, -1, axis=0), axis=1) < 1e-7).any():
            raise ValueError('Degenerate polygon edge')
        self.feature = np.concatenate([[self.scene_type], np.column_stack(
            [self.vertices, self.enterable]).ravel(), [self.expanded]]).astype(np.float32)

    def spec(self):
        return dict(polygon=self.vertices.tolist(), scene_type=self.scene_type,
                    enterable=self.enterable.tolist(), is_expanded=self.expanded,
                    required_time_seconds=self.required_time)

    def augment_entry(self, seed):
        """Reference lane-change augmentation; geometry and other scenes unchanged."""
        if self.scene_type != 1 or self.enterable.sum() == 16:
            return BranchCorridor(self.spec())
        rng = np.random.default_rng(seed)
        distance = np.linalg.norm(self.vertices, axis=-1)
        offset = 0 if distance[:8].mean() < distance[8:].mean() else 8
        points = int(rng.integers(1, 8)) + 1
        start = int(rng.integers(offset, offset + 9 - points))
        flags = np.zeros(16); flags[start:start + points] = 1
        spec = self.spec(); spec['enterable'] = flags.tolist()
        return BranchCorridor(spec)

    @torch.no_grad()
    def labels(self, trajectories, dt=.1):
        x = torch.as_tensor(trajectories)
        if x.ndim != 3 or x.shape[-1] != 2 or x.shape[1] < 2 or not torch.isfinite(x).all():
            raise ValueError('Expected finite [N,T,2] trajectories')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive')
        x = x.to(dtype=torch.float64)
        v = torch.as_tensor(self.vertices, device=x.device, dtype=x.dtype)
        end = v.roll(-1, 0); edge = end - v
        rel = x[:, :, None] - v
        cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        eps = 1e-6
        on = ((cross(rel, edge).abs() <= eps * edge.norm(dim=-1)) &
              ((rel * edge).sum(-1) >= -eps) &
              ((rel * edge).sum(-1) <= edge.square().sum(-1) + eps)).any(-1)
        a_y, b_y, p_y = v[:, 1], end[:, 1], x[:, :, None, 1]
        crosses = (a_y > p_y) != (b_y > p_y)
        denominator = b_y - a_y
        denominator = torch.where(denominator.abs() > eps, denominator, torch.ones_like(denominator))
        ix = v[:, 0] + (p_y - a_y) * edge[:, 0] / denominator
        inside = on | ((crosses & (x[:, :, None, 0] < ix)).sum(-1) % 2 == 1)
        touch = inside.any(-1)
        exits = inside[:, :-1] & ~inside[:, 1:]
        entries = ~inside[:, :-1] & inside[:, 1:]

        def crossing_edge(transitions):
            step = transitions.long().argmax(-1)
            idx = torch.arange(len(x), device=x.device)
            start = x[idx, step]; direction = x[idx, step + 1] - start
            q = v[None] - start[:, None]
            den = cross(direction[:, None], edge)
            safe = torch.where(den.abs() > eps, den, torch.ones_like(den))
            t = cross(q, edge) / safe
            u = cross(q, direction[:, None]) / safe
            hit = (den.abs() > eps) & (t >= -eps) & (t <= 1 + eps) & (u >= -eps) & (u <= 1 + eps)
            first = torch.where(hit, t.clamp(0, 1), torch.full_like(t, float('inf'))).argmin(-1)
            return torch.where(hit.any(-1), first, torch.full_like(first, -1))

        count = exits.sum(-1)
        good = touch & ((count == 0) | ((count == 1) & (crossing_edge(exits) == 7)))
        if self.required_time is not None:
            fraction = self.required_time / dt
            index = round(fraction)
            if abs(fraction-index) > 1e-6 or not 0 <= index < x.shape[1]:
                raise ValueError('Required time must be on the sample grid and within the horizon')
            good &= inside[:, index]
        if self.scene_type == 1:
            entry = crossing_edge(entries)
            flags = torch.as_tensor(self.enterable > .5, device=x.device)
            allowed_edges = flags & flags.roll(-1)
            valid_entry = inside[:, 0] | (entries.any(-1) & (entry >= 0) & allowed_edges[entry.clamp_min(0)])
            good &= valid_entry
        return good
