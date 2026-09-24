"""Prepared map predicates with bounded tensor-only queries.

Preparation accepts CPU Shapely/cKDTree context once per map or frame. Queries
never call a CPU reference or transfer results to the host. Float64 ray parity
is boundary inclusive, but is not a proof of GEOS robust-predicate equivalence.

Direction ties are not assigned an invented cKDTree ordering. ``bad`` reports
certain rejection; ``tie_ambiguous`` reports candidates whose answer depends on
an unresolved nearest-vertex tie. A caller must resolve that flag before
claiming parity or accepting the candidate as direction-safe.
"""

import math

import numpy as np
import torch


def _positive_chunk(value, name):
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class PolygonCache:
    """Inclusive point coverage of one valid Polygon/MultiPolygon.

    ``covers(points[..., 2])`` returns bool ``points.shape[:-1]`` on the cache
    device. CPU is accepted only when explicitly selected for portable tests.
    Temporary edge matrices are at most ``query_chunk * edge_chunk`` elements.
    """

    def __init__(self, geometry, device="cuda", query_chunk=2048, edge_chunk=256):
        self.query_chunk = _positive_chunk(query_chunk, "query_chunk")
        self.edge_chunk = _positive_chunk(edge_chunk, "edge_chunk")
        self._empty = torch.empty(0, device=device, dtype=torch.float64)
        self.device = self._empty.device
        self.polygons = []
        if geometry.is_empty:
            return
        if geometry.geom_type not in ("Polygon", "MultiPolygon") or not geometry.is_valid:
            raise ValueError("Map cache requires a valid Polygon/MultiPolygon")
        parts = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        for part in parts:
            rings = []
            for ring in [part.exterior, *part.interiors]:
                a = np.asarray(ring.coords, dtype=np.float64)[:, :2]
                if len(a) > 1 and np.array_equal(a[0], a[-1]):
                    a = a[:-1]
                if len(a) < 3 or not np.isfinite(a).all():
                    raise ValueError("Map rings require at least three finite vertices")
                b = np.roll(a, -1, axis=0)
                # Edges and bounds are static; do not rebuild them per query.
                packed = np.column_stack((a, b - a, np.minimum(a, b), np.maximum(a, b), b[:, 1]))
                rings.append(torch.as_tensor(packed, device=self.device, dtype=torch.float64))
            self.polygons.append(rings)

    def _check_points(self, points):
        if points.ndim < 1 or points.shape[-1] != 2:
            raise ValueError("Expected points[..., 2]")
        if points.device != self.device or points.dtype not in (torch.float32, torch.float64):
            raise ValueError("Points must be float32/float64 on the cache device")

    def _ring(self, points, edges):
        x, y = points[:, 0, None], points[:, 1, None]
        parity = torch.zeros(len(points), device=self.device, dtype=torch.bool)
        boundary = torch.zeros_like(parity)
        for start in range(0, len(edges), self.edge_chunk):
            edge = edges[start:start + self.edge_chunk]
            ax, ay, dx, dy, xmin, ymin, xmax, ymax, by = edge.unbind(1)
            cross = (x - ax) * dy - (y - ay) * dx
            boundary |= ((cross == 0) & (x >= xmin) & (x <= xmax)
                         & (y >= ymin) & (y <= ymax)).any(1)
            crossing = (ay > y) != (by > y)
            cut = ax + (y - ay) * dx / torch.where(dy != 0, dy, 1.)
            parity ^= ((crossing & (x < cut)).sum(1) % 2).bool()
        return parity, boundary

    @torch.inference_mode()
    def covers(self, points):
        self._check_points(points)
        shape = points.shape[:-1]
        flat = points.reshape(-1, 2).double()
        result = torch.zeros(len(flat), device=self.device, dtype=torch.bool)
        for start in range(0, len(flat), self.query_chunk):
            q = flat[start:start + self.query_chunk]
            covered = torch.zeros(len(q), device=self.device, dtype=torch.bool)
            for rings in self.polygons:
                inside, edge = self._ring(q, rings[0])
                part = inside | edge
                for hole in rings[1:]:
                    hole_inside, hole_edge = self._ring(q, hole)
                    part &= ~(hole_inside & ~hole_edge)
                covered |= part
            result[start:start + len(q)] = covered & torch.isfinite(q).all(1)
        return result.reshape(shape)

    @torch.inference_mode()
    def query(self, points):
        """Coverage plus an explicit per-point nonfinite flag for adapters."""
        return dict(covered=self.covers(points), invalid=~torch.isfinite(points).all(-1))


class RoadCache:
    """The legacy four-corner road predicate, without added buffers or margins."""

    def __init__(self, region_world, ego, *, half_length, half_width,
                 device="cuda", query_chunk=2048, edge_chunk=256):
        if not all(math.isfinite(v) for v in
                   (ego.x, ego.y, ego.heading, half_length, half_width)):
            raise ValueError("Nonfinite road frame or vehicle dimensions")
        if min(half_length, half_width) <= 0:
            raise ValueError("Positive vehicle half dimensions required")
        self.polygon = PolygonCache(region_world, device, query_chunk, edge_chunk)
        c, s = np.cos(ego.heading), np.sin(ego.heading)
        self.rotation = torch.as_tensor([[c, s], [-s, c]], device=self.polygon.device,
                                        dtype=torch.float64)
        self.origin = torch.as_tensor([ego.x, ego.y], device=self.polygon.device,
                                      dtype=torch.float64)
        self.half_length, self.half_width = half_length, half_width

    @torch.inference_mode()
    def outside(self, features):
        centers = features["box_centers"]
        forward, side = features["forward"], features["side"]
        if centers.ndim != 3 or centers.shape[-1] != 2:
            raise ValueError("Road features must have shape [candidate,time,2]")
        if forward.shape != centers.shape or side.shape != centers.shape:
            raise ValueError("Road feature shapes differ")
        # Match NumPy: construct local corners at candidate precision FIRST.
        corners = torch.stack([centers + l * self.half_length * forward
                               + w * self.half_width * side
                               for l, w in ((1, 1), (1, -1), (-1, -1), (-1, 1))], 2)
        world = corners.double() @ self.rotation + self.origin
        return ~self.polygon.covers(world).flatten(1).all(1)

    @torch.inference_mode()
    def evaluate(self, features):
        invalid = torch.zeros(len(features["box_centers"]), device=self.polygon.device,
                              dtype=torch.bool)
        for key in ("box_centers", "forward", "side"):
            invalid |= ~torch.isfinite(features[key]).flatten(1).all(1)
        return dict(outside=self.outside(features) | invalid, invalid=invalid)


class DirectionCache:
    """Nearest baseline VERTEX direction with lane-overlap forward exemption.

    Context is the existing ``direction_guard.prepare`` result. Extra context
    cache fields are ignored. The cached offset is informational: centers must
    already use that offset, as in device_features.direction_centers.
    """

    def __init__(self, context, device="cuda", query_chunk=1024, vertex_chunk=256,
                 edge_chunk=256):
        self.query_chunk = _positive_chunk(query_chunk, "query_chunk")
        self.vertex_chunk = _positive_chunk(vertex_chunk, "vertex_chunk")
        self.offset = float(context[1])
        if not math.isfinite(self.offset):
            raise ValueError("Nonfinite direction center offset")
        self.device = torch.empty(0, device=device).device
        self.entries = []
        for polygon, index, tangents in context[0]:
            vertices = np.asarray(index.data, dtype=np.float64)
            tangent = np.asarray(tangents, dtype=np.float64)
            if (vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 1
                    or tangent.shape != vertices.shape or not np.isfinite(vertices).all()
                    or not np.isfinite(tangent).all()):
                raise ValueError("Finite matching baseline vertices/tangents required")
            self.entries.append((
                PolygonCache(polygon, self.device, query_chunk, edge_chunk),
                torch.as_tensor(vertices, device=self.device, dtype=torch.float64),
                torch.as_tensor(tangent, device=self.device, dtype=torch.float64)))

    def _nearest_classes(self, points, direction, vertices, tangents):
        """Return all possible F/R classifications among exactly nearest vertices."""
        best = torch.full((len(points),), torch.inf, device=self.device, dtype=torch.float64)
        any_forward = torch.zeros(len(points), device=self.device, dtype=torch.bool)
        all_forward = torch.zeros_like(any_forward)
        any_reverse = torch.zeros_like(any_forward)
        all_reverse = torch.zeros_like(any_forward)
        for start in range(0, len(vertices), self.vertex_chunk):
            vertex = vertices[start:start + self.vertex_chunk]
            tangent = tangents[start:start + self.vertex_chunk]
            distance = (points[:, None] - vertex).square().sum(-1)
            local_best = distance.amin(1)
            tied = distance == local_best[:, None]
            dot = (direction[:, None] * tangent).sum(-1)
            forward, reverse = dot >= 0, dot < -.1
            af = (tied & forward).any(1)
            ar = (tied & reverse).any(1)
            ef = (~tied | forward).all(1)
            er = (~tied | reverse).all(1)
            closer, equal = local_best < best, local_best == best
            any_forward = torch.where(closer, af, any_forward | (equal & af))
            any_reverse = torch.where(closer, ar, any_reverse | (equal & ar))
            all_forward = torch.where(closer, ef, all_forward & (~equal | ef))
            all_reverse = torch.where(closer, er, all_reverse & (~equal | er))
            best = torch.minimum(best, local_best)
        return any_forward, all_forward, any_reverse, all_reverse

    @torch.inference_mode()
    def evaluate_tensors(self, centers, unit, norm):
        if (centers.ndim != 3 or centers.shape[-1] != 2 or unit.shape != centers.shape
                or norm.shape != centers.shape[:-1]):
            raise ValueError("Expected matching [candidate,time,2] and [candidate,time]")
        for tensor in (centers, unit, norm):
            if tensor.device != self.device or tensor.dtype not in (torch.float32, torch.float64):
                raise ValueError("Direction features must be floats on the cache device")
        shape = norm.shape
        flat, direction = centers.reshape(-1, 2).double(), unit.reshape(-1, 2).double()
        possible_forward = torch.zeros(len(flat), device=self.device, dtype=torch.bool)
        certain_forward = torch.zeros_like(possible_forward)
        possible_reverse = torch.zeros_like(possible_forward)
        certain_reverse = torch.zeros_like(possible_forward)
        for polygon, vertices, tangents in self.entries:
            for start in range(0, len(flat), self.query_chunk):
                q = flat[start:start + self.query_chunk]
                sl = slice(start, start + len(q))
                inside = polygon.covers(q)
                af, ef, ar, er = self._nearest_classes(q, direction[sl], vertices, tangents)
                possible_forward[sl] |= inside & af
                certain_forward[sl] |= inside & ef
                possible_reverse[sl] |= inside & ar
                certain_reverse[sl] |= inside & er
        moving = norm > .005
        certain = (certain_reverse & ~possible_forward).reshape(shape) & moving
        possible = (possible_reverse & ~certain_forward).reshape(shape) & moving
        tie_points = possible & ~certain
        invalid = (~torch.isfinite(centers).flatten(1).all(1)
                   | ~torch.isfinite(unit).flatten(1).all(1)
                   | ~torch.isfinite(norm).all(1))
        bad = certain.any(1) | invalid
        return dict(bad=bad, tie_ambiguous=tie_points.any(1) & ~bad,
                    tie_points=tie_points, invalid=invalid)

    @torch.inference_mode()
    def evaluate(self, features):
        return self.evaluate_tensors(features["direction_centers"],
                                     features["direction_unit"], features["direction_norm"])
