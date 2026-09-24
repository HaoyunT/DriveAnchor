"""R3: legacy-builder-compatible explicit-libdevice-RN Triton replacement for device_map's CUDA query hot paths.

One full-grid launch per ring, and one nearest/classification/accumulation
launch per lane. No [all-points, all-edges] tensor or per-query host loop.
CPU explicitly falls back to device_map for portable tests. CUDA requires
Triton and never silently falls back. Critical float64 arithmetic uses
explicit libdevice round-to-nearest operations; no unsupported launch flags.

Import alongside the original device_map.py; the public cache constructors and
query results are drop-in compatible. Validate before changing production imports.
"""

import torch

from device_map import PolygonCache as _PolygonCache
from device_map import RoadCache as _RoadCache
from device_map import DirectionCache as _DirectionCache

try:
    import triton
    import triton.language as tl
    from triton.language import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None


if triton is not None:
    @triton.jit
    def _ring_kernel(P, E, INSIDE, BOUNDARY, N, NE,
                     BQ: tl.constexpr, BE: tl.constexpr):
        q = tl.program_id(0) * BQ + tl.arange(0, BQ)
        x = tl.load(P + q * 2, mask=q < N, other=0.).to(tl.float64)
        y = tl.load(P + q * 2 + 1, mask=q < N, other=0.).to(tl.float64)
        valid = (q < N) & (tl.abs(x) < float("inf")) & (tl.abs(y) < float("inf"))
        parity = tl.full((BQ,), 0, tl.int32)
        boundary = tl.full((BQ,), 0, tl.int32)
        ei = tl.arange(0, BE)
        for start in range(0, NE, BE):
            e = start + ei
            live = e < NE
            ax = tl.load(E + e * 9, live, other=0.)
            ay = tl.load(E + e * 9 + 1, live, other=0.)
            dx = tl.load(E + e * 9 + 2, live, other=0.)
            dy = tl.load(E + e * 9 + 3, live, other=0.)
            xmin = tl.load(E + e * 9 + 4, live, other=0.)
            ymin = tl.load(E + e * 9 + 5, live, other=0.)
            xmax = tl.load(E + e * 9 + 6, live, other=0.)
            ymax = tl.load(E + e * 9 + 7, live, other=0.)
            by = tl.load(E + e * 9 + 8, live, other=0.)
            xx, yy = x[:, None], y[:, None]
            cross = libdevice.sub_rn(
                libdevice.mul_rn(libdevice.sub_rn(xx, ax[None, :]), dy[None, :]),
                libdevice.mul_rn(libdevice.sub_rn(yy, ay[None, :]), dx[None, :]))
            touching = (live[None, :] & (cross == 0)
                        & (xx >= xmin[None, :]) & (xx <= xmax[None, :])
                        & (yy >= ymin[None, :]) & (yy <= ymax[None, :]))
            boundary = boundary | (tl.sum(touching.to(tl.int32), axis=1) > 0).to(tl.int32)
            crossing = (ay[None, :] > yy) != (by[None, :] > yy)
            denominator = tl.where(dy != 0., dy, 1.)
            cut = libdevice.add_rn(ax[None, :], libdevice.div_rn(
                libdevice.mul_rn(libdevice.sub_rn(yy, ay[None, :]), dx[None, :]),
                denominator[None, :]))
            count = tl.sum((live[None, :] & crossing & (xx < cut)).to(tl.int32), axis=1)
            parity = parity ^ (count & 1)
        tl.store(INSIDE + q, (parity != 0) & valid, mask=q < N)
        tl.store(BOUNDARY + q, (boundary != 0) & valid, mask=q < N)


    @triton.jit
    def _nearest_kernel(P, U, V, T, INSIDE, AF, EF, AR, ER, N, NV, THRESHOLD,
                        BQ: tl.constexpr, BV: tl.constexpr, ACCUMULATE: tl.constexpr):
        q = tl.program_id(0) * BQ + tl.arange(0, BQ)
        x = tl.load(P + q * 2, mask=q < N, other=0.).to(tl.float64)
        y = tl.load(P + q * 2 + 1, mask=q < N, other=0.).to(tl.float64)
        ux = tl.load(U + q * 2, mask=q < N, other=0.).to(tl.float64)
        uy = tl.load(U + q * 2 + 1, mask=q < N, other=0.).to(tl.float64)
        active = tl.load(INSIDE + q, mask=q < N, other=0).to(tl.int1)
        active = (active & (q < N) & (tl.abs(x) < float("inf"))
                  & (tl.abs(y) < float("inf")) & (tl.abs(ux) < float("inf"))
                  & (tl.abs(uy) < float("inf")))
        best = tl.full((BQ,), float("inf"), tl.float32).to(tl.float64)
        af = tl.full((BQ,), 0, tl.int32)
        ef = tl.full((BQ,), 0, tl.int32)
        ar = tl.full((BQ,), 0, tl.int32)
        er = tl.full((BQ,), 0, tl.int32)
        # This GPU branch skips whole point tiles outside this lane. No host
        # nonzero, count download or dynamic candidate compaction is needed.
        if tl.sum(active.to(tl.int32), axis=0) > 0:
            vi = tl.arange(0, BV)
            threshold = tl.load(THRESHOLD)
            for start in range(0, NV, BV):
                v = start + vi
                live = v < NV
                vx = tl.load(V + v * 2, live, other=0.)
                vy = tl.load(V + v * 2 + 1, live, other=0.)
                tx = tl.load(T + v * 2, live, other=0.)
                ty = tl.load(T + v * 2 + 1, live, other=0.)
                dx = libdevice.sub_rn(x[:, None], vx[None, :])
                dy = libdevice.sub_rn(y[:, None], vy[None, :])
                distance = tl.where(live[None, :], libdevice.add_rn(
                    libdevice.mul_rn(dx, dx), libdevice.mul_rn(dy, dy)), float("inf"))
                local_best = tl.min(distance, axis=1)
                tied = live[None, :] & (distance == local_best[:, None])
                dot = libdevice.add_rn(libdevice.mul_rn(ux[:, None], tx[None, :]),
                                      libdevice.mul_rn(uy[:, None], ty[None, :]))
                forward, reverse = dot >= 0., dot < threshold
                laf = (tl.sum((tied & forward).to(tl.int32), axis=1) > 0).to(tl.int32)
                lar = (tl.sum((tied & reverse).to(tl.int32), axis=1) > 0).to(tl.int32)
                lef = (tl.sum((tied & ~forward).to(tl.int32), axis=1) == 0).to(tl.int32)
                ler = (tl.sum((tied & ~reverse).to(tl.int32), axis=1) == 0).to(tl.int32)
                closer, equal = local_best < best, local_best == best
                af = tl.where(closer, laf, af | (equal.to(tl.int32) & laf))
                ar = tl.where(closer, lar, ar | (equal.to(tl.int32) & lar))
                ef = tl.where(closer, lef, ef & ((~equal).to(tl.int32) | lef))
                er = tl.where(closer, ler, er & ((~equal).to(tl.int32) | ler))
                best = tl.minimum(best, local_best)
        af, ef = (af != 0) & active, (ef != 0) & active
        ar, er = (ar != 0) & active, (er != 0) & active
        if ACCUMULATE:
            af = af | tl.load(AF + q, mask=q < N, other=0).to(tl.int1)
            ef = ef | tl.load(EF + q, mask=q < N, other=0).to(tl.int1)
            ar = ar | tl.load(AR + q, mask=q < N, other=0).to(tl.int1)
            er = er | tl.load(ER + q, mask=q < N, other=0).to(tl.int1)
        tl.store(AF + q, af, mask=q < N)
        tl.store(EF + q, ef, mask=q < N)
        tl.store(AR + q, ar, mask=q < N)
        tl.store(ER + q, er, mask=q < N)


def _require_backend(device):
    if device.type == "cuda":
        if triton is None or libdevice is None:
            raise RuntimeError("CUDA fused map queries require Triton libdevice; no implicit fallback")
        if not all(hasattr(libdevice, name) for name in ("add_rn", "sub_rn", "mul_rn", "div_rn")):
            raise RuntimeError("This Triton lacks required explicit float64 RN operations")


def _adopt_polygon(cache):
    """Reuse prepared edge tensors; do not re-upload geometry during promotion."""
    result = PolygonCache.__new__(PolygonCache)
    result.__dict__.update(cache.__dict__)
    _require_backend(result.device)
    if result.device.type == "cuda":
        result.query_chunk = 1 << 62
        result.polygons = [[edge.contiguous() for edge in rings] for rings in result.polygons]
    return result


class PolygonCache(_PolygonCache):
    def __init__(self, geometry, device="cuda", query_chunk=2048, edge_chunk=256):
        super().__init__(geometry, device, query_chunk, edge_chunk)
        _require_backend(self.device)
        if self.device.type == "cuda":
            self.query_chunk = 1 << 62  # grid tiles, never a full pairwise allocation
            self.polygons = [[edge.contiguous() for edge in rings] for rings in self.polygons]

    def _ring(self, points, edges):
        if self.device.type != "cuda":
            return super()._ring(points, edges)
        n = len(points)
        inside = torch.empty(n, device=self.device, dtype=torch.bool)
        boundary = torch.empty_like(inside)
        if n:
            _ring_kernel[(triton.cdiv(n, 16),)](
                points.contiguous(), edges.contiguous(), inside, boundary, n, len(edges),
                BQ=16, BE=64, num_warps=4)
        return inside, boundary


class RoadCache(_RoadCache):
    def __init__(self, region_world, ego, *, half_length, half_width,
                 device="cuda", query_chunk=2048, edge_chunk=256):
        super().__init__(region_world, ego, half_length=half_length, half_width=half_width,
                         device=device, query_chunk=query_chunk, edge_chunk=edge_chunk)
        self.polygon = _adopt_polygon(self.polygon)


class DirectionCache(_DirectionCache):
    def __init__(self, context, device="cuda", query_chunk=1024, vertex_chunk=256,
                 edge_chunk=256):
        super().__init__(context, device, query_chunk, vertex_chunk, edge_chunk)
        _require_backend(self.device)
        # Old builders have no get_fp64: upload the exact Python-double threshold once.
        self._direction_threshold = torch.tensor([-.1], device=self.device, dtype=torch.float64)
        self.entries = [(_adopt_polygon(p), v.contiguous(), t.contiguous())
                        for p, v, t in self.entries]
        if self.device.type == "cuda":
            self.query_chunk = 1 << 62

    def _nearest_classes(self, points, direction, vertices, tangents):
        if self.device.type != "cuda":
            return super()._nearest_classes(points, direction, vertices, tangents)
        n = len(points)
        result = [torch.empty(n, device=self.device, dtype=torch.bool) for _ in range(4)]
        if n:
            inside = torch.ones(n, device=self.device, dtype=torch.bool)
            _nearest_kernel[(triton.cdiv(n, 16),)](
                points.contiguous(), direction.contiguous(), vertices.contiguous(), tangents.contiguous(),
                inside, *result, n, len(vertices), self._direction_threshold, BQ=16, BV=64,
                ACCUMULATE=False, num_warps=4)
        return tuple(result)

    @torch.inference_mode()
    def evaluate_tensors(self, centers, unit, norm):
        if self.device.type != "cuda":
            return super().evaluate_tensors(centers, unit, norm)
        if (centers.ndim != 3 or centers.shape[-1] != 2 or unit.shape != centers.shape
                or norm.shape != centers.shape[:-1]):
            raise ValueError("Expected matching [candidate,time,2] and [candidate,time]")
        for tensor in (centers, unit, norm):
            if tensor.device != self.device or tensor.dtype not in (torch.float32, torch.float64):
                raise ValueError("Direction features must be floats on the cache device")
        shape = norm.shape
        flat = centers.reshape(-1, 2).double().contiguous()
        direction = unit.reshape(-1, 2).double().contiguous()
        n = len(flat)
        flags = [torch.zeros(n, device=self.device, dtype=torch.bool) for _ in range(4)]
        if n:
            for polygon, vertices, tangents in self.entries:
                inside = polygon.covers(flat)
                _nearest_kernel[(triton.cdiv(n, 16),)](
                    flat, direction, vertices, tangents, inside, *flags, n, len(vertices),
                    self._direction_threshold,
                    BQ=16, BV=64, ACCUMULATE=True, num_warps=4)
        possible_forward, certain_forward, possible_reverse, certain_reverse = flags
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
