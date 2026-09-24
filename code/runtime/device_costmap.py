"""Shared device occupancy broadphase for CVField, with 0.5 s time bins.

Build once per frame AND geometry variant; changed actor dimensions invalidate
the field. Candidate coordinates are ego BOX centers: apply the rear-axle
offset upstream exactly once. All tensors stay on their input device. CPU is
accepted for independent tests; production uses the existing CUDA CVField.

Each bool layer conservatively contains the actor center polyline over its
entire time interval, inflated by actor/ego circumradii and one half-cell
diagonal. Axis-aligned bounding rectangles deliberately admit false positives.
Only an empty query can skip the corresponding CV exact predicate; hits MUST
still use strict four-axis SAT. This is neither TTC nor a replacement for
learned/candidate-conditioned DTPP, final +/-reserve, road constraints, or ego
motion between candidate samples. Retiming requires a fresh query of all
relevant layers; a first-occupancy scalar is not an equivalent representation.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SharedCostmap:
    layers: torch.Tensor             # [L,H,W], bool
    origin: torch.Tensor             # [2], same floating dtype as CVField
    upper: torch.Tensor              # [2], exclusive spatial upper bound
    invalid: torch.Tensor            # scalar bool, fail closed for bad field
    resolution: float
    layer_dt: float
    prediction_dt: float
    horizon: float                   # (T-1)*dt, not rounded up to a full bin

    def _indices(self, xy):
        if not isinstance(xy, torch.Tensor) or xy.ndim < 1 or xy.shape[-1] != 2:
            raise ValueError('query positions must be a tensor with final dimension 2')
        if xy.device != self.origin.device or xy.dtype != self.origin.dtype:
            raise ValueError('query positions must match the field device and dtype')
        finite = torch.isfinite(xy).all(-1)
        invalid = (~finite | (xy < self.origin).any(-1)
                   | (xy >= self.upper).any(-1) | self.invalid)
        # Clamp floating coordinates BEFORE conversion: NaN, Inf and enormous
        # out-of-grid coordinates must never create unsafe integer indices.
        cell = torch.nan_to_num((xy - self.origin) / self.resolution,
                                nan=0., posinf=0., neginf=0.)
        h, w = self.layers.shape[-2:]
        ix = cell[..., 0].floor().clamp(0, w - 1).long()
        iy = cell[..., 1].floor().clamp(0, h - 1).long()
        return ix, iy, invalid

    def _times(self, times, shape):
        if not isinstance(times, torch.Tensor):
            raise ValueError('explicit query times must be device tensors')
        if times.device != self.origin.device or times.dtype != self.origin.dtype:
            raise ValueError('query times must match the field device and dtype')
        return torch.broadcast_to(times, shape)

    @torch.inference_mode()
    def query(self, box_centers, times=None):
        """Return device bool ``hit`` and ``invalid``, both shape [...].

        box_centers normally has shape [N,T,2]. With times=None, use the CV
        convention t=0,dt,... along its penultimate dimension. Explicit times
        are broadcastable to box_centers.shape[:-1]. Out-of-grid/time and all
        invalid input points are occupied AND marked invalid, even if A=0.
        """
        ix, iy, invalid = self._indices(box_centers)
        if times is None:
            if box_centers.ndim < 2:
                raise ValueError('a single position requires an explicit query time')
            times = (torch.arange(box_centers.shape[-2], device=box_centers.device,
                                  dtype=box_centers.dtype) * self.prediction_dt)
        times = self._times(times, box_centers.shape[:-1])
        invalid = (invalid | ~torch.isfinite(times) | (times < 0)
                   | (times > self.horizon))
        layer = torch.nan_to_num(times / self.layer_dt, nan=0., posinf=0., neginf=0.)
        layer = layer.floor().clamp(0, len(self.layers) - 1).long()
        return dict(hit=self.layers[layer, iy, ix] | invalid, invalid=invalid)

    @torch.inference_mode()
    def query_interval(self, box_centers, start_times, end_times):
        """OR all bins intersecting each closed [start,end] time interval.

        Position is held fixed during each interval. This provides the map
        broadphase needed for a time-reserve query, not the final exact reserve
        predicate or a check of an ego segment moving between two positions.
        Truncated prediction horizons fail closed rather than silently clipping.
        """
        ix, iy, invalid = self._indices(box_centers)
        shape = box_centers.shape[:-1]
        start_times = self._times(start_times, shape)
        end_times = self._times(end_times, shape)
        invalid = (invalid | ~torch.isfinite(start_times) | ~torch.isfinite(end_times)
                   | (start_times < 0) | (end_times > self.horizon)
                   | (end_times < start_times))
        starts = torch.arange(len(self.layers), device=box_centers.device,
                              dtype=box_centers.dtype) * self.layer_dt
        ends = (starts + self.layer_dt).clamp_max(self.horizon)
        view = (len(self.layers),) + (1,) * len(shape)
        overlaps = ((end_times[None] >= starts.reshape(view))
                    & (start_times[None] <= ends.reshape(view)))
        hit = (self.layers[:, iy, ix] & overlaps).any(0)
        return dict(hit=hit | invalid, invalid=invalid)


@torch.inference_mode()
def build_costmap(field, *, half_length, half_width,
                  bounds=(-150., -150., 150., 150.), resolution=1., layer_dt=.5):
    """Rasterize CVField polylines using four-corner rectangle difference sums.

    Work is O(T*A + L*A + L*H*W), not O(T*A*H*W). No per-actor full-grid
    distances, dynamic nonzero extraction, tensor scalar reads, or CPU copies.
    Boundary samples are included in both adjacent bins. A final partial bin
    ends at (T-1)*field.dt; an exactly aligned endpoint gets an instant layer.
    ``static`` membership never overrides the supplied predicted positions.
    """
    if len(bounds) != 4 or any(not math.isfinite(x) for x in bounds):
        raise ValueError('bounds must be finite (xmin,ymin,xmax,ymax)')
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError('bounds must have positive width and height')
    if any(not math.isfinite(x) or x <= 0 for x in
           (half_length, half_width, resolution, layer_dt, field.dt)):
        raise ValueError('ego half dimensions, resolution and time steps must be positive finite')
    centers = field.centers
    if (centers.ndim != 3 or centers.shape[-1] != 2 or centers.shape[0] < 1
            or centers.dtype not in (torch.float32, torch.float64)):
        raise ValueError('field centers must be floating [T,A,2], T>=1')
    steps, actors, _ = centers.shape
    for value in (field.heading, field.length, field.width):
        if (value.shape != (actors,) or value.device != centers.device
                or value.dtype != centers.dtype):
            raise ValueError('actor geometry must be [A] on the field device and dtype')
    if (field.static.shape != (actors,) or field.static.dtype != torch.bool
            or field.static.device != centers.device or field.invalid.ndim != 0
            or field.invalid.dtype != torch.bool or field.invalid.device != centers.device):
        raise ValueError('static [A] and invalid scalar must be device bool tensors')
    origin = centers.new_tensor(bounds[:2])
    upper = centers.new_tensor(bounds[2:])
    width = math.ceil((bounds[2] - bounds[0]) / resolution)
    height = math.ceil((bounds[3] - bounds[1]) / resolution)
    horizon = (steps - 1) * field.dt
    count = math.floor(horizon / layer_dt) + 1
    # Radius is around BOX center, not rear axle; no second ego inflation later.
    radius = (torch.hypot(field.length / 2, field.width / 2)
              + math.hypot(half_length, half_width) + resolution * math.sqrt(.5))
    invalid = (field.invalid | ~torch.isfinite(centers).all()
               | ~torch.isfinite(field.heading).all() | ~torch.isfinite(radius).all()
               | ~torch.isfinite(field.length).all() | ~torch.isfinite(field.width).all()
               | (field.length <= 0).any() | (field.width <= 0).any())
    safe_centers = torch.nan_to_num(centers, nan=0., posinf=0., neginf=0.)
    safe_radius = torch.nan_to_num(radius, nan=0., posinf=0., neginf=0.)
    lower, upper_rect = [], []
    for layer in range(count):
        start = layer * layer_dt
        end = min(start + layer_dt, horizon)
        # Including bracketing samples also conservatively handles dt values
        # not aligned to layer_dt, without assuming constant actor velocity.
        first = max(0, min(steps - 1, math.floor(start / field.dt)))
        last = max(first, min(steps - 1, math.ceil(end / field.dt)))
        segment = safe_centers[first:last + 1]
        lower.append(segment.amin(0) - safe_radius[:, None])
        upper_rect.append(segment.amax(0) + safe_radius[:, None])
    low = (torch.stack(lower) - origin) / resolution - .5
    high = (torch.stack(upper_rect) - origin) / resolution - .5
    # Grid centers within the inflated AABB. The half-cell diagonal included
    # above makes every point in a queried cell conservatively represented.
    x0 = low[..., 0].ceil().clamp(0, width).long()
    y0 = low[..., 1].ceil().clamp(0, height).long()
    x1 = (high[..., 0].floor() + 1).clamp(0, width).long()
    y1 = (high[..., 1].floor() + 1).clamp(0, height).long()
    pitch = width + 1
    indices = torch.stack((y0 * pitch + x0, y0 * pitch + x1,
                           y1 * pitch + x0, y1 * pitch + x1), -1)
    signs = torch.tensor((1, -1, -1, 1), device=centers.device, dtype=torch.int32)
    difference = torch.zeros((count, (height + 1) * pitch),
                             device=centers.device, dtype=torch.int32)
    difference.scatter_add_(1, indices.reshape(count, -1),
                            signs.expand(count, actors, 4).reshape(count, -1))
    coverage = difference.reshape(count, height + 1, pitch)
    coverage = coverage.cumsum(1, dtype=torch.int32).cumsum(2, dtype=torch.int32)
    layers = coverage[:, :height, :width] > 0
    return SharedCostmap(layers, origin, upper, invalid, float(resolution),
                         float(layer_dt), float(field.dt), float(horizon))
