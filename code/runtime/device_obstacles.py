"""Device-native CV obstacle provider, preserving legacy strict four-axis SAT.

Frame data is uploaded by the caller ONCE. All returned fields are CUDA tensors.
Static membership must come from existing static/stopped-track classification;
zero observed speed alone is not a replacement for the >3s stopped-track rule.
This provider does not replace DTPP, time-reserve, or between-sample sweep checks.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CVField:
    centers: torch.Tensor       # [T,A,2], ego-local vehicle BOX centers
    heading: torch.Tensor       # [A]
    length: torch.Tensor        # [A]
    width: torch.Tensor         # [A]
    static: torch.Tensor        # [A], caller-supplied static/stopped classification
    invalid: torch.Tensor       # scalar bool, must fail closed at integration gate
    dt: float
    # Original velocity is retained on device for continuous time-reserve
    # queries.  It is optional to keep the validation helpers and old callers
    # source-compatible with fields constructed position-only.
    velocity: torch.Tensor = None


def _same_device_dtype(reference, *values):
    for x in values:
        if x.device != reference.device or x.dtype != reference.dtype:
            raise ValueError('floating tensors must have identical device and dtype')


@torch.inference_mode()
def prepare_cv_field(position, velocity, deceleration, heading, length, width,
                     static, *, steps, dt):
    """Prepare once per frame/geometry variant (lead inflation is a new variant).

    Positions/velocities must already use the same ego-local frame as candidates.
    Float64 matches existing collision_gpu.py. Deceleration is causal observed
    braking, not estimated from future ground truth. Static flags partition risk
    and do not alter motion: exactly preserve the supplied legacy velocities.
    """
    if not position.is_cuda or position.dtype != torch.float64:
        raise ValueError('CUDA float64 actor tensors required')
    if position.ndim != 2 or position.shape[1] != 2:
        raise ValueError('position must have shape [A,2]')
    a = len(position)
    if velocity.shape != position.shape or any(x.shape != (a,) for x in
                                               (deceleration, heading, length, width, static)):
        raise ValueError('actor shape mismatch')
    _same_device_dtype(position, velocity, deceleration, heading, length, width)
    if static.device != position.device or static.dtype != torch.bool:
        raise ValueError('static membership must be CUDA bool on the actor device')
    import math
    if not isinstance(steps, int) or steps < 2 or not math.isfinite(dt) or dt <= 0:
        raise ValueError('positive finite dt and at least two samples required')
    invalid = (~torch.isfinite(position).all() | ~torch.isfinite(velocity).all()
               | ~torch.isfinite(deceleration).all() | ~torch.isfinite(heading).all()
               | ~torch.isfinite(length).all() | ~torch.isfinite(width).all()
               | (length <= 0).any() | (width <= 0).any())
    times = torch.arange(steps, device=position.device, dtype=position.dtype) * dt
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    brake = (deceleration < 0) & (speed >= 1e-6)
    denominator = torch.where(brake, deceleration, -torch.ones_like(deceleration))
    elapsed = torch.minimum(times[:, None], -speed[None] / denominator[None])
    distance = speed[None] * elapsed + .5 * deceleration[None] * elapsed.square()
    braking = distance[..., None] * (velocity / speed.clamp_min(1e-6)[:, None])[None]
    displacement = torch.where(brake[None, :, None], braking,
                               times[:, None, None] * velocity[None])
    return CVField(position[None] + displacement, heading, length, width,
                   static, invalid, float(dt), velocity)


@torch.inference_mode()
def temporal_reserve_masks_tensor(centers, heading, field, *, half_length,
                                  half_width, center_offset, reserve_s,
                                  dt=None, actor_chunk=32):
    """Continuous reserve-window SAT on CUDA.

    ``centers`` are ego box centers at the sampled times.  For each sampled
    pose, this evaluates the same four separating axes and shared time
    interval used by ``cv_temporal_clearance.conflict_mask`` while keeping
    candidates, actors, and interval arithmetic on the device.  The actor
    motion is the causal constant-velocity prediction used by the legacy
    final temporal check; braking-adjusted collision remains in
    ``collision_masks_tensor``.

    The returned mask has shape [N,T].  This is an advisory reserve check,
    never a replacement for strict oriented-footprint collision or the final
    speed/path checks.  It is intentionally chunked by actor to bound memory.
    """
    import math
    if (not isinstance(centers, torch.Tensor) or not centers.is_cuda
            or centers.dtype != torch.float64 or centers.ndim != 3
            or centers.shape[-1] != 2):
        raise ValueError('CUDA float64 centers [N,T,2] required')
    if (heading.shape != centers.shape[:2] or heading.device != centers.device
            or heading.dtype != centers.dtype):
        raise ValueError('matching CUDA float64 headings required')
    if field.centers.device != centers.device or field.centers.dtype != centers.dtype:
        raise ValueError('field/device/dtype mismatch')
    if field.centers.ndim != 3 or field.centers.shape[-1] != 2:
        raise ValueError('invalid CV field')
    if not math.isfinite(reserve_s) or reserve_s < 0:
        raise ValueError('reserve_s must be finite and nonnegative')
    if not isinstance(actor_chunk, int) or actor_chunk < 1:
        raise ValueError('actor_chunk must be positive')
    dt = field.dt if dt is None else float(dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be positive and finite')
    if field.velocity is None:
        if field.centers.shape[0] < 2:
            velocity = torch.zeros((field.centers.shape[1], 2),
                                   device=centers.device, dtype=centers.dtype)
        else:
            velocity = (field.centers[1] - field.centers[0]) / field.dt
    else:
        velocity = field.velocity
    if velocity.shape != field.centers.shape[1:] or velocity.device != centers.device:
        raise ValueError('field velocity shape/device mismatch')

    n, steps, _ = centers.shape
    ego_forward = torch.stack((heading.cos(), heading.sin()), -1)
    ego_lateral = torch.stack((-heading.sin(), heading.cos()), -1)
    ego_center = centers + float(center_offset) * ego_forward
    times = torch.arange(steps, device=centers.device,
                         dtype=centers.dtype) * dt
    low0 = -torch.minimum(times, centers.new_tensor(float(reserve_s)))
    high0 = centers.new_full((steps,), float(reserve_s))
    mask = torch.zeros((n, steps), device=centers.device, dtype=torch.bool)
    actor_pos = field.centers[0]
    for start in range(0, field.centers.shape[1], actor_chunk):
        end = min(start + actor_chunk, field.centers.shape[1])
        af = torch.stack((field.heading[start:end].cos(),
                          field.heading[start:end].sin()), -1)
        al = torch.stack((-af[:, 1], af[:, 0]), -1)
        vel = velocity[start:end]
        # Match the legacy reserve check: the sampled ego pose is held fixed
        # while the actor continues with its observed constant velocity over
        # the reserve interval.
        delta = (ego_center[:, :, None, :]
                 - (actor_pos[None, None, start:end, :]
                    + times[None, :, None, None] * vel[None, None, :, :]))
        low = low0[None, :, None].expand(n, steps, end - start).clone()
        high = high0[None, :, None].expand_as(low).clone()
        possible = torch.ones_like(low, dtype=torch.bool)
        nominal = torch.ones_like(low, dtype=torch.bool)
        axes = (ego_forward[:, :, None, :], ego_lateral[:, :, None, :],
                af[None, None, :, :], al[None, None, :, :])
        for axis in axes:
            projection = (delta * axis).sum(-1)
            rate = -(vel[None, None, :, :] * axis).sum(-1)
            extent = (
                float(half_length) * (ego_forward[:, :, None, :] * axis).sum(-1).abs()
                + float(half_width) * (ego_lateral[:, :, None, :] * axis).sum(-1).abs()
                + field.length[start:end][None, None, :] / 2
                  * (af[None, None, :, :] * axis).sum(-1).abs()
                + field.width[start:end][None, None, :] / 2
                  * (al[None, None, :, :] * axis).sum(-1).abs())
            moving = rate.abs() > 1e-12
            denominator = torch.where(moving, rate, torch.ones_like(rate))
            a = (-extent - projection) / denominator
            b = (extent - projection) / denominator
            low = torch.maximum(low, torch.where(moving,
                                torch.minimum(a, b), -torch.inf))
            high = torch.minimum(high, torch.where(moving,
                                  torch.maximum(a, b), torch.inf))
            possible &= moving | (projection.abs() < extent)
            nominal &= projection.abs() < extent
        if reserve_s == 0:
            mask |= nominal.any(-1)
        else:
            mask |= (possible & (low < high)).any(-1)
    return mask


@torch.inference_mode()
def temporal_clearance_tensor(centers, heading, field, *, half_length,
                              half_width, center_offset, dt=None,
                              actor_chunk=64):
    """GPU minimum circumradius clearance used only for candidate ranking."""
    import math
    if centers.ndim != 3 or centers.shape[-1] != 2 or not centers.is_cuda:
        raise ValueError('CUDA [N,T,2] centers required')
    if heading.shape != centers.shape[:2]:
        raise ValueError('matching headings required')
    dt = field.dt if dt is None else float(dt)
    times = torch.arange(centers.shape[1], device=centers.device,
                         dtype=centers.dtype) * dt
    forward = torch.stack((heading.cos(), heading.sin()), -1)
    ego = centers + float(center_offset) * forward
    velocity = field.velocity
    if velocity is None:
        velocity = ((field.centers[1] - field.centers[0]) / field.dt
                    if field.centers.shape[0] > 1 else torch.zeros_like(field.centers[0]))
    actor_center = field.centers[0][None, None] + times[None, :, None, None] * velocity[None, None]
    ego_radius = math.hypot(float(half_length), float(half_width))
    actor_radius = torch.hypot(field.length / 2, field.width / 2)
    best = torch.full((len(centers),), torch.inf, device=centers.device,
                      dtype=centers.dtype)
    for start in range(0, field.centers.shape[1], actor_chunk):
        end = min(start + actor_chunk, field.centers.shape[1])
        distance = torch.linalg.vector_norm(ego[:, :, None] - actor_center[:, :, start:end], dim=-1)
        margin = distance.amin(1) - (ego_radius + actor_radius[start:end])[None]
        best = torch.minimum(best, margin.amin(1))
    return best


@torch.inference_mode()
def collision_masks_tensor(centers, heading, field, *, half_length, half_width,
                           candidate_chunk=256, actor_chunk=32):
    """Return separate static/dynamic collision masks and invalid-input flags.

    centers [N,T,2] are ego BOX centers (rear-axle offset applied upstream ONCE).
    Strict overlap is the legacy CV convention: tangency alone is not collision.
    No host copy, scalar extraction, dynamic nonzero compaction, or actor objects.
    The caller must OR `invalid` into rejection independently of overlap flags.
    """
    import math
    if not centers.is_cuda or centers.dtype != torch.float64:
        raise ValueError('CUDA float64 candidate tensors required')
    if centers.ndim != 3 or centers.shape[-1] != 2 or heading.shape != centers.shape[:2]:
        raise ValueError('candidate shape mismatch')
    _same_device_dtype(centers, heading, field.centers, field.heading, field.length, field.width)
    n, t, _ = centers.shape
    if t != field.centers.shape[0]:
        raise ValueError('actor/candidate timestamps must match')
    if any(not math.isfinite(x) or x <= 0 for x in (half_length, half_width)):
        raise ValueError('positive finite ego half dimensions required')
    if candidate_chunk < 1 or actor_chunk < 1:
        raise ValueError('positive chunk sizes required')
    static_hit = torch.zeros((n, t), device=centers.device, dtype=torch.bool)
    dynamic_hit = torch.zeros_like(static_hit)
    invalid = (~torch.isfinite(centers).flatten(1).all(1)
               | ~torch.isfinite(heading).all(1) | field.invalid)
    for start in range(0, n, candidate_chunk):
        end = min(start + candidate_chunk, n)
        ec = centers[start:end, :, None]
        eh = heading[start:end, :, None]
        ce, se = eh.cos(), eh.sin()
        for j in range(0, len(field.heading), actor_chunk):
            k = min(j + actor_chunk, len(field.heading))
            angle = eh - field.heading[j:k]
            ca, sa = angle.cos().abs(), angle.sin().abs()
            delta = ec - field.centers[None, :, j:k]
            c, s = field.heading[j:k].cos(), field.heading[j:k].sin()
            x = delta[..., 0] * c + delta[..., 1] * s
            y = -delta[..., 0] * s + delta[..., 1] * c
            hl = field.length[j:k] / 2 + ca * half_length + sa * half_width
            hw = field.width[j:k] / 2 + sa * half_length + ca * half_width
            ex = delta[..., 0] * ce + delta[..., 1] * se
            ey = -delta[..., 0] * se + delta[..., 1] * ce
            ehl = half_length + ca * field.length[j:k] / 2 + sa * field.width[j:k] / 2
            ehw = half_width + sa * field.length[j:k] / 2 + ca * field.width[j:k] / 2
            hit = ((x.abs() < hl) & (y.abs() < hw) & (ex.abs() < ehl) & (ey.abs() < ehw))
            static_hit[start:end] |= (hit & field.static[j:k]).any(-1)
            dynamic_hit[start:end] |= (hit & ~field.static[j:k]).any(-1)
    return dict(static_collision=static_hit.any(1), dynamic_collision=dynamic_hit.any(1),
                static_by_time=static_hit, dynamic_by_time=dynamic_hit, invalid=invalid)
