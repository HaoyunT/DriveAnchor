"""Candidate CUDA features with explicitly distinct legacy heading conventions.

Source: /workdir/driveanchor_selector_frozen_v1/selection.py
SHA256 60fffe5b17e902caa61d53d83d72ac1add7de88c53bcdee79e113311b9b6f235.
Keep candidate dtype at input precision; converting float32 xy to double before
feature derivation changes legacy rounding. Convert for downstream SAT AFTER
derivation if its reference previously converted precomputed NumPy features.
"""
import math
import torch


@torch.jit.script
def _stable_heading_sequential(raw: torch.Tensor, moving: torch.Tensor):
    # Same time recurrence as reference, all candidates processed together.
    yaw = torch.zeros_like(raw)
    for t in range(1, raw.shape[1]):
        previous = yaw[:, t - 1]
        turn = torch.remainder(raw[:, t] - previous + math.pi / 2, math.pi) - math.pi / 2
        yaw[:, t] = torch.where(moving[:, t], previous + turn, previous)
    return yaw


@torch.inference_mode()
def candidate_features(xy, *, dt=.1, heading_hold_speed=.5, center_offset=1.461):
    """One batched computation shared by downstream providers; no host outputs.

    xy includes sample t0. Segment derivatives preserve old selector semantics;
    center derivatives preserve old DTPP semantics. Direction vectors preserve
    direction_guard's np.gradient WITHOUT dividing by dt (norm gate .005).
    Finite flags must be included in every eventual eligibility gate.
    """
    if not xy.is_cuda or xy.dtype not in (torch.float32, torch.float64):
        raise ValueError('CUDA float32/float64 candidates required')
    if xy.ndim != 3 or xy.shape[-1] != 2 or xy.shape[1] < 3:
        raise ValueError('candidate shape must be [N,T>=3,2]')
    if not all(math.isfinite(v) for v in (dt, heading_hold_speed, center_offset)) or dt <= 0 or heading_hold_speed < 0:
        raise ValueError('invalid feature parameters')
    finite = torch.isfinite(xy).flatten(1).all(1)
    segment_delta = xy[:, 1:] - xy[:, :-1]
    centered_delta = torch.cat((segment_delta[:, :1], xy[:, 2:] - xy[:, :-2], segment_delta[:, -1:]), 1)
    # Reference np.r_[dt, repeated2dt, dt] is float64 even for float32 xy.
    duration = torch.full((xy.shape[1],), 2 * dt, device=xy.device, dtype=torch.float64)
    duration[0] = dt
    duration[-1] = dt
    moving = torch.linalg.vector_norm(centered_delta, dim=-1).to(torch.float64) / duration >= heading_hold_speed
    raw = torch.atan2(centered_delta[..., 1], centered_delta[..., 0])
    yaw = _stable_heading_sequential(raw, moving)
    segment_velocity = segment_delta / dt
    segment_acceleration = (segment_velocity[:, 1:] - segment_velocity[:, :-1]) / dt
    gradient_velocity = torch.gradient(xy, spacing=dt, dim=1)[0]
    # A separate gradient is deliberate: direction_guard normalizes unscaled
    # np.gradient, which differs from heading holding and uses a .005 norm gate.
    direction_delta = torch.gradient(xy, dim=1)[0]
    direction_norm = torch.linalg.vector_norm(direction_delta, dim=-1)
    direction_unit = direction_delta / direction_norm.clamp_min(1e-8)[..., None]
    forward = torch.stack((yaw.cos(), yaw.sin()), -1)
    side = torch.stack((-yaw.sin(), yaw.cos()), -1)
    speed = torch.linalg.vector_norm(gradient_velocity, dim=-1)
    six = torch.stack((xy[..., 0], xy[..., 1], yaw, speed,
                       torch.gradient(speed, spacing=dt, dim=1)[0],
                       torch.gradient(yaw, spacing=dt, dim=1)[0] / speed.clamp_min(.1)), -1)
    return dict(xy=xy, finite=finite, stable_heading=yaw, forward=forward, side=side,
                box_centers=xy + center_offset * forward,
                segment_velocity=segment_velocity, segment_acceleration=segment_acceleration,
                gradient_velocity=gradient_velocity, dtpp_six=six,
                direction_unit=direction_unit, direction_norm=direction_norm,
                direction_centers=xy + center_offset * direction_unit)
