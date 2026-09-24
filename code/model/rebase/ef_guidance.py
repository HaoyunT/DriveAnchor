"""EF neighbourhood supervision adapted from walle guide-flow training.

Reference: guide_flow_training_v6.0_rebased at 2dad4ee807c6,
decoder/_guide_flow_ef.py:229-299. Geometry is injected through ``labels`` so
training and evaluation can use the same corridor definition. This module
does not define an obstacle-collision or vehicle-dynamics guarantee.

The reference adds independent N(0, 1 m^2) noise to every XY coordinate.
``preserve_t0=True`` explicitly adapts this to nuPlan trajectories starting
at the ego origin. Normalized Smooth L1 is also an optional adaptation;
the default loss uses physical metres and the reference weighting.
"""

from dataclasses import dataclass
import math
from typing import Optional, Protocol

import torch
import torch.nn.functional as F


class TrajectoryClassifier(Protocol):
    def labels(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Return one good/bad label for each [T, 2] trajectory."""


@dataclass(frozen=True)
class GuidanceTargets:
    target_displacements: torch.Tensor
    noisy_good: torch.Tensor
    clean_good: torch.Tensor
    valid: torch.Tensor
    nearest_good_indices: torch.Tensor
    nearest_good_distances: torch.Tensor


def _validate_trajectories(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.ndim != 3 or value.shape[-1] != 2 or value.shape[1] < 2:
        raise ValueError(f"{name} must have shape [N, T >= 2, 2]")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite coordinates")


@torch.no_grad()
def sample_noisy_anchors(
    anchors: torch.Tensor,
    *,
    noise_std: float = 1.0,
    seed: int,
    preserve_t0: bool = True,
) -> torch.Tensor:
    """Sample reproducible noise without consuming the global RNG state.

    ``seed`` is required: callers should use a different recorded seed for
    each training update, and fixed seeds for validation. With t0 preserved,
    anchors must start at the ego origin; no other coordinates are changed
    or projected. This noise is a supervision distribution, not a claim of
    dynamically feasible exploration.
    """
    _validate_trajectories(anchors, "anchors")
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    if preserve_t0 and not torch.allclose(
        anchors[:, 0], torch.zeros_like(anchors[:, 0]), atol=1e-6, rtol=0
    ):
        raise ValueError("preserve_t0 requires anchors starting at the ego origin")
    generator = torch.Generator(device=anchors.device).manual_seed(seed)
    noise = torch.randn(
        anchors.shape, dtype=anchors.dtype, device=anchors.device,
        generator=generator,
    ) * noise_std
    if preserve_t0:
        noise[:, 0] = 0
    return anchors + noise


def _labels(classifier: TrajectoryClassifier, trajectories: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(classifier.labels(trajectories), device=trajectories.device)
    if values.shape != (len(trajectories),):
        raise ValueError("classifier.labels must return shape [N]")
    if not torch.all((values == 0) | (values == 1)):
        raise ValueError("classifier.labels must return Boolean or binary labels")
    return values.bool()


@torch.no_grad()
def build_guidance_targets(
    noisy_anchors: torch.Tensor,
    clean_anchors: torch.Tensor,
    classifier: TrajectoryClassifier,
    *,
    source_chunk_size: int = 64,
    reference_chunk_size: int = 256,
) -> GuidanceTargets:
    """Relabel noisy inputs and assign nearest *clean good* ADE targets.

    Good noisy inputs have zero target displacement, even if their source
    clean anchor was bad. Bad noisy inputs are valid only when a clean good
    reference exists. Invalid targets are finite zeros and must be excluded
    by ``valid``. The nearest distance is mean_t ||query_t-reference_t||_2;
    it is not flattened trajectory L2. Ties choose the first vocabulary ID.

    Query and reference vocabulary sizes may differ, but time grids, frame,
    units, dtype and device must agree.
    """
    _validate_trajectories(noisy_anchors, "noisy_anchors")
    _validate_trajectories(clean_anchors, "clean_anchors")
    if noisy_anchors.shape[1:] != clean_anchors.shape[1:]:
        raise ValueError("noisy and clean trajectories must share the time grid")
    if noisy_anchors.device != clean_anchors.device or noisy_anchors.dtype != clean_anchors.dtype:
        raise ValueError("noisy and clean trajectories must share device and dtype")
    if source_chunk_size < 1 or reference_chunk_size < 1:
        raise ValueError("chunk sizes must be positive")
    noisy_good = _labels(classifier, noisy_anchors)
    clean_good = _labels(classifier, clean_anchors)
    good_ids = clean_good.nonzero(as_tuple=True)[0]
    nearest_ids = torch.full((len(noisy_anchors),), -1, device=noisy_anchors.device, dtype=torch.long)
    # Compute distance in float32, matching reference training even under AMP.
    nearest_distances = torch.full(
        (len(noisy_anchors),), float("inf"), device=noisy_anchors.device, dtype=torch.float32
    )
    for start in range(0, len(noisy_anchors), source_chunk_size):
        stop = min(start + source_chunk_size, len(noisy_anchors))
        query = noisy_anchors[start:stop].float()
        for ref_start in range(0, len(good_ids), reference_chunk_size):
            ids = good_ids[ref_start:ref_start + reference_chunk_size]
            reference = clean_anchors[ids].float()
            distances = (query[:, None] - reference[None]).norm(dim=-1).mean(dim=-1)
            best_distances, best_positions = distances.min(dim=1)
            improve = best_distances < nearest_distances[start:stop]
            nearest_distances[start:stop] = torch.where(
                improve, best_distances, nearest_distances[start:stop]
            )
            nearest_ids[start:stop] = torch.where(
                improve, ids[best_positions], nearest_ids[start:stop]
            )
    valid_bad = ~noisy_good & (nearest_ids >= 0)
    valid = noisy_good | valid_bad
    displacement = torch.zeros_like(noisy_anchors)
    if valid_bad.any():
        displacement[valid_bad] = clean_anchors[nearest_ids[valid_bad]] - noisy_anchors[valid_bad]
    return GuidanceTargets(
        target_displacements=displacement,
        noisy_good=noisy_good,
        clean_good=clean_good,
        valid=valid,
        nearest_good_indices=nearest_ids,
        nearest_good_distances=nearest_distances,
    )


def guidance_loss(
    prediction: torch.Tensor,
    targets: GuidanceTargets,
    *,
    residual_scale: Optional[torch.Tensor] = None,
    beta: float = 1.0,
    loss_weight: float = 0.005,
) -> torch.Tensor:
    """Valid-anchor Smooth L1 average times 2*T*loss_weight.

    The optional positive [2] residual scale normalizes XY before Smooth L1.
    Its beta is therefore in normalized units, and this option is not
    numerically identical to the reference loss in metres. Good and bad
    samples are averaged together as in the reference, without rebalancing.
    """
    _validate_trajectories(prediction, "prediction")
    if prediction.shape != targets.target_displacements.shape:
        raise ValueError("prediction and target shapes must agree")
    if targets.valid.shape != (len(prediction),):
        raise ValueError("targets.valid must have shape [N]")
    if not math.isfinite(beta) or beta < 0 or not math.isfinite(loss_weight) or loss_weight < 0:
        raise ValueError("beta and loss_weight must be finite and nonnegative")
    pred = prediction.float()
    target = targets.target_displacements.detach().float()
    if residual_scale is not None:
        scale = torch.as_tensor(residual_scale, device=prediction.device, dtype=torch.float32).detach()
        if scale.shape != (2,) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("residual_scale must contain two finite positive XY scales")
        pred, target = pred / scale, target / scale
    if not targets.valid.any():
        return pred.sum() * 0.0
    per_anchor = F.smooth_l1_loss(pred, target, beta=beta, reduction="none").mean(dim=(1, 2))
    return per_anchor[targets.valid].mean() * (2 * prediction.shape[1] * loss_weight)
