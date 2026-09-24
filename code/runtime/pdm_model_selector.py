"""GPU candidate ranking helpers.

The PDM weighted scorer is retained below for compatibility and ablations.  The
native default selector uses the simpler production contract: apply the hard
feasibility mask, then rank feasible candidates by FM2 ``vnorm`` (ascending),
with anchor ID as a deterministic tie breaker.  TTC is intentionally not a
selector ranking term; final downstream safety checks remain unchanged.
"""

from __future__ import annotations

import torch


# Same relative PDMScorer weights for the three native terms.
PDM_PROGRESS_WEIGHT = 5.0
PDM_TTC_WEIGHT = 5.0
PDM_COMFORT_WEIGHT = 2.0
DEFAULT_MODEL_PRIOR_WEIGHT = 1.0


@torch.inference_mode()
def vnorm_topk(vnorm: torch.Tensor, k: int = 20) -> torch.Tensor:
    """Return deterministic original-pool indices for the smallest vnorms."""

    if vnorm.ndim != 1 or not vnorm.is_floating_point():
        raise ValueError("vnorm must be a one-dimensional floating tensor")
    if k < 1:
        raise ValueError("k must be positive")
    finite = torch.isfinite(vnorm)
    # Stable sorting preserves anchor ID order for equal norms.  Invalid model
    # outputs are never allowed into the shortlist.
    order = torch.argsort(torch.where(finite, vnorm, torch.inf), stable=True)
    return order[: min(int(k), int(vnorm.numel()))]


@torch.inference_mode()
def select_vnorm_feasible(
    *,
    hard_violation: torch.Tensor,
    vnorm: torch.Tensor,
    anchor_ids: torch.Tensor,
):
    """Select the lowest-FM2-vnorm feasible candidate on the input device.

    ``hard_violation`` is the already-computed GPU mask for collision, road,
    signal, speed, acceleration and direction constraints.  No TTC quantity
    is accepted here by design, so it cannot accidentally affect ordering.
    Equal norms are resolved by the smallest original anchor ID.  The returned
    ``order`` is the complete stable feasible-first order, useful for
    diagnostics without another device round trip.
    """

    if hard_violation.ndim != 1 or vnorm.ndim != 1 or anchor_ids.ndim != 1:
        raise ValueError("candidate quantities must be one-dimensional")
    n = int(vnorm.numel())
    if int(hard_violation.numel()) != n or int(anchor_ids.numel()) != n:
        raise ValueError("candidate quantities must be aligned")
    if not vnorm.is_floating_point():
        raise ValueError("vnorm must be floating point")

    finite = torch.isfinite(vnorm)
    valid = (~hard_violation.bool()) & finite
    # Stable sort by anchor ID first, then stable-sort that result by vnorm.
    # This is a CUDA-side lexicographic sort: vnorm is the primary key and the
    # original anchor ID is used only when norms are equal.
    safe_vnorm = torch.where(valid, vnorm, torch.inf)
    order_id = torch.argsort(anchor_ids, stable=True)
    order_v = torch.argsort(safe_vnorm[order_id], stable=True)
    order = order_id[order_v]
    found = valid.any()
    sentinel = torch.iinfo(anchor_ids.dtype).max
    best_vnorm = torch.where(valid, vnorm, torch.inf).amin()
    tied = valid & (vnorm == best_vnorm)
    best_id = torch.where(tied, anchor_ids, sentinel).amin()
    index = torch.argmax((tied & (anchor_ids == best_id)).to(torch.int8))
    return {
        "index": index,
        "found": found,
        "valid": valid,
        "anchor_id": best_id,
        "order": order,
        "vnorm": vnorm,
    }


@torch.inference_mode()
def score_pdm_model(
    *,
    hard_violation: torch.Tensor,
    progress: torch.Tensor,
    min_ttc_s: torch.Tensor,
    comfortable: torch.Tensor,
    vnorm: torch.Tensor,
    ttc_horizon_s: float = 3.0,
    model_prior_weight: float = DEFAULT_MODEL_PRIOR_WEIGHT,
):
    """Compute PDM-style scores and return all ranking components.

    ``hard_violation`` contains only hard selector checks.  TTC is deliberately
    a weighted term here, matching PDMScorer's aggregate rather than turning a
    model prediction into an unconditional rejection.  Scores stay on the
    input device; only the final selected index needs to cross to Python.
    """

    tensors = (hard_violation, progress, min_ttc_s, comfortable, vnorm)
    if not tensors[0].is_cuda:
        # This function is also used by parity tests.  Production calls remain
        # on CUDA through v6_select.
        pass
    n = int(progress.numel())
    if any(x.ndim != 1 or int(x.numel()) != n for x in tensors):
        raise ValueError("all candidate quantities must be one-dimensional and aligned")
    if ttc_horizon_s <= 0 or model_prior_weight < 0:
        raise ValueError("invalid score weights or TTC horizon")

    # ``comfortable`` is a boolean mask, so finiteness is defined by the two
    # floating score inputs and the model prior only.
    finite = torch.isfinite(progress) & torch.isfinite(min_ttc_s) & torch.isfinite(vnorm)
    safe = (~hard_violation.bool()) & finite

    # PDMScorer normalizes progress by the best feasible raw progress.  Keep a
    # zero-progress fallback so stationary/short-route frames remain defined.
    raw_progress = torch.where(safe, progress.clamp_min(0), torch.zeros_like(progress))
    max_progress = raw_progress.max() if n else progress.new_tensor(0)
    progress_score = torch.where(
        max_progress > 0,
        raw_progress / max_progress.clamp_min(torch.finfo(progress.dtype).eps),
        torch.where(safe, torch.ones_like(progress), torch.zeros_like(progress)),
    )

    # PDM's TTC contribution is a bounded quality term.  A predicted collision
    # should therefore score low, while the hard mask remains reserved for
    # physical/map violations supplied by the caller.
    ttc_score = min_ttc_s.clamp_min(0).clamp_max(ttc_horizon_s) / ttc_horizon_s
    comfort_score = comfortable.bool().to(progress.dtype)

    # Deterministic FM has no density/log-likelihood head.  Lower vnorm is the
    # existing confidence ordering, converted to [0, 1] only within this
    # shortlist.  Diagnostics must call this model_prior_proxy, never prob.
    finite_v = torch.where(finite, vnorm, torch.inf)
    vmin = finite_v.min() if n else vnorm.new_tensor(0)
    vmax = torch.where(finite, vnorm, -torch.inf).max() if n else vnorm.new_tensor(0)
    vrange = (vmax - vmin).clamp_min(torch.finfo(vnorm.dtype).eps)
    model_prior_proxy = ((vmax - vnorm) / vrange).clamp(0, 1)
    model_prior_proxy = torch.where(finite, model_prior_proxy, torch.zeros_like(model_prior_proxy))

    denom = PDM_PROGRESS_WEIGHT + PDM_TTC_WEIGHT + PDM_COMFORT_WEIGHT + model_prior_weight
    weighted = (
        PDM_PROGRESS_WEIGHT * progress_score
        + PDM_TTC_WEIGHT * ttc_score
        + PDM_COMFORT_WEIGHT * comfort_score
        + model_prior_weight * model_prior_proxy
    ) / denom
    score = torch.where(safe, weighted, torch.zeros_like(weighted))
    return {
        "score": score,
        "safe": safe,
        "progress_score": progress_score,
        "ttc_score": ttc_score,
        "comfort_score": comfort_score,
        "model_prior_proxy": model_prior_proxy,
    }


@torch.inference_mode()
def select_pdm_model(scored, anchor_ids: torch.Tensor):
    """Select the best safe candidate with deterministic anchor-ID ties."""

    score = scored["score"]
    safe = scored["safe"] & torch.isfinite(score)
    if anchor_ids.ndim != 1 or anchor_ids.numel() != score.numel():
        raise ValueError("anchor_ids and scores must be aligned")
    best = torch.where(safe, score, -torch.inf).amax()
    tied = safe & (score == best)
    sentinel = torch.iinfo(anchor_ids.dtype).max
    best_id = torch.where(tied, anchor_ids, sentinel).amin()
    index = torch.argmax((tied & (anchor_ids == best_id)).to(torch.int8))
    return {"index": index, "found": safe.any(), "valid": safe, "anchor_id": best_id}
