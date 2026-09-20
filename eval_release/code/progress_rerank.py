"""Rerank the score head's candidates by longitudinal progress along the route.

The head's label is ``softmax(-distance_to_expert/tau)``: it ranks by geometric
similarity to what the expert did and knows nothing else.  Closed loop shows
where that breaks.  Nine of 200 Val14 NR scenarios score zero on
``ego_is_making_progress`` with progress 0.03-0.20 against a 0.20 gate, four of
them ``starting_right_turn``; the car is stationary and picks a stationary
trajectory.  The reason is structural: ``ego_current_state`` -- speed,
acceleration, steer, yaw rate -- never enters the encoder, the context, or the
head, so no amount of training on this input set can teach it to pull away.

The planner does have that state.  ``_selection`` is handed ``tracks``,
``route`` and ``state.dynamic_car_state.speed``, which is exactly the signal the
model is missing, so a few lines here can supply what retraining cannot.

Two things this is deliberately *not*:

``score x progress``  The head emits raw logits from ``nn.Linear(hidden, 1)``,
    not probabilities.  A negative logit times a larger progress is a *smaller*
    product, so multiplying inverts the ranking wherever the logit is negative.
    Multiplying the softmax instead is ``log p + log(progress)``, and a
    stationary candidate has zero progress, so ``log 0 = -inf`` would make
    stopping unselectable.  That is the wrong direction: at-fault collisions are
    the *larger* zero class (11 of 32 NR, 14 of 33 R) and several sit at
    progress 1.00 in ``stopping_with_lead`` and ``waiting_for_pedestrian``,
    i.e. the planner already fails to stop when it should.

``an unconditional bias toward moving``  ``lambda`` is gated on measured
    clearance down the route corridor.  With something ahead it decays to zero
    and the selection is the head's argmax, unchanged.  Without that gate this
    would trade progress zeros for collision zeros at a worse exchange rate.

So: additive in log space, saturating in progress, gated on free space.

    final(c) = log_softmax(logits)[c] + lambda * min(s(c), s_ref) / s_ref
    lambda   = lambda0 * clip(gap_ahead / d_ref, 0, 1)

``s_ref`` saturates the reward so this cannot turn into "faster is always
better" -- speed-limit compliance is at 99.6% and is not worth spending.

A second, unconditional term subtracts predicted collision risk:

    final(c) = ... - mu * risk(c)

This is not gated, unlike the progress bonus: a candidate should never be
preferred *because* it is more likely to hit something, regardless of how
clear the road looks elsewhere.  ``risk`` extrapolates each tracked object at
constant velocity over the same 0.1 s grid the candidates are sampled on --
the planner has no future prediction for other agents at this point, and the
offline safety-mask labels this head's collision cases were built from used
the same constant-velocity-free approximation (a static centre-distance
threshold against the logged future).  The threshold, ``collision_m``,
defaults to the value ``build_safety_masks.py`` calibrated against the
logged expert: 3.0 m flags 6% of human-driven closest-approaches, 2.0 m flags
0.5%, so a value inside that band does not fire on ordinary lane-keeping.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

from comfort_progress import route_progress


def _env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _project(points: np.ndarray, route: np.ndarray):
    """Project N points onto a polyline at once: returns (arc_length, lateral).

    The per-agent version of this called ``route_progress`` in a Python loop --
    a full O(route) projection per agent per frame.  At ~40 agents and ~150
    frames that is thousands of passes over the polyline and it ran a scenario
    5x over budget (1 of 6 smoke cases in 20 minutes, 100% of one core).  One
    batched pass costs the same as the single largest of those.
    """
    segments = np.diff(route, axis=0)
    length = np.linalg.norm(segments, axis=1)
    keep = length > 1e-8
    if not keep.any():
        raise ValueError("Degenerate route")
    starts, segments, length = route[:-1][keep], segments[keep], length[keep]
    arc = np.r_[0.0, np.cumsum(length[:-1])]
    fraction = np.clip(((points[:, None] - starts) * segments).sum(-1) / length ** 2, 0.0, 1.0)
    offset = starts + fraction[..., None] * segments - points[:, None]
    squared = (offset ** 2).sum(-1)
    nearest = squared.argmin(1)
    rows = np.arange(len(points))
    return (arc[nearest] + fraction[rows, nearest] * length[nearest],
            np.sqrt(squared[rows, nearest]))


def corridor_clearance(tracks, ego, route, *, half_width_m: float,
                       horizon_m: float) -> float:
    """Metres of clear route ahead, capped at ``horizon_m``.

    Agents are projected onto the same polyline the candidates are scored
    against, so "ahead" means ahead *along the route* rather than ahead in the
    ego's current heading -- at the start of a turn those differ by most of the
    turn angle, and the heading version would call a blocked intersection clear.

    Lateral offset decides relevance: a car in the next lane is not in the way.
    ``half_width_m`` defaults to roughly one lane half-width plus the ego's
    half-width, so an oncoming vehicle on the far side of a 3.8 m lane does not
    suppress a legitimate pull-away.
    """
    if route is None or len(route) < 2:
        return 0.0
    centres = np.asarray([[obj.center.x, obj.center.y] for obj in tracks], dtype=float)
    if not len(centres):
        return horizon_m
    c, s = math.cos(ego.heading), math.sin(ego.heading)
    delta = centres - np.asarray([ego.x, ego.y], dtype=float)
    local = np.stack([c * delta[:, 0] + s * delta[:, 1],
                      -s * delta[:, 0] + c * delta[:, 1]], axis=1)
    # The ego sits at the origin of this frame; its own arc length is the datum
    # that turns absolute positions on the route into "ahead" and "behind".
    route = np.asarray(route, dtype=float)
    arc, lateral = _project(np.vstack([[0.0, 0.0], local]), route)
    ahead = arc[1:] - arc[0]
    blocking = (ahead > 0.0) & (ahead < horizon_m) & (lateral[1:] <= half_width_m)
    return float(ahead[blocking].min()) if blocking.any() else horizon_m


def collision_risk(xy: np.ndarray, tracks, ego, *, collision_m: float,
                   dt: float = 0.1) -> np.ndarray:
    """Per-candidate collision risk in [0, 1]: [K].

    ``xy`` is [K, T, 2] in the ego-local frame the candidates are already in.
    Each tracked object is advanced at its current (constant) velocity to the
    same waypoint times, so waypoint t is compared against where that agent is
    predicted to be at t, not a frozen snapshot -- a car crossing the ego's
    path is only flagged at the moment their paths actually cross, not for the
    whole horizon it happens to be nearby.

    A stationary object (parked car, waiting pedestrian) keeps velocity zero
    under this model and stays exactly where it is, which is the correct
    prediction for exactly the cases that matter most here: ``stopping_with_lead``
    and ``waiting_for_pedestrian_to_cross`` are stationary-obstacle scenes.

    Risk ramps linearly from 0 at ``collision_m`` to 1 at zero separation,
    rather than a hard 0/1 flag: a candidate that clears every agent by 10 cm
    should be ranked below one that clears by 3 m, and a step function cannot
    express that -- it would make this term as blind to margin as the
    geometric head already is.
    """
    if not len(tracks):
        return np.zeros(len(xy))
    c, s = math.cos(ego.heading), math.sin(ego.heading)
    centres = np.asarray([[t.center.x, t.center.y] for t in tracks], dtype=float)
    vel = np.asarray([[getattr(t.velocity, "x", 0.0), getattr(t.velocity, "y", 0.0)]
                      for t in tracks], dtype=float)
    delta = centres - np.asarray([ego.x, ego.y], dtype=float)
    local = np.stack([c * delta[:, 0] + s * delta[:, 1],
                      -s * delta[:, 0] + c * delta[:, 1]], axis=1)
    local_vel = np.stack([c * vel[:, 0] + s * vel[:, 1],
                          -s * vel[:, 0] + c * vel[:, 1]], axis=1)
    steps = np.arange(xy.shape[1]) * dt
    # [agent, time, xy]: where each tracked object is predicted to be at each
    # waypoint time, under the same constant-velocity model the offline safety
    # masks were labelled with.
    predicted = local[:, None, :] + local_vel[:, None, :] * steps[None, :, None]
    gap = np.linalg.norm(xy[:, None, :, :] - predicted[None, :, :, :], axis=-1)  # [K,agent,T]
    closest = gap.min(axis=(1, 2))
    return np.clip((collision_m - closest) / collision_m, 0.0, 1.0)


@torch.no_grad()
def rerank(logits: torch.Tensor, candidate_xy: torch.Tensor, route,
           tracks, ego) -> tuple[torch.Tensor, dict]:
    """Pick one anchor from the head's top-K after the progress term.

    ``logits`` is the head's score over the full 3000-anchor vocabulary;
    ``candidate_xy`` the matching trajectories.  Returns the winning index as a
    1-element tensor so the caller's top-1 short-circuit is unaffected, plus a
    diagnostic dict that records what the term actually did -- without it a run
    cannot distinguish "the gate never fired" from "the gate fired and did not
    help", and those call for opposite next steps.
    """
    k = int(_env("DRIVEANCHOR_RERANK_K", 50))
    lambda0 = _env("DRIVEANCHOR_RERANK_LAMBDA", 1.0)
    s_ref = _env("DRIVEANCHOR_RERANK_SREF", 8.0)
    d_ref = _env("DRIVEANCHOR_RERANK_DREF", 15.0)
    half_width = _env("DRIVEANCHOR_RERANK_HALFWIDTH", 2.5)
    mu = _env("DRIVEANCHOR_RERANK_MU", 0.0)
    collision_m = _env("DRIVEANCHOR_RERANK_COLLISION_M", 3.0)

    k = max(1, min(k, logits.numel()))
    top = logits.topk(k).indices
    head_choice = int(top[0])
    if k == 1 or (lambda0 <= 0.0 and mu <= 0.0):
        return top[:1], dict(rerank_applied=False, reason="disabled")

    xy = candidate_xy[top].detach().cpu().numpy()
    progress = np.asarray(route_progress(xy, np.asarray(route, dtype=float)), dtype=float)
    gap = corridor_clearance(tracks, ego, route,
                             half_width_m=half_width, horizon_m=d_ref)
    lam = lambda0 * float(np.clip(gap / max(d_ref, 1e-6), 0.0, 1.0)) if lambda0 > 0 else 0.0
    phi = np.clip(progress, 0.0, s_ref) / s_ref
    risk = collision_risk(xy, tracks, ego, collision_m=collision_m) if mu > 0 else np.zeros(len(top))
    base = torch.log_softmax(logits.float(), dim=0)[top].detach().cpu().numpy()
    final = base + lam * phi - mu * risk
    winner = int(np.argmax(final))
    chosen = int(top[winner])
    debug = os.environ.get("DRIVEANCHOR_RERANK_DEBUG")
    if debug:
        # The first closed-loop run came back byte-identical to the baseline,
        # which says the term never flipped an argmax but not why: lambda could
        # be zero from a blocked corridor, or the head's log-prob spread could
        # simply dwarf a bonus capped at lambda0.  Those need opposite fixes, so
        # record the two scales side by side rather than guess.
        import json
        with open(debug, "a") as handle:
            handle.write(json.dumps(dict(
                clearance_m=round(gap, 2), lam=round(lam, 4),
                logprob_top1=round(float(base[0]), 3),
                logprob_spread_top1_to_topk=round(float(base[0] - base[-1]), 3),
                max_progress_bonus=round(float(lam * phi.max()), 4),
                mu=mu, risk_top1=round(float(risk[0]), 4), risk_max=round(float(risk.max()), 4),
                risk_min=round(float(risk.min()), 4),
                risk_argmin_logprob_gap=round(float(base[0] - base[int(np.argmin(risk))]), 3),
                progress_top1_m=round(float(progress[0]), 2),
                progress_best_m=round(float(progress.max()), 2),
                # What lambda0 would it take to reach the best-progress
                # candidate at all?  That is the gap the term has to close,
                # and it is the number that decides whether this is a
                # tuning problem or a candidate-pool problem.
                logprob_gap_to_best_progress=round(
                    float(base[0] - base[int(np.argmax(progress))]), 3),
                k_used=int(len(top)),
                changed=bool(chosen != head_choice))) + "\n")
    return top[winner:winner + 1], dict(
        rerank_applied=True,
        lambda_effective=round(lam, 4),
        mu=mu,
        clearance_m=round(gap, 2),
        head_choice_id=head_choice,
        chosen_id=chosen,
        changed=bool(chosen != head_choice),
        head_choice_progress_m=round(float(progress[0]), 3),
        chosen_progress_m=round(float(progress[winner]), 3),
        head_choice_risk=round(float(risk[0]), 4),
        chosen_risk=round(float(risk[winner]), 4),
        logprob_given_up=round(float(base[0] - base[winner]), 4),
    )
