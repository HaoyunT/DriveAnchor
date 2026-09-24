"""Deployment-only candidate ranking; no oracle or logged-future inputs."""
import numpy as np
from dataclasses import dataclass,asdict
import hashlib,json

@dataclass(frozen=True)
class SelectorRules:
    version: str = "feasible_quality_v1"
    dt: float = .1
    points: int = 40
    speed_limit: float = 25.
    acceleration_limit: float = 4.5
    heading_hold_speed: float = .5
    center_offset: float = 1.461
    half_length: float = 2.588
    half_width: float = 1.1485
    route_weight: float = 1.
    speed_match_weight: float = .2
    acceleration_weight: float = .01
    endpoint_distance_weight: float = .15
    jerk_limit: None = None
    collision: str = "current-object constant velocity; conservative object-axis overlap; strict overlap"
    road: str = "all sampled vehicle corners within unbuffered nearby lane/connector union, including boundary"
    red: str = "hold current red; rear axle polygon entry including boundary; initial-inside excluded; missing connector skipped"
    yaw: str = "ego-local initial zero; low-speed hold; modulo-pi continuity"
    quality: str = "negative mean squared nearest-route-sample distance minus initial speed mismatch cost minus mean squared acceleration cost plus final endpoint norm benefit"
    fallback: str = "fewest failed constraint categories, then highest quality"
    tie: str = "first candidate in unchanged input order"

RULES=SelectorRules()
RULES_SHA256=hashlib.sha256(json.dumps(asdict(RULES),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def stable_heading(xy):
    delta=np.concatenate([xy[:,1:2]-xy[:,:1],xy[:,2:]-xy[:,:-2],xy[:,-1:]-xy[:,-2:-1]],axis=1)
    moving=np.linalg.norm(delta,axis=-1)/np.r_[RULES.dt,np.full(xy.shape[1]-2,2*RULES.dt),RULES.dt]>=RULES.heading_hold_speed
    raw=np.arctan2(delta[...,1],delta[...,0]);yaw=np.zeros_like(raw)
    for t in range(1,xy.shape[1]):
        turn=(raw[:,t]-yaw[:,t-1]+np.pi/2)%np.pi-np.pi/2
        yaw[:,t]=yaw[:,t-1]+np.where(moving[:,t],turn,0.)
    return yaw


def constrained_choice(quality,violations):
    """Maximize quality over feasible candidates; explicit minimum-count fallback."""
    quality=np.asarray(quality,float);violations=np.asarray(violations,bool)
    if quality.ndim!=1 or not len(quality) or violations.ndim!=2 or violations.shape[0]!=len(quality):
        raise ValueError('Expected nonempty quality[N], violations[N,C]')
    finite=np.isfinite(quality)
    if not finite.any():raise ValueError('No finite candidate score')
    counts=violations.sum(1);feasible=finite & (counts==0)
    pool=feasible if feasible.any() else finite & (counts==counts[finite].min())
    index=int(np.argmax(np.where(pool,quality,-np.inf)))
    return index,dict(selector_version=RULES.version,selector_rules_sha256=RULES_SHA256,feasible_count=int(feasible.sum()),candidate_count=len(quality),selection_fallback=not bool(feasible.any()),selected_quality=float(quality[index]),selected_violation_count=int(counts[index]))
