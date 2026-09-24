"""Shared GPU prediction field and exact collision/TTC candidate filter.

Inputs stay on CUDA. Predictions must share a common ego conditioning reference.
This is a new prediction protocol, not equivalent to per-candidate DTPP decoding.
Static actors use constant poses; all actors (not only predicted nearest ten) may
be supplied. Missing predictions must be populated by the caller, never dropped.
"""
import torch


@torch.inference_mode()
def filter_candidates(xy, yaw, velocity, poses, lengths, widths,
                      ego_length, ego_width, center_offset, dt=.1,
                      ttc_horizon=3., ttc_threshold=.95, chunk=512):
    """Exact shared-field OBB collision and frozen-motion swept TTC on GPU.

    xy/velocity [N,T,2], yaw [N,T], poses [A,T,3] at the same times.
    Results remain on GPU. Contact counts as collision. Chunking bounds memory;
    no Python loop over individual trajectories or actors. TTC is evaluated at
    EVERY trajectory sample, not just its endpoint. Constant actor poses give
    zero velocity. Outputs distinguish overlap from future frozen-motion risk.
    """
    if not (xy.is_cuda and poses.is_cuda):
        raise ValueError('CUDA tensors required')
    n, t, _ = xy.shape
    if poses.shape[1:] != (t, 3) or t < 2 or dt <= 0:
        raise ValueError('aligned prediction timestamps and at least two points required')
    if yaw.shape != (n,t) or velocity.shape != xy.shape:
        raise ValueError('candidate shape mismatch')
    a = len(poses)
    if lengths.shape != (a,) or widths.shape != (a,):
        raise ValueError('actor size mismatch')
    collision = torch.zeros(n, device=xy.device, dtype=torch.bool)
    min_ttc = torch.full((n,), ttc_horizon, device=xy.device, dtype=xy.dtype)
    first = torch.full_like(min_ttc, float('inf'))
    invalid = ~(torch.isfinite(xy).flatten(1).all(1) & torch.isfinite(yaw).all(1)
                & torch.isfinite(velocity).flatten(1).all(1))
    field_bad = (~torch.isfinite(poses).all() | ~torch.isfinite(lengths).all()
                 | ~torch.isfinite(widths).all() | (lengths<=0).any() | (widths<=0).any())
    if a:
        oc = poses[...,:2].transpose(0,1)[None]
        oy = poses[...,2].transpose(0,1)[None]
        av = torch.gradient(poses[...,:2], spacing=dt, dim=1)[0].transpose(0,1)[None]
        u = torch.stack((oy.cos(), oy.sin()), -1)
        v = torch.stack((-u[...,1],u[...,0]),-1)
        times = torch.arange(t,device=xy.device,dtype=xy.dtype)*dt
        for start in range(0,n,chunk):
            end=min(start+chunk,n)
            h=yaw[start:end]
            e=torch.stack((h.cos(),h.sin()),-1)[:,:,None]
            f=torch.stack((-e[...,1],e[...,0]),-1)
            d=oc-(xy[start:end]+center_offset*e[:,:,0])[:,:,None]
            dv=av-velocity[start:end,:,None]
            enter=torch.zeros((end-start,t,a),device=xy.device,dtype=xy.dtype)
            leave=torch.full_like(enter,float('inf'))
            overlap=torch.ones_like(enter,dtype=torch.bool)
            for axis in (e,f,u,v):
                radius=(ego_length/2*abs((e*axis).sum(-1))
                        +ego_width/2*abs((f*axis).sum(-1))
                        +lengths/2*abs((u*axis).sum(-1))
                        +widths/2*abs((v*axis).sum(-1)))
                p=(d*axis).sum(-1); speed=(dv*axis).sum(-1)
                inside=abs(p)<=radius; overlap &= inside
                moving=abs(speed)>1e-9
                den=torch.where(moving,speed,1.)
                lo=(-radius-p)/den; hi=(radius-p)/den
                enter=torch.maximum(enter,torch.where(moving,torch.minimum(lo,hi),
                                    torch.where(inside,-float('inf'),float('inf'))))
                leave=torch.minimum(leave,torch.where(moving,torch.maximum(lo,hi),
                                    torch.where(inside,float('inf'),-float('inf'))))
            hit=overlap.any(-1)
            collision[start:end]=hit.any(-1)
            first[start:end]=torch.where(hit,times,float('inf')).amin(-1)
            tt=torch.where(enter<=leave,enter,float('inf'))
            min_ttc[start:end]=tt.amin((1,2)).clamp(max=ttc_horizon)
    invalid |= field_bad
    rejected=collision | (min_ttc<ttc_threshold) | invalid
    return dict(collision=collision,min_ttc=min_ttc,first_overlap_s=first,
                invalid=invalid,rejected=rejected,eligible=~rejected)


@torch.inference_mode()
def shortlist(result, scores, k=16):
    """Fixed-size GPU shortlist; validity mask MUST be honored by the caller.

    Higher score is better. No forced safety pass when all candidates fail.
    Invalid/nonfinite scores cannot enter the valid shortlist. Selection itself
    has no host copy or data-dependent nonzero synchronization.
    """
    valid=result['eligible'] & torch.isfinite(scores)
    values, ids=torch.topk(torch.where(valid,scores,-float('inf')),
                          min(k,len(scores)),sorted=True)
    return dict(ids=ids,valid=valid[ids],scores=values,
                eligible_count=valid.sum(),rejected_count=(~valid).sum())
