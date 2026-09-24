"""Device-native decision core; caller must supply complete geometry/risk masks.
No CPU output/NumPy bridge inside choose_tensor. Not a replacement for missing gates.
"""
import torch
from candidate_costs_gpu import comfort
class RouteCache:
 def __init__(self,route):
  # Prepare once per route/frame outside timed candidate core.
  import numpy as np
  r=np.asarray(route,float);s=np.diff(r,axis=0);length=np.linalg.norm(s,axis=1);valid=length>1e-8
  if not valid.any():raise ValueError('degenerate route')
  self.starts=torch.as_tensor(r[:-1][valid],device='cuda',dtype=torch.float64)
  self.segments=torch.as_tensor(s[valid],device='cuda',dtype=torch.float64)
  self.length=torch.as_tensor(length[valid],device='cuda',dtype=torch.float64)
  self.arc=torch.as_tensor(np.r_[0.,np.cumsum(length[valid][:-1])],device='cuda',dtype=torch.float64)
 @torch.inference_mode()
 def progress(self,xy):
  points=xy[:,[0,-1]].reshape(-1,2);out=[]
  for q in points.split(1024):
   f=(((q[:,None]-self.starts)*self.segments).sum(-1)/self.length.square()).clamp(0,1)
   projection=self.starts+f[...,None]*self.segments
   nearest=(projection-q[:,None]).square().sum(-1).argmin(1)
   out.append(self.arc[nearest]+f.gather(1,nearest[:,None])[:,0]*self.length[nearest])
  s=torch.cat(out).reshape(-1,2);return s[:,1]-s[:,0]
@torch.inference_mode()
def choose_tensor(xy,heading,route_cache,complete_safety_mask,needs_retime,unreachable,remaining,anchor_ids,initial_velocity,initial_accel,past_accel,past_heading,center_offset=1.461,costs=None):
 if costs is None:
  good,stats,comfort_flags=comfort(xy,heading,initial_velocity,initial_accel,past_accel,past_heading,center_offset=center_offset)
  progress=route_cache.progress(xy)
 else:good,stats,comfort_flags,progress=costs
 finite=torch.isfinite(xy).flatten(1).all(1)&torch.isfinite(heading).all(1)&torch.isfinite(initial_velocity).all()&torch.isfinite(initial_accel).all()&torch.isfinite(past_accel).all()&torch.isfinite(past_heading).all()
 valid=finite & complete_safety_mask & ~unreachable & remaining & torch.isfinite(progress) & (good|needs_retime)
 # needs_retime does not override an independently failing complete safety mask.
 best=torch.where(valid,progress,-float('inf')).amax()
 tied=valid&(progress==best);sentinel=torch.iinfo(anchor_ids.dtype).max
 bestid=torch.where(tied,anchor_ids,sentinel).amin()
 index=torch.argmax((tied&(anchor_ids==bestid)).to(torch.int8))
 return dict(index=index,found=valid.any(),valid=valid,progress=progress,comfort_stats=stats,comfort_flags=comfort_flags,needs_retime=needs_retime,anchor_id=bestid)
