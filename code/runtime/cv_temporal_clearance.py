"""GPU four-axis continuous time-interval intersection; CPU-compatible adapter."""
import numpy as np
import torch
@torch.inference_mode()
def conflict_mask(xy,heading,actors,*,ego_length,ego_width,center_offset,dt=.1,reserve_s):
 xy=np.asarray(xy,float);heading=np.asarray(heading,float)
 if xy.ndim==2:xy=xy[None];heading=heading[None]
 if xy.ndim!=3 or xy.shape[-1]!=2 or heading.shape!=xy.shape[:-1]:raise ValueError('shape')
 if not np.isfinite(xy).all() or not np.isfinite(heading).all() or dt<=0 or reserve_s<0:raise ValueError('invalid trajectory')
 if not actors:return np.zeros(xy.shape[:-1],bool)
 def t(x):return torch.as_tensor(x,device='cuda',dtype=torch.float64)
 # Preserve reference trigonometry and frame conversion during migration.
 f=t(np.stack([np.cos(heading),np.sin(heading)],-1))[:,:,None];l=torch.stack([-f[...,1],f[...,0]],-1)
 center=t(xy)[:,:,None]+center_offset*f;times=t(np.arange(xy.shape[1])*dt)
 mask=torch.zeros(xy.shape[:-1],device='cuda',dtype=torch.bool)
 for begin in range(0,len(actors),32):
  group=actors[begin:begin+32]
  pos=np.asarray([o['position'] for o in group],float);vel=np.asarray([o['velocity'] for o in group],float);yaw=np.asarray([o['heading'] for o in group],float);sizes=np.asarray([[o['length'],o['width']] for o in group],float)
  if not all(np.isfinite(v).all() for v in [pos,vel,yaw,sizes]) or (sizes<=0).any():raise ValueError('invalid actor geometry')
  af=t(np.stack([np.cos(yaw),np.sin(yaw)],-1))[None,None];al=torch.stack([-af[...,1],af[...,0]],-1);v=t(vel)[None,None];size=t(sizes)
  delta=center-(t(pos)[None,None]+times[None,:,None,None]*v)
  shape=delta.shape[:-1];low=(-torch.minimum(times,t(reserve_s)))[None,:,None].expand(shape).clone();high=torch.full_like(low,reserve_s);possible=torch.ones_like(low,dtype=torch.bool);nominal=possible.clone()
  for axis in [f,l,af,al]:
   p=(delta*axis).sum(-1);rate=-(v*axis).sum(-1)
   extent=ego_length/2*(f*axis).sum(-1).abs()+ego_width/2*(l*axis).sum(-1).abs()+size[:,0]/2*(af*axis).sum(-1).abs()+size[:,1]/2*(al*axis).sum(-1).abs()
   moving=rate.abs()>1e-12;den=torch.where(moving,rate,1.);a=(-extent-p)/den;b=(extent-p)/den
   low=torch.maximum(low,torch.where(moving,torch.minimum(a,b),-float('inf')));high=torch.minimum(high,torch.where(moving,torch.maximum(a,b),float('inf')))
   possible &= moving|(p.abs()<extent);nominal &= p.abs()<extent
  mask|=(nominal if reserve_s==0 else possible&(low<high)).any(-1)
 return mask.cpu().numpy()
