"""Device-resident route projection and comfort reductions.
Small Savitzky-Golay operators are built once from the reference filter, then cached on GPU.
"""
import torch
_FILTERS={}
LIMITS=dict(lon_accel_min=-4.05,lon_accel_max=2.4,lat_accel_abs=4.89,lon_jerk_abs=4.13,mag_jerk_abs=8.37,yaw_rate_abs=.95,yaw_accel_abs=1.93)
def unwrap(a):
 d=a[:,1:]-a[:,:-1];dd=torch.remainder(d+torch.pi,2*torch.pi)-torch.pi;dd=torch.where((dd==-torch.pi)&(d>0),torch.pi,dd)
 correction=torch.where(d.abs()<torch.pi,0.,dd-d)
 return torch.cat((a[:,:1],a[:,1:]+correction.cumsum(1)),1)
def smooth(a,window,poly,deriv=0,dt=.1):
 key=(a.shape[1],window,poly,deriv,dt,str(a.device),a.dtype)
 if key not in _FILTERS:
  import numpy as np
  from scipy.signal import savgol_filter
  matrix=savgol_filter(np.eye(a.shape[1]),window,poly,deriv=deriv,delta=dt,axis=1)
  _FILTERS[key]=torch.as_tensor(matrix,device=a.device,dtype=a.dtype)
 return a@_FILTERS[key]
@torch.inference_mode()
def comfort(xy,heading,initial_velocity,initial_accel,past_accel,past_heading,center_offset=1.461,dt=.1):
 assert xy.is_cuda
 h=heading.clone();h[:,0]=0.;h=unwrap(h);v=torch.gradient(xy,spacing=dt,dim=1,edge_order=2)[0];v[:,0]=initial_velocity
 acc=torch.gradient(v,spacing=dt,dim=1,edge_order=2)[0];omega=torch.gradient(h,spacing=dt,dim=1,edge_order=2)[0];alpha=torch.gradient(omega,spacing=dt,dim=1,edge_order=2)[0]
 f=torch.stack((h.cos(),h.sin()),-1);side=torch.stack((-h.sin(),h.cos()),-1);acc=acc+center_offset*(alpha[...,None]*side-omega[...,None]**2*f)
 lon=(acc*f).sum(-1);lat=(acc*side).sum(-1);lon[:,0]=initial_accel[0];lat[:,0]=initial_accel[1]
 offset=len(past_heading)
 def prefix(a,p):return torch.cat((p[None].expand(len(a),-1),a),1)
 lon=prefix(lon,past_accel[:,0]);lat=prefix(lat,past_accel[:,1]);mag=(lon**2+lat**2).sqrt();angles=unwrap(prefix(h,past_heading))
 slon=smooth(lon,8,2);slat=smooth(lat,8,2);smag=smooth(mag,8,2)
 stats={'lon_accel_min':slon[:,offset:].amin(1),'lon_accel_max':slon[:,offset:].amax(1),'lat_accel_abs':slat[:,offset:].abs().amax(1),'lon_jerk_abs':smooth(slon,15,2,1)[:,offset:].abs().amax(1),'mag_jerk_abs':smooth(smag,15,2,1)[:,offset:].abs().amax(1),'yaw_rate_abs':smooth(angles,15,2,1)[:,offset:].abs().amax(1),'yaw_accel_abs':smooth(angles,15,3,2)[:,offset:].abs().amax(1)}
 violation=torch.stack([stats[k]<v if k.endswith('_min') else stats[k]>v for k,v in LIMITS.items()],1)
 return ~violation.any(1),stats,violation
@torch.inference_mode()
def route_progress(xy,route,chunk=256):
 assert xy.is_cuda and route.is_cuda
 seg=route[1:]-route[:-1];length=torch.linalg.vector_norm(seg,dim=-1);valid=length>1e-8;starts=route[:-1][valid];seg=seg[valid];length=length[valid]
 if not len(length):raise ValueError('Degenerate route')
 arc=torch.cat((length.new_zeros(1),length.cumsum(0)[:-1]));points=xy[:,[0,-1]].reshape(-1,2);out=[]
 for q in points.split(chunk):
  f=(((q[:,None]-starts)*seg).sum(-1)/(length**2)).clamp(0,1);proj=starts+f[...,None]*seg;nearest=((proj-q[:,None])**2).sum(-1).argmin(1);out.append(arc[nearest]+f.gather(1,nearest[:,None])[:,0]*length[nearest])
 s=torch.cat(out).reshape(-1,2);return s[:,1]-s[:,0]
