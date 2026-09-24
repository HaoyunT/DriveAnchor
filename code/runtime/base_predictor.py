import sys,pathlib,time
import numpy as np
import torch
from types import SimpleNamespace
from scipy.optimize import linear_sum_assignment
R=pathlib.Path('/autocar/.cache_pt_pytorch2/nuplan_recovery_20260910')
sys.path.insert(0,str(R/'open_source_dtpp_v1'))
from scenario_tree_prediction import Encoder,Decoder
import obs_adapter
from driveanchor_planner import local
from selection import stable_heading
from risk import swept_ttc
# This host PyTorch lacks CPU LAPACK; closed-form SE(2) inversion is equivalent.
def local_se2(states,origin,precision=torch.float64):
 states=states.to(precision);origin=origin.to(precision);d=states[:,:2]-origin[:2];c=torch.cos(origin[2]);sn=torch.sin(origin[2]);yaw=states[:,2]-origin[2]
 return torch.stack([d[:,0]*c+d[:,1]*sn,-d[:,0]*sn+d[:,1]*c,torch.atan2(torch.sin(yaw),torch.cos(yaw))],-1)
obs_adapter.global_state_se2_tensor_to_local=local_se2
# CPU-only feature preprocessing fallback, same least-squares system and cutoff.
original_lstsq=torch.linalg.lstsq
def cpu_lstsq(a,b,rcond=None,**kwargs):
 if a.device.type!='cpu':return original_lstsq(a,b,rcond=rcond,**kwargs)
 assert not a.requires_grad and not b.requires_grad
 if rcond is None:rcond=torch.finfo(a.dtype).eps*max(a.shape[-2:])
 solution,residuals,rank,singular=np.linalg.lstsq(a.numpy(),b.numpy(),rcond=rcond)
 return (torch.as_tensor(solution,dtype=a.dtype),torch.as_tensor(residuals),torch.as_tensor(rank),torch.as_tensor(singular))
torch.linalg.lstsq=cpu_lstsq
original_inv=torch.linalg.inv
def cpu_inv(a,**kwargs):
 if a.device.type!='cpu':return original_inv(a,**kwargs)
 assert not a.requires_grad
 return torch.from_numpy(np.linalg.inv(a.numpy())).to(a.dtype)
torch.linalg.inv=cpu_inv

class Predictor:
 def __init__(self):
  import hashlib
  p=R/'open_source_dtpp_v1/base_model.pth'
  assert hashlib.sha256(p.read_bytes()).hexdigest()=='aba84322e6e26ca4a3f33133a4feb43901e7c666d2db17234255cc832a49c073'
  weights=torch.load(p,map_location='cpu',weights_only=True)
  self.enc=Encoder().eval();self.dec=Decoder().eval()
  self.enc.load_state_dict(weights['encoder']);self.dec.load_state_dict(weights['decoder'])
  self.device='cuda';self.enc.to(self.device);self.dec.to(self.device)
 @torch.inference_mode()
 def prepare(self,inp,initialization):
  h=inp.history
  history=SimpleNamespace(ego_state_buffer=list(h.ego_states)[-22:],observation_buffer=list(h.observations)[-22:],current_state=h.current_state)
  self.features=obs_adapter.observation_adapter(history,list(inp.traffic_light_data or []),initialization.map_api,initialization.route_roadblock_ids,'cpu')
  self.state=h.ego_states[-1]
  types=[obs_adapter.TrackedObjectType.VEHICLE,obs_adapter.TrackedObjectType.PEDESTRIAN,obs_adapter.TrackedObjectType.BICYCLE]
  tracked=h.observations[-1].tracked_objects
  current=tracked.get_tracked_objects_of_types(types)
  slot=self.features['neighbor_agents_past'][0,:10,-1].numpy();present=np.flatnonzero(abs(slot).sum(-1)>0);self.mapping={}
  if len(present):
   anchor=obs_adapter.sampled_past_ego_states_to_tensor(history.ego_state_buffer)[-1].clone()
   tensor,_,_=obs_adapter.extract_agent_tensor(tracked,{},types)
   rel=obs_adapter.convert_absolute_quantities_to_relative(tensor,anchor,'agent')
   centers=rel[:,[obs_adapter.AgentInternalIndex.x(),obs_adapter.AgentInternalIndex.y()]].numpy()
   cost=np.linalg.norm(slot[present,None,:2]-centers[None],axis=-1);a,b=linear_sum_assignment(cost)
   assert len(a)==len(present) and np.max(cost[a,b])<.02
   self.mapping={current[k].track_token:int(present[j]) for j,k in zip(a,b)}
  self.tracks=[o for o in current if o.track_token in self.mapping]
  self.features={k:v.to(self.device) for k,v in self.features.items()}
  self.context=self.enc(self.features)
 @torch.inference_mode()
 def evaluate(self,xy):
  start=time.perf_counter();xy=np.asarray(xy);n=len(xy);state=self.state;ego=state.rear_axle;vehicle=state.car_footprint.vehicle_parameters
  yaw=stable_heading(xy);vel=np.gradient(xy,.1,axis=1);speed=np.linalg.norm(vel,axis=-1)
  six=np.stack([xy[...,0],xy[...,1],yaw,speed,np.gradient(speed,.1,axis=1),np.gradient(yaw,.1,axis=1)/np.maximum(speed,.1)],-1)
  predicted=[]
  for off in range(0,n,30):
   count=min(30,n-off);inputs=torch.zeros(1,30,80,6);inputs[0,:count,:30]=torch.from_numpy(six[off:off+count,1:31]).float()
   pred=self.dec(self.context,inputs.to(self.device),self.features['neighbor_agents_past'],30)[0][0,:count,:,:30].cpu().numpy()
   predicted.append(pred)
  pred=np.concatenate(predicted);assert np.isfinite(pred).all()
  tracks=self.tracks;m=len(tracks);minimum=np.full(n,np.inf);collision=np.zeros(n,bool)
  if m:
   poses=np.empty((n,m,31,3));ol=np.array([o.box.length for o in tracks]);ow=np.array([o.box.width for o in tracks]);times=np.arange(31)*.1
   for j,o in enumerate(tracks):
    pos=local([[o.center.x,o.center.y]],ego)[0];v=local([[ego.x+o.velocity.x,ego.y+o.velocity.y]],ego)[0] if hasattr(o,'velocity') else np.zeros(2)
    poses[:,j,:,:2]=pos+times[:,None]*v;poses[:,j,:,2]=o.center.heading-ego.heading
    if o.track_token in self.mapping:poses[:,j,1:]=pred[:,self.mapping[o.track_token]]
   av=np.gradient(poses[...,:2],.1,axis=2)
   centers=xy[:,:31]+vehicle.rear_axle_to_center*np.stack([np.cos(yaw[:,:31]),np.sin(yaw[:,:31])],-1)
   # Future-state TTC extrapolates up to 3s from each predicted state; it is a proxy.
   for t in range(31):
    tt=swept_ttc(centers[:,t],yaw[:,t],vel[:,t],poses[:,:,t,:2],poses[:,:,t,2],av[:,:,t],vehicle.length,vehicle.width,ol,ow)
    minimum=np.minimum(minimum,tt.min(-1));collision|=(tt==0).any(-1)
  bad=(minimum<=.95)|collision
  return bad,dict(min_ttc_capped_s=np.minimum(minimum,3.).tolist(),predicted_collision=collision.tolist(),rejected_count=int(bad.sum()),dtpp_agents=len(self.mapping),total_objects=m,seconds=time.perf_counter()-start,scope='DTPP first3s; current-state and predicted-state swept-box TTC horizon3s; nearest up-to10 dynamic actors only; not official TTC',future_GT_input=False)
