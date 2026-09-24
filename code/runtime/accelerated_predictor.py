import pathlib,sys,time
import numpy as np
import torch
from base_predictor import Predictor as Base,local,stable_heading
ROOT=pathlib.Path('/autocar/.cache_pt_pytorch2/nuplan_recovery_20260910/dtpp_predictor_extracted_v1')
sys.path.insert(0,str(ROOT))
from prediction_parallel import Decoder,Encoder
from shared_filter import filter_candidates
class Predictor(Base):
 def __init__(self):
  w=torch.load(ROOT/'prediction_only.pth',map_location='cpu',weights_only=True)
  self.device='cuda';self.enc=Encoder().eval();self.dec=Decoder().eval()
  self.enc.load_state_dict(w['encoder'],strict=True);self.dec.load_state_dict(w['decoder'],strict=True)
  self.enc.cuda();self.dec.cuda()
 def prepare(self,*args,**kwargs):
  self._shared_cache={}
  return super().prepare(*args,**kwargs)
 @torch.inference_mode()
 def evaluate(self,xy):
  begin=time.perf_counter();xy=np.asarray(xy);n=len(xy);v=self.state.car_footprint.vehicle_parameters;ego=self.state.rear_axle
  steps=80 if xy.shape[1]==81 else 30
  # Preserve the validated heading convention and finite-difference preprocessing.
  yaw=stable_heading(xy);vel=np.gradient(xy,.1,axis=1);speed=np.linalg.norm(vel,axis=-1)
  six=np.stack([xy[...,0],xy[...,1],yaw,speed,np.gradient(speed,.1,axis=1),np.gradient(yaw,.1,axis=1)/np.maximum(speed,.1)],-1)
  six=torch.as_tensor(six,device='cuda',dtype=torch.float32);pred=[]
  # Shared field conditioned on current-speed straight ego reference.
  # Final single-path safety checks preserve candidate-conditioned decoding.
  x=torch.zeros(1,30,80,6,device='cuda')
  if n==1:
   x[0,0,:steps]=six[0,1:steps+1]
   pred=self.dec(self.context,x,self.features['neighbor_agents_past'],steps)[0,:1,:,:steps]
  else:
   if not hasattr(self,'_shared_cache'): self._shared_cache={}
   key=(id(self.context),steps)
   if key not in self._shared_cache:
    self._shared_cache={}
    speed0=float(self.state.dynamic_car_state.speed)
    x[0,0,:steps,0]=torch.arange(1,steps+1,device='cuda')*.1*speed0
    x[0,0,:steps,3]=speed0
    self._shared_cache[key]=self.dec(self.context,x,self.features['neighbor_agents_past'],steps)[0,:1,:,:steps]
   pred=self._shared_cache[key]
  if getattr(self,'capture_dp_lines',False):
   assert n==1
   ego=self.state.rear_axle
   self.dp_lines={o.track_token:np.concatenate([np.array([[*local([[o.center.x,o.center.y]],ego)[0],o.center.heading-ego.heading]]),pred[0,self.mapping[o.track_token]].detach().cpu().numpy()],axis=0) for o in self.tracks}

  m=len(self.tracks)
  def tensor(x):return torch.as_tensor(x,device='cuda',dtype=torch.float64)
  slots=[self.mapping[o.track_token] for o in self.tracks]
  current=np.asarray([[*local([[o.center.x,o.center.y]],ego)[0],o.center.heading-ego.heading] for o in self.tracks]).reshape(m,3)
  poses=torch.cat([tensor(current)[:,None],pred[0,slots].double()],1)
  result=filter_candidates(tensor(xy[:,:steps+1]),tensor(yaw[:,:steps+1]),tensor(vel[:,:steps+1]),poses,
    tensor([o.box.length for o in self.tracks]),tensor([o.box.width for o in self.tracks]),v.length,v.width,v.rear_axle_to_center)
  minimum=result['min_ttc'];collision=result['collision']
  finite=torch.isfinite(pred).all();assert bool(finite),'Nonfinite predictions'
  capped=minimum.clamp(max=3).cpu().numpy();hit=collision.cpu().numpy();bad=(capped<=.95)|hit
  return bad,dict(min_ttc_capped_s=capped.tolist(),predicted_collision=hit.tolist(),rejected_count=int(bad.sum()),dtpp_agents=m,total_objects=m,seconds=time.perf_counter()-begin,scope='experimental shared DTPP current-speed straight reference for batches; candidate-conditioned single-path final check; GPU exact filter; nearest10; static/CV guard retained separately',future_GT_input=False)
