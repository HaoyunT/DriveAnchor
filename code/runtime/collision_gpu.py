from observed_braking import displacement
"""Full four-axis oriented rectangle SAT overlap, float64; strict contact semantics."""
import numpy as np
import torch
@torch.inference_mode()
def collision_mask(centers,heading,tracks,ego,local,rules,decelerations=None):
 if not tracks:return np.zeros(len(centers),bool)
 def t(x):return torch.as_tensor(x,dtype=torch.float64,device='cuda')
 # Preserve NumPy local-coordinate conversion and timestamp calculations.
 predictions=[];yaw=[];length=[];width=[]
 for o in tracks:
  pos=local([[o.center.x,o.center.y]],ego)[0]
  vel=local([[ego.x+o.velocity.x,ego.y+o.velocity.y]],ego)[0] if hasattr(o,'velocity') else np.zeros(2)
  predictions.append(pos+displacement(vel,(decelerations or {}).get(getattr(o,"track_token",None),0.),np.arange(centers.shape[1])*rules.dt))
  yaw.append(o.center.heading-ego.heading);length.append(o.box.length);width.append(o.box.width)
 pred=t(np.asarray(predictions).transpose(1,0,2));yaw=t(yaw);length=t(length);width=t(width)
 center=t(centers)[:,:,None];head=t(heading)[:,:,None];collision=torch.zeros(len(centers),dtype=torch.bool,device='cuda')
 for j in range(0,len(tracks),32):
  angle=head-yaw[j:j+32];delta=center-pred[None,:,j:j+32]
  c=yaw[j:j+32].cos();s=yaw[j:j+32].sin()
  x=delta[...,0]*c+delta[...,1]*s;y=-delta[...,0]*s+delta[...,1]*c
  hl=length[j:j+32]/2+angle.cos().abs()*rules.half_length+angle.sin().abs()*rules.half_width
  hw=width[j:j+32]/2+angle.sin().abs()*rules.half_length+angle.cos().abs()*rules.half_width
  # Both rectangles contribute separating axes. Object axes alone can
  # falsely reject disjoint rotated boxes near a corner.
  ex=delta[...,0]*head.cos()+delta[...,1]*head.sin()
  ey=-delta[...,0]*head.sin()+delta[...,1]*head.cos()
  ehl=rules.half_length+angle.cos().abs()*length[j:j+32]/2+angle.sin().abs()*width[j:j+32]/2
  ehw=rules.half_width+angle.sin().abs()*length[j:j+32]/2+angle.cos().abs()*width[j:j+32]/2
  collision|=((x.abs()<hl)&(y.abs()<hw)&(ex.abs()<ehl)&(ey.abs()<ehw)).any(-1).any(-1)
 return collision.cpu().numpy()
