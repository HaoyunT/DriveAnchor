from observed_braking import displacement
"""Run the existing extracted DTPP decoder with all 80 future conditioning steps."""
import numpy as np
import torch


@torch.inference_mode()
def predict_lines(predictor,xy):
    xy=np.asarray(xy,float)
    if xy.shape!=(81,2) or not np.isfinite(xy).all():raise ValueError('8 seconds requires 81 points')
    from accelerated_predictor import stable_heading,local
    yaw=stable_heading(xy[None])[0]
    vel=np.gradient(xy,.1,axis=0);speed=np.linalg.norm(vel,axis=-1)
    six=np.stack([xy[:,0],xy[:,1],yaw,speed,np.gradient(speed,.1),np.gradient(yaw,.1)/np.maximum(speed,.1)],-1)
    x=torch.zeros(1,30,80,6,device=predictor.device)
    x[0,0]=torch.as_tensor(six[1:],device=predictor.device,dtype=x.dtype)
    output=predictor.dec(predictor.context,x,predictor.features['neighbor_agents_past'],80)
    pred=output[0,0].detach().cpu().numpy()
    if pred.shape[1:]!=(80,3) or not np.isfinite(pred).all():raise ValueError('Invalid DTPP 8s output')
    ego=predictor.state.rear_axle
    return {o.track_token:np.vstack([[*local([[o.center.x,o.center.y]],ego)[0],o.center.heading-ego.heading],pred[predictor.mapping[o.track_token]]]) for o in predictor.tracks}


def build(predictor,xy,tracks,ego):
    predictions=predict_lines(predictor,xy)
    c,s=np.cos(ego.heading),np.sin(ego.heading);rotation=np.array([[c,-s],[s,c]])
    times=np.arange(81)*.1;out=[];counts=dict(predicted=0,uncovered_dynamic=0,static=0)
    for obj in tracks:
        pos=(np.array([obj.center.x,obj.center.y])-[ego.x,ego.y])@rotation
        yaw=obj.center.heading-ego.heading
        poses=np.tile([*pos,yaw],(81,1));inflation=np.zeros(81)
        if obj.track_token in predictions:
            poses=predictions[obj.track_token];counts['predicted']+=1
        elif getattr(obj.tracked_object_type,'name','UNKNOWN') in ('VEHICLE','PEDESTRIAN','BICYCLE','UNKNOWN'):
            v=getattr(obj,'velocity',None)
            if v is not None and np.isfinite([v.x,v.y]).all():poses[:,:2]+=displacement(np.array([v.x,v.y])@rotation,getattr(predictor,"observed_decelerations",{}).get(obj.track_token,0.),times)
            else:inflation=10*times+1.5*times**2
            counts['uncovered_dynamic']+=1
        else:counts['static']+=1
        out.append(dict(times=times,poses=poses,length=obj.box.length,width=obj.box.width,inflation=inflation,
                        native_predicted=obj.track_token in predictions))
    return out,dict(counts,prediction_horizon_s=8.,native_prediction_steps=80,dt=.1,
        covered_tail='native DTPP, no CV tail',uncovered_model='observed braking when supported by history, else CV; unknown velocity envelope',future_GT_input=False)
