"""DTPP lines for DP; explicit uncertainty envelopes outside model coverage."""
import numpy as np


def build(predictor, selected, tracks, ego):
    predictor.capture_dp_lines = True
    try:
        predictor.evaluate(selected[None])
        predictions = predictor.dp_lines
    finally:
        predictor.capture_dp_lines = False
    c,s=np.cos(ego.heading),np.sin(ego.heading)
    rotation=np.array([[c,-s],[s,c]])
    times=np.arange(40)*.1
    out=[];counts={'predicted':0,'uncovered_dynamic':0,'static':0}
    for obj in tracks:
        pos=(np.array([obj.center.x,obj.center.y])-[ego.x,ego.y])@rotation
        yaw=obj.center.heading-ego.heading
        kind=getattr(getattr(obj,'tracked_object_type',None),'name','UNKNOWN')
        moving=kind in ('VEHICLE','PEDESTRIAN','BICYCLE','UNKNOWN')
        poses=np.tile([*pos,yaw],(40,1));inflation=np.zeros(40)
        if obj.track_token in predictions:
            poses[:31]=predictions[obj.track_token]
            tail=times[31:]-3.
            velocity=(poses[30,:2]-poses[29,:2])/.1
            poses[31:]=poses[30]
            poses[31:,:2]+=tail[:,None]*velocity
            # Same CV assumption as finite-horizon selector extrapolation.
            # This is a prediction assumption, not a reachability guarantee.
            counts['predicted']+=1
        elif moving:
            v=getattr(obj,'velocity',None)
            if v is not None:
                velocity=np.array([v.x,v.y])@rotation
                poses[:,:2]+=times[:,None]*velocity
            else:
                # Keep conservative uncertainty when velocity is unavailable.
                inflation=10.*times+1.5*times**2
            counts['uncovered_dynamic']+=1
        else:counts['static']+=1
        out.append(dict(times=times,poses=poses,length=obj.box.length,width=obj.box.width,inflation=inflation))
    return out,dict(counts,prediction_horizon_s=3.,tail_model='last predicted velocity extrapolation; uncovered actors observed CV; unknown velocity retains reachability envelope',conditioned_on='selected model trajectory')
