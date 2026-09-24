"""Observed leaders impose a stopping station, independent of learned forecasts.

Recomputed each frame, so the wall advances with the actor and is removed when it leaves
the geometric path. No scenario ID, recorded future or map-specific rule.
"""
import numpy as np


def stop_station(path,tracks,ego,vehicle,clearance=1.):
    path=np.asarray(path,float)
    ds=np.linalg.norm(np.diff(path,axis=0),axis=1)
    keep=np.r_[True,ds>1e-6];path=path[keep]
    if len(path)<2:return None
    delta=np.diff(path,axis=0);length=np.linalg.norm(delta,axis=1)
    tangent=delta/length[:,None];arc=np.r_[0.,np.cumsum(length)]
    c,s=np.cos(ego.heading),np.sin(ego.heading);rotation=np.array([[c,-s],[s,c]])
    best=None
    for obj in tracks:
        if getattr(obj.tracked_object_type,'name','')!='VEHICLE':continue
        velocity=getattr(obj,'velocity',None)
        if velocity is None or not np.isfinite([velocity.x,velocity.y]).all():continue
        pos=(np.array([obj.center.x,obj.center.y])-[ego.x,ego.y])@rotation
        along=((pos-path[:-1])*tangent).sum(1)
        closest=path[:-1]+np.clip(along,0,length)[:,None]*tangent
        k=int(np.argmin(np.linalg.norm(pos-closest,axis=1)))
        yaw=obj.center.heading-ego.heading-np.arctan2(tangent[k,1],tangent[k,0])
        if np.cos(yaw)<.7:continue
        lateral=abs(np.cross(tangent[k],pos-path[k]))
        half_l=.5*(obj.box.length*abs(np.cos(yaw))+obj.box.width*abs(np.sin(yaw)))
        half_w=.5*(obj.box.width*abs(np.cos(yaw))+obj.box.length*abs(np.sin(yaw)))
        if lateral>half_w+vehicle.width/2+.1:continue
        station=arc[k]+along[k]
        if station<=0:continue
        local_velocity=np.array([velocity.x,velocity.y])@rotation
        forward_speed=max(0.,float(local_velocity@tangent[k]))
        # Allow only the leader's distance under immediate 3 m/s^2 braking.
        # Low-speed observation noise retains the previous stationary wall.
        extension=forward_speed**2/6. if forward_speed>.2 else 0.
        limit=station+extension-half_l-vehicle.length/2-vehicle.rear_axle_to_center-clearance
        if best is None or limit<best['station']:
            best=dict(station=float(limit),track_token=obj.track_token,observed_speed=float(np.hypot(velocity.x,velocity.y)),lead_stop_extension_m=float(extension),lead_braking_mps2=3.)
    return best
