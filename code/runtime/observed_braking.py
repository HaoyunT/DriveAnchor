"""Causal braking estimate from recent observed track speeds; no future GT."""
import numpy as np


def estimate(history):
    observations=list(history.observations)[-6:]
    states=list(history.ego_states)[-len(observations):]
    tracks={}
    for state,observation in zip(states,observations):
        t=state.time_point.time_us*1e-6
        for obj in observation.tracked_objects:
            v=getattr(obj,'velocity',None)
            if v is None or not np.isfinite([v.x,v.y]).all():continue
            tracks.setdefault(obj.track_token,[]).append((t,np.hypot(v.x,v.y)))
    groups={}
    for token,values in tracks.items():
        if len(values)>=4:groups.setdefault(len(values),[]).append((token,values))
    estimates={}
    for group in groups.values():
        values=np.asarray([x[1] for x in group])
        t=values[:,:,0]-values[:,-1:,0];speed=values[:,:,1]
        slopes=np.diff(speed,axis=1)/np.maximum(np.diff(t,axis=1),1e-6)
        acceleration=np.median(slopes,axis=1)
        eligible=(t[:,-1]-t[:,0]>=.25)&(acceleration<-.2)&(np.mean(slopes<0,axis=1)>=.8)
        for i in np.flatnonzero(eligible):estimates[group[i][0]]=max(-3.,float(acceleration[i]))
    return {token:estimates[token] for token in tracks if token in estimates}



def displacement(velocity,deceleration,times):
    velocity=np.asarray(velocity,float);times=np.asarray(times,float)
    speed=np.linalg.norm(velocity)
    if deceleration>=0 or speed<1e-6:return times[:,None]*velocity
    u=np.minimum(times,-speed/deceleration)
    distance=speed*u+.5*deceleration*u*u
    return distance[:,None]*(velocity/speed)
