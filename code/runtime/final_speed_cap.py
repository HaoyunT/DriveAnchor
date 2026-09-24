"""Path-preserving final speed cap with bounded acceleration and jerk.

Applied after every selection/fallback branch. An initially infeasible speed
is recovered dynamically rather than hidden by clipping measured state.
"""
import numpy as np

def scalar_clip(x,low,high):
    return low if x<low else high if x>high else x

def apply(path,speed,acceleration,limit_at,dt=.1,count=40,stop_distance=None,extra_speed_envelope=None,time_reference=None):
    if isinstance(time_reference,dict):
        from time_reference import from_option
        time_reference=from_option(time_reference,dt=dt)
    path=np.asarray(path,float)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    original_speed=np.gradient(arc,dt)
    keep=np.r_[True,np.diff(arc)>1e-7];arc,path,original_speed=arc[keep],path[keep],original_speed[keep]
    if len(arc)<2:
        diagnostic=dict(reason='stationary')
        if time_reference is not None:diagnostic.update(time_reference_applied=True,hold_count=len(getattr(time_reference,'holds',[])),endpoint_exhausted_with_motion=bool(speed>1e-3))
        return np.repeat(path[:1],count,axis=0),diagnostic
    grid=np.linspace(0,arc[-1],max(3,int(np.ceil(arc[-1]/.2))+1))
    points=np.column_stack([np.interp(grid,arc,path[:,k]) for k in [0,1]])
    legal=np.asarray(limit_at(points),float)
    tangent=np.gradient(points,grid,axis=0)
    heading=np.unwrap(np.arctan2(tangent[:,1],tangent[:,0]))
    curvature=np.abs(np.gradient(heading,grid))
    turn_limit=np.minimum(np.sqrt(3.5/np.maximum(curvature,1e-6)),.8/np.maximum(curvature,1e-6))
    spatial_target=np.minimum(np.maximum(legal-.15,0.),turn_limit)
    target=np.minimum(np.interp(grid,arc,original_speed),spatial_target) if time_reference is None else spatial_target
    if stop_distance is not None:
        if not np.isfinite(stop_distance):raise ValueError('Nonfinite stopping distance')
        target=np.minimum(target,np.sqrt(3.*np.maximum(float(stop_distance)-grid,0.)))
    if extra_speed_envelope is not None:
        extra=np.asarray(extra_speed_envelope(points),float)
        if extra.shape != target.shape or np.isnan(extra).any() or (extra<0).any():
            raise ValueError('Invalid additional speed envelope')
        target=np.minimum(target,extra)
    # Need no invented motion beyond the end of the available geometric path.
    target[-1]=0.
    for i in range(len(grid)-2,-1,-1):target[i]=min(target[i],np.sqrt(target[i+1]**2+3.*(grid[i+1]-grid[i])))
    station=0.;v=max(0.,float(speed));a=float(acceleration)
    stations=[0.];speeds=[v];accels=[a];h=.01
    endpoint_exhausted=False
    for i in range((count-1)*10):
        cap=float(min(np.interp(station,grid,target),np.interp(station+max(v,0.)*.6,grid,target)))
        if stop_distance is not None:
            cap=min(cap,float(np.sqrt(3.*max(float(stop_distance)-station-v*.6,0.))))
        if time_reference is not None:
            reference=time_reference.at(i*h)
            ref_speed=float(reference['reference_speed'])
            if not np.isfinite(ref_speed) or ref_speed<0:raise ValueError('Invalid reference speed')
            cap=min(cap,ref_speed)
            hold_station=reference.get('hold_stop_station')
            if hold_station is not None:
                if not np.isfinite(hold_station):raise ValueError('Nonfinite hold station')
                cap=min(cap,float(np.sqrt(3.*max(float(hold_station)-station-v*.6,0.))))
        desired=float(scalar_clip((cap-v)/.45,-3.,2.))
        if a>0 and v+a*a/(2*3.5)>=cap-.015:desired=min(desired,0.)
        next_a=a+float(scalar_clip(desired-a,-3.5*h,3.5*h))
        next_a=float(scalar_clip(next_a,-3.,2.))
        next_v=max(0.,v+next_a*h)
        if time_reference is not None and station+.5*(v+next_v)*h>arc[-1] and next_v>1e-3:
            endpoint_exhausted=True
        station=min(arc[-1],station+.5*(v+next_v)*h)
        v,a=next_v,next_a
        if (i+1)%10==0:stations.append(station);speeds.append(v);accels.append(a)
    xy=np.column_stack([np.interp(stations,arc,path[:,k]) for k in [0,1]])
    excess=np.asarray(speeds)-np.interp(stations,grid,legal)
    diagnostic=dict(reason='final_map_speed_cap',initial_speed=float(speed),initial_limit=float(legal[0]),max_predicted_excess=float(np.maximum(excess,0).max()),min_acceleration=float(min(accels)),max_acceleration=float(max(accels)),station=float(stations[-1]),stop_distance=None if stop_distance is None else float(stop_distance))

    if time_reference is not None:
        diagnostic.update(time_reference_applied=True,hold_count=len(getattr(time_reference,'holds',[])),endpoint_exhausted_with_motion=bool(endpoint_exhausted))
    return xy,diagnostic
