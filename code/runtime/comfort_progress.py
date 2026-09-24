"""Candidate comfort proxy then signed route progress, with safety-first fallback.

Thresholds match the current native config. Candidate kinematics are a proxy;
the native metric still evaluates the controller's actual closed-loop states.
"""
import numpy as np
from scipy.signal import savgol_filter

LIMITS=dict(lon_accel_min=-4.05,lon_accel_max=2.4,lat_accel_abs=4.89,
            lon_jerk_abs=4.13,mag_jerk_abs=8.37,yaw_rate_abs=.95,yaw_accel_abs=1.93)
VERSION='safe_comfort_then_route_progress_v1'


def comfort_statistics(xy,heading,initial_velocity,initial_accel,history_accel=None,
                       history_heading=None,dt=.1,center_offset=1.461):
    xy=np.asarray(xy,dtype=float);heading=np.asarray(heading,dtype=float).copy()
    if xy.ndim!=3 or (xy.shape[2]!=2 or xy.shape[1]<15) or heading.shape!=xy.shape[:2]:
        raise ValueError('Expected N x40 x2 and N x40 headings')
    if not np.isfinite(xy).all() or not np.isfinite(heading).all():raise ValueError('Nonfinite trajectory')
    n=len(xy)
    # The current heading is observed, not the tangent guessed by the candidate.
    heading[:,0]=0.;heading=np.unwrap(heading,axis=1)
    velocity=np.gradient(xy,dt,axis=1,edge_order=2)
    velocity[:,0]=np.asarray(initial_velocity)
    rear_acc=np.gradient(velocity,dt,axis=1,edge_order=2)
    yaw_rate=np.gradient(heading,dt,axis=1,edge_order=2)
    yaw_acc=np.gradient(yaw_rate,dt,axis=1,edge_order=2)
    forward=np.stack([np.cos(heading),np.sin(heading)],axis=-1)
    side=np.stack([-np.sin(heading),np.cos(heading)],axis=-1)
    center_acc=rear_acc+center_offset*(yaw_acc[...,None]*side-yaw_rate[...,None]**2*forward)
    lon=(center_acc*forward).sum(-1);lat=(center_acc*side).sum(-1)
    lon[:,0]=initial_accel[0];lat[:,0]=initial_accel[1]
    past=np.asarray(history_accel if history_accel is not None else [],dtype=float).reshape(-1,2)[-14:]
    past_heading=np.asarray(history_heading if history_heading is not None else [],dtype=float)[-14:]
    if len(past)!=len(past_heading):raise ValueError('History lengths differ')
    offset=len(past)
    def prepend(values,prefix):return np.concatenate([np.broadcast_to(prefix,(n,len(prefix))),values],axis=1)
    lon=prepend(lon,past[:,0]);lat=prepend(lat,past[:,1])
    magnitude=np.sqrt(lon**2+lat**2)
    angles=np.unwrap(prepend(heading,past_heading),axis=1)
    # Native acceleration filter window8/poly2; derivative filter15/poly2/3.
    smooth=lambda x:savgol_filter(x,8,2,axis=1)
    slon,slat,smag=map(smooth,[lon,lat,magnitude])
    jlon=savgol_filter(slon,15,2,deriv=1,delta=dt,axis=1)
    jmag=savgol_filter(smag,15,2,deriv=1,delta=dt,axis=1)
    omega=savgol_filter(angles,15,2,deriv=1,delta=dt,axis=1)
    alpha=savgol_filter(angles,15,3,deriv=2,delta=dt,axis=1)
    future=slice(offset,None)
    stats=dict(lon_accel_min=slon[:,future].min(1),lon_accel_max=slon[:,future].max(1),
               lat_accel_abs=np.abs(slat[:,future]).max(1),lon_jerk_abs=np.abs(jlon[:,future]).max(1),
               mag_jerk_abs=np.abs(jmag[:,future]).max(1),yaw_rate_abs=np.abs(omega[:,future]).max(1),
               yaw_accel_abs=np.abs(alpha[:,future]).max(1))
    violation=np.column_stack([stats[k]<v if k.endswith('_min') else stats[k]>v for k,v in LIMITS.items()])
    return ~violation.any(1),stats,violation


def route_progress(xy,route):
    # Exact projection onto ordered native route segments, not Euclidean length.
    xy=np.asarray(xy,dtype=float);route=np.asarray(route,dtype=float)
    if route.ndim!=2 or route.shape[1]!=2 or len(route)<2:raise ValueError('Route needs two points')
    segments=np.diff(route,axis=0);length=np.linalg.norm(segments,axis=1);valid=length>1e-8
    if not valid.any():raise ValueError('Degenerate route')
    starts=route[:-1][valid];segments=segments[valid];length=length[valid]
    arc=np.r_[0.,np.cumsum(length[:-1])]
    points=xy[:,[0,-1]].reshape(-1,2)
    fraction=np.clip(((points[:,None]-starts)*segments).sum(-1)/(length**2),0.,1.)
    projection=starts+fraction[...,None]*segments
    nearest=((projection-points[:,None])**2).sum(-1).argmin(1)
    s=arc[nearest]+fraction[np.arange(len(points)),nearest]*length[nearest]
    s=s.reshape(-1,2)
    return s[:,1]-s[:,0]


def choose_index(quality,safety_violations,comfortable,progress,anchor_ids,baseline_index):
    safety=np.asarray(safety_violations,dtype=bool);comfortable=np.asarray(comfortable,dtype=bool)
    eligible=~safety.any(1)&comfortable
    if eligible.any():
        ids=np.flatnonzero(eligible)
        # User-requested progress-only selection, stable original ID tie.
        import torch
        gp=torch.as_tensor(np.asarray(progress)[ids],device='cuda',dtype=torch.float64)
        gid=torch.as_tensor(np.asarray(anchor_ids)[ids],device='cuda',dtype=torch.int64)
        best=gp.max();tie=gp==best
        best_id=torch.where(tie,gid,torch.iinfo(torch.int64).max).min()
        picked=torch.argmax((tie & (gid==best_id)).to(torch.int8))
        return int(ids[int(picked)]),eligible,False
    # Never trade an original safety violation for one less comfort violation.
    return int(baseline_index),eligible,True


def evaluate_cached(planner,xy,heading,route):
    key=xy.tobytes()
    if not hasattr(planner,'_comfort_progress_cache'):planner._comfort_progress_cache={}
    if key not in planner._comfort_progress_cache:
        history=planner._comfort_history;state=history[-1]
        velocity=state.dynamic_car_state.rear_axle_velocity_2d
        accel=state.dynamic_car_state.center_acceleration_2d
        previous=history[-15:-1]
        past_accel=[[s.dynamic_car_state.center_acceleration_2d.x,s.dynamic_car_state.center_acceleration_2d.y] for s in previous]
        past_heading=[s.rear_axle.heading-state.rear_axle.heading for s in previous]
        import torch,time
        from candidate_costs_gpu import comfort as gpu_comfort,route_progress as gpu_progress
        started=time.perf_counter()
        def tensor(x):return torch.as_tensor(x,device='cuda',dtype=torch.float64)
        gx=planner._candidate_xy_cuda.double() if key==getattr(planner,'_candidate_xy_key',None) else tensor(xy)
        gh,gr=tensor(heading),tensor(route)
        gv,ga=tensor([velocity.x,velocity.y]),tensor([accel.x,accel.y])
        gpast=tensor(past_accel).reshape(-1,2);gph=tensor(past_heading)
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
        good,stats,violations=gpu_comfort(gx,gh,gv,ga,gpast,gph,center_offset=state.car_footprint.vehicle_parameters.rear_axle_to_center)
        progress=gpu_progress(gx,gr)
        end.record()
        if not hasattr(planner,'_gpu_tensor_costs'):planner._gpu_tensor_costs={}
        planner._gpu_tensor_costs[key]=(gx,gh,gv,ga,gpast,gph,(good,stats,violations,progress))
        # Compatibility bridge remains until selector masks/diagnostics become device-resident.
        planner._comfort_progress_cache[key]=(good.cpu().numpy(),{k:v.cpu().numpy() for k,v in stats.items()},violations.cpu().numpy(),progress.cpu().numpy())
        planner._gpu_cost_timing=dict(core_ms=start.elapsed_time(end),adapter_wall_ms=(time.perf_counter()-started)*1000)
    return planner._comfort_progress_cache[key]
