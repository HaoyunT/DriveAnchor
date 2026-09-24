"""Eight-second (s,v,a) beam DP on a geometric path; explicit constant jerk edges."""
import numpy as np
from scipy.interpolate import CubicSpline
from route_sl_dp import oriented_overlap

def ttc_risk(xy,heading,velocity,op,oy,ov,hl,hw,ego_hl,ego_hw,offset):
    e=np.stack([np.cos(heading),np.sin(heading)],-1);f=np.stack([-e[...,1],e[...,0]],-1)
    u=np.stack([np.cos(oy),np.sin(oy)],-1);side=np.stack([-u[...,1],u[...,0]],-1)
    delta=op-(xy+offset*e);dv=ov-velocity
    enter=np.zeros(heading.shape);leave=np.full(heading.shape,np.inf)
    for axis in (e,f,u,side):
        radius=ego_hl*np.abs((e*axis).sum(-1))+ego_hw*np.abs((f*axis).sum(-1))+hl*np.abs((u*axis).sum(-1))+hw*np.abs((side*axis).sum(-1))
        pos=(delta*axis).sum(-1);speed=(dv*axis).sum(-1);moving=np.abs(speed)>1e-9
        den=np.where(moving,speed,1.)
        a=(-radius-pos)/den;b=(radius-pos)/den
        enter=np.maximum(enter,np.where(moving,np.minimum(a,b),np.where(np.abs(pos)<=radius,-np.inf,np.inf)))
        leave=np.minimum(leave,np.where(moving,np.maximum(a,b),np.where(np.abs(pos)<=radius,np.inf,-np.inf)))
    return (enter<=leave)&(enter<=.95)

def search(path,speed,acceleration,obstacles,half_length,half_width,rear_to_center,beam=48,forbidden=None,max_curvature=.2,speed_limit=None):
    path=np.asarray(path,float)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    keep=np.r_[True,np.diff(arc)>1e-5];arc,path=arc[keep],path[keep]
    if len(arc)<4 or speed<0 or not -4.05<=acceleration<=2.4:return [],dict(reason='invalid_initial_state')
    curve=CubicSpline(arc,path,bc_type='natural')
    # Cubic Bezier control hull conservatively bounds every interpolated point.
    lengths=np.diff(arc)[:,None]
    controls=np.vstack([path,path[:-1]+curve(arc[:-1],1)*lengths/3,path[1:]-curve(arc[1:],1)*lengths/3])
    low=controls.min(0);high=controls.max(0)
    # State columns: station, speed, acceleration, accumulated cost.
    states=np.array([[0.,speed,acceleration,0.]])
    sequences=np.zeros((1,81));histories=np.zeros((1,40))
    jerk_controls=np.array([-3.5,-2.,-1.,0.,1.,2.,3.5])
    cache=[]
    times=np.arange(81)*.1
    for obj in obstacles:
        tg=np.asarray(obj['times']);poses=np.asarray(obj['poses'])
        if tg[-1]<8.-1e-8:return [],dict(reason='short_prediction')
        pp=np.stack([np.interp(times,tg,poses[:,k]) for k in range(2)],1)
        yaw=np.interp(times,tg,np.unwrap(poses[:,2]))
        extra=.3+np.interp(times,tg,obj.get('inflation',np.zeros(len(tg))))
        radius=np.hypot(half_length,half_width)+rear_to_center+np.hypot(obj['length']/2+extra.max(),obj['width']/2+extra.max())
        if not obj.get('native_predicted',True) and (np.any(pp.max(0)<low-radius) or np.any(pp.min(0)>high+radius)):
            continue
        cache.append((pp,yaw,obj['length']/2+extra,obj['width']/2+extra,np.gradient(pp,.1,axis=0),obj.get('native_predicted',True)))
    expanded=0
    rejected=dict(kinematics=0,path_end=0,collision=0,ttc=0,curvature=0,red_entry=0)
    if cache:
        objects=[np.stack([o[k] for o in cache],axis=1) for k in range(5)]
        predicted=np.asarray([o[5] for o in cache],bool)
    for step in range(40):
        parents=np.repeat(np.arange(len(states)),len(jerk_controls));j=np.tile(jerk_controls,len(states))
        old=states[parents];s,v,a=old[:,:3].T
        tau=np.array([.1,.2]);ss=s[:,None]+v[:,None]*tau+.5*a[:,None]*tau**2+j[:,None]*tau**3/6
        vv=v[:,None]+a[:,None]*tau+.5*j[:,None]*tau**2;aa=a[:,None]+j[:,None]*tau
        control_ok=np.ones(len(parents),bool)
        # Stop primitives reach v=a=0 smoothly. A fixed jerk alphabet almost
        # never lands exactly at rest, so it otherwise loses a feasible wait.
        for duration in [.2,.4,.8,1.2,2.]:
            p=np.arange(len(states));base=states[p];sv,v0,a0=base[:,:3].T
            b=(-3*v0-2*a0*duration)/duration**2;c=(2*v0+a0*duration)/duration**3
            st=sv[:,None]+v0[:,None]*tau+.5*a0[:,None]*tau**2+b[:,None]*tau**3/3+c[:,None]*tau**4/4
            vt=v0[:,None]+a0[:,None]*tau+b[:,None]*tau**2+c[:,None]*tau**3
            at=a0[:,None]+2*b[:,None]*tau+3*c[:,None]*tau**2
            jj=np.maximum(np.abs(2*b),np.abs(2*b+6*c*.2))
            if duration==.2:
                vt[:,-1]=0.;at[:,-1]=0.
            parents=np.r_[parents,p];old=np.vstack([old,base]);j=np.r_[j,jj]
            ss=np.vstack([ss,st]);vv=np.vstack([vv,vt]);aa=np.vstack([aa,at])
            control_ok=np.r_[control_ok,jj<=3.5+1e-8]
        s,v,a=old[:,:3].T
        good=(vv.min(1)>=0)&(aa.min(1)>=-3.)&(aa.max(1)<=2.)&(ss[:,-1]<=arc[-1])
        # Permit recovery from an observed acceleration outside preferred bounds.
        good|=((vv.min(1)>=0)&(aa.min(1)>=-4.05)&(aa.max(1)<=2.4)&(ss[:,-1]<=arc[-1])&
               (np.abs(aa[:,-1])<np.abs(a))&(step<10))
        good&=control_ok
        rejected['path_end']+=int((ss[:,-1]>arc[-1]).sum())
        rejected['kinematics']+=int((~good).sum())
        points=curve(ss);d=curve(ss,1);dd=curve(ss,2)
        if forbidden is not None:
            red=forbidden(points).any(1);rejected['red_entry']+=int((good&red).sum());good&=~red
        heading=np.arctan2(d[...,1],d[...,0])
        curvature=(d[...,0]*dd[...,1]-d[...,1]*dd[...,0])/np.maximum(np.linalg.norm(d,axis=-1)**3,1e-8)
        safe_turn=(np.abs(curvature).max(1)<=max_curvature)&(np.abs(vv*curvature).max(1)<=.8)&(np.abs(vv**2*curvature).max(1)<=3.5)
        rejected['curvature']+=int((good&~safe_turn).sum());good&=safe_turn
        if speed_limit is not None:
            within=(vv<=speed_limit(points)+1e-6).all(1)
            rejected['speed_limit']=rejected.get('speed_limit',0)+int((good&~within).sum());good&=within
        indices=np.arange(step*2+1,step*2+3)
        velocity=d/np.maximum(np.linalg.norm(d,axis=-1,keepdims=True),1e-8)*vv[...,None]
        if cache:
            pp,yy,hl,hw,ov=[o[indices] for o in objects]
            overlap=oriented_overlap(points[:,:,None,:],heading[:,:,None],pp[None],yy[None],hl[None],hw[None],half_length,half_width,rear_to_center)
            hit=overlap.any((1,2));rejected['collision']+=int((good&hit).sum());good&=~hit
            if predicted.any():
                risk=ttc_risk(points[:,:,None,:],heading[:,:,None],velocity[:,:,None,:],pp[None,:,predicted],yy[None,:,predicted],ov[None,:,predicted],hl[None,:,predicted],hw[None,:,predicted],half_length,half_width,rear_to_center).any((1,2))
                rejected['ttc']+=int((good&risk).sum());good&=~risk
        expanded+=len(parents)
        ids=np.flatnonzero(good)
        if not len(ids):return [],dict(reason='no_speed_path',reached_s=float(states[:,0].max()),reached_time=step*.2,expanded=expanded,rejected=rejected)
        # Prefer progress while charging acceleration and jerk. Diversity bins
        # retain slower arrivals that can survive a later conflict.
        cost=old[:,3]+.02*aa[:,-1]**2+.03*j**2
        order=ids[np.lexsort((cost[ids],-ss[ids,-1]))]
        unique={};groups={}
        for q in order:
            resting=abs(vv[q,-1])<1e-8 and abs(aa[q,-1])<1e-8
            key=(round(ss[q,-1]/.2),round(vv[q,-1]/.1),round(aa[q,-1]/.1),resting)
            if key in unique:continue
            unique[key]=True
            groups.setdefault((round(vv[q,-1]/.5),round(aa[q,-1]/.2),resting),[]).append(q)
        # Round-robin speed strata: a single progress-ranked beam prunes all
        # early braking alternatives before the obstacle becomes immediate.
        chosen=[]
        keys=list(groups)
        keys=[keys[i//2] if i%2==0 else keys[-1-i//2] for i in range(len(keys))]
        for depth in range(beam):
            for key in keys:
                if depth<len(groups[key]):chosen.append(groups[key][depth])
                if len(chosen)>=beam:break
            if len(chosen)>=beam:break
        q=np.asarray(chosen);parent=parents[q]
        sequences=sequences[parent].copy();sequences[:,step*2+1:step*2+3]=ss[q]
        histories=histories[parent].copy();histories[:,step]=j[q]
        states=np.column_stack([ss[q,-1],vv[q,-1],aa[q,-1],cost[q]])
    ranked=np.lexsort((states[:,3],-states[:,0]))
    groups={}
    for q in ranked:groups.setdefault((round(states[q,0]/2.),round(states[q,1]/.5)),[]).append(q)
    keys=list(groups);keys=[keys[i//2] if i%2==0 else keys[-1-i//2] for i in range(len(keys))]
    chosen=[]
    for depth in range(16):
        for key in keys:
            if depth<len(groups[key]):chosen.append(groups[key][depth])
            if len(chosen)>=16:break
        if len(chosen)>=16:break
    order=np.asarray(chosen)
    return [curve(sequences[q]) for q in order],dict(reason='speed_candidates',count=len(order),expanded=expanded,horizon_s=8.,end_progress_m=states[order,0].tolist(),rejected=rejected)
