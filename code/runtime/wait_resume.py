"""Explicit stop/wait/resume timing candidates on an unchanged polyline.

No obstacle future is assumed here. Caller must condition DTPP on each
candidate and apply full collision/TTC/dynamics/route checks before selection.
"""
import numpy as np


def candidates(path,speed,acceleration,horizon=8.,dt=.1):
    path=np.asarray(path,float)
    if path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or not np.isfinite(path).all():
        raise ValueError('finite XY path required')
    if speed<0 or not np.isfinite([speed,acceleration,horizon,dt]).all() or horizon<=0 or dt<=0:
        raise ValueError('invalid initial state')
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    keep=np.r_[True,np.diff(arc)>1e-6];path=path[keep];arc=arc[keep]
    if len(path)<2:return []
    t=np.arange(int(round(horizon/dt))+1)*dt
    out=[]
    for stop in [1.,1.5,2.,2.5,3.,4.,5.]:
        v0=float(speed);a0=float(acceleration)
        b=(-3*v0-2*a0*stop)/stop**2;c=(2*v0+a0*stop)/stop**3
        # Continuous extrema, not just sampled frames, certify longitudinal
        # braking bounds. v extrema occur where a=0, a extrema where jerk=0.
        points=[0.,stop]
        if abs(c)>1e-12:
            root=-b/(3*c)
            if 0<root<stop:points.append(root)
        roots=np.roots([3*c,2*b,a0]) if abs(c)>1e-12 else np.roots([2*b,a0]) if abs(b)>1e-12 else []
        points.extend(float(z.real) for z in roots if abs(z.imag)<1e-9 and 0<z.real<stop)
        q=np.asarray(points);vv=v0+a0*q+b*q*q+c*q**3;aa=a0+2*b*q+3*c*q*q
        if vv.min()<-1e-8 or aa.min()<-3.-1e-8 or aa.max()>2.+1e-8 or max(abs(2*b),abs(2*b+6*c*stop))>3.5+1e-8:continue
        z=np.minimum(t,stop)
        base_s=v0*z+.5*a0*z*z+b*z**3/3+c*z**4/4
        base_v=np.where(t<stop,v0+a0*z+b*z*z+c*z**3,0.)
        base_a=np.where(t<stop,a0+2*b*z+3*c*z*z,0.)
        if base_s[-1]<=arc[-1]+1e-8:
            xy=np.column_stack([np.interp(base_s,arc,path[:,k]) for k in (0,1)])
            out.append(dict(xy=xy,station=base_s,speed=base_v,acceleration=base_a,
                            stop_s=stop,wait_s=horizon-stop,resume_s=None,resume_accel=0.,timing_source='stop_hold'))
        for wait in [0.,.5,1.,2.,3.]:
            resume=stop+wait
            if resume>=horizon:continue
            for accel in [.5,1.,1.5]:
                # Ramp acceleration with bounded jerk, then hold acceleration.
                jerk=1.;u=np.maximum(t-resume,0.);ramp=accel/jerk;z=np.minimum(u,ramp);tail=np.maximum(u-ramp,0.)
                displacement=jerk*z**3/6+.5*jerk*ramp*ramp*tail+.5*accel*tail**2
                station=base_s+displacement
                velocity=base_v+.5*jerk*z*z+accel*tail
                acc=base_a+jerk*z
                if station[-1]>arc[-1]+1e-8:continue
                xy=np.column_stack([np.interp(station,arc,path[:,k]) for k in (0,1)])
                out.append(dict(xy=xy,station=station,speed=velocity,acceleration=acc,stop_s=stop,wait_s=wait,resume_s=resume,resume_accel=accel))
    return out
