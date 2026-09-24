"""Jerk-limited partial braking candidates; no future observations required."""
import numpy as np


def candidates(path, speed, acceleration, dt=.1, horizon=8.):
    path=np.asarray(path,float)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    keep=np.r_[True,np.diff(arc)>1e-6];arc=arc[keep];path=path[keep]
    if len(path)<2:return []
    out=[]
    for target in [1.,.5,0.,-.5,-1.,-1.5,-2.,-3.]:
        for duration in [1.,2.,4.,8.]:
            v=max(0.,float(speed));a=float(acceleration);station=0.
            positions=[0.];velocities=[v];accelerations=[a]
            # Integrate at 10 ms, retaining the 100 ms planner grid.
            for k in range(round(horizon/.01)):
                desired=target if k*.01<duration else 0.
                nxt=a+np.clip(desired-a,-.01,.01)
                nv=max(0.,v+.5*(a+nxt)*.01)
                station+=.5*(v+nv)*.01
                v=nv;a=nxt if v>0 else 0.
                if (k+1)%round(dt/.01)==0:
                    positions.append(station);velocities.append(v);accelerations.append(a)
            if station>arc[-1]+1e-6:continue
            xy=np.column_stack([np.interp(positions,arc,path[:,j]) for j in (0,1)])
            out.append(dict(xy=xy,target=target,duration=duration,speed=velocities,acceleration=accelerations))
    return out
