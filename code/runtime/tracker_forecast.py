"""Finite frozen-plan longitudinal rollout; approximate comfort risk, no future GT.

Uses native velocity fitting, LQR gain and actuator lag. Applies nuPlan's two
Savitzky-Golay stages to actual past + predicted rear-axle acceleration. Lateral
controller coupling/center acceleration and future replanning remain unmodelled.
"""
from functools import lru_cache
import numpy as np
from scipy.signal import savgol_filter

@lru_cache(None)
def velocity_weights(samples):
    m=samples-1;dt=.1
    H=np.zeros((m,m));H[:,0]=dt;H[:,1:]=np.tril(np.ones((m,m-1)),-1)*dt**2
    R=np.column_stack([np.zeros(m-2),np.diff(np.eye(m-1),axis=0)])
    fit=np.linalg.pinv(H.T@H+1e-4*R.T@R)@H.T
    V=np.zeros((m,m));V[:,0]=1;V[:,1:]=np.tril(np.ones((m,m-1)),-1)*dt
    return V@fit

def risk(xy,heading,speed,acceleration,past=(),initial_boundary=False):
    xy=np.asarray(xy);heading=np.asarray(heading)
    f=np.stack([np.cos(heading[:-1]),np.sin(heading[:-1])],-1)
    profile=velocity_weights(len(xy))@((np.diff(xy,axis=0)*f).sum(-1))
    a=float(acceleration);v=float(speed);values=(list(past) if initial_boundary else list(past)[-14:])+[a];start=len(values)-1
    for step in range(14):
        ref=profile[10+step];gain=.5 if ref<=.2 and v<=.2 else 10./11.
        a+=(gain*(ref-v)-a)/3.;v+=a*.1;values.append(a)
    smooth=savgol_filter(values,8,2)
    jerk=savgol_filter(smooth,15,2,deriv=1,delta=.1)
    return float(np.max(np.abs(jerk[0 if initial_boundary else max(0,start-2):start+8])))
