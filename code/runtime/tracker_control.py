"""Exact longitudinal next-step prediction for the frozen native LQR config.

The controller tracks velocity 1 s ahead, then applies an acceleration lag of
0.2 s. This is a next-step control estimate, not a full closed-loop metric.
"""
from functools import lru_cache
import numpy as np

@lru_cache(None)
def reference_weights(samples):
    m=samples-1;dt=.1
    H=np.zeros((m,m));H[:,0]=dt;H[:,1:]=np.tril(np.ones((m,m-1)),-1)*dt**2
    R=np.column_stack([np.zeros(m-2),np.diff(np.eye(m-1),axis=0)])
    fit=np.linalg.pinv(H.T@H+1e-4*(R.T@R))@H.T
    at_one_second=np.zeros(m);at_one_second[0]=1.;at_one_second[1:11]=dt
    return at_one_second@fit

def estimate(xy,heading,speed,acceleration):
    xy=np.asarray(xy,float);heading=np.asarray(heading,float)
    displacement=np.diff(xy,axis=-2)
    forward=np.stack([np.cos(heading[...,:-1]),np.sin(heading[...,:-1])],-1)
    projected=(displacement*forward).sum(-1)
    reference=projected@reference_weights(xy.shape[-2])
    gain=np.where((reference<=.2)&(speed<=.2),.5,10./11.)
    command=gain*(reference-speed)
    next_acceleration=acceleration+(command-acceleration)/3.
    return next_acceleration,(next_acceleration-acceleration)/.1
