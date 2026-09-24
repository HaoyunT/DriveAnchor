"""Candidate-conditioned swept-box TTC; velocities are vectors, not heading speeds."""
import numpy as np

def swept_ttc(ec, ey, ev, oc, oy, ov, length, width, ol, ow, horizon=3.):
    # Candidate N by actor M, all positions in current ego coordinates.
    e=np.stack([np.cos(ey),np.sin(ey)],-1)[:,None];f=np.stack([-e[...,1],e[...,0]],-1)
    u=np.stack([np.cos(oy),np.sin(oy)],-1);v=np.stack([-u[...,1],u[...,0]],-1)
    d=oc-ec[:,None];dv=ov-ev[:,None]
    enter=np.zeros(oy.shape);leave=np.full(oy.shape,np.inf)
    for axis in (e,f,u,v):
        rad=length/2*abs((e*axis).sum(-1))+width/2*abs((f*axis).sum(-1))+ol/2*abs((u*axis).sum(-1))+ow/2*abs((v*axis).sum(-1))
        pos=(d*axis).sum(-1);vel=(dv*axis).sum(-1);moving=abs(vel)>1e-9;den=np.where(moving,vel,1)
        a=(-rad-pos)/den;b=(rad-pos)/den
        enter=np.maximum(enter,np.where(moving,np.minimum(a,b),np.where(abs(pos)<=rad,-np.inf,np.inf)))
        leave=np.minimum(leave,np.where(moving,np.maximum(a,b),np.where(abs(pos)<=rad,np.inf,-np.inf)))
    return np.where((enter<=leave)&(enter<horizon),enter,np.inf)

def test():
    args=(np.array([[0.,0.]]),np.array([0.]),np.array([[5.,0.]]),np.array([[[10.,0.]]]),np.array([[np.pi]]),np.array([[[-5.,0.]]]),2,2,np.array([2.]),np.array([2.]))
    assert np.allclose(swept_ttc(*args),.8)
    args=list(args);args[2]=np.zeros((1,2));args[5]=np.array([[[-5.,0.]]]);assert np.allclose(swept_ttc(*args),1.6)
    args[3]=np.array([[[10.,10.]]]);assert np.isinf(swept_ttc(*args)).all()
    args[3]=np.array([[[0.,0.]]]);assert swept_ttc(*args)[0,0]==0
