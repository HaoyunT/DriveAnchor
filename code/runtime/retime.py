"""Conservative path-preserving longitudinal retiming inspired by Walle refiner.
Native safety/comfort revalidation is performed by the caller after retiming.
"""
import numpy as np
from scipy.optimize import minimize,LinearConstraint
VERSION='path_preserving_accel_jerk_retime_v1'

def retime(xy,speed,acceleration,dt=.1):
    xy=np.asarray(xy,dtype=float)
    if xy.shape!=(40,2) or not np.isfinite(xy).all():raise ValueError('Finite40x2 path required')
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    if arc[-1]<1. or speed<0 or not -4.05<=acceleration<=2.4:return None,dict(reason='ineligible_initial_state_or_short_path')
    # Optimize39 interval accelerations. Velocity and distance follow discrete
    # constant-acceleration integration, tied to observed v0 and a0.
    n=39;V=dt*np.tril(np.ones((n,n)));S=np.zeros((n,n))
    for k in range(n):
        for j in range(k+1):S[k,j]=dt*dt*(k-j+.5)
    time=np.arange(1,40)*dt;base_s=speed*time
    J=np.diff(np.eye(n),axis=0)/dt
    desired=arc[1:]
    # Position fit and jerk penalty are convex; all constraints are linear.
    H=S.T@S+.002*np.eye(n)+.0002*(J.T@J)
    q=S.T@(base_s-desired)
    fun=lambda a:.5*float(a@H@a)+float(q@a)
    jac=lambda a:H@a+q
    rows=np.vstack([np.eye(n),V,S,J,np.eye(n)[0:1]])
    low=np.r_[np.full(n,-4.05),np.full(n,-speed),np.maximum(0,desired-.75)-base_s,np.full(n-1,-3.5),acceleration]
    high=np.r_[np.full(n,2.4),np.full(n,np.inf),np.minimum(arc[-1],desired+.75)-base_s,np.full(n-1,3.5),acceleration]
    # Require at least98% of original final path distance.
    low[2*n+n-1]=max(low[2*n+n-1],.98*arc[-1]-base_s[-1])
    x0=np.clip(np.gradient(np.r_[speed,np.diff(arc)/dt],dt)[:n],-4.05,2.4);x0[0]=acceleration
    result=minimize(fun,x0,jac=jac,constraints=[LinearConstraint(rows,low,high)],method='SLSQP',options=dict(maxiter=60,ftol=1e-7))
    values=rows@result.x
    if not result.success or not np.isfinite(result.x).all() or np.any(values<low-1e-5) or np.any(values>high+1e-5):return None,dict(reason='infeasible_or_unsolved',solver_message=str(result.message))
    s=np.r_[0.,base_s+S@result.x];valid=np.r_[True,np.diff(arc)>1e-8]
    refined=np.stack([np.interp(s,arc[valid],xy[valid,k]) for k in (0,1)],axis=1)
    delta=float(np.linalg.norm(refined-xy,axis=1).max())
    if delta<.005:return None,dict(reason='negligible_change',max_shift_m=delta)
    return refined,dict(reason='candidate',version=VERSION,max_shift_m=delta,progress_retention=float(s[-1]/arc[-1]),accel_min=float(result.x.min()),accel_max=float(result.x.max()),max_jerk=float(np.abs(J@result.x).max()))
