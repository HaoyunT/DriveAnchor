"""Smooth route path tied to current rear-axle heading and curvature."""
import numpy as np
from route_sl_dp import Reference,edge

def build(route,speed,curvature):
    ref=Reference(route);s,l=ref.project([[0.,0.]])
    start=float(s[0]);l0=float(l[0]);r1,r2,r3=ref.curve(start,1),ref.curve(start,2),ref.curve(start,3)
    v=float(np.linalg.norm(r1));dv=float(r1@r2)/v
    theta1=float(np.cross(r1,r2))/(v*v);theta2=float(np.cross(r1,r3))/(v*v)-2*theta1*dv/v
    heading=float(np.arctan2(r1[1],r1[0]));A=v-theta1*l0
    if abs(heading)>.7 or abs(l0)>3.5 or A<=1e-4:return None
    slope=A*np.tan(-heading)
    second=(curvature*(A*A+slope*slope)**1.5-A*A*theta1+slope*(dv-theta2*l0-2*theta1*slope))/A
    distance=min(ref.arc[-1]-start,max(50.,speed*8+16.))
    if distance<10:return None
    transition=min(distance,max(15.,speed*3.))
    ss,ll=edge(start,start+transition,l0,slope,0.,0.,second,0.)
    if distance>transition:
        rest=np.linspace(start+transition,start+distance,max(2,int((distance-transition)/.2)+1))[1:]
        ss=np.r_[ss,rest];ll=np.r_[ll,np.zeros(len(rest))]
    path=ref.xy(ss,ll);path[0]=0.
    return path
