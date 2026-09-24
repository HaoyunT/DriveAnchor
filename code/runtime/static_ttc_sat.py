"""Continuous four-axis rectangle sweep; current static observations only."""
import numpy as np

def clearances(points, headings, rectangles, length, width, rear_to_center, max_distance=30.):
    points=np.asarray(points,float);h=np.asarray(headings,float);rect=np.asarray(rectangles,float)
    if rect.size==0:return np.full(len(points),np.inf)
    rect=rect.reshape(-1,4,2)
    edges=rect[:,1:3]-rect[:,:2];norm=np.linalg.norm(edges,axis=-1)
    if np.any(norm<1e-9):raise ValueError('Degenerate rectangle')
    object_axes=edges/norm[...,None]
    forward=np.stack([np.cos(h),np.sin(h)],-1);lateral=np.stack([-np.sin(h),np.cos(h)],-1)
    centers=points+rear_to_center*forward
    axes=np.concatenate([np.broadcast_to(np.stack([forward,lateral],1)[:,None],(len(points),len(rect),2,2)),np.broadcast_to(object_axes[None],(len(points),len(rect),2,2))],axis=2)
    projected=np.einsum('bkaq,kvq->bkav',axes,rect)
    ego_center=np.einsum('bkaq,bq->bka',axes,centers)
    velocity=np.einsum('bkaq,bq->bka',axes,forward)
    radius=length*.5*np.abs(velocity)+width*.5*np.abs(np.einsum('bkaq,bq->bka',axes,lateral))
    low=projected.min(-1)-ego_center-radius;high=projected.max(-1)-ego_center+radius
    moving=np.abs(velocity)>1e-10
    denominator=np.where(moving,velocity,1.)
    a,b=low/denominator,high/denominator
    entry=np.where(moving,np.minimum(a,b),-np.inf)
    leave=np.where(moving,np.maximum(a,b),np.inf)
    stationary_separated=(~moving)&((low>0)|(high<0))
    start=np.maximum(entry.max(-1),0.);end=leave.min(-1)
    valid=(~stationary_separated.any(-1))&(end>=start)&(start<=max_distance)
    return np.where(valid,start,np.inf).min(-1)

class Envelope:
    def __init__(self,rectangles,length,width,rear_to_center,horizon=1.6):
        self.rectangles=rectangles;self.length=length;self.width=width;self.offset=rear_to_center
        if horizon<=0:raise ValueError('Positive horizon required')
        self.horizon=horizon
    def __call__(self,points):
        delta=np.gradient(points,axis=0);h=np.arctan2(delta[:,1],delta[:,0])
        return clearances(points,h,self.rectangles,self.length,self.width,self.offset)/self.horizon
