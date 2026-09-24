"""Fast sufficient three-disc test with exact footprint fallback."""
import numpy as np
import shapely

def dimensions(length,width,rear_to_center):
    if length<=0 or width<=0:raise ValueError('Positive vehicle dimensions required')
    return rear_to_center+np.array([-length/3,0.,length/3]),float(np.hypot(width/2,length/6))

class ThreeCircleBoundary:
    def __init__(self,region,length,width,rear_to_center,margin=0.):
        self.offsets,self.radius=dimensions(length,width,rear_to_center)
        self.region=region.buffer(-self.radius-margin)
        self.footprint_region=region.buffer(-margin) if margin else region
        self.length=length;self.width=width;self.rear_to_center=rear_to_center
        shapely.prepare(self.region);shapely.prepare(self.footprint_region)
    def centers(self,xy,heading):
        direction=np.stack([np.cos(heading),np.sin(heading)],axis=-1)
        return np.asarray(xy)[...,None,:]+direction[...,None,:]*self.offsets[:,None]
    def feasible(self,xy,heading):
        xy=np.asarray(xy);heading=np.asarray(heading)
        centers=self.centers(xy,heading)
        good=shapely.intersects_xy(self.region,centers[...,0],centers[...,1]).all(axis=-1)
        flat=good.reshape(-1);idx=np.flatnonzero(~flat)
        if len(idx):
            p=xy.reshape(-1,2)[idx];h=heading.reshape(-1)[idx];c=np.cos(h);s=np.sin(h)
            cx=p[:,0]+self.rear_to_center*c;cy=p[:,1]+self.rear_to_center*s
            corners=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*[self.length/2,self.width/2]
            x=cx[:,None]+corners[None,:,0]*c[:,None]-corners[None,:,1]*s[:,None]
            y=cy[:,None]+corners[None,:,0]*s[:,None]+corners[None,:,1]*c[:,None]
            polygons=shapely.polygons(np.stack([x,y],-1))
            flat[idx]=shapely.covers(self.footprint_region,polygons)
        return good
    def paths_feasible(self,paths):
        paths=np.asarray(paths);delta=np.gradient(paths,axis=-2)
        heading=np.arctan2(delta[...,1],delta[...,0])
        return self.feasible(paths,heading).all(axis=-1)
