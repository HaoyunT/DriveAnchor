"""Center-pose map speed limit; lane preferred, connector uses adjacent limits."""
import numpy as np
import shapely

def limits(points,lanes,ego,rear_to_center):
    points=np.asarray(points)
    d=np.gradient(points,axis=-2)
    length=np.linalg.norm(d,axis=-1,keepdims=True)
    direction=np.divide(d,length,out=np.zeros_like(d),where=length>1e-8)
    direction=np.where(length>1e-8,direction,np.array([1.,0.]))
    center=points+rear_to_center*direction
    c,s=np.cos(ego.heading),np.sin(ego.heading)
    world=center@np.array([[c,s],[-s,c]])+[ego.x,ego.y]
    lane_limits=np.full(points.shape[:-1],np.inf)
    connector_limits=np.zeros(points.shape[:-1])
    for lane in lanes:
        inside=shapely.intersects_xy(lane.polygon,world[...,0],world[...,1])
        if 'Connector' in type(lane).__name__:
            values=[x.speed_limit_mps for x in lane.incoming_edges+lane.outgoing_edges]
            if values and all(values):connector_limits=np.where(inside,np.maximum(connector_limits,max(values)),connector_limits)
        elif lane.speed_limit_mps is not None and lane.speed_limit_mps>0:
            lane_limits=np.where(inside,np.minimum(lane_limits,lane.speed_limit_mps),lane_limits)
    return np.where(np.isfinite(lane_limits),lane_limits,np.where(connector_limits>0,connector_limits,np.inf))
