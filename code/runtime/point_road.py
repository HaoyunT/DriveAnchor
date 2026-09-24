"""Center-point broad phase in width-eroded road; full body remains authority."""
import numpy as np
import shapely

def attach(full_check,region,ego,half_width,rear_to_center):
    if half_width<=0:raise ValueError('Positive half width required')
    eroded=region.buffer(-half_width)
    shapely.prepare(eroded)
    c,s=np.cos(ego[2]),np.sin(ego[2]);rotation=np.array([[c,s],[-s,c]])
    origin=np.asarray(ego[:2])
    def point_check(xy,heading):
        center=np.asarray(xy)+rear_to_center*np.stack([np.cos(heading),np.sin(heading)],axis=-1)
        world=center@rotation+origin
        return bool(shapely.intersects_xy(eroded,world[...,0],world[...,1]).all())
    full_check.point_check=point_check
    return full_check
