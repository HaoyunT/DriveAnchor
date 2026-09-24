"""Extend a repaired path using its full geometry, retaining the timed prefix."""
import numpy as np
from horizon import extend_seed

def extend(route, selected, geometry=None, tolerance=.01):
    selected=np.asarray(selected,float)
    if geometry is None:return extend_seed(route,selected)
    path=np.asarray(geometry,float)
    if selected.shape!=(40,2) or path.ndim!=2 or path.shape[1]!=2 or not np.isfinite(path).all() or not np.isfinite(selected).all():
        raise ValueError('Invalid timed prefix or geometric path')
    if len(path)<2:return extend_seed(route,selected)
    length=np.linalg.norm(np.diff(path,axis=0),axis=1)
    path=path[np.r_[True,length>1e-9]]
    if len(path)<2:return extend_seed(route,selected)
    delta=np.diff(path,axis=0);length=np.linalg.norm(delta,axis=1)
    arc=np.r_[0.,np.cumsum(length)]
    offset=selected[:,None]-path[None,:-1]
    fraction=np.clip(np.sum(offset*delta,axis=-1)/(length*length),0.,1.)
    distance=np.linalg.norm(offset-fraction[...,None]*delta,axis=-1)
    nearest=distance.argmin(axis=1);error=distance[np.arange(40),nearest]
    station=arc[nearest]+fraction[np.arange(40),nearest]*length[nearest]
    # A later fallback may have selected genuinely different geometry. Do not
    # splice that prefix onto a stale DP path.
    if error.max()>tolerance or np.any(np.diff(station)<-.01):
        result,info=extend_seed(route,selected)
        return result,dict(info,geometry_seed_not_applicable=True,prefix_path_error_m=float(error.max()))
    speed=max(0.,float((station[-1]-station[-6])/.5))
    tail_station=np.minimum(station[-1]+speed*np.arange(1,42)*.1,arc[-1])
    tail=np.column_stack([np.interp(tail_station,arc,path[:,j]) for j in (0,1)])
    return np.vstack([selected,tail]),dict(reason='extended_geometry_seed',horizon_s=8.,samples=81,tail_source='full repaired geometry',terminal_speed=speed,available_path_m=float(arc[-1]),prefix_path_error_m=float(error.max()),path_end_reached=bool(tail_station[-1]>=arc[-1]))
