"""Online direction guard using observed map lanes, never future GT."""
import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.affinity import affine_transform


def prepare(lanes,ego,offset):
    c,s=np.cos(ego.heading),np.sin(ego.heading)
    transform=[c,s,-s,c,-c*ego.x-s*ego.y,s*ego.x-c*ego.y]
    data=[]
    for lane in lanes:
        poly=affine_transform(lane.polygon,transform);shapely.prepare(poly)
        p=np.array(affine_transform(lane.baseline_path.linestring,transform).coords)
        if len(p)<2:continue
        d=np.gradient(p,axis=0);d/=np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-8)
        data.append((poly,cKDTree(p),d))
    return data,offset,{}


def _bad_uncached(xy,context):
    xy=np.asarray(xy);d=np.gradient(xy,axis=1);norm=np.linalg.norm(d,axis=-1)
    heading=d/np.maximum(norm[...,None],1e-8)
    center=xy+context[1]*heading
    flat=center.reshape(-1,2);direction=heading.reshape(-1,2)
    forward=np.zeros(len(flat),bool);reverse=np.zeros(len(flat),bool)
    for poly,index,tangents in context[0]:
        ids=np.flatnonzero(shapely.intersects_xy(poly,flat[:,0],flat[:,1]))
        if not len(ids):continue
        _,k=index.query(flat[ids]);dot=(direction[ids]*tangents[k]).sum(1)
        forward[ids]|=dot>=0;reverse[ids]|=dot<-.1
    violation=(reverse&~forward).reshape(norm.shape)&(norm>.005)
    return violation.any(1)

def bad(xy,context):
    xy=np.asarray(xy)
    count=len(xy)
    if xy.strides[0]==0 and count>1:
        return np.repeat(bad(xy[:1],context),count)
    if len(context)<3:return _bad_uncached(xy,context)
    key=(xy.shape,xy.dtype.str,xy.tobytes())
    cache=context[2]
    if key not in cache:
        result=_bad_uncached(xy,context)
        if len(cache)>=8:cache.clear()
        cache[key]=result.copy()
    return cache[key].copy()
