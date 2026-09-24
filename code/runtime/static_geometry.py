"""Current static obstacle footprints, transformed to the ego path frame."""
import math
from shapely.affinity import affine_transform
from shapely.ops import unary_union

def build(tracks, ego, now, memory):
    observations=[];by_token={}
    for obj in tracks:
        token=obj.track_token;kind=obj.tracked_object_type.name
        velocity=getattr(obj,'velocity',None)
        speed=math.hypot(velocity.x,velocity.y) if velocity is not None else 0.
        observations.append(dict(token=token,kind=kind,x=obj.center.x,y=obj.center.y,speed=speed))
        by_token[token]=obj
    tokens=memory.update(now,observations)
    c,s=math.cos(ego.heading),math.sin(ego.heading)
    transform=[c,s,-s,c,-c*ego.x-s*ego.y,s*ego.x-c*ego.y]
    shapes=[affine_transform(by_token[t].box.geometry,transform) for t in tokens]
    return (unary_union(shapes) if shapes else None),tokens
