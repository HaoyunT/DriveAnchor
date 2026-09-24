"""Build a speed envelope from this frame's causally eligible static boxes."""
import numpy as np
from static_ttc_sat import Envelope

def build(tracks, static_tokens, ego, vehicle_parameters):
    eligible=set(static_tokens)
    if not eligible:return None
    c,s=np.cos(ego.heading),np.sin(ego.heading)
    rotation=np.array([[c,-s],[s,c]])
    rectangles=[]
    for obj in tracks:
        if obj.track_token not in eligible:continue
        corners=np.asarray(obj.box.geometry.exterior.coords,dtype=float)[:-1]
        if corners.shape!=(4,2):raise ValueError('Static speed envelope requires an observed rectangle')
        rectangles.append((corners-[ego.x,ego.y])@rotation)
    if not rectangles:return None
    vp=vehicle_parameters
    return Envelope(np.asarray(rectangles),vp.length,vp.width,vp.rear_axle_to_center,horizon=1.6)
