"""Causal static-set membership; disappearance never leaves phantom obstacles."""
import math

class StaticTrackHistory:
    def __init__(self, duration=3., speed_threshold=.2, max_gap=.25, drift=.3):
        self.duration=duration;self.speed_threshold=speed_threshold
        self.max_gap=max_gap;self.drift=drift;self.history={};self.time=None
    def update(self, now, objects):
        """Objects: token, kind, x, y, speed. Returns current eligible tokens.

        Pedestrians/bicycles are deliberately excluded from stopped-vehicle
        freezing: low-speed VRUs require separate uncertainty modeling.
        """
        if self.time is not None and now<self.time:self.history.clear()
        self.time=now;seen=set();eligible=[]
        for obj in objects:
            token=obj['token'];seen.add(token);kind=obj['kind']
            if kind in ('TRAFFIC_CONE','BARRIER','GENERIC_OBJECT'):
                eligible.append(token);self.history.pop(token,None);continue
            if kind!='VEHICLE':self.history.pop(token,None);continue
            x,y,speed=obj['x'],obj['y'],obj['speed']
            if not all(math.isfinite(v) for v in (x,y,speed)) or speed>self.speed_threshold:
                self.history.pop(token,None);continue
            old=self.history.get(token)
            if old is None or now-old[1]>self.max_gap or math.hypot(x-old[2],y-old[3])>self.drift:
                old=(now,now,x,y)
            else:old=(old[0],now,old[2],old[3])
            self.history[token]=old
            if now-old[0]>self.duration:eligible.append(token)
        self.history={k:v for k,v in self.history.items() if k in seen}
        return eligible
