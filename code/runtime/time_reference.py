"""Uncollapsed timed station reference; optional input for a future cap mode.

This supplies constraints only. It does not integrate, clip positions, or
claim to enforce safety. Do not combine with the old station-indexed original
speed: that representation retains the very stop bottleneck being replaced.
"""
import numpy as np

class TimeReference:
    def __init__(self,station,dt=.1,min_hold=.2):
        self.station=np.asarray(station,float)
        if self.station.ndim!=1 or len(self.station)<2 or not np.isfinite(self.station).all() or not np.isfinite(dt) or dt<=0:raise ValueError('Invalid timed station reference')
        if abs(float(self.station[0]))>1e-6:raise ValueError('Reference must start at station zero')
        if np.any(np.diff(self.station)<-1e-8):raise ValueError('Station must be monotonic')
        self.times=np.arange(len(self.station))*dt
        self.speed=np.maximum(np.gradient(self.station,dt),0.)
        stopped=np.diff(self.station)<=1e-8;self.holds=[];i=0
        while i<len(stopped):
            if not stopped[i]:i+=1;continue
            start=i
            while i<len(stopped) and stopped[i]:i+=1
            if (i-start)*dt>=min_hold:
                self.holds.append(dict(station=float(self.station[start]),start=float(self.times[start]),release=float(self.times[i]) if i<len(stopped) else float('inf')))
    def at(self,time):
        if time<0:raise ValueError('Negative rollout time')
        wall=next((h['station'] for h in self.holds if time<h['release']),None)
        return dict(reference_speed=float(np.interp(time,self.times,self.speed)),reference_station=float(np.interp(time,self.times,self.station)),hold_stop_station=wall)

def from_option(option,dt=.1):
    """Adapt existing wait_resume station/speed/stop_s/resume_s metadata."""
    reference=TimeReference(option['station'],dt=dt)
    if 'speed' in option:
        speed=np.asarray(option['speed'],float)
        if speed.shape!=reference.station.shape or not np.isfinite(speed).all() or (speed<0).any():raise ValueError('Invalid reference speed')
        reference.speed=speed
    if option.get('stop_s') is not None:
        stop=float(option['stop_s']);release=option.get('resume_s')
        release=float('inf') if release is None else float(release)
        if not np.isfinite(stop) or stop<0 or stop>reference.times[-1] or np.isnan(release) or release<stop:raise ValueError('Invalid hold timing')
        reference.holds=[dict(station=float(np.interp(stop,reference.times,reference.station)),start=stop,release=release)]
    return reference
