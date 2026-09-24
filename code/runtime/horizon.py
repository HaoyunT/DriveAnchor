"""Explicit 8 s planning seed. Learned points retain their original timestamps."""
import numpy as np
from route_sl_dp import Reference

DT=.1
HORIZON=8.
SAMPLES=81  # t=0 plus 80 future points


def extend_seed(route, selected):
    selected=np.asarray(selected,float)
    if selected.shape!=(40,2) or not np.isfinite(selected).all():
        raise ValueError('Expected original 40 points at t=0..3.9 s')
    ref=Reference(route)
    ss,ll=ref.project(selected)
    # Refine the sampled-reference projection before estimating speed: a
    # 0.2 m lookup grid otherwise biases the 0.5 s finite difference.
    for _ in range(4):
        delta=ref.curve(ss)-selected;d=ref.curve(ss,1)
        denom=(d*d).sum(1)+(delta*ref.curve(ss,2)).sum(1)
        ss=np.clip(ss-(delta*d).sum(1)/np.maximum(denom,1e-8),0,ref.arc[-1])
    d=ref.curve(ss,1);normal=np.stack([-d[:,1],d[:,0]],1)/np.linalg.norm(d,axis=1)[:,None]
    ll=((selected-ref.curve(ss))*normal).sum(1)
    if np.any(np.diff(ss)<-.5):return None,dict(reason='backward_seed')
    # This is a reference-route continuation, not an FM prediction. Terminal
    # forward speed is estimated over 0.5 s to reduce projection quantization.
    terminal_speed=max(0.,float((ss[-1]-ss[-6])/.5))
    tail_t=np.arange(1,SAMPLES-len(selected)+1)*DT
    station=ss[-1]+terminal_speed*tail_t
    if station[-1]>ref.arc[-1]-1e-4:
        return None,dict(reason='insufficient_route_for_8s',required_station=float(station[-1]),available_station=float(ref.arc[-1]))
    # Match position, velocity and acceleration at the 3.9 s join. A tail
    # starting with zero lateral derivative creates an artificial curvature
    # spike when the learned path is still turning.
    duration=tail_t[-1];u=tail_t/duration
    sample_t=np.arange(-5,1)*DT
    fit=np.polynomial.polynomial.polyfit(sample_t,selected[-6:],3)
    velocity=fit[1];acceleration=2*fit[2]
    end=station[-1];r1=ref.curve(end,1);r2=ref.curve(end,2)
    coefficients=np.zeros((6,2));coefficients[0]=selected[-1]
    coefficients[1]=velocity*duration;coefficients[2]=.5*acceleration*duration**2
    end_velocity=r1*terminal_speed;end_acceleration=r2*terminal_speed**2
    rhs=np.stack([ref.curve(end)-coefficients[:3].sum(0),
        end_velocity*duration-coefficients[1]-2*coefficients[2],
        end_acceleration*duration**2-2*coefficients[2]])
    coefficients[3:]=np.linalg.solve(np.array([[1,1,1],[3,4,5],[6,12,20]]),rhs)
    tail=np.stack([np.polynomial.polynomial.polyval(u,coefficients[:,k]) for k in range(2)],1)
    result=np.vstack([selected,tail])
    return result,dict(reason='extended_seed',horizon_s=8.,samples=81,
        learned_horizon_s=3.9,tail_source='C2 quintic join to reference route',terminal_speed=terminal_speed)
