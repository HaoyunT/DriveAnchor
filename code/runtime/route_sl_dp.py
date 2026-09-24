"""Route-relative lattice DP prototype, strictly fail-closed on infeasibility.

Route and obstacles are in rear-axle local coordinates. Quintic edges carry
lateral position and slope; zero lateral second derivative at lattice nodes
is a restricted curvature state, not a full vehicle-dynamics optimizer.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from collections import Counter


class Reference:
    def __init__(self, route):
        p = np.asarray(route, float)
        if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
            raise ValueError('invalid_route')
        p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-4]]
        if len(p) < 3:
            raise ValueError('short_route')
        self.arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        self.curve = CubicSpline(self.arc, p, bc_type='natural')
        self.grid = np.linspace(0, self.arc[-1], max(3, int(self.arc[-1]/.2)+1))
        self.points = self.curve(self.grid)
        self._point_index=cKDTree(self.points)

    def project(self, points):
        q = np.asarray(points).reshape(-1, 2)
        distance,nearest=self._point_index.query(q,k=2)
        i=nearest[:,0].copy()
        # Match np.argmin's first-index tie rule, including near-rounded ties.
        tolerance=np.maximum(1.,distance[:,0])*1e-12
        ambiguous=np.flatnonzero(distance[:,1]-distance[:,0]<=tolerance)
        for row in ambiguous:
            candidates=np.asarray(sorted(self._point_index.query_ball_point(q[row],distance[row,1]+tolerance[row])))
            squared=((q[row]-self.points[candidates])**2).sum(-1)
            i[row]=candidates[squared.argmin()]
        s = self.grid[i].copy()
        lo=self.grid[np.maximum(i-1,0)];hi=self.grid[np.minimum(i+1,len(self.grid)-1)]
        # Refine the coarse KD-tree seed to a continuous orthogonal projection.
        # Otherwise snapping the constructed path start to ego creates a kink.
        for _ in range(12):
            residual=self.curve(s)-q;first=self.curve(s,1);second=self.curve(s,2)
            derivative=(first*first).sum(1)+(residual*second).sum(1)
            step=(residual*first).sum(1)/np.maximum(derivative,1e-8)
            updated=np.clip(s-step,lo,hi)
            if np.max(np.abs(updated-s))<1e-10:
                s=updated;break
            s=updated
        tang = self.curve(s, 1)
        tang /= np.linalg.norm(tang, axis=1)[:, None]
        normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
        return s, ((q-self.curve(s))*normal).sum(1)

    def xy(self, s, l):
        d = self.curve(s, 1)
        return self.curve(s) + np.stack([-d[:, 1], d[:, 0]], axis=1) / np.linalg.norm(d, axis=1)[:, None] * l[:, None]


def edge(s0, s1, l0, slope0, l1, slope1, second0=0., second1=0.):
    h = s1-s0
    # Quintic Hermite interpolation with zero second derivative at both ends.
    a = np.zeros(6); a[:3] = [l0, slope0*h, .5*second0*h*h]
    a[3:] = np.linalg.solve(np.array([[1,1,1],[3,4,5],[6,12,20]]),
                            [l1-a[0]-a[1]-a[2], slope1*h-a[1]-2*a[2], second1*h*h-2*a[2]])
    u = np.linspace(0, 1, max(9, int(h/.2)+1))
    return s0+u*h, np.polynomial.polynomial.polyval(u, a)


def oriented_overlap(xy,heading,obstacle_xy,obstacle_heading,obstacle_hl,obstacle_hw,ego_hl,ego_hw,rear_to_center):
    xy=np.asarray(xy);heading=np.asarray(heading)
    e=np.stack([np.cos(heading),np.sin(heading)],-1);side=np.stack([-e[...,1],e[...,0]],-1)
    if np.asarray(obstacle_xy).ndim==3:xy=xy[:,None];e=e[:,None];side=side[:,None]
    u=np.stack([np.cos(obstacle_heading),np.sin(obstacle_heading)],-1);uside=np.stack([-u[...,1],u[...,0]],-1)
    delta=obstacle_xy-(xy+rear_to_center*e)
    c=np.abs((e*u).sum(-1));s=np.abs((e*uside).sum(-1))
    return ((np.abs((delta*e).sum(-1))<=ego_hl+obstacle_hl*c+obstacle_hw*s)&
            (np.abs((delta*side).sum(-1))<=ego_hw+obstacle_hl*s+obstacle_hw*c)&
            (np.abs((delta*u).sum(-1))<=obstacle_hl+ego_hl*c+ego_hw*s)&
            (np.abs((delta*uside).sum(-1))<=obstacle_hw+ego_hl*s+ego_hw*c))


def solve(route, selected, speed, initial_curvature, obstacles, half_length, half_width,
          rear_to_center, max_curvature, road_check=None, dt=.1):
    """obstacles: timestamped predicted poses and dimensions; caller reruns DTPP.

    Returns a path on the input time grid plus diagnostics, or None. DP maximizes
    reachable route station first; final acceptance retains original progress.
    """
    try:
        ref = Reference(route)
    except ValueError as e:
        return None, {'reason': str(e)}
    selected = np.asarray(selected, float)
    if selected.ndim != 2 or selected.shape[1] != 2 or len(selected)<3 or not np.isfinite(selected).all() or dt<=0:
        raise ValueError('Invalid selected trajectory or timestep')
    horizon=(len(selected)-1)*dt
    for obstacle in obstacles:
        tg=np.asarray(obstacle['times'])
        if len(tg)<2 or tg[0]>0 or tg[-1]<horizon-1e-8 or np.any(np.diff(tg)<=0):
            return None,dict(reason='insufficient_obstacle_time_coverage',horizon_s=horizon)
    ss, ll = ref.project(selected)
    s0, l0 = ref.project([[0., 0.]])
    start = float(s0[0]); l0 = float(l0[0])
    heading = float(np.arctan2(*ref.curve(start,1)[::-1]))
    if abs(heading) > .7 or abs(l0) > 3.5 or np.any(np.diff(ss) < -.5):
        return None, {'reason': 'ambiguous_route_or_backward_projection'}
    distance = float(ss[-1]-start)
    if distance < 6 or speed < 1:
        return None, {'reason': 'short_or_low_speed'}
    stations = np.linspace(start, min(ref.arc[-1], start+distance), max(6,int(np.ceil(horizon/.8))+1))
    time_s = np.maximum.accumulate(ss-start)
    valid = np.r_[True, np.diff(time_s)>1e-5]
    ts, times = time_s[valid], np.arange(len(selected))[valid]*dt
    # Work in Cartesian coordinates: clamped SL station loses longitudinal
    # distance for actors beyond route endpoints. Preserve uncertainty inflation.
    obstacle_data=[]
    for o in obstacles:
        poses=np.asarray(o['poses'],float)
        obstacle_data.append((np.asarray(o['times']),poses[:,:2],np.unwrap(poses[:,2]),
            np.asarray(o.get('inflation',np.zeros(len(poses)))),float(o['length'])/2,float(o['width'])/2))
    r1,r2,r3=ref.curve(start,1),ref.curve(start,2),ref.curve(start,3)
    v=float(np.linalg.norm(r1));dv=float(r1@r2)/v
    theta1=float(np.cross(r1,r2))/(v*v)
    theta2=float(np.cross(r1,r3))/(v*v)-2*theta1*dv/v
    A=v-theta1*l0
    if A<=1e-4:return None,{'reason':'singular_offset_geometry'}
    slope0=A*np.tan(-heading)
    second0=(initial_curvature*(A*A+slope0*slope0)**1.5-A*A*theta1+slope0*(dv-theta2*l0-2*theta1*slope0))/A
    nodes=[(l0,slope0,0.,[np.array([[0.,0.]])],second0)]
    reached=0; expanded=0;rejects=Counter()
    for layer in range(1,len(stations)):
        # All edges in a layer share the same longitudinal grid and timestamps.
        # Keep exact native arithmetic; reuse spline queries and obstacle lookups.
        h_layer=stations[layer]-stations[layer-1]
        station_grid=stations[layer-1]+np.linspace(0,1,max(9,int(h_layer/.2)+1))*h_layer
        reference_xy=ref.curve(station_grid)
        tangent=ref.curve(station_grid,1)
        normal=np.stack([-tangent[:,1],tangent[:,0]],axis=1)/np.linalg.norm(tangent,axis=1)[:,None]
        eta=np.interp(station_grid-start,ts,times)
        layer_objects=[]
        for tg,positions,yaws,inflation,hl,hw in obstacle_data:
            pos=np.stack([np.interp(eta,tg,positions[:,j]) for j in range(2)],1)
            yaw=np.interp(eta,tg,yaws);extra=.3+np.interp(eta,tg,inflation)
            layer_objects.append((pos,yaw,hl+extra,hw+extra))
        if layer_objects:
            object_pos=np.stack([o[0] for o in layer_objects],axis=1)
            object_yaw=np.stack([o[1] for o in layer_objects],axis=1)
            object_hl=np.stack([o[2] for o in layer_objects],axis=1)
            object_hw=np.stack([o[3] for o in layer_objects],axis=1)
        pref=np.interp(station_grid,ss,ll)
        nxt=[]
        for lateral in np.arange(-2.,2.01,.5):
            for slope in ([0.] if layer==len(stations)-1 else [-.12,0.,.12]):
                best=None
                for prev_l,prev_slope,cost,parts,prev_second in nodes:
                    expanded+=1
                    s,l=edge(stations[layer-1],stations[layer],prev_l,prev_slope,lateral,slope,prev_second,0.)
                    xy=reference_xy+normal*l[:,None]
                    d=np.gradient(xy,axis=0); h=np.unwrap(np.arctan2(d[:,1],d[:,0]))
                    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
                    k=np.gradient(h)/np.maximum(np.gradient(arc),1e-5)
                    if np.max(np.abs(k))>max_curvature:rejects['curvature']+=1;continue
                    if np.max(np.abs(np.diff(k))/np.maximum(np.diff(arc),1e-5))>.12:rejects['curvature_rate']+=1;continue
                    if layer==1 and (np.linalg.norm(xy[0])>.25 or abs(h[0])>.08 or abs(k[0]-initial_curvature)>.08):
                        rejects['start_state']+=1;continue
                    blocked=bool(layer_objects) and oriented_overlap(xy,h,object_pos,object_yaw,object_hl,object_hw,half_length,half_width,rear_to_center).any()
                    if blocked:rejects['obstacle_sat']+=1;continue
                    if road_check is not None and not getattr(road_check,'point_check',road_check)(xy,h):rejects['road_point']+=1;continue
                    total=cost+float(np.mean((l-pref)**2)+.1*np.mean(l*l)+8*np.mean(k*k))
                    if best is None or total<best[2]:best=(lateral,slope,total,parts+[xy[1:]],0.)
                if best is not None:nxt.append(best)
        if not nxt:break
        nodes=nxt;reached=layer
    diag={'horizon_s':horizon,'samples':len(selected),'reached_layer':reached,'expanded_edges':expanded,'blocked_objects':len(obstacle_data),'edge_rejections':dict(rejects)}
    if reached!=len(stations)-1:
        return None, dict(diag,reason='no_full_progress_feasible_path')
    def finalize(best):
        path=np.concatenate(best[3])
        # Validate the assembled path too: per-edge checks miss join artifacts.
        path_d=np.gradient(path,axis=0);path_h=np.unwrap(np.arctan2(path_d[:,1],path_d[:,0]))
        path_s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        path_k=np.gradient(path_h)/np.maximum(np.gradient(path_s),1e-5)
        if np.abs(path_k).max()>max_curvature or (np.abs(np.diff(path_k))/np.maximum(np.diff(path_s),1e-5)).max()>.12:
            return None,dict(diag,reason='assembled_curvature_or_rate')
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        desired=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(selected,axis=0),axis=1))]
        sample=desired/max(desired[-1],1e-6)*arc[-1]
        xy=np.stack([np.interp(sample,arc,path[:,k]) for k in (0,1)],axis=1)
        end_heading=np.arctan2(*(xy[-1]-xy[-2])[::-1])
        end_ref=np.arctan2(*ref.curve(stations[-1],1)[::-1])
        error=float(np.arctan2(np.sin(end_heading-end_ref),np.cos(end_heading-end_ref)))
        if abs(error)>.15:return None,dict(diag,reason='terminal_heading',terminal_heading_error=error)
        motion=np.gradient(xy,dt,axis=0);velocity=np.linalg.norm(motion,axis=1)
        final_heading=np.unwrap(np.arctan2(motion[:,1],motion[:,0]))
        if road_check is not None and (not road_check(path,path_h) or not road_check(xy,final_heading)):
            return None,dict(diag,reason='full_body_road_recheck')
        yaw=np.unwrap(np.arctan2(motion[:,1],motion[:,0]));omega=np.gradient(yaw,dt)
        max_yaw=float(np.abs(omega).max());max_lat=float(np.abs(velocity*omega).max())
        if max_yaw>.95 or max_lat>4.89:
            return None,dict(diag,reason='speed_related_turn_limit',max_yaw_rate=max_yaw,max_lateral_acceleration=max_lat)
        return xy,dict(diag,reason='candidate' ,terminal_heading_error=error,route_progress_m=distance)
    # Point-model search can admit a body-infeasible path. Try other final
    # lattice states before abandoning the candidate, retaining cost ordering.
    failures=Counter()
    for best in sorted(nodes,key=lambda node:node[2]):
        candidate,detail=finalize(best)
        if candidate is not None:return candidate,dict(detail,final_rejections=dict(failures))
        failures[detail['reason']]+=1
    return None,dict(diag,reason='all_final_states_rejected',final_rejections=dict(failures))
