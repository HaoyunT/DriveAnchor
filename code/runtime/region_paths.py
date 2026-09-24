from three_circle_boundary import ThreeCircleBoundary
from curvature_check import evaluate as native_evaluate
"""Bounded lateral lattice within forward route-connected lane polygons.

Generates geometric paths for the existing 8 s speed solver. Every final
candidate still passes the planner's native footprint/TTC/dynamics gates.
"""
import time
import shapely
from shapely.prepared import prep
import numpy as np
from shapely.geometry import Point,LineString
from shapely.affinity import affine_transform
from corridor_union import road_union
from route_sl_dp import Reference,edge


def generate(route,lanes,route_ids,ego,speed,curvature,half_width,tier=0,vehicle_length=None,rear_to_center=None,max_curvature=None,static_obstacles=None):
    begin=time.perf_counter();diag=dict(tier=tier,generated=0)
    ref=Reference(route);rids=set(map(str,route_ids))
    c,s=np.cos(ego.heading),np.sin(ego.heading)
    transform=[c,s,-s,c,-c*ego.x-s*ego.y,s*ego.x-c*ego.y]
    nearby=[l for l in lanes if str(l.get_roadblock_id()) in rids]
    # Include a connector only when its incoming and outgoing lanes are on route.
    for l in lanes:
        if l in nearby:continue
        if ('Connector' in type(l).__name__ and
            any(str(x.get_roadblock_id()) in rids for x in l.incoming_edges) and
            any(str(x.get_roadblock_id()) in rids for x in l.outgoing_edges)):
            nearby.append(l)
    approved=[]
    for l in nearby:
        line=affine_transform(l.baseline_path.linestring,transform)
        pos=line.interpolate(min(line.length/2,15.))
        ss,_=ref.project([[pos.x,pos.y]])
        a=line.interpolate(max(0,min(line.length/2,15.)-.3));b=line.interpolate(min(line.length,min(line.length/2,15.)+.3))
        d=np.array([b.x-a.x,b.y-a.y]);t=ref.curve(ss[0],1)
        aligned=d@t/max(np.linalg.norm(d)*np.linalg.norm(t),1e-8)>.5
        # At a turn entrance, projecting a distant lane midpoint onto the
        # clipped route can compare tangents from different road sections.
        # Classify the occupied lane by its local directed tangent instead.
        if l.polygon.distance(Point(ego.x,ego.y))<.5:
            local_s=line.project(Point(0.,0.))
            a=line.interpolate(max(0.,local_s-.3))
            b=line.interpolate(min(line.length,local_s+.3))
            local_d=np.array([b.x-a.x,b.y-a.y])
            aligned=local_d[0]/max(np.linalg.norm(local_d),1e-8)>.5
        if aligned:
            approved.append(l)
    # Restrict to directed reachability from an initial same-direction lane.
    pool={l.id:l for l in approved}
    starts=[l for l in approved if l.polygon.distance(Point(ego.x,ego.y))<.5]
    reachable=set();queue=list(starts)
    while queue:
        l=queue.pop()
        if l.id in reachable:continue
        reachable.add(l.id)
        queue.extend(x for x in l.outgoing_edges if x.id in pool and x.id not in reachable)
        # Same roadblock, same-direction internal lane changes are soft.
        queue.extend(x for x in approved if str(x.get_roadblock_id())==str(l.get_roadblock_id()) and x.id not in reachable)
    polys=[affine_transform(pool[k].polygon,transform) for k in reachable]
    if not polys:return [],dict(diag,reason='no_forward_route_region')
    assert vehicle_length is not None and rear_to_center is not None
    road=road_union(polys)
    if static_obstacles is not None and not static_obstacles.is_empty:
        road=road.difference(static_obstacles)
    body=ThreeCircleBoundary(road,vehicle_length,2*half_width,rear_to_center,margin=.15 if tier==0 else 0.)
    region=body.region
    prepared_region=prep(region)
    diag.update(lane_ids=sorted(reachable),region_area=region.area,body_model='three_circles_then_exact_footprint',circle_radius=body.radius,circle_offsets=body.offsets.tolist())
    start,l0=ref.project([[0.,0.]]);start=float(start[0]);l0=float(l0[0])
    r1,r2,r3=ref.curve(start,1),ref.curve(start,2),ref.curve(start,3)
    v=np.linalg.norm(r1);dv=r1@r2/v
    th1=np.cross(r1,r2)/v**2;th2=np.cross(r1,r3)/v**2-2*th1*dv/v
    heading=np.arctan2(r1[1],r1[0]);A=v-th1*l0
    if A<=1e-4 or abs(heading)>.9:return [],dict(diag,reason='initial_direction_mismatch')
    slope=A*np.tan(-heading)
    second=(curvature*(A*A+slope*slope)**1.5-A*A*th1+slope*(dv-th2*l0-2*th1*slope))/A
    end=min(ref.arc[-1],start+max(35.,speed*8+15.))
    if end-start<10:return [],dict(diag,reason='short_route')
    # Give the initial heading/curvature transition enough room in a turn.
    stations=np.linspace(start,end,max(3,int(np.ceil((end-start)/12.))+1))
    states=[(l0,slope,second,0.,[np.array([[0.,0.]])])];trace=[];rejected={'road':0,'curvature':0}
    edge_cache={}
    for i in range(1,len(stations)):
        # Dense road-derived lateral samples, no fixed +/-2 m limit.
        lateral=np.arange(-7.,7.01,.5);xy=ref.xy(np.full(len(lateral),stations[i]),lateral)
        tangent=ref.curve(stations[i],1);heading_ref=np.arctan2(tangent[1],tangent[0])
        # Keep a lateral sample if any terminal steering state is feasible.
        permitted=np.zeros(len(lateral),dtype=bool)
        for ds in [-.3,-.12,0.,.12,.3]:
            permitted|=body.feasible(xy,np.full(len(lateral),heading_ref+np.arctan(ds)))
        lateral=[float(l) for l,ok in zip(lateral,permitted) if ok]
        # Every edge in a station layer samples the identical route stations.
        grid,_=edge(stations[i-1],stations[i],0.,0.,0.,0.)
        derivative=ref.curve(grid,1)
        base_xy=ref.curve(grid)
        normal=np.stack([-derivative[:,1],derivative[:,0]],axis=1)/np.linalg.norm(derivative,axis=1)[:,None]
        # Match edge() arithmetic exactly, but reuse its invariant sample grid.
        h=stations[i]-stations[i-1]
        u=np.linspace(0,1,max(9,int(h/.2)+1))
        matrix=np.array([[1,1,1],[3,4,5],[6,12,20]])
        def lateral_edge(l0,slope0,l1,slope1,second0):
            key=(float(h),l0,slope0,l1,slope1,second0)
            if key in edge_cache:return edge_cache[key]
            a=np.zeros(6);a[:3]=[l0,slope0*h,.5*second0*h*h]
            a[3:]=np.linalg.solve(matrix,
                [l1-a[0]-a[1]-a[2],slope1*h-a[1]-2*a[2],0.*h*h-2*a[2]])
            result=np.polynomial.polynomial.polyval(u,a)
            edge_cache[key]=result
            return result
        # Batch exact GEOS predicates after curvature screening. Enumeration and
        # per-target strict comparison retain the original stable tie order.
        pending=[]
        for l in lateral:
            for slope1 in [-.3,-.12,0.,.12,.3]:
                for pl,ps,psecond,cost,parts in states:
                    ll=lateral_edge(pl,ps,l,slope1,psecond)
                    path=base_xy+normal*ll[:,None];feasible,k_cost=native_evaluate(path,max_curvature)
                    if not feasible:
                        rejected['curvature']+=1;continue
                    pending.append((l,slope1,cost,parts,path,ll,k_cost))
        best_by_target={}
        if pending:
            covered=body.paths_feasible(np.stack([x[4] for x in pending]))
            for item,inside in zip(pending,covered):
                if not inside:
                    rejected['road']+=1;continue
                l,slope1,cost,parts,path,ll,k_cost=item
                value=cost+float(np.mean(ll**2)*.05+k_cost*8+slope1**2)
                key=(l,slope1);best=best_by_target.get(key)
                if best is None or value<best[3]:
                    best_by_target[key]=(l,slope1,0.,value,parts+[path[1:]])
        nxt=list(best_by_target.values())
        nxt.sort(key=lambda x:x[3]);states=nxt[:18]
        trace.append(dict(s=float(stations[i]),l=[x[0] for x in states]))
        if not states:break
        if time.perf_counter()-begin>1.5:return [],dict(diag,reason='geometric_search_budget',trace=trace)
    result=[]
    if len(trace)==len(stations)-1 and states:
        # Distinct terminal offsets before filling with other steering states.
        seen=set()
        for state in states:
            key=round(state[0],1)
            if key in seen:continue
            seen.add(key);result.append(np.concatenate(state[4]))
            if len(result)==4:break
    if not result and tier==1:
        from xy_path_search import search
        result,xy_diag=search(route,body,curvature,max_curvature,distance=min(25.,end-start))
        diag['xy_fallback']=xy_diag
    return result,dict(diag,reason='paths' if result else 'no_geometric_path',generated=len(result),trace=trace,rejected=rejected,seconds=time.perf_counter()-begin)
