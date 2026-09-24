"""Bounded forward geometric search after a failed route-relative lattice."""
import time
import numpy as np
from scipy.spatial import cKDTree
from curvature_check import evaluate as native_evaluate


def search(route, body, initial_curvature, max_curvature, distance=25., budget=1.2):
    begin=time.perf_counter();tree=cKDTree(route)
    state=np.array([[0.,0.,0.,float(initial_curvature)]])
    paths=[np.array([[0.,0.]])];seen=set();trace=[]
    step=.5;increments=np.array([-.15,-.075,0.,.075,.15]);best=[]
    for depth in range(int(np.ceil(distance/step))):
        parent=np.repeat(np.arange(len(state)),len(increments));old=state[parent]
        k=np.clip(state[:,3,None]+increments,-max_curvature,max_curvature).ravel()
        points=[];heads=[]
        # Continuous steering transition: the first sample starts at measured
        # curvature; integrate a linear curvature ramp over the edge.
        x=old[:,0].copy();y=old[:,1].copy();h=old[:,2].copy()
        for j in range(1,5):
            ds=.125
            kmid=old[:,3]+(k-old[:,3])*((j-.5)/4.)
            scale=ds*np.sinc(ds*kmid/(2*np.pi))
            x=x+scale*np.cos(h+ds*kmid/2)
            y=y+scale*np.sin(h+ds*kmid/2)
            h=h+ds*kmid
            points.append(np.column_stack([x.copy(),y.copy()]));heads.append(h.copy())
        samples=np.stack(points,1);headings=np.stack(heads,1)
        good=body.feasible(samples,headings).all(1)
        end=samples[:,-1];h=headings[:,-1]
        dist,idx=tree.query(end)
        # Directed local route tangent prevents a forward search looping around.
        tangent=np.asarray(route)[np.minimum(idx+1,len(route)-1)]-np.asarray(route)[np.maximum(idx-1,0)]
        good &= (tangent[:,0]*np.cos(h)+tangent[:,1]*np.sin(h))>0
        ids=np.flatnonzero(good)
        if not len(ids):break
        cost=dist[ids]+.2*abs(k[ids])
        keep=[]
        for j in ids[np.argsort(cost,kind='stable')]:
            key=(round(end[j,0]/.2),round(end[j,1]/.2),round(h[j]/.05),round(k[j]/.075))
            if key in seen:continue
            seen.add(key);keep.append(j)
            if len(keep)>=600:break
        if not keep:break
        keep=np.asarray(keep)
        paths=[np.vstack([paths[parent[j]],samples[j]]) for j in keep]
        state=np.column_stack([end[keep],h[keep],k[keep]])
        trace.append(len(keep))
        if (depth+1)*step>=distance-1e-6:
            for path in paths:
                ok,_=native_evaluate(path,max_curvature)
                if ok and body.paths_feasible(path[None])[0]:
                    best.append(path)
                    if len(best)==4:break
            break
        if time.perf_counter()-begin>budget:break
    return best,dict(generated=len(best),layers=len(trace),states=trace,
                     seconds=time.perf_counter()-begin,method='forward_xy_exact_footprint',
                     reason='paths' if best else 'no_validated_xy_path')
