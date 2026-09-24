from pathlib import Path
import importlib.util,time,json
import numpy as np,shapely
from shapely.geometry import box
from scipy.spatial import cKDTree
r=Path('/autocar/.cache_pt_pytorch2/nuplan_recovery_20260910')
def load(name,d):
 s=importlib.util.spec_from_file_location(name,r/d/'direction_guard.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=load('old','gpu_fixed500_case_v1_code');new=load('new','gpu_selector_v1_code');poly=box(-20,-5,100,5);shapely.prepare(poly);pts=np.c_[np.arange(-20,101),np.zeros(121)];data=[(poly,cKDTree(pts),np.tile([1.,0.],(121,1)))];rng=np.random.default_rng(58);context=(data,1.3,{})
for n in [1,500]:
 for mode in range(12):
  xy=rng.normal(size=(n,40,2))*10
  if mode==0:xy=np.broadcast_to(xy[:1],(n,40,2))
  ref=old.bad(xy,(data,1.3))
  for _ in range(2):np.testing.assert_array_equal(ref,new.bad(xy,context))
xy=rng.normal(size=(500,40,2))*10;new.bad(xy,context);times=[[],[]]
for i in range(5):
 for k in ([0,1] if i%2==0 else [1,0]):
  t=time.perf_counter();[old,new][k].bad(xy,[(data,1.3),context][k]);times[k].append((time.perf_counter()-t)*1000)
print(json.dumps(dict(exact_comparisons=48,repeated_query_median_ms=[float(np.median(t)) for t in times])))
