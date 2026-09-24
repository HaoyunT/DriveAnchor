"""Bulk tensorization preserving upstream IDs, order, float32 rounding."""
import torch,numpy as np
import obs_adapter

def extract_agent_tensor(tracked_objects,track_token_ids,object_types):
 agents=tracked_objects.get_tracked_objects_of_types(object_types)
 values=[];types=[];I=obs_adapter.AgentInternalIndex
 for a in agents:
  if a.track_token not in track_token_ids:track_token_ids[a.track_token]=len(track_token_ids)
  row=[0.]*I.dim()
  for index,value in [(I.track_token(),track_token_ids[a.track_token]),(I.vx(),a.velocity.x),(I.vy(),a.velocity.y),(I.heading(),a.center.heading),(I.width(),a.box.width),(I.length(),a.box.length),(I.x(),a.center.x),(I.y(),a.center.y)]:row[index]=value
  values.append(row);types.append(a.tracked_object_type)
 return torch.from_numpy(np.asarray(values,dtype=np.float32).reshape(-1,I.dim())),track_token_ids,types

def filter_agents_tensor(agents,reverse=False):
 I=obs_adapter.AgentInternalIndex
 ids=(agents[-1] if reverse else agents[0])[:,I.track_token()]
 for i,a in enumerate(agents):
  if a.ndim!=2 or a.shape[1]!=I.dim():raise ValueError('Invalid agent shape')
  keep=torch.isin(a[:,I.track_token()],ids)
  agents[i]=a[keep]
 return agents

_lane_cache={}
def get_lane_polylines(map_api,point,radius):
 import nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils as v
 layers=[v.SemanticMapLayer.LANE,v.SemanticMapLayer.LANE_CONNECTOR]
 nearby=map_api.get_proximal_map_objects(point,radius,layers)
 objs=[o for layer in layers for o in nearby[layer]]
 objs.sort(key=lambda o:float(v.get_distance_between_map_object_and_point(point,o)))
 mid=[];left=[];right=[];ids=[]
 for o in objs:
  key=(id(map_api),type(o).__name__,o.id)
  if key not in _lane_cache:
   if len(_lane_cache)>4096:_lane_cache.clear()
   _lane_cache[key]=(map_api,tuple([v.Point2D(n.x,n.y) for n in path.discrete_path] for path in [o.baseline_path,o.left_boundary,o.right_boundary]))
  a,b,c=_lane_cache[key][1];mid.append(a);left.append(b);right.append(c);ids.append(o.id)
 return v.MapObjectPolylines(mid),v.MapObjectPolylines(left),v.MapObjectPolylines(right),v.LaneSegmentLaneIDs(ids)

def install():
 obs_adapter.extract_agent_tensor=extract_agent_tensor
 obs_adapter.filter_agents_tensor=filter_agents_tensor
 obs_adapter.get_lane_polylines=get_lane_polylines

# Native history observations are immutable snapshots. Retain references to prevent
# Python id reuse, and copy cached world tensors before relative transforms.
from collections import OrderedDict
_frame_cache=OrderedDict()
def sampled_tracked_objects_to_tensor_list(past):
 types=[obs_adapter.TrackedObjectType.VEHICLE,obs_adapter.TrackedObjectType.PEDESTRIAN,obs_adapter.TrackedObjectType.BICYCLE]
 ids={};out=[];out_types=[]
 for obs in past:
  key=id(obs)
  if key not in _frame_cache:
   t,local_ids,at=extract_agent_tensor(obs.tracked_objects,{},types)
   _frame_cache[key]=(obs,t,tuple(local_ids),at)
   if len(_frame_cache)>64:_frame_cache.popitem(last=False)
  _,cached,tokens,at=_frame_cache[key]
  row_ids=[]
  for token in tokens:
   if token not in ids:ids[token]=len(ids)
   row_ids.append(ids[token])
  t=cached.clone();t[:,obs_adapter.AgentInternalIndex.track_token()]=torch.tensor(row_ids,dtype=t.dtype)
  out.append(t);out_types.append(at)
 return out,out_types

_previous_install=install
def install():
 _previous_install()
 obs_adapter.sampled_tracked_objects_to_tensor_list=sampled_tracked_objects_to_tensor_list

_base_install=install
def install():
 _base_install()
 from tensor_map import map_process
 from route_map import get_route_lane_polylines_from_roadblock_ids
 obs_adapter.map_process=map_process
 obs_adapter.get_route_lane_polylines_from_roadblock_ids=get_route_lane_polylines_from_roadblock_ids
