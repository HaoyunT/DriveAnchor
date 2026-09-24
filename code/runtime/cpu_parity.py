import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['NUPLAN_DATA_ROOT']='/root/nuplan/dataset'
os.environ['NUPLAN_MAPS_ROOT']='/workdir/nuplan_data/maps'
import sys,json,pathlib,time,hashlib
R=pathlib.Path('/autocar/.cache_pt_pytorch2/nuplan_recovery_20260910');out=R/'fm2_top250_cpu_parity_v3';out.mkdir(exist_ok=True)
m=json.loads((R/'stratified10_visuals_v1/manifest.json').read_text());prior=json.loads(pathlib.Path('/workdir/driveanchor_joint_three_20260910/pilot_first14_v2/closed_loop_nonreactive_agents/manifest.json').read_text())
sys.path.extend(prior['python_paths']);sys.path.insert(0,'/tmp/driveanchor_runtime_repairs_v5_20260910')
import torch,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from hydra.utils import instantiate
from hydra.core.utils import setup_globals
from omegaconf import OmegaConf
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
import repair_planner as rp
from driveanchor_planner import local
setup_globals();torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
cfg=OmegaConf.load(prior['reference_config']);OmegaConf.set_struct(cfg,False)
tokens=[r['scenario'] for r in m['scenarios']]
report=pd.read_parquet(pathlib.Path(prior['reference_config']).parents[2]/'runner_report.parquet')
logs=report[report.scenario_name.isin(tokens)].log_name.unique().tolist();assert len(report[report.scenario_name.isin(tokens)])==10
cfg.scenario_builder.db_files=['/root/nuplan/dataset/nuplan-v1.1/trainval/'+x+'.db' for x in logs]
cfg.scenario_builder.max_workers=2
cfg.scenario_filter.scenario_tokens=tokens;cfg.scenario_filter.log_names=logs;cfg.scenario_filter.shuffle=False
cfg.scenario_filter.timestamp_threshold_s=None
builder=instantiate(cfg.scenario_builder);scenarios=builder.get_scenarios(instantiate(cfg.scenario_filter),Sequential());by={s.token:s for s in scenarios};assert set(by)==set(tokens),(len(by),tokens)
checkpoint=R/'ttc_weight1_extend_v1/weight1/model_5508.pt'
# CPU-only parity: no CUDA initialization and no GPU reservation.
sys.path.insert(0,str(R/"open_source_dtpp_v1"))
import obs_adapter
original_extract=obs_adapter.extract_agent_tensor
original_history=obs_adapter.sampled_tracked_objects_to_tensor_list
original_filter=obs_adapter.filter_agents_tensor
original_lanes=obs_adapter.get_lane_polylines
original_map_process=obs_adapter.map_process
from fm2_planner import FM2Top250Planner
from cached_planner import CachedGeometryPlanner
from indexed_features import IndexedFeaturesPlanner
import fast_features
from shapely.geometry import box
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.state_representation import Point2D
planner=IndexedFeaturesPlanner.__new__(IndexedFeaturesPlanner);planner.device='cpu'
fast_features.install()
import spatial_index
checks=[]
for row in m['scenarios'][:3]:
 s=by[row['scenario']]
 sim=Simulation(simulation_setup=SimulationSetup(time_controller=instantiate(cfg.simulation_time_controller,scenario=s),observations=instantiate(cfg.observation,scenario=s),ego_controller=instantiate(cfg.ego_controller,scenario=s),scenario=s),simulation_history_buffer_duration=cfg.simulation_history_buffer_duration)
 init=sim.initialize();inp=sim.get_planner_input();planner.map=init.map_api;spatial_index.install(planner.map)
 history=inp.history;ego=history.ego_states[-1].rear_axle
 for offset in (0,1,3):
  past=list(history.observations)[offset:]
  # Original sampled-history global must also use original extraction.
  obs_adapter.extract_agent_tensor=original_extract
  expected,expected_types=original_history(past)
  obs_adapter.extract_agent_tensor=fast_features.extract_agent_tensor
  actual,actual_types=fast_features.sampled_tracked_objects_to_tensor_list(past)
  assert len(expected)==len(actual) and expected_types==actual_types
  assert all(torch.equal(a,b) for a,b in zip(expected,actual))
  for reverse in (False,True):
   a=original_filter([x.clone() for x in expected],reverse=reverse)
   b=fast_features.filter_agents_tensor([x.clone() for x in actual],reverse=reverse)
   assert all(torch.equal(x,y) for x,y in zip(a,b))
 layers=[SemanticMapLayer.LANE,SemanticMapLayer.LANE_CONNECTOR,SemanticMapLayer.INTERSECTION,SemanticMapLayer.CROSSWALK,SemanticMapLayer.STOP_LINE,SemanticMapLayer.WALKWAYS,SemanticMapLayer.CARPARK_AREA]
 for dx,dy in [(0,0),(25,-20),(-30,30)]:
  for radius in (20,60,100):
   patch=box(ego.x+dx-radius,ego.y+dy-radius,ego.x+dx+radius,ego.y+dy+radius)
   for layer in layers:
    a=type(planner.map)._get_proximity_map_object(planner.map,patch,layer)
    b=planner.map._get_proximity_map_object(patch,layer)
    assert [x.id for x in a]==[x.id for x in b],(layer,radius)
 for repeat in range(2):
  a=CachedGeometryPlanner.features(planner,history)
  b=IndexedFeaturesPlanner.features(planner,history)
  assert a[0].keys()==b[0].keys()
  assert all(torch.equal(a[0][k],b[0][k]) for k in a[0]),{k:float((a[0][k]-b[0][k]).abs().max()) for k in a[0]}
  assert torch.equal(a[1],b[1])
  assert [x.id for x in a[2]]==[x.id for x in b[2]]
  assert [x.track_token for x in a[3]]==[x.track_token for x in b[3]]
 spatial_index.update_roi(planner.map,ego)
 coords,tl=obs_adapter.get_neighbor_vector_set_map(planner.map,['LANE','ROUTE_LANES','CROSSWALK'],Point2D(ego.x,ego.y),80,init.route_roadblock_ids,inp.traffic_light_data)
 args=(ego,coords,tl,['LANE','ROUTE_LANES','CROSSWALK'],{'LANE':40,'ROUTE_LANES':10,'CROSSWALK':5},{'LANE':50,'ROUTE_LANES':50,'CROSSWALK':30},'linear')
 for repeat in range(3):
  expected=original_map_process(*args);actual=obs_adapter.map_process(*args)
  assert all(torch.equal(expected[k],actual[k]) for k in expected)
 # Verify clipped-query contract with an oversized query.
 patch=box(ego.x-300,ego.y-300,ego.x+300,ego.y+300)
 for layer in layers:
  expected=type(planner.map)._get_proximity_map_object(planner.map,patch.intersection(planner.map._planning_local_roi),layer)
  actual=planner.map._get_proximity_map_object(patch,layer)
  assert [o.id for o in expected]==[o.id for o in actual]
 planner.map._planning_local_roi=None
 a=original_lanes(planner.map,Point2D(ego.x,ego.y),60)
 b=fast_features.get_lane_polylines(planner.map,Point2D(ego.x,ego.y),60)
 assert repr(a)==repr(b),(repr(a)[:200],repr(b)[:200])
 from driveanchor_planner import DriveAnchorPlanner
 from choose_dtpp import make_fast_choose as cpu_choose_factory
 from choose_gpu import make_fast_choose as fast_choose_factory
 old_choose=cpu_choose_factory(DriveAnchorPlanner.choose);new_choose=fast_choose_factory(DriveAnchorPlanner.choose)
 route=np.stack([np.linspace(0,60,100),np.zeros(100)],axis=1)
 for n in (250,500):
  xy=np.broadcast_to(np.stack([np.linspace(0,10,40),np.zeros(40)],axis=1),(n,40,2))
  for bad in (False,True):
   def fake_ttc(x):return np.full(len(x),bad),dict(min_ttc_capped_s=[0.5 if bad else 3.0]*len(x),predicted_collision=[False]*len(x),rejected_count=int(bad)*len(x))
   planner.dtpp_evaluate=fake_ttc
   old=old_choose(planner,xy,route,[],ego,0,[],IndexedFeaturesPlanner.features(planner,history)[2])
   expected=dict(planner.selection_diagnostic)
   planner._score_cache={};planner._invariant_cache={}
   new=new_choose(planner,xy,route,[],ego,0,[],IndexedFeaturesPlanner.features(planner,history)[2])
   actual=planner.selection_diagnostic
   assert old==new
   for key in ['constraint_mask_sha256','candidate_count','feasible_count','selected_quality','selected_violations','rejected_by_constraint']:
    assert expected[key]==actual[key],(key,expected[key],actual[key])
 checks.append(dict(token=s.token,history=True,filter=True,map_queries=63,features_exact=True,lanes=True))
 print(checks[-1],flush=True)
 assert not torch.cuda.is_initialized()
(out/'summary.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2))
