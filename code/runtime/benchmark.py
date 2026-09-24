import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['NUPLAN_DATA_ROOT']='/root/nuplan/dataset'
os.environ['NUPLAN_MAPS_ROOT']='/workdir/nuplan_data/maps'
import sys,json,pathlib,time,hashlib
R=pathlib.Path('/autocar/.cache_pt_pytorch2/nuplan_recovery_20260910');out=R/'fm2_top250_v1';out.mkdir(exist_ok=True)
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
import fcntl
lock=open('/tmp/driveanchor_gpu_training.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
from fm2_planner import FM2Top250Planner as DTPPTTCPlanner
cpu=DTPPTTCPlanner(checkpoint=str(checkpoint),device='cuda',use_ef=True,fm_steps=2,chunk_size=128,enable_guard=True)
gpu=DTPPTTCPlanner(checkpoint=str(checkpoint),device='cuda',use_ef=True,fm_steps=2,chunk_size=128,enable_guard=True)
from choose_dtpp import make_fast_choose
from driveanchor_planner import DriveAnchorPlanner
import types
cpu.choose=types.MethodType(make_fast_choose(DriveAnchorPlanner.choose),cpu)
assert next(gpu.model.parameters()).is_cuda
captured={};original=rp.generate_full_pool
def capture(*a,**kw):
 paths=original(*a,**kw);captured['xy']=paths['FM2'].detach().cpu().numpy();return paths
rp.generate_full_pool=capture
import fm2_planner
fm2_planner.generate_full_pool=capture
results=[]
for row in m['scenarios'][:3]:
 s=by[row['scenario']]
 sim=Simulation(simulation_setup=SimulationSetup(time_controller=instantiate(cfg.simulation_time_controller,scenario=s),observations=instantiate(cfg.observation,scenario=s),ego_controller=instantiate(cfg.ego_controller,scenario=s),scenario=s),simulation_history_buffer_duration=cfg.simulation_history_buffer_duration)
 init=sim.initialize();inp=sim.get_planner_input();reference=None
 for label,planner in [('FM2_top250_CPUgeometry',cpu),('FM2_top250_GPUgeometry',gpu)]:
  for repeat in range(4):
   planner.initialize(init);torch.cuda.synchronize();start=time.perf_counter();planner.compute_planner_trajectory(inp);torch.cuda.synchronize();elapsed=time.perf_counter()-start
   diag=planner.last_full_pool_diagnostic;xy=captured['xy'].copy()
   expected=np.argsort(np.linalg.norm(xy.reshape(500,-1),axis=1),kind='stable')[:250]
   assert np.array_equal(expected,planner._last_kept_ids)
   assert diag['selected_anchor_id'] is None or diag['selected_anchor_id'] in expected
   assert diag['output_stage']=='FM2'
   if label=='FM2_top250_CPUgeometry':reference=(xy,diag)
   a=reference[1]['dtpp_ttc'];b=diag['dtpp_ttc']
   parity=dict(max_xy_error_m=float(abs(xy-reference[0]).max()),same_selected=diag['selected_anchor_id']==reference[1]['selected_anchor_id'],same_TTC_mask=bool(np.array_equal(np.asarray(a['min_ttc_capped_s'])<=.95,np.asarray(b['min_ttc_capped_s'])<=.95)),same_selected_violations=diag['constraint_mask_sha256']==reference[1]['constraint_mask_sha256'])
   result=dict(token=s.token,variant=label,repeat=repeat,warmup=repeat==0,wall_seconds=elapsed,logged_planner_seconds=diag['total_planner_seconds'],selector_seconds=diag['selector_seconds'],dtpp_seconds=b['seconds'],parity=parity)
   results.append(result);(out/'partial.json').write_text(json.dumps(results,indent=2));print(json.dumps(result),flush=True)
   assert parity['max_xy_error_m']<.001 and parity['same_selected'] and parity['same_TTC_mask'] and parity['same_selected_violations'],parity
summary=dict(complete=True,results=results,scope='3 initial scenes x500; full wall includes DTPP prepare and synchronization; first repetition warmup; no closed-loop restart')
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',flush=True)
