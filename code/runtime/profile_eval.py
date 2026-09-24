"""Evaluate an explicitly validated A-derived RL checkpoint with frozen v5 inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

INFERENCE_ROOT=Path('/tmp/driveanchor_runtime_repairs_v5_20260910')
V5_MANIFEST=Path('/workdir/driveanchor_joint_three_20260910/runtime_repairs_v5/guard_fixed14_parallel/closed_loop_nonreactive_agents/manifest.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_checkpoint(path,reference):
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    assert not torch.cuda.is_initialized()
    source=torch.load(reference['path'],map_location='cpu')
    candidate=torch.load(path,map_location='cpu')
    root=Path('/autocar/.cache_pt_pytorch2/nuplan_recovery_20260910')
    allowed={4356:(root/'b_scene_sampling_v1/random/model_4356.pt','bc738c9560ce3820f115228aff01f8a1290e5f1be169216c5f945c4f98b8af8d'),5508:(root/'ttc_weight1_extend_v1/weight1/model_5508.pt','46cedae5f12adf0b1ee754a1f11cc9bbed8319a4875a4e043c5eca54aec85ccf')}
    step=candidate['step']
    assert step in allowed and Path(path).resolve()==allowed[step][0].resolve()
    assert sha(path)==allowed[step][1]
    assert candidate['architecture']==source['architecture'] and candidate['config']==source['config']
    assert candidate['train_keys']==source['train_keys'] and candidate['val_keys']==source['val_keys']
    assert set(candidate['model'])==set(source['model'])
    changed=0
    for name,value in candidate['model'].items():
        old=source['model'][name]
        assert value.shape==old.shape and value.dtype==old.dtype and torch.isfinite(value).all()
        if not name.startswith('denoiser.'):
            assert torch.equal(value,old), 'Frozen tensor changed: '+name
        else:changed+=int(not torch.equal(value,old))
    assert (changed>0)==(candidate['step']>0)
    return dict(path=str(Path(path).resolve()),sha256=sha(path),step=candidate['step'],
        stage=candidate['stage'],changed_FM_tensors=changed,frozen_encoder_EF_buffers_equal_A=True)


def write(path, value):
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,float) and not __import__('math').isfinite(x):return None
        return x
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(clean(value),indent=2,default=str,allow_nan=False));tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--variant',choices=['geometry','guard'],required=True)
    p.add_argument('--tokens',nargs='+',required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--detach',action='store_true')
    p.add_argument('--max-seconds',type=int,default=1800)
    p.add_argument('--gpu-lock-fd',type=int)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--fm-steps',type=int,choices=(2,),default=2,help='FM1 is an explicit diagnostic; historical v5 reference uses FM2')
    args=p.parse_args();out=Path(args.output)
    if args.gpu_lock_fd is not None:
        assert args.execute and not args.detach and args.gpu_lock_fd>=3
    assert 'closed_loop_nonreactive_agents' in out.parts and 1<=args.max_seconds<=21600
    assert not out.exists()
    root=Path('/tmp/driveanchor_quick_pilot_v2_20260910/experiments/ef_branch_training')
    api_path=root/'joint_three_schemes/quick_pilot/runner.py'
    spec=importlib.util.spec_from_file_location('_frozen_native_repair_api',api_path)
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    prior=api.load_json('/workdir/driveanchor_joint_three_20260910/pilot_first14_v2/closed_loop_nonreactive_agents/manifest.json')
    api.check_hashes(prior['sources_sha256'])
    import pandas as pd
    fixed=json.loads(Path('/autocar/.cache_pt_pytorch2/nuplan_recovery_20260910/stratified10_visuals_v1/manifest.json').read_text())
    report=pd.read_parquet(Path(prior['reference_config']).parents[2]/'runner_report.parquet')
    rows={}
    for row in fixed['scenarios']:
        matches=report[report.scenario_name==row['scenario']];assert len(matches)==1
        rows[row['scenario']]={**row,'token':row['scenario'],'log_name':matches.iloc[0].log_name}
    assert len(rows)==10

    assert len(args.tokens)==len(set(args.tokens)) and set(args.tokens)<=set(rows) and len(args.tokens)>0
    reference=prior['models']['A']
    assert sha(reference['path'])==reference['sha256']
    source=checked_checkpoint(args.checkpoint,reference)
    v5=json.loads(V5_MANIFEST.read_text())
    code={str(Path(__file__).resolve()):sha(__file__)}
    for name in ['repair_planner.py','geometry.py','braking.py']:
        path=INFERENCE_ROOT/name
        assert sha(path)==v5['repair_source_sha256'][str(path)]
        code[str(path)]=sha(path)
    contract=dict(variant=args.variant,source_model=source,cohort=[rows[t] for t in args.tokens],
        repair_source_sha256=code,frozen_reference_contract_sha256=prior['contract_sha256'],
        native_protocol_unchanged=True,learned_candidates=500,EF_calls=1,FM_steps=args.fm_steps,
        separate_brake_candidate_enabled=args.variant=='guard',training=False)
    contract['selector_change']='EF+FM+FM; norm(FM2 raw meter output) ascending top250 out of500; nearest10 DTPP TTC and existing rules; original anchor IDs preserved'
    contract['DTPP_code_sha256']={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    contract['inference_devices']={'EF_FM':'cuda','DTPP':'cuda','geometry':'cpu','TF32':False,'precision':'float32'}
    contract['inherited_shared_GPU_reservation']=args.gpu_lock_fd is not None
    contract.update(reference_A=reference,inference_identical_to_v5=False,
        v5_reference_manifest_sha256=sha(V5_MANIFEST))
    if args.detach:
        assert not args.execute
        launch=out.parent/(out.name+'_launch');launch.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,'-u',str(Path(__file__).resolve()),*[
            x for x in sys.argv[1:] if x!='--detach'],'--execute']
        with (launch/'run.log').open('xb') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(launch/'launch.json',dict(pid=child.pid,command=command,contract=contract,
            started_at=time.time(),start_new_session=True,stdin='DEVNULL',finite=True))
        print(json.dumps(dict(pid=child.pid,launch=str(launch))));return
    if not args.execute:
        print(json.dumps(contract,indent=2));return
    out.mkdir(parents=True);write(out/'manifest.json',contract)
    sys.path.extend(prior['python_paths'])
    sys.path.insert(0,str(INFERENCE_ROOT))
    os.environ.setdefault('NUPLAN_DATA_ROOT','/root/nuplan/dataset')
    os.environ.setdefault('NUPLAN_MAPS_ROOT','/workdir/nuplan_data/maps')
    import gc
    import numpy as np
    import pandas as pd
    import torch
    from omegaconf import OmegaConf
    from fm2_planner import FM2Top250Planner as RepairPlanner
    original_compute=RepairPlanner.compute_planner_trajectory
    def profiled_compute(self,inp):
        if inp.iteration.index==28:
            import cProfile
            prof=cProfile.Profile();prof.enable()
            try:return original_compute(self,inp)
            finally:
                prof.disable();prof.dump_stats(str(out/'frame28.prof'))
        return original_compute(self,inp)
    RepairPlanner.compute_planner_trajectory=profiled_compute
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    spec=importlib.util.spec_from_file_location('_repair_native_streaming',prior['streaming'])
    streaming=importlib.util.module_from_spec(spec);spec.loader.exec_module(streaming)
    started=time.monotonic();results=[]
    def alarm(signum,frame):raise TimeoutError('Finite repair evaluation deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(args.max_seconds)
    try:
        assert args.gpu_lock_fd is None
        lock_handle=open('/tmp/driveanchor_gpu_training.lock','a+')
        if args.gpu_lock_fd is not None:
            inherited=os.fstat(lock_handle.fileno());expected=os.stat(prior['gpu_lock'])
            assert (inherited.st_dev,inherited.st_ino)==(expected.st_dev,expected.st_ino)
        with lock_handle as lock:
            write(out/'status.json',dict(phase='starting_CPU',pid=os.getpid()))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            api.check_hashes(code)
            planner=RepairPlanner(checkpoint=source['path'],device='cuda',use_ef=True,fm_steps=args.fm_steps,
                                  chunk_size=500,enable_guard=args.variant=='guard')
            original={k:v.detach().cpu().clone() for k,v in planner.model.state_dict().items()}
            for token in args.tokens:
                if time.monotonic()-started>=args.max_seconds:raise TimeoutError('Evaluation budget')
                row=rows[token];case=out/token;case.mkdir()
                planner.diagnostic_path=case/'planner_diagnostics.jsonl'
                planner.last_full_pool_diagnostic=None
                cfg=api.config_for_case(prior['reference_config'],row,case)
                OmegaConf.save(cfg,case/'config.yaml')
                write(out/'status.json',dict(phase='running',token=token,completed=len(results)))
                tick=time.monotonic();streaming.run_simulation_streaming(cfg,planners=planner)
                reports=pd.read_parquet(case/'runner_report.parquet')
                assert len(reports)==1 and reports.iloc[0].scenario_name==token
                assert all(torch.equal(v,planner.model.state_dict()[k].detach().cpu()) for k,v in original.items())
                success=bool(reports.iloc[0].succeeded)
                result=dict(token=token,scenario_type=row['scenario_type'],succeeded=success,
                    seconds=time.monotonic()-tick,model_unchanged=True,score=None,
                    error=None if success else str(reports.iloc[0].error_message))
                diag=[json.loads(line) for line in planner.diagnostic_path.read_text().splitlines()] if planner.diagnostic_path.exists() else []
                result.update(planner_calls=len(diag),geometry_repairs=sum(r['geometry']['repaired'] for r in diag),
                    selection_sources={k:sum(r['safety_guard']['selection_source']==k for r in diag)
                                       for k in ['learned','learned_clearance','bounded_brake']})
                if success:
                    assert len(diag) in (149,150) and [r['iteration'] for r in diag]==list(range(len(diag)))
                    assert all(r['learned_candidate_count']==500 and r['future_GT_input'] is False for r in diag)
                    assert {x.name for x in (case/'metrics').glob('*.parquet')}==set(prior['reference_metric_files'])
                    scores=[r for path in (case/'aggregator_metric').glob('*.parquet')
                            for r in pd.read_parquet(path).to_dict('records') if r.get('scenario')==token]
                    assert len(scores)==1 and np.isfinite(scores[0]['score'])
                    result.update(score=float(scores[0]['score']),official_metrics=scores[0],
                        mean_planner_seconds=float(np.mean([r['total_planner_seconds'] for r in diag])),
                        selected_OOL_fraction=float(np.mean([r['selected_any_ool'] for r in diag])))
                result['artifact_sha256']={str(f):sha(f) for f in case.rglob('*') if f.is_file() and
                    (f.suffix=='.parquet' or f.name in ['planner_diagnostics.jsonl','geometry_failure.json','config.yaml'])}
                write(case/'complete.json',result);results.append(result)
                write(out/'results.json',results)
                print(json.dumps(dict(token=token,succeeded=success,score=result['score'],geometry_repairs=result['geometry_repairs'])),flush=True)
                gc.collect()
            api.check_hashes(code);api.check_hashes(prior['sources_sha256'])
            write(out/'status.json',dict(phase='complete',attempts=len(results),successful=sum(r['succeeded'] for r in results),
                seconds=time.monotonic()-started,model_unchanged=True,training=False))
    except BaseException as error:
        write(out/'status.json',dict(phase='failed',error=repr(error),completed=len(results),retry=False));raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
