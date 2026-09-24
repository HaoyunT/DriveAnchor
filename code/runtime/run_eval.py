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
V5_MANIFEST=Path(os.environ.get('DRIVEANCHOR_V5_MANIFEST','/workdir/driveanchor_joint_three_20260910/runtime_repairs_v5/guard_fixed14_parallel/closed_loop_nonreactive_agents/manifest.json'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_checkpoint(path,reference):
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    root=Path('/autocar/.cache_pt_pytorch2/nuplan_recovery_20260910')
    expected=root/'ef_vocab3000_continue_v15_run/model_08000.pt'
    assert Path(path).resolve()==expected.resolve()
    assert sha(path)=='d8efeb769d599e17d83ac5435664f2c7e8bb4f85a4942bb4f729a0713575c1d8'
    candidate=torch.load(path,map_location='cpu')
    source=torch.load(root/'ef_vocab3000_fast_v14_run/model_04000.pt',map_location='cpu')
    assert candidate['architecture']==source['architecture'] and candidate['config']==source['config']
    assert candidate['step']==8000 and set(candidate['model'])==set(source['model'])
    assert candidate['train_keys']==source['train_keys'] and candidate['val_keys']==source['val_keys']
    for name,value in candidate['model'].items():
        assert value.shape==source['model'][name].shape and value.dtype==source['model'][name].dtype and torch.isfinite(value).all()
        if not name.startswith('ef_head.'):assert torch.equal(value,source['model'][name]),name
    import numpy as np
    assert torch.equal(candidate['model']['anchors'],torch.from_numpy(np.load(root/'anchor_dense_all_v1/cluster3000/anchors.npy')))
    assert candidate['model']['anchors'].shape==(3000,40,2)
    return dict(path=str(expected),sha256=sha(path),step=8000,stage=candidate['stage'],frozen_FM_encoder_equal_v14=True,EF_calls=2,anchors=3000)


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
    p.add_argument('--cohort-manifest')
    p.add_argument('--output',required=True);p.add_argument('--variant',choices=['geometry','guard'],required=True)
    p.add_argument('--tokens',nargs='+',required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--detach',action='store_true')
    p.add_argument('--max-seconds',type=int,default=1800)
    p.add_argument('--gpu-lock-fd',type=int)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--fm-steps',type=int,choices=(2,),default=2,help='FM1 is an explicit diagnostic; historical v5 reference uses FM2')
    p.add_argument('--vnorm-topk',type=int,choices=range(1,3001),default=20,
                   help='shortlist size after the full 3000-anchor FM2 pass')
    p.add_argument('--selector',choices=('legacy','pdm_model'),default='pdm_model',
                   help='candidate ranking protocol (PDM by default; legacy is explicit)')
    args=p.parse_args();out=Path(args.output)
    os.environ['DRIVEANCHOR_VNORM_TOPK']=str(args.vnorm_topk)
    os.environ['DRIVEANCHOR_SELECTOR_MODE']=args.selector
    if args.gpu_lock_fd is not None:
        assert args.execute and not args.detach and args.gpu_lock_fd>=3
    assert 'closed_loop_nonreactive_agents' in out.parts and 1<=args.max_seconds<=21600
    assert not out.exists()
    root=Path('/tmp/driveanchor_quick_pilot_v2_20260910/experiments/ef_branch_training')
    api_path=root/'joint_three_schemes/quick_pilot/runner.py'
    spec=importlib.util.spec_from_file_location('_frozen_native_repair_api',api_path)
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    # The historical pilot manifest was an implementation detail and is not
    # present on fresh machines.  The new contract is self-contained.
    contract_path=Path(os.environ.get('DRIVEANCHOR_CONTRACT',''))
    prior=api.load_json(str(contract_path)) if contract_path.exists() else None
    api.check_hashes(prior.get('sources_sha256', {}))
    import pandas as pd
    if prior is None:
        raise RuntimeError('DRIVEANCHOR_CONTRACT must point to a self-contained polygon contract')
    rows={row['token']:dict(row,scenario=row['token']) for row in prior['cohort']}
    if args.cohort_manifest:
        manifest_path=Path(args.cohort_manifest)
        assert manifest_path.resolve()==(Path(__file__).parent/'common210_manifest.json').resolve()
        full=json.loads(manifest_path.read_text())
        assert len(full['scenarios'])==210
        dp=full['baseline_sources']['dp']
        report_path=Path(dp['root'])/'runner_report.parquet'
        assert sha(report_path)==dp['runner_sha256']
        dp_report=pd.read_parquet(report_path)
        rows={}
        for row in full['scenarios']:
            matches=dp_report[dp_report.scenario_name==row['token']]
            assert len(matches)==1 and matches.iloc[0].log_name==row['log_name']
            assert isinstance(row['scenario_type'],str) and row['scenario_type']
            rows[row['token']]=dict(row,scenario=row['token'])
        assert len(rows)==210

    assert len(args.tokens)==len(set(args.tokens)) and set(args.tokens)<=set(rows) and len(args.tokens)>0
    reference={'path': prior.get('reference_config') or prior.get('checkpoint',{}).get('path')}
    if not reference['path'] or not Path(reference['path']).exists():
        raise RuntimeError('contract has no usable reference_config')
    source=checked_checkpoint(args.checkpoint,reference)
    v5=json.loads(V5_MANIFEST.read_text()) if V5_MANIFEST.exists() else {'repair_source_sha256':{}}
    code={str(Path(__file__).resolve()):sha(__file__)}
    for name in ['repair_planner.py','geometry.py','braking.py']:
        path=INFERENCE_ROOT/name
        if not path.exists():
            raise RuntimeError(f'missing inference repair module: {path}')
        expected=v5.get('repair_source_sha256',{}).get(str(path))
        if expected is not None: assert sha(path)==expected
        code[str(path)]=sha(path)
    contract=dict(variant=args.variant,source_model=source,cohort=[rows[t] for t in args.tokens],
        repair_source_sha256=code,frozen_reference_contract_sha256=prior['contract_sha256'],
        native_protocol_unchanged=True,learned_candidates=3000,EF_calls=2,FM_steps=args.fm_steps,
        vnorm_prefilter_topk=args.vnorm_topk,selector_mode=args.selector,
        separate_brake_candidate_enabled=args.variant=='guard',training=False)
    subset_path=Path(__file__).parent/'subset.json'
    contract['fixed_subset']=json.loads(subset_path.read_text()) if subset_path.exists() else {'tokens': args.tokens}
    contract['cohort_manifest_sha256']=sha(args.cohort_manifest) if args.cohort_manifest else None
    contract['comfort_progress_selector']=('PDM hard-feasibility mask followed by FM2 vnorm ascending; '
        'TTC is diagnostic/final-safety only and does not rank candidates' if args.selector=='pdm_model' else
        'Native-threshold candidate comfort proxy hard filter after safety; signed route progress descending; original safety-first fallback')
    contract['selector_change']=('EF twice; anchor->EF1->EF2->FM1->FM2; all3000 model outputs; '
        f'FM2 vnorm ascending top{args.vnorm_topk}; hard-feasible candidates only; TTC not used for ranking; original anchor IDs preserved'
        if args.selector=='pdm_model' else
        'EF twice; anchor->EF1->EF2->FM1->FM2; all3000 original IDs before EF; nearest10 DTPP TTC and existing rules; original anchor IDs preserved')
    contract['DTPP_code_sha256']={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    contract['inference_devices']={'EF_FM':'cuda','DTPP':'cuda','geometry':'cpu','TF32':False,'precision':'float32'}
    contract['inherited_shared_GPU_reservation']=args.gpu_lock_fd is not None
    contract.update(reference_A=reference,inference_identical_to_v5=False,
        v5_reference_manifest_sha256=sha(V5_MANIFEST) if V5_MANIFEST.exists() else None)
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
    sys.path.extend(prior.get('python_paths', []))
    sys.path.insert(0,str(INFERENCE_ROOT))
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    os.environ.setdefault('NUPLAN_DATA_ROOT','/root/nuplan/dataset')
    os.environ.setdefault('NUPLAN_MAPS_ROOT','/workdir/nuplan_data/maps')
    import gc
    import numpy as np
    import pandas as pd
    import torch
    from omegaconf import OmegaConf
    from fm2_planner import FM2Top250Planner as RepairPlanner
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    streaming_path=prior.get('streaming') or os.environ.get('DRIVEANCHOR_STREAMING')
    if not streaming_path or not Path(streaming_path).exists():
        raise RuntimeError('contract must provide streaming module via streaming or DRIVEANCHOR_STREAMING')
    spec=importlib.util.spec_from_file_location('_repair_native_streaming',streaming_path)
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
                    assert all(r['learned_candidate_count']==3000 and r['EF_calls_this_frame']==2 and r['future_GT_input'] is False for r in diag)
                    if args.selector=='pdm_model':
                        # PDM may receive one extra CPU IDM proposal alongside
                        # the CUDA model shortlist; the FM prefilter count must
                        # still match the requested vnorm top-k exactly.
                        assert all(r['vnorm_topk']==args.vnorm_topk and r['model_prefilter_count']==args.vnorm_topk and r['pdm_candidate_count'] in (args.vnorm_topk,args.vnorm_topk+1) for r in diag)
                        assert all(r['selector_version']=='v8_pdm_vnorm_feasible_gpu' for r in diag)
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
            api.check_hashes(code);api.check_hashes(prior.get('sources_sha256', {}))
            write(out/'status.json',dict(phase='complete',attempts=len(results),successful=sum(r['succeeded'] for r in results),
                seconds=time.monotonic()-started,model_unchanged=True,training=False))
    except BaseException as error:
        write(out/'status.json',dict(phase='failed',error=repr(error),completed=len(results),retry=False));raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
