"""Bounded matched native comparison; one process per case, shared GPU lock in driver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from metrics import aggregate, collect_case

R=Path('/autocar/.cache_pt_pytorch2/nuplan_recovery_20260910')
CHECKPOINTS={4356:R/'b_scene_sampling_v1/random/model_4356.pt',
             5508:R/'ttc_weight1_extend_v1/weight1/model_5508.pt',
             6020:R/'ttc_weight1_extend2_v1/weight1/model_6020.pt'}


def write(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,nargs='+',default=[4356,5508,6020]);p.add_argument('--tokens',nargs='+')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    code=Path(__file__).parent;snapshot=json.loads((code/'snapshot.json').read_text())
    for name,expected in snapshot['snapshot_sha256'].items():
        assert hashlib.sha256((code/name).read_bytes()).hexdigest()==expected
    cohort=json.loads((R/'stratified10_visuals_v1/manifest.json').read_text())['scenarios']
    tokens=[r['scenario'] for r in cohort]
    if a.tokens:
        assert len(set(a.tokens))==len(a.tokens) and set(a.tokens)<=set(tokens)
        tokens=a.tokens
    assert set(a.steps)<=set(CHECKPOINTS) and len(set(a.steps))==len(a.steps)
    prior=json.loads(Path('/workdir/driveanchor_joint_three_20260910/pilot_first14_v2/closed_loop_nonreactive_agents/manifest.json').read_text())
    write(out/'manifest.json',dict(snapshot=snapshot,steps=a.steps,tokens=tokens,
          checkpoint_sha256={str(s):hashlib.sha256(CHECKPOINTS[s].read_bytes()).hexdigest() for s in a.steps},
          started=time.time(),pid=os.getpid(),max_seconds_per_case=900,training=False))
    results={s:[] for s in a.steps}
    # Token-first pairs keep checkpoints exposed to the same scene order.
    for token in tokens:
        for step in a.steps:
            destination=out/str(step)/token/'closed_loop_nonreactive_agents'
            destination.parent.mkdir(parents=True,exist_ok=False)
            command=[sys.executable,'-u',str(code/'run_eval.py'),'--output',str(destination),
                     '--variant','guard','--tokens',token,'--checkpoint',str(CHECKPOINTS[step]),
                     '--fm-steps','2','--max-seconds','900','--execute']
            write(out/'status.json',dict(phase='evaluating',step=step,token=token,
                  completed=sum(map(len,results.values())),total=len(tokens)*len(a.steps)))
            tick=time.time()
            try:
                with (destination.parent/'run.log').open('xb') as log:
                    child=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=960)
                if child.returncode:raise RuntimeError('Evaluator exit '+str(child.returncode))
                result=collect_case(destination/token,token,prior['reference_metric_files'])
            except Exception as error:
                result=dict(token=token,succeeded=False,error=repr(error),log=str(destination.parent/'run.log'))
            result.update(step=step,wall_seconds=time.time()-tick)
            results[step].append(result)
            write(out/'results.json',{str(k):v for k,v in results.items()})
            write(out/'summary.json',{str(k):aggregate(v,tokens) for k,v in results.items()})
            print(json.dumps(result),flush=True)
    summaries={str(k):aggregate(v,tokens) for k,v in results.items()}
    write(out/'complete.json',dict(attempts_complete=True,
          training_gate_passed=all(v['complete'] for v in summaries.values()),summary=summaries))
    write(out/'status.json',dict(phase='complete',completed=len(tokens)*len(a.steps),
          all_cases_scored=all(v['complete'] for v in summaries.values())))


if __name__=='__main__':main()
