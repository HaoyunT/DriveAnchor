"""Fixed100 evaluation, first case is integration smoke; detached resumable progress."""
from pathlib import Path
import json,os,sys,time,fcntl,subprocess,hashlib,signal,traceback
R=Path('/autocar/.cache_pt_pytorch2/nuplan_recovery_20260910');CODE=R/'ef2fm2_single_v2_code';OUT=R/'ef2fm2_single_v2_run';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(CODE));from metrics import collect_case,aggregate
CK=R/'ef_vocab3000_continue_v15_run/model_08000.pt'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(x,indent=2,default=str));t.replace(p)
def status(phase,**kw):write(OUT/'status.json',dict(phase=phase,pid=os.getpid(),updated=time.time(),**kw))
def main():
 with (OUT/'pipeline.lock').open('a+') as lk:
  fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
  tokens=read(R/'dp_fixed100/manifest.json')['tokens'];assert len(tokens)==len(set(tokens))==100
  tokens=tokens[:1]
  full=read(CODE/'common210_manifest.json');assert set(tokens)<={r['token'] for r in full['scenarios']}
  snap=read(CODE/'snapshot.json')
  for n,h in snap['snapshot_sha256'].items():assert hashlib.sha256((CODE/n).read_bytes()).hexdigest()==h
  prior=read('/workdir/driveanchor_joint_three_20260910/pilot_first14_v2/closed_loop_nonreactive_agents/manifest.json')
  manifest=dict(tokens=tokens,source=str(CK),snapshot=snap,first_case_integration_smoke=True,official_protocol='closed_loop_nonreactive_agents',case_timeout_s=1800,automatic_deployment=False,known_seen_eval_frames=True)
  write(OUT/'manifest.json',manifest)
  results=read(OUT/'results.json') if (OUT/'results.json').exists() else []
  assert [r['token'] for r in results]==tokens[:len(results)]
  for i,token in enumerate(tokens[len(results):],len(results)):
   status('waiting_gpu_lock',completed=i,total=1,token=token)
   with open('/tmp/driveanchor_gpu_training.lock','a+') as gpu:fcntl.flock(gpu,fcntl.LOCK_EX)
   case=OUT/token/'closed_loop_nonreactive_agents';case.parent.mkdir(exist_ok=False)
   cmd=[sys.executable,'-u',str(CODE/'run_eval.py'),'--output',str(case),'--variant','guard','--tokens',token,'--checkpoint',str(CK),'--cohort-manifest',str(CODE/'common210_manifest.json'),'--fm-steps','2','--max-seconds','1800','--execute']
   status('smoke_case' if i==0 else 'running',completed=i,total=1,token=token)
   tick=time.time()
   with (case.parent/'run.log').open('x') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    try:rc=child.wait(timeout=1860)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();raise RuntimeError('case timeout '+token)
   if rc:raise RuntimeError('evaluator exit '+str(rc)+' token '+token)
   result=collect_case(case/token,token,prior['reference_metric_files']);result['wall_seconds']=time.time()-tick;result['case_index']=i+1
   results.append(result);write(OUT/'results.json',results);write(OUT/'summary.json',aggregate(results,tokens))
   assert result['succeeded'],'simulation failed '+token
   if i==0:write(OUT/'smoke_passed.json',dict(token=token,official=result['official'],note='Pipeline compatibility passed; score is not a promotion gate.'))
  status('complete',completed=1,total=1)
if __name__=='__main__':
 try:main()
 except BaseException as e:status('failed',error=repr(e));traceback.print_exc();raise
