"""Finite native A/B/C pilot; the unchanged streaming function owns simulation.

Schema is CPU only. Run must be explicitly authorized externally. A complete
first case is reusable by --resume; partial outputs never silently retry.
"""
import argparse
import copy
import fcntl
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

VERSION = 'native_full500_ABC_pilot_v1'
RUNS = '/workdir/driveanchor_joint_three_20260910/runs_v3_full_pool_20260910'
REFERENCE = '/root/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner/val14/local_eval_200/model_20260908_stream/code/hydra/config.yaml'
DP_BASELINE = '/workdir/driveanchor_full_pool_dp_eval/dp_pilot_first14_native_v1'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    def clean(v):
        if isinstance(v, dict): return {k:clean(x) for k,x in v.items()}
        if isinstance(v, (tuple,list)): return [clean(x) for x in v]
        if isinstance(v, float) and not math.isfinite(v): return None
        return v
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False, default=str))
    temp.replace(path)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_hashes(hashes):
    for path, digest in hashes.items():
        if sha(path) != digest: raise RuntimeError('Source/input changed: ' + path)


def load_json(path): return json.loads(Path(path).read_text())


def paths(args):
    root = Path(args.code_root)
    return [root / 'full_pool_acceleration', root / 'full_pool_inference',
        Path('/workdir/driveanchor_selector_frozen_v1'),
        Path(args.cached_planner).parent, Path(args.native_geometry).parent,
        Path(args.rebase_root),
        Path('/workdir/nuplan-devkit'), Path('/workdir/diffusion_planner_repro')]


def source_files(args):
    root = Path(args.code_root)
    files = [*sorted(Path(__file__).parent.glob('*.py')), root/'full_pool_acceleration/fast_planner.py',
        root/'full_pool_acceleration/fast_selector.py',
        *[root/'full_pool_inference'/name for name in
          ('full_pool_planner.py','full_pool_selector.py','source_binding.py')],
        Path(args.cached_planner), Path(args.native_geometry), Path(args.streaming),
        Path('/workdir/driveanchor_selector_frozen_v1/driveanchor_planner.py'),
        Path('/workdir/driveanchor_selector_frozen_v1/selection.py')]
    files += sorted((Path(args.rebase_root)/'rebase').rglob('*.py'))
    return files


def schema(args):
    import torch
    import yaml
    torch.set_num_threads(2)
    out = Path(args.output)
    if 'closed_loop_nonreactive_agents' not in out.parts:
        raise ValueError('Output must contain a closed_loop_nonreactive_agents directory: official callback filters by challenge path')
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'manifest.json'
    cohort = load_json(args.cohort_manifest)
    filt = yaml.safe_load(Path(args.cohort_filter).read_text())
    rows = cohort['scenarios']
    assert len(rows) == cohort['scenario_count'] == 14
    assert len({r['token'] for r in rows}) == 14
    assert len({r['scenario_type'] for r in rows}) == 14
    assert set(filt['scenario_tokens']) == {r['token'] for r in rows}
    assert filt['shuffle'] is False
    assert cohort['filter_sha256'] == sha(args.cohort_filter)
    config = yaml.safe_load(Path(args.reference_config).read_text())
    assert config['run_metric'] is True
    assert config['observation']['_target_'].endswith('.TracksObservation')
    assert config['ego_controller']['_target_'].endswith('.TwoStageController')
    assert config['simulation_time_controller']['_target_'].endswith('.StepSimulationTimeController')
    assert 'simulation_log_callback' in config['callback']
    assert list(config['main_callback']).index('metric_file_callback') < list(config['main_callback']).index('metric_aggregator_callback')
    hashes = {str(p.resolve()): sha(p) for p in source_files(args)}
    hashes.update({str(Path(p).resolve()): sha(p) for p in
        (args.reference_config,args.cohort_filter,args.cohort_manifest)})
    baseline_results=Path(args.baseline_root)/'results.json'
    baseline_csv=Path(args.baseline_root)/'per_scenario.csv'
    assert sha(baseline_results)=='3808485135ab2f34e84619b1ca0c933cf3b1cc7a2534f45d9df5213a5bb73047'
    assert sha(baseline_csv)=='b712f12e7a5414a78dec065036b8630b00c92efcd55fec0ea6bb18fd9ea22023'
    models = {}
    for arm in args.arms:
        path = Path(args.runs)/arm/'model_4000.pt'
        if not path.exists(): raise FileNotFoundError('Required final4000 missing, never substitute best: ' + str(path))
        ckpt = torch.load(path, map_location='cpu')
        assert ckpt['stage'] == 'joint_three_scheme_' + arm
        step = ckpt.get('step', ckpt.get('successful_updates'))
        assert step == 4000, (arm, step)
        joint = ckpt['manifest']['joint_three']
        assert joint['scheme'] == arm
        assert joint['protocol']['version'] == 'joint_three_v3_full_pool_selection'
        assert joint['contract']['protocol_sha256'] == 'df99cba29aea9c4366c7f904f56c890f576596d09bc845fbfd96d4cb26560e70'
        complete_path = path.parent/'complete.json'
        complete = load_json(complete_path)
        assert complete['successful_updates'] == 4000
        assert complete['encoder_anchors_scales_bitwise_unchanged'] and complete['all_buffers_bitwise_unchanged']
        assert all(complete['changed_parameter_tensors'].values())
        models[arm] = dict(path=str(path), sha256=sha(path), step=step, stage=ckpt['stage'],
            completion_path=str(complete_path), completion_sha256=sha(complete_path),
            joint_training_contract=joint['contract'])
        del ckpt
    # Instantiate once on CPU, without a forward. Constructor source bindings
    # catch shadow imports and strict state-loading incompatibilities early.
    for path in reversed(paths(args)): sys.path.insert(0,str(path))
    from fast_planner import FastFullPoolPlanner
    probe = FastFullPoolPlanner(checkpoint=models[args.arms[0]]['path'], device='cpu',
        use_ef=True, fm_steps=2, chunk_size=64)
    assert len(probe.model.anchors) == 500 and probe.model.ef_head is not None
    assert not any(module.training for module in probe.model.modules())
    assert all(p.device.type == 'cpu' for p in probe.model.parameters())
    del probe
    reference_metrics = Path(args.reference_config).parents[2]/'metrics'
    metric_names = sorted(p.name for p in reference_metrics.glob('*.parquet'))
    if len(metric_names) != 16: raise RuntimeError('Expected historical full16 native metrics: ' + str(metric_names))
    contract = dict(version=VERSION, arms=args.arms, cohort=rows, models=models,
        DP_baseline=dict(results_path=str(baseline_results),results_sha256=sha(baseline_results),
            per_scenario_path=str(baseline_csv),per_scenario_sha256=sha(baseline_csv)),
        sources_sha256=hashes, reference_config=str(Path(args.reference_config).resolve()),
        reference_metric_files=metric_names, python_paths=[str(p) for p in paths(args)],
        streaming=str(Path(args.streaming).resolve()),
        inference=dict(EF_calls=1, FM_steps=2, candidates=500, candidate_prefilter='none',
            selector='FastFullPoolPlanner, parity-equivalent frozen quality rules', chunk_size=64),
        native_protocol='Original reference config, controllers, observation, scenario mapping, metrics and simulation logs; only token/log subset, planner and output change.',
        cohort_scope='Fixed first14 development pilot; no claim of full210 or unbiased generalization.',
        source_config_no_time_override=True, gpu_lock=args.gpu_lock)
    contract['contract_sha256'] = canonical_sha(contract)
    if manifest_path.exists():
        if load_json(manifest_path) != contract: raise RuntimeError('Existing output belongs to a different immutable contract')
    else:
        if any(out.iterdir()): raise RuntimeError('Refuse nonempty output without manifest')
        write(manifest_path, contract)
    write(out/'schema_result.json', dict(passed=True, GPU_initialized=torch.cuda.is_initialized(),
        training=False, contract_sha256=contract['contract_sha256'], cases=14*len(args.arms)))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(schema='PASS', output=str(out), contract_sha256=contract['contract_sha256'])), flush=True)
    return contract


def config_for_case(reference, row, destination):
    """Only subset and bookkeeping changes; all simulation settings are copied."""
    from omegaconf import OmegaConf
    from hydra.core.utils import setup_globals
    setup_globals()
    cfg = OmegaConf.load(reference)
    OmegaConf.set_struct(cfg, False)
    cfg.scenario_filter.scenario_tokens = [row['token']]
    cfg.scenario_filter.scenario_types = [row['scenario_type']]
    cfg.scenario_filter.log_names = [row['log_name']]
    cfg.scenario_filter.shuffle = False
    cfg.scenario_builder.db_files = ['/root/nuplan/dataset/nuplan-v1.1/trainval/' + row['log_name'] + '.db']
    cfg.output_dir = str(destination)
    cfg.experiment_uid = 'driveanchor/' + VERSION + '/' + destination.parent.name + '/' + row['token']
    # The original streaming API explicitly accepts a preconstructed planner and
    # removes this key itself. Do not instantiate the historical DP planner.
    OmegaConf.set_struct(cfg, True)
    return cfg


def verify_case(case, contract):
    complete = case/'complete.json'
    if not complete.exists(): raise RuntimeError('Partial/unknown case requires explicit audit; refusing retry: ' + str(case))
    result = load_json(complete)
    if result['contract_sha256'] != contract['contract_sha256'] or result.get('terminal_status') not in ('success','native_scenario_failure'):
        raise RuntimeError('Invalid complete-case contract: ' + str(case))
    if not result['interface_passed'] and result['terminal_status'] != 'native_scenario_failure':
        raise RuntimeError('Unclassified incomplete case')
    check_hashes(result['artifact_sha256'])
    return result


def validate_case(case, row, contract, elapsed, model_unchanged, metadata, allow_terminal_failure=False):
    import numpy as np
    import pandas as pd
    reports = pd.read_parquet(case/'runner_report.parquet')
    own = reports[reports.scenario_name == row['token']]
    if len(reports) != 1 or len(own) != 1: raise RuntimeError('Exact unique native scenario report missing')
    assert own.iloc[0].log_name == row['log_name']
    assert model_unchanged
    if not bool(own.iloc[0].succeeded):
        if not allow_terminal_failure: raise RuntimeError('Firstcase native runner failed; interface smoke stops')
        result = dict(contract_sha256=contract['contract_sha256'], interface_passed=False,
            terminal_status='native_scenario_failure', token=row['token'],log_name=row['log_name'],
            scenario_type=row['scenario_type'], native_metadata=metadata, elapsed_s=elapsed,
            score=None, fixed_denominator_operational_score=0.0,
            error=str(own.iloc[0].get('error_message')),model_state_unchanged=True,
            # Native logging can retain a FileHandler across cases. Preserve
            # those logs as supplemental evidence, but hash immutable artifacts.
            artifact_sha256={str(p):sha(p) for p in case.rglob('*') if p.is_file() and
                (p.suffix == '.parquet' or p.name in ('config.yaml','planner_diagnostics.jsonl'))})
        write(case/'complete.json', result)
        return result
    files = sorted((case/'metrics').glob('*.parquet'))
    expected = set(contract['reference_metric_files'])
    if {p.name for p in files} != expected: raise RuntimeError('Native16 metric file coverage mismatch')
    for p in files:
        frame = pd.read_parquet(p)
        if set(frame.scenario_name) != {row['token']}: raise RuntimeError('Unexpected metric scenario: ' + str(p))
    diagnostics_path = case/'planner_diagnostics.jsonl'
    diagnostics = [json.loads(line) for line in diagnostics_path.read_text().splitlines() if line.strip()]
    if not diagnostics: raise RuntimeError('No native planner diagnostics')
    for r in diagnostics:
        assert r['candidate_count'] == 500 and r['prefilter'] == 'none'
        assert r['output_stage'] == 'FM2'
        assert 'candidate_any_ool_fraction' in r and 'selected_any_ool' in r and 'executed_point_ool' in r
        assert r['selector_rules_sha256'] == '296fdeb1fb8bac5bf87a0af289e29dfeb02464bc287b9b5ce342c2cd026ee1ba'
        assert r['future_GT_input'] is False
    assert metadata['scenario_type'] == row['scenario_type']
    assert len(diagnostics) == metadata['iterations'] - 1
    assert [r['iteration'] for r in diagnostics] == list(range(metadata['iterations'] - 1))
    assert abs(metadata['interval_s'] - 0.1) < 1e-9
    assert metadata['iterations'] in (150,151), metadata
    assert abs((metadata['last_time_us']-metadata['first_time_us'])/1e6 -
               (metadata['iterations']-1)*metadata['interval_s']) < 0.01
    aggregation_files = sorted((case/'aggregator_metric').glob('*.parquet'))
    scores = []
    for p in aggregation_files:
        for r in pd.read_parquet(p).to_dict('records'):
            if r.get('scenario') == row['token']:
                assert np.isfinite(r['score']), 'A score of zero is valid; nonfinite is not'
                scores.append({k: (v.item() if isinstance(v,np.generic) else v) for k,v in r.items()})
    if len(scores) != 1: raise RuntimeError('Expected exactly one official scenario aggregate score')
    assert model_unchanged
    required = files + aggregation_files + [case/'runner_report.parquet', diagnostics_path, case/'config.yaml']
    result = dict(contract_sha256=contract['contract_sha256'], interface_passed=True,terminal_status='success',
        token=row['token'], log_name=row['log_name'], scenario_type=row['scenario_type'],
        elapsed_s=elapsed, planner_calls=len(diagnostics), native_metadata=metadata,
        score=float(scores[0]['score']), official_metrics=scores[0], model_state_unchanged=model_unchanged,
        selector_fallback_calls=sum(bool(r['selection_fallback']) for r in diagnostics),
        mean_feasible_candidates=float(np.mean([r['feasible_count'] for r in diagnostics])),
        mean_selector_seconds=float(np.mean([r['selector_seconds'] for r in diagnostics])),
        mean_planner_seconds=float(np.mean([r['total_planner_seconds'] for r in diagnostics])),
        OOL=dict(candidate_fraction_mean=float(np.mean([r['candidate_any_ool_fraction'] for r in diagnostics])),
            selected_fraction=float(np.mean([r['selected_any_ool'] for r in diagnostics])),
            executed_point_fraction=float(np.mean([r['executed_point_ool'] for r in diagnostics]))),
        artifact_sha256={str(p):sha(p) for p in required})
    write(case/'complete.json', result)
    return result


def official_aggregate(metric_dir, reference_config, destination):
    """Identical public aggregator API used by the verified DP reaggregation."""
    import pandas as pd
    from omegaconf import OmegaConf
    from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame
    from nuplan.planning.metrics.aggregator.weighted_average_metric_aggregator import WeightedAverageMetricAggregator
    cfg = OmegaConf.load(reference_config)
    spec = OmegaConf.to_container(cfg.metric_aggregator.closed_loop_nonreactive_agents_weighted_average, resolve=True)
    frames = {}
    for path in sorted(Path(metric_dir).glob('*.parquet')):
        item = MetricStatisticsDataFrame.load_parquet(path)
        if item.metric_statistic_name in frames: raise RuntimeError('Duplicate metric statistic')
        frames[item.metric_statistic_name] = item
    options = {k:v for k,v in spec.items() if not k.startswith('_')}
    options['file_name'] = 'pilot_first14_native'
    aggregator = WeightedAverageMetricAggregator(aggregator_save_path=destination, **options)
    aggregator(frames)
    return dict(path=str(Path(destination)/'pilot_first14_native.parquet'),
        score=float(aggregator.final_metric_score), metric_count=len(frames))


def aggregate_arm(arm_path, contract):
    import pandas as pd
    results = [verify_case(arm_path/r['token'], contract) for r in contract['cohort']]
    successful_rows = [r for r in contract['cohort'] if verify_case(arm_path/r['token'],contract)['interface_passed']]
    merged = arm_path/'merged'
    if merged.exists():
        if (arm_path/'complete.json').exists():
            result=load_json(arm_path/'complete.json')
            assert result['contract_sha256']==contract['contract_sha256']
            if result['aggregate_file'] is not None:
                assert sha(result['aggregate_file'])==result['aggregate_sha256']
            return result
        raise RuntimeError('Partial merge cannot be silently overwritten: ' + str(merged))
    (merged/'metrics').mkdir(parents=True)
    for name in contract['reference_metric_files']:
        if not successful_rows: break
        dfs = [pd.read_parquet(arm_path/r['token']/'metrics'/name) for r in successful_rows]
        frame = pd.concat(dfs, ignore_index=True)
        assert set(frame.scenario_name) == {r['token'] for r in successful_rows}
        assert not frame.duplicated(['scenario_name','planner_name']).any()
        frame.to_parquet(merged/'metrics'/name)
    aggregate = official_aggregate(merged/'metrics', contract['reference_config'], merged/'aggregator_metric') if successful_rows else None
    result = dict(arm=arm_path.name, contract_sha256=contract['contract_sha256'], cases=len(results),
        successful_cases=len(successful_rows),failed_cases=len(results)-len(successful_rows),
        aggregate_file=aggregate['path'] if aggregate else None,
        native_final_score=aggregate['score'] if len(successful_rows)==14 else None,
        official_success_subset_score=aggregate['score'] if aggregate else None,
        operational_fixed14_score=sum(r['score'] or 0.0 for r in results)/14,
        operational_definition='Arithmetic mean on frozen14 denominator, terminal scenario failures count0; not complete-cohort native score when failures exist.',
        aggregate_sha256=sha(aggregate['path']) if aggregate else None,
        total_case_seconds=sum(r['elapsed_s'] for r in results), official_aggregation=True)
    write(arm_path/'complete.json', result)
    return result


def run(args, contract):
    import torch
    from omegaconf import OmegaConf
    for p in reversed(contract['python_paths']): sys.path.insert(0, p)
    from fast_planner import FastFullPoolPlanner
    spec = importlib.util.spec_from_file_location('frozen_native_streaming', contract['streaming'])
    streaming = importlib.util.module_from_spec(spec); spec.loader.exec_module(streaming)
    output = Path(args.output)
    existing = list(output.glob('[ABC]/*'))
    if existing and not args.resume: raise RuntimeError('Case outputs exist; use explicit --resume after audit')
    torch.set_num_threads(2)
    new_cases = 0
    start = time.monotonic()
    with open(contract['gpu_lock'], 'a+') as lock:
        write(output/'status.json', dict(phase='waiting_for_GPU_lock', pid=os.getpid()))
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); break
            except BlockingIOError:
                if time.monotonic()-start > args.max_wall_seconds: raise TimeoutError('Finite GPU-lock wait expired')
                time.sleep(2)
        check_hashes(contract['sources_sha256'])
        for arm in contract['arms']:
            model = contract['models'][arm]
            if sha(model['path']) != model['sha256']: raise RuntimeError('Checkpoint changed')
            arm_path = output/arm
            arm_path.mkdir(exist_ok=True)
            pending = [r for r in contract['cohort'] if not (arm_path/r['token']).exists()]
            for row in contract['cohort']:
                if (arm_path/row['token']).exists(): verify_case(arm_path/row['token'], contract)
            if pending:
                planner = FastFullPoolPlanner(checkpoint=model['path'], device='cuda', use_ef=True,
                    fm_steps=2, chunk_size=64, diagnostic_path=None)
                original = {k:v.detach().cpu().clone() for k,v in planner.model.state_dict().items()}
                for row in pending:
                    if new_cases >= args.max_new_cases:
                        write(output/'status.json', dict(phase='case_budget_reached', new_cases=new_cases)); return
                    if time.monotonic()-start > args.max_wall_seconds: raise TimeoutError('Finite pilot wall budget expired')
                    case = arm_path/row['token']; case.mkdir()
                    planner.diagnostic_path = case/'planner_diagnostics.jsonl'
                    planner.last_full_pool_diagnostic = None
                    cfg = config_for_case(contract['reference_config'], row, case)
                    OmegaConf.save(cfg, case/'config.yaml')
                    write(output/'status.json', dict(phase='running_case', arm=arm, token=row['token'], new_cases=new_cases))
                    metadata = {}
                    execute = streaming.execute_runners
                    def observed_execute(runners, *pos, **kw):
                        assert len(runners) == 1
                        scenario = runners[0].scenario
                        assert scenario.token == row['token'] and scenario.log_name == row['log_name']
                        assert scenario.scenario_type == row['scenario_type']
                        metadata.update(iterations=scenario.get_number_of_iterations(),
                            interval_s=float(scenario.database_interval), scenario_type=scenario.scenario_type,
                            first_time_us=scenario.get_time_point(0).time_us,
                            last_time_us=scenario.get_time_point(scenario.get_number_of_iterations()-1).time_us)
                        return execute(runners, *pos, **kw)
                    streaming.execute_runners = observed_execute
                    tick = time.perf_counter()
                    try:
                        streaming.run_simulation_streaming(cfg, planners=planner)
                        torch.cuda.synchronize()
                        unchanged = all(torch.equal(v, planner.model.state_dict()[k].detach().cpu()) for k,v in original.items())
                        result = validate_case(case, row, contract, time.perf_counter()-tick, unchanged, metadata,args.allow_terminal_failure)
                        new_cases += 1
                        print(json.dumps(dict(event='CASE_COMPLETE', arm=arm, token=row['token'], score=result['score'],
                            terminal_status=result['terminal_status'],seconds=result['elapsed_s'], planner_calls=result.get('planner_calls'))), flush=True)
                    except BaseException as error:
                        write(case/'failure.json', dict(error=repr(error), elapsed_s=time.perf_counter()-tick, retry_forbidden=True))
                        raise
                    finally:
                        streaming.execute_runners = execute
                        gc.collect()
                del original, planner
                torch.cuda.empty_cache()
            aggregate_arm(arm_path, contract)
        check_hashes(contract['sources_sha256'])
        from comparison import build_comparison
        build_comparison(output,contract)
        write(output/'status.json', dict(phase='complete', new_cases=new_cases,
            arms=contract['arms'], wall_seconds=time.monotonic()-start))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['schema','run'], default='schema')
    p.add_argument('--output', required=True)
    p.add_argument('--code-root', required=True, help='Deployed experiments/ef_branch_training directory')
    p.add_argument('--runs', default=RUNS)
    p.add_argument('--arms', nargs='+', choices=['A','B','C'], default=['A','B','C'])
    p.add_argument('--cohort-manifest', default='/tmp/driveanchor_pilot14_20260910/pilot_first14_manifest.json')
    p.add_argument('--cohort-filter', default='/tmp/driveanchor_pilot14_20260910/pilot_first14.yaml')
    p.add_argument('--reference-config', default=REFERENCE)
    p.add_argument('--baseline-root',default=DP_BASELINE)
    p.add_argument('--streaming', default='/workdir/nuplan_streaming.py')
    p.add_argument('--cached-planner', default='/tmp/ef_epoch3_hard10_v1/steps_epoch_03/FM1/cached_planner.py')
    p.add_argument('--native-geometry', default='/tmp/ef_delta_v4_20260909/code/native_geometry.py')
    p.add_argument('--rebase-root', default='/tmp/ef_delta_v4_20260909')
    p.add_argument('--gpu-lock', default='/tmp/driveanchor_gpu_training.lock')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--allow-terminal-failure', action='store_true', help='Only after successful firstcase: record explicit native succeeded=False and continue fixed list')
    p.add_argument('--max-new-cases', type=int, default=42)
    p.add_argument('--max-wall-seconds', type=int, default=21600)
    args = p.parse_args()
    assert args.arms == sorted(set(args.arms))
    assert 1 <= args.max_new_cases <= 42 and 1 <= args.max_wall_seconds <= 21600
    os.environ.setdefault('NUPLAN_DATA_ROOT','/root/nuplan/dataset')
    os.environ.setdefault('NUPLAN_MAPS_ROOT','/workdir/nuplan_data/maps')
    os.environ.setdefault('NUPLAN_EXP_ROOT','/root/nuplan/exp')
    contract = schema(args)
    if args.mode == 'run': run(args, contract)


if __name__ == '__main__': main()
