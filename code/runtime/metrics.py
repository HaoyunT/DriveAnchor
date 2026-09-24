"""Collect native results without dropping failed or missing cohort members."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from progress import shape_native_reward

SAFETY = ('no_ego_at_fault_collisions', 'drivable_area_compliance',
          'driving_direction_compliance', 'time_to_collision_within_bound')
SCORES = SAFETY + ('ego_is_making_progress', 'ego_progress_along_expert_route',
                   'ego_is_comfortable', 'speed_limit_compliance')


def collect_case(case, token, expected_metric_files):
    case = Path(case)
    report = pd.read_parquet(case / 'runner_report.parquet')
    if len(report) != 1 or str(report.iloc[0].scenario_name) != token:
        raise ValueError('Runner report does not match exactly one requested token')
    if not bool(report.iloc[0].succeeded):
        raise ValueError(str(report.iloc[0].get('error_message', 'Native runner failed')))
    files = {p.name for p in (case / 'metrics').glob('*.parquet')}
    if files != set(expected_metric_files):
        raise ValueError('Incomplete native metric set')
    score_rows = [r for p in (case / 'aggregator_metric').glob('*.parquet')
                  for r in pd.read_parquet(p).to_dict('records') if r.get('scenario') == token]
    if len(score_rows) != 1:
        raise ValueError('Expected one native aggregated scenario score')
    score_row = score_rows[0]
    official = {k: float(score_row[k]) for k in SCORES + ('score',)}
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in official.values()):
        raise ValueError('Invalid native score')
    progress = pd.read_parquet(case / 'metrics/ego_progress_along_expert_route.parquet')
    progress = progress[progress.scenario_name == token]
    if len(progress) != 1:
        raise ValueError('Missing exact-token native progress statistics')
    stats = progress.iloc[0].to_dict()
    reward = shape_native_reward(official['score'], stats)
    diag = [json.loads(s) for s in (case / 'planner_diagnostics.jsonl').read_text().splitlines()]
    if len(diag) not in (149, 150) or [d['iteration'] for d in diag] != list(range(len(diag))):
        raise ValueError('Incomplete 15-second closed loop')
    if not all(d['future_GT_input'] is False for d in diag):
        raise ValueError('Future information used as planner input')
    times = np.array([d['total_planner_seconds'] for d in diag])
    return dict(token=token, succeeded=True, official=official, shaped_reward=reward['reward'],
                progress_penalty=reward['progress_penalty'],
                ego_route_progress_m=float(stats['ego_total_progress_along_route_stat_value']),
                expert_route_progress_m=float(stats['expert_total_progress_along_route_stat_value']),
                official_progress_ratio=float(stats['ego_expert_progress_along_route_ratio_stat_value']),
                planner_calls=len(diag),
                selector_fallback_fraction=float(np.mean([d['safety_guard']['original_diagnostic']['selection_fallback'] for d in diag])),
                bounded_brake_fraction=float(np.mean([d['safety_guard']['selection_source'] == 'bounded_brake' for d in diag])),
                selected_OOL_fraction=float(np.mean([d['selected_any_ool'] for d in diag])),
                logged_planner_ms=dict(p50=float(np.quantile(times, .5)*1000),
                                       p95=float(np.quantile(times, .95)*1000),
                                       max=float(times.max()*1000)),
                timing_scope='Logged planner time; not an independent end-to-end latency benchmark')


def aggregate(rows, expected_tokens):
    expected_tokens = list(expected_tokens)
    if len(expected_tokens) != len(set(expected_tokens)) or not expected_tokens:
        raise ValueError('Expected a nonempty unique frozen cohort')
    by = {r['token']: r for r in rows}
    if len(by) != len(rows) or not set(by) <= set(expected_tokens):
        raise ValueError('Duplicate or unexpected scenario')
    good = [by[t] for t in expected_tokens if t in by and by[t]['succeeded']]
    n = len(expected_tokens)
    all_complete = len(good) == n
    return dict(expected=n, attempted=len(rows), successful=len(good),
                missing=[t for t in expected_tokens if t not in by],
                failed=[t for t in expected_tokens if t in by and not by[t]['succeeded']],
                complete=all_complete,
                official_mean=float(np.mean([r['official']['score'] for r in good])) if all_complete else None,
                observed_success_mean=float(np.mean([r['official']['score'] for r in good])) if good else None,
                fixed_denominator_score_failures_zero=sum(r['official']['score'] for r in good)/n,
                fixed_denominator_label='Development accounting; execution failures are not official driving scores',
                safety_failures={k:sum(r['official'][k] < 1-1e-8 for r in good) for k in SAFETY},
                progress_failures=sum(r['official']['ego_is_making_progress'] < 1-1e-8 for r in good),
                comfort_failures=sum(r['official']['ego_is_comfortable'] < 1-1e-8 for r in good))
