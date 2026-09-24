"""Continuous diagnostics from native signed route progress, not path length.
Native scores/thresholds remain unchanged; shaping is a separate training reward.
"""
import math

def progress_terms(ego_m, expert_m, threshold=.2, floor_m=2.):
    if not all(math.isfinite(x) for x in (ego_m,expert_m,threshold,floor_m)):
        raise ValueError('Finite native progress statistics required')
    if not 0<threshold<=1 or floor_m<=0:raise ValueError('Invalid threshold')
    denominator=max(expert_m,floor_m)
    ratio=0. if ego_m < -floor_m else min(1.,max(ego_m,floor_m)/denominator)
    # The official 2m floor makes short/expert-stationary scenes special.
    required=-floor_m if threshold*denominator<=floor_m else threshold*denominator
    deficit=max(0.,required-ego_m)
    return dict(ego_route_progress_m=ego_m,expert_route_progress_m=expert_m,
                official_ratio=ratio,official_pass=ratio>=threshold,
                minimum_signed_progress_for_pass_m=required,
                distance_deficit_m=deficit,
                continuous_completion_to_threshold=min(1.,ratio/threshold),
                proposed_normalized_deficit_penalty=-deficit/denominator)

if __name__=='__main__':
    p=progress_terms(7.523312101308297,74.45668875476046)
    assert not p['official_pass'] and abs(p['distance_deficit_m']-7.368025649643796)<1e-9
    assert progress_terms(14.891337750952093,74.45668875476046)['official_pass']
    assert progress_terms(0,0)['official_pass']
    assert not progress_terms(-2.001,0)['official_pass']
    assert progress_terms(8,74)['distance_deficit_m']<progress_terms(7,74)['distance_deficit_m']
    print(p)


def shape_native_reward(native_score, progress_statistics, weight=1.):
    """Penalize remaining native route distance; preserve raw evaluation score.

    Uses a completed native rollout, never a 4s path-length approximation.
    Honor native passing cases (including no-route and short-expert conventions).
    """
    if not math.isfinite(native_score) or not 0 <= native_score <= 1:
        raise ValueError('Expected official score in [0,1]')
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Expected nonnegative finite weight')
    ratio=float(progress_statistics['ego_expert_progress_along_route_ratio_stat_value'])
    if not math.isfinite(ratio):raise ValueError('Missing official progress ratio')
    if ratio >= .2:
        terms={'official_pass':True,'distance_deficit_m':0.,
               'proposed_normalized_deficit_penalty':0.}
    else:
        terms=progress_terms(
            float(progress_statistics['ego_total_progress_along_route_stat_value']),
            float(progress_statistics['expert_total_progress_along_route_stat_value']))
        if terms['official_pass']:raise ValueError('Native progress contract mismatch')
    penalty=weight*terms['proposed_normalized_deficit_penalty']
    return dict(native_score=float(native_score), reward=float(native_score+penalty),
                progress_weight=weight, progress_penalty=penalty, progress=terms)
