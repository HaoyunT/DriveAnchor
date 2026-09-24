"""Ordering only for explicitly unresolved-risk fallback, never a safe selector."""
import numpy as np

def order(original_risky, original_ttc, candidate_ttc, progress):
    t=np.asarray(candidate_ttc,float);p=np.asarray(progress,float)
    if t.ndim!=1 or p.shape!=t.shape:raise ValueError('matching candidate vectors required')
    if not original_risky or not np.isfinite(original_ttc):return np.empty(0,dtype=int)
    valid=np.isfinite(t)&np.isfinite(p)&(t>float(original_ttc)+1e-6)
    ids=np.flatnonzero(valid)
    return ids[np.lexsort((ids,-p[ids],-t[ids]))]

def only_ttc_violation(diagnostic):
    names=diagnostic.get('constraint_names',[]);values=diagnostic.get('selected_violations',[])
    # Unknown/missing scorer schema cannot silently authorize a fallback.
    required={'predicted_collision','lane_footprint_exit','current_red_entry','speed','acceleration','dtpp_ttc','candidate_comfort'}
    if len(names)!=len(values) or len(set(names))!=len(names) or not required.issubset(names):return False
    return all(not bool(v) or name=='dtpp_ttc' for name,v in zip(names,values))
