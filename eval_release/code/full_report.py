"""Emit the comparison table in the shape the paper table uses."""
import pickle, statistics as st
import pandas as pd
from pathlib import Path
from collections import defaultdict

GATES = ["no_ego_at_fault_collisions", "drivable_area_compliance",
         "ego_is_making_progress", "driving_direction_compliance"]
WEIGHTED = {"ego_progress_along_expert_route": 5.0, "time_to_collision_within_bound": 5.0,
            "speed_limit_compliance": 4.0, "ego_is_comfortable": 2.0}
CN = {"no_ego_at_fault_collisions": "无责碰撞", "drivable_area_compliance": "可行驶区域",
      "ego_is_making_progress": "有效前进", "driving_direction_compliance": "行驶方向",
      "ego_progress_along_expert_route": "路线进度", "time_to_collision_within_bound": "TTC",
      "speed_limit_compliance": "限速合规", "ego_is_comfortable": "舒适性"}
BASE = [("Diffusion Planner", 89.87, 82.80), ("PLUTO w/o refine.*", 88.89, 78.11),
        ("PlanTF", 84.72, 76.95), ("UrbanDriver", 68.57, 64.11),
        ("STR2-CPKS-800M w/o refine.*", 65.16, None), ("PDM-Open*", 53.53, 54.24),
        ("Diffusion-es w/o LLM", 50.00, None), ("GameFormer w/o refine.", 13.32, 8.69)]


def load(root):
    per = defaultdict(dict)
    for p in Path(root).rglob("*.pickle.temp"):
        try:
            rows = pickle.load(p.open("rb"))
        except Exception:
            continue
        for r in rows:
            s = r.get("metric_score")
            if s is not None:
                per[r["scenario_name"]][r["metric_computator"]] = float(s)
    for p in Path(root).rglob("metrics/*.parquet"):
        try:
            f = pd.read_parquet(p)
        except Exception:
            continue
        for r in f.itertuples():
            s = getattr(r, "metric_score", None)
            if s is None or s != s:
                continue
            per[r.scenario_name][r.metric_computator] = float(s)
    return {n: m for n, m in per.items()
            if all(g in m for g in GATES) and all(k in m for k in WEIGHTED)}


def total(m):
    mult = 1.0
    for g in GATES:
        mult *= m[g]
    return mult * sum(WEIGHTED[k] * m[k] for k in WEIGHTED) / sum(WEIGHTED.values())


A, B = load("/tmp/dp_eval/exp_val14"), load("/tmp/dp_eval/exp_val14r")
va = [total(m) for m in A.values()]
vb = [total(m) for m in B.values()]
ours = (100 * st.mean(va), 100 * st.mean(vb))

rows = BASE + [("本方法", ours[0], ours[1])]
print(f"\n{'方法':<30}{'Val14 (NR)':>12}{'Val14 (R)':>12}")
for name, nr, r in sorted(rows, key=lambda x: -(x[1] or 0)):
    mark = "  <-" if name == "本方法" else ""
    print(f"{name:<30}{nr:>12.2f}{(f'{r:.2f}' if r is not None else '—'):>12}{mark}")

for tag, v in (("NR", va), ("R", vb)):
    nz = [x for x in v if x > 1e-9]
    print(f"\n{tag}: n={len(v)}  总均值 {100*st.mean(v):.2f}  "
          f"非零均值 {100*st.mean(nz):.2f}  零分 {len(v)-len(nz)} ({100*(len(v)-len(nz))/len(v):.0f}%)")

print(f"\n{'指标':<14}{'权重':>5}{'NR':>9}{'R':>9}")
for g in GATES:
    print(f"{CN[g]:<14}{'门':>5}"
          f"{100*st.mean(1 if A[n][g] > 1e-9 else 0 for n in A):>8.1f}%"
          f"{100*st.mean(1 if B[n][g] > 1e-9 else 0 for n in B):>8.1f}%")
for k, w in sorted(WEIGHTED.items(), key=lambda x: -x[1]):
    print(f"{CN[k]:<14}{w:>5.0f}{100*st.mean(A[n][k] for n in A):>9.2f}"
          f"{100*st.mean(B[n][k] for n in B):>9.2f}")
