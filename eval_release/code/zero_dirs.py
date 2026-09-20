"""List every zero-score scenario with the path its metrics were written to.

A zero is always a gate, never the weighted average, so the useful thing to
print next to the path is *which* gate fired -- that is the difference between
"drove into a car" and "never left the kerb", and they need different fixes.

The run has no simulation_log directory to point at: the shim no-ops
``SimulationLogCallback.on_simulation_end`` because its pickle of the planner
raised and took the aggregator down with it.  So what exists per scenario is the
metric temp file, and that is what is reported here.  Replaying these cases in
nuBoard needs a re-run with the callback restored; see ``--rerun-filter``, which
writes a scenario_filter containing exactly these tokens.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

GATES = ["no_ego_at_fault_collisions", "drivable_area_compliance",
         "ego_is_making_progress", "driving_direction_compliance"]
WEIGHTED = {"ego_progress_along_expert_route": 5.0, "time_to_collision_within_bound": 5.0,
            "speed_limit_compliance": 4.0, "ego_is_comfortable": 2.0}
LABEL = {"no_ego_at_fault_collisions": "碰撞", "drivable_area_compliance": "出可行驶区",
         "ego_is_making_progress": "无进度", "driving_direction_compliance": "逆行"}


def collect(root: Path):
    metrics: dict[str, dict[str, float]] = defaultdict(dict)
    where: dict[str, Path] = {}
    info: dict[str, dict] = {}
    for path in root.rglob("*.pickle.temp"):
        try:
            rows = pickle.load(path.open("rb"))
        except Exception:
            continue
        for row in rows:
            score = row.get("metric_score")
            if score is None:
                continue
            name = row["scenario_name"]
            metrics[name][row["metric_computator"]] = float(score)
            where[name] = path
            info[name] = {"log_name": str(row.get("log_name")),
                          "scenario_type": str(row.get("scenario_type"))}
    try:
        import pandas as pd
        for path in root.rglob("metrics/*.parquet"):
            try:
                frame = pd.read_parquet(path)
            except Exception:
                continue
            for row in frame.itertuples():
                score = getattr(row, "metric_score", None)
                if score is None or score != score:
                    continue
                metrics[row.scenario_name][row.metric_computator] = float(score)
                # A parquet holds one metric across every scenario of the shard,
                # so naming one file would point at whichever metric happened to
                # be read first.  The shard's metrics directory is the honest
                # anchor; the scenario is a row inside each parquet there.
                where[row.scenario_name] = path.parent
                info[row.scenario_name] = {"log_name": str(row.log_name),
                                           "scenario_type": str(row.scenario_type)}
    except ImportError:
        pass
    return metrics, where, info


def zeros(tag: str, root: Path) -> list[dict]:
    metrics, where, info = collect(root)
    out = []
    for name, entry in metrics.items():
        if not all(g in entry for g in GATES) or not all(k in entry for k in WEIGHTED):
            continue
        failed = [g for g in GATES if entry[g] <= 1e-9]
        if not failed:
            continue
        out.append({
            "arm": tag,
            "token": name.split("_")[-1] if False else name,
            "scenario_type": info[name]["scenario_type"],
            "log_name": info[name]["log_name"],
            "failed_gates": failed,
            "failed_zh": "+".join(LABEL[g] for g in failed),
            # Progress separates the two failure modes that both end at zero:
            # ~1.0 means it drove the whole route and then hit something,
            # ~0.0 means it never moved.
            "progress": round(entry["ego_progress_along_expert_route"], 3),
            "ttc": round(entry["time_to_collision_within_bound"], 3),
            "shard": (where[name] if where[name].is_dir() else where[name].parent).parent.name,
            "source": "temp" if where[name].is_file() else "parquet",
            "path": str(where[name]),
        })
    return sorted(out, key=lambda r: (r["failed_zh"], r["scenario_type"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=Path, default=Path("/tmp/dp_eval/exp_val14"))
    ap.add_argument("--reactive", type=Path, default=Path("/tmp/dp_eval/exp_val14r"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/dp_eval/zero_cases_dirs.json"))
    ap.add_argument("--rerun-filter", type=Path,
                    help="write a scenario_filter yaml holding exactly these tokens")
    a = ap.parse_args()

    rows = zeros("NR", a.nr) + zeros("R", a.reactive)
    a.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")

    for arm in ("NR", "R"):
        group = [r for r in rows if r["arm"] == arm]
        print(f"\n===== {arm}  {len(group)} 个零分 =====")
        by_reason = defaultdict(list)
        for r in group:
            by_reason[r["failed_zh"]].append(r)
        for reason, items in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            print(f"\n-- {reason}  ({len(items)}) --")
            for r in items:
                print(f"  {r['token']}")
                print(f"     类型 {r['scenario_type']}  进度 {r['progress']:.2f}  "
                      f"分片 {r['shard']}  log {r['log_name']}")
                print(f"     {r['path']}"
                      + ("" if r["source"] == "temp" else "/   <- 该场景是各 parquet 里的一行"))

    if a.rerun_filter:
        tokens = sorted({r["token"] for r in rows})
        body = "\n".join(f"- '{t}'" for t in tokens)
        a.rerun_filter.write_text(
            "scenario_tokens:\n" + body + "\n"
            "log_names: null\nscenario_types: null\nnum_scenarios_per_type: null\n"
            "limit_total_scenarios: null\ntimestamp_threshold_s: null\n"
            "ego_displacement_minimum_m: null\nexpand_scenarios: false\n"
            "remove_invalid_goals: true\nshuffle: false\n"
            "ego_start_speed_threshold: null\nego_stop_speed_threshold: null\n"
            "speed_noise_tolerance: null\n")
        print(f"\n重跑用 filter: {a.rerun_filter}  ({len(tokens)} tokens)")
    print(f"\n完整清单: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
