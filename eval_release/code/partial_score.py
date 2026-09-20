"""Score the scenarios finished so far, before nuPlan writes its aggregator.

nuPlan keeps per-scenario metrics as ``*.pickle.temp`` and only converts them to
parquet and runs the weighted-average aggregator when the whole run ends.  For a
three-hour shard that means no visibility until it is over.  This reads the temp
files directly and applies the same formula the official aggregator does, so a
partial mean is available while the run continues.

The formula is nuPlan's ``closed_loop_nonreactive_agents_weighted_average``:
four multiplicative gates, then a weighted average over five scored metrics.
A gate at zero zeroes the scenario regardless of the rest, which is why the
zero count matters more than the mean when comparing configurations.

Numbers from this script are provisional: they cover only the scenarios that
have finished, and finishing order is not random -- shards work through their
own token list in sequence, so an early partial mean is biased by whatever
scenario types happen to sit at the front of those lists.
"""
from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

GATES = (
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "ego_is_making_progress",
    "driving_direction_compliance",
)
WEIGHTED = {
    "ego_progress_along_expert_route": 5.0,
    "time_to_collision_within_bound": 5.0,
    "speed_limit_compliance": 4.0,
    "ego_is_comfortable": 2.0,
}


def scenario_scores(root: Path) -> dict[str, float]:
    """Read both storage forms nuPlan uses for per-scenario metrics.

    A run writes ``*.pickle.temp`` as each scenario finishes, then converts them
    to parquet and deletes the temps when the run ends.  Counting only the temps
    made completed shards look like they had lost data -- the total dropped from
    139 to 108 as a shard finished.  Reading both keeps the tally monotonic.
    """
    per_scenario: dict[str, dict[str, float]] = defaultdict(dict)
    for path in root.rglob("*.pickle.temp"):
        try:
            rows = pickle.load(path.open("rb"))
        except Exception:
            continue  # still being written
        for row in rows:
            score = row.get("metric_score")
            if score is None:
                continue
            per_scenario[row["scenario_name"]][row["metric_computator"]] = float(score)

    import pandas as pd
    for path in root.rglob("metrics/*.parquet"):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        for row in frame.itertuples():
            score = getattr(row, "metric_score", None)
            if score is None or score != score:  # NaN
                continue
            per_scenario[row.scenario_name][row.metric_computator] = float(score)

    out: dict[str, float] = {}
    for name, metrics in per_scenario.items():
        # Only score scenarios whose full metric set has landed; a partially
        # written scenario would otherwise look like a zero on a missing gate.
        if not all(g in metrics for g in GATES):
            continue
        if not all(m in metrics for m in WEIGHTED):
            continue
        multiplier = 1.0
        for gate in GATES:
            multiplier *= metrics[gate]
        total = sum(WEIGHTED[m] * metrics[m] for m in WEIGHTED)
        out[name] = multiplier * total / sum(WEIGHTED.values())
    return out


def report(tag: str, root: Path, reference: str) -> None:
    scores = scenario_scores(root)
    if not scores:
        print(f"{tag}: 尚无完整场景")
        return
    values = list(scores.values())
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    se = (variance / n) ** 0.5
    zeros = sum(1 for v in values if v <= 1e-9)
    print(f"{tag}: {n}/228 场景 | 均值 {100*mean:.2f} ± {100*se:.2f} | "
          f"零分 {zeros} ({100*zeros/n:.0f}%) | 参照 {reference}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=Path, default=Path("/tmp/dp_eval/exp_val14"))
    ap.add_argument("--reactive", type=Path, default=Path("/tmp/dp_eval/exp_val14r"))
    a = ap.parse_args()
    report("Val14 NR", a.nr, "DP 89.87 / PLUTO 88.89 / PlanTF 84.72")
    report("Val14 R ", a.reactive, "DP 82.80 / PLUTO 78.11 / PlanTF 76.95")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
