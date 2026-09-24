"""Native FullPoolPlanner with the parity-checked point-predicate selector."""
from driveanchor_planner import DriveAnchorPlanner
from full_pool_planner import FullPoolPlanner
from fast_selector import VERSION, make_fast_choose


# nuPlan's simulation log pickles the planner instance. A factory-local
# function cannot be part of its instance state; keep this immutable binding
# in the importable module instead. The same audited function and rules run
# before and after loading a log, without discarding any parent planner state.
_PREPARED_CHOOSE = make_fast_choose(DriveAnchorPlanner.choose)


class FastFullPoolPlanner(FullPoolPlanner):
    def choose(self, xy, route, tracks, ego, speed, lights, lanes):
        return _PREPARED_CHOOSE(self, xy, route, tracks, ego, speed, lights, lanes)

    def name(self):
        return super().name() + '_' + VERSION
