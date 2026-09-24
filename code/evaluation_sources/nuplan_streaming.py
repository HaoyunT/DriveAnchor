import gc
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nuplan.common.utils.distributed_scenario_filter import DistributedMode, DistributedScenarioFilter
from nuplan.common.utils.file_backed_barrier import distributed_sync
from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.script.builders.metric_builder import build_metrics_engines
from nuplan.planning.script.builders.observation_builder import build_observations
from nuplan.planning.script.builders.planner_builder import build_planners
from nuplan.planning.script.builders.simulation_callback_builder import (
    build_callbacks_worker,
    build_simulation_callbacks,
)
from nuplan.planning.script.builders.simulation_builder import build_simulations
from nuplan.planning.script.builders.utils.utils_type import is_target_type
from nuplan.planning.script.utils import save_runner_reports, set_default_path, set_up_common_builder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.runner.executor import execute_runners
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.simulation_time_controller.abstract_simulation_time_controller import (
    AbstractSimulationTimeController,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_default_path()

CONFIG_PATH = os.path.join("/workdir/nuplan-devkit/nuplan/planning/script", "config/simulation")
CONFIG_NAME = "default_simulation"
BATCH_SIZE = 20


def run_simulation_streaming(cfg: DictConfig, planners: Optional[Union[AbstractPlanner, List[AbstractPlanner]]] = None) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    profiler_name = "building_simulation"
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)
    # The historical planner contains dynamically imported modules that cannot
    # be pickled by nuPlan's simulation-log callback. Metrics and aggregation
    # remain enabled; replay serialization is optional for evaluation.
    OmegaConf.set_struct(cfg, False)
    if "callback" in cfg:
        cfg.callback.pop("simulation_log_callback", None)
    OmegaConf.set_struct(cfg, True)
    callbacks_worker_pool = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(cfg=cfg, output_dir=common_builder.output_dir, worker=callbacks_worker_pool)
    # SimulationLogCallback serializes the live planner at scenario end.  The
    # production selector owns CUDA/GEOS runtime objects (and an IDM worker)
    # that are intentionally not part of evaluation artifacts, so that
    # callback can fail with a PyCapsule/lock pickle error after all metrics
    # have already been computed.  Metrics and the official aggregator are the
    # required outputs; filter only the optional replay-log callback here.
    callbacks = [callback for callback in callbacks
                 if callback.__class__.__name__ != 'SimulationLogCallback']

    if planners and "planner" in cfg.keys():
        logger.info("Using pre-instantiated planner. Ignoring planner in config")
        OmegaConf.set_struct(cfg, False)
        cfg.pop("planner")
        OmegaConf.set_struct(cfg, True)
    if isinstance(planners, AbstractPlanner):
        planners = [planners]

    scenario_filter = DistributedScenarioFilter(
        cfg=cfg, worker=common_builder.worker,
        node_rank=int(os.environ.get("NODE_RANK", 0)),
        num_nodes=int(os.environ.get("NUM_NODES", 1)),
        synchronization_path=cfg.output_dir,
        timeout_seconds=cfg.distributed_timeout_seconds,
        distributed_mode=DistributedMode[cfg.distributed_mode],
    )
    scenarios = scenario_filter.get_scenarios()
    logger.info("Streaming mode: %d scenarios, batch_size=%d", len(scenarios), BATCH_SIZE)

    metric_engines_map = {}
    if cfg.run_metric:
        metric_engines_map = build_metrics_engines(cfg=cfg, scenarios=scenarios)

    all_reports = []
    for batch_start in range(0, len(scenarios), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(scenarios))
        batch_scenarios = scenarios[batch_start:batch_end]
        batch_idx = (batch_start // BATCH_SIZE) + 1
        num_batches = (len(scenarios) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info("=== BATCH [%d/%d] scenarios %d-%d (%d) ===",
                    batch_idx, num_batches, batch_start + 1, batch_end, len(batch_scenarios))
        batch_reports = []
        for idx_in_batch, scenario in enumerate(batch_scenarios):
            global_idx = batch_start + idx_in_batch + 1
            logger.info("=== [%d/%d] scenario %s ===", global_idx, len(scenarios), scenario.scenario_name)
            runner = None
            try:
                if planners is None:
                    build_planners_list = build_planners(cfg.planner, scenario)
                else:
                    build_planners_list = planners
                for planner in build_planners_list:
                    from hydra.utils import instantiate
                    ego_controller = instantiate(cfg.ego_controller, scenario=scenario)
                    simulation_time_controller = instantiate(cfg.simulation_time_controller, scenario=scenario)
                    observations = build_observations(cfg.observation, scenario=scenario)
                    metric_engine = metric_engines_map.get(scenario.scenario_type, None)
                    stateful_callbacks = []
                    if metric_engine is not None:
                        stateful_callbacks.append(MetricCallback(metric_engine=metric_engine, worker_pool=callbacks_worker_pool))
                    # Keep simulation log serialization disabled for this
                    # runtime; see the callback filter above.  It is optional
                    # for scoring and cannot pickle the GPU/GEOS planner.
                    simulation_setup = SimulationSetup(
                        time_controller=simulation_time_controller,
                        observations=observations,
                        ego_controller=ego_controller,
                        scenario=scenario,
                    )
                    simulation = Simulation(
                        simulation_setup=simulation_setup,
                        callback=MultiCallback(callbacks + stateful_callbacks),
                        simulation_history_buffer_duration=cfg.simulation_history_buffer_duration,
                    )
                    runner = SimulationRunner(simulation, planner)
                    reports = execute_runners(
                        runners=[runner], worker=common_builder.worker,
                        num_gpus=cfg.number_of_gpus_allocated_per_simulation,
                        num_cpus=cfg.number_of_cpus_allocated_per_simulation,
                        exit_on_failure=False, verbose=cfg.verbose,
                    )
                    batch_reports.extend(reports)
            except Exception as e:
                import traceback
                logger.warning("Scenario %s failed: %s", scenario.scenario_name, e)
                logger.warning(traceback.format_exc())
            finally:
                del runner
                gc.collect()
        all_reports.extend(batch_reports)
        logger.info("=== BATCH DONE [%d/%d], %d scenarios, %d successful ===",
                    batch_idx, num_batches, len(batch_reports), sum(1 for r in batch_reports if r.succeeded))
        gc.collect()

    logger.info("All scenarios done. %d reports", len(all_reports))
    save_runner_reports(all_reports, common_builder.output_dir, cfg.runner_report_file)
    distributed_sync(Path(cfg.output_dir / Path("barrier")), cfg.distributed_timeout_seconds)
    if int(os.environ.get("NODE_RANK", 0)) == 0:
        common_builder.multi_main_callback.on_run_simulation_end()
    if common_builder.profiler:
        common_builder.profiler.save_profiler("running_simulation")
    num_ok = sum(1 for r in all_reports if r.succeeded)
    logger.info("Successful: %d / %d", num_ok, len(all_reports))


def clean_up_s3_artifacts() -> None:
    working_path = os.getcwd()
    s3_dirname = "s3:"
    s3_ind = working_path.find(s3_dirname)
    if s3_ind != -1:
        from shutil import rmtree
        local_s3_path = working_path[: working_path.find(s3_dirname) + len(s3_dirname)]
        rmtree(local_s3_path)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    run_simulation_streaming(cfg=cfg)
    if is_s3_path(Path(cfg.output_dir)):
        clean_up_s3_artifacts()


if __name__ == "__main__":
    main()
