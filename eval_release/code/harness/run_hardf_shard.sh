#!/usr/bin/env bash
set -euo pipefail
SHARD=$1
CHALLENGE=${2:-closed_loop_nonreactive_agents}
export HYDRA_FULL_ERROR=1
# The planner config pins device: cuda, i.e. the first visible card.  Without
# this every shard lands on device 0 and 24 of them exhaust its 42 GB -- the
# earlier run died with CUDACachingAllocator OOM.  Pinning one card per shard
# spreads them four ways; the smoke case used 1.9 GB, so six per card fits.
export CUDA_VISIBLE_DEVICES=$(( $1 % 3 ))
export NUPLAN_DEVKIT_ROOT=/home/user/third_party/nuplan-devkit
export NUPLAN_DATA_ROOT=/tmp/fag_repro/nuplan_data
export NUPLAN_MAPS_ROOT=/home/user/nuplan_fullpool_8s/raw/maps
export NUPLAN_EXP_ROOT=/tmp/dp_eval/exp_hardf
export DRIVEANCHOR_AUTOCAR_ROOT=/tmp/fag_repro/autocar/.cache_pt_pytorch2/nuplan_recovery_20260910
export DRIVEANCHOR_PREFILTER=scorehead DRIVEANCHOR_VNORM_TOPK=1
R=/tmp/driveanchor_selector_v8_full_runtime
# worker=sequential, not ray: the planner holds CUDA state and ray fails with
# "Could not serialize the argument SimulationRunner".  Parallelism comes from
# running 24 of these processes, one per shard yaml.
PYTHONPATH=/tmp/dp_eval/shim_all:$R:/tmp/fag_repro/workdir/driveanchor_selector_frozen_v1:/tmp/driveanchor_runtime_repairs_v5_20260910:/tmp/driveanchor_quick_pilot_v2_20260910/experiments/ef_branch_training/full_pool_acceleration:/tmp/driveanchor_quick_pilot_v2_20260910/experiments/ef_branch_training/full_pool_inference:/tmp/ef_delta_v4_20260909:/tmp/ef_delta_v4_20260909/code:/tmp/ef_epoch3_hard10_v1/steps_epoch_03/FM1:/home/user/da-fix-zero/current/dependencies/root_004:/home/user/da-fix-zero/current/dependencies/root_010:/home/user/third_party/Diffusion-Planner \
/home/user/anaconda3/envs/pytorch_env/bin/python3.10 $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
  +simulation=$CHALLENGE planner=driveanchor_fag \
  scenario_builder=nuplan_challenge scenario_filter=hardf_s$SHARD \
  experiment_uid=driveanchor_fag/test14hardf/$CHALLENGE/s$SHARD \
  verbose=false worker=sequential distributed_mode=SINGLE_NODE \
  number_of_gpus_allocated_per_simulation=0.15 enable_simulation_progress_bar=false \
  "hydra.searchpath=[file:///tmp/dp_eval/cfg, pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
