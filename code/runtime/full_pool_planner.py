"""Versioned nuPlan adapter: native context -> all500 EF/FM -> frozen Q."""
import inspect
import json
import logging
import time
from pathlib import Path
import numpy as np
import torch
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.vectorized import contains
from cached_planner import CachedGeometryPlanner
import cached_planner
import driveanchor_planner
import selection
from native_geometry import route_corridor
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from source_binding import checked_file,RULES_SHA,PLANNER_SHA,CACHED_SHA,NATIVE_GEOMETRY_SHA
from full_pool_selector import generate_full_pool,select_full_pool,INFERENCE_VERSION


class FullPoolPlanner(CachedGeometryPlanner):
    def __init__(self,*args,fm_steps=2,chunk_size=64,use_ef=True,diagnostic_path=None,**kwargs):
        if not use_ef:raise ValueError('This protocol requires EF once before FM')
        if fm_steps not in (1,2):raise ValueError('Only FM1 or FM2 comparison supported')
        checked_file(selection.__file__,RULES_SHA)
        checked_file(driveanchor_planner.__file__,PLANNER_SHA)
        checked_file(cached_planner.__file__,CACHED_SHA)
        checked_file(inspect.getfile(route_corridor),NATIVE_GEOMETRY_SHA)
        super().__init__(*args,use_ef=True,**kwargs)
        self.model.eval().requires_grad_(False)
        self.fm_steps,self.chunk_size=fm_steps,chunk_size
        self.diagnostic_path=Path(diagnostic_path) if diagnostic_path else None
        self.last_full_pool_diagnostic=None
        if self.model.ef_head is None or len(self.model.anchors)!=3000:
            raise ValueError('Checkpoint must contain trained EF and the full3000 vocabulary')

    def name(self):return INFERENCE_VERSION+'_FM'+str(self.fm_steps)

    @torch.no_grad()
    def compute_planner_trajectory(self,current_input):
        begin=time.perf_counter();history=current_input.history
        state=history.ego_states[-1];ego=state.rear_axle
        features,mask,lanes,tracks=self.features(history)
        route=self.route_line(lanes,state)
        # Route correction above precedes native corridor construction. A
        # missing corridor is an explicit failure, never a fabricated feature.
        corridor,geometry=route_corridor(lanes,self.route_ids,state,
            state.car_footprint.vehicle_parameters.width/2)
        context,valid=self.model.encode(features,mask)
        paths=generate_full_pool(self.model,corridor,context,valid,self.fm_steps,self.chunk_size)
        stage='FM'+str(self.fm_steps)
        xy=paths[stage].cpu().numpy()
        tick=time.perf_counter()
        index,diagnostic=select_full_pool(self,xy,route,tracks,ego,state.dynamic_car_state.speed,
            list(current_input.traffic_light_data or []),lanes,stage)
        selector_seconds=time.perf_counter()-tick
        selected=xy[index]
        poses=np.column_stack([selected[1:],selection.stable_heading(selected[None])[0,1:]])
        trajectory=InterpolatedTrajectory(transform_predictions_to_states(poses,history.ego_states,3.9,.1))
        self.selection_diagnostic=diagnostic
        # Preserve the previous diagnostic's exact rear-axle OOL definition.
        # This buffered sampled-point check is separate from official DAC and
        # from the selector's unbuffered all-corner lane-union constraint.
        region=unary_union([lane.polygon for lane in lanes]).buffer(.3)
        c,s=np.cos(ego.heading),np.sin(ego.heading)
        world=xy@np.array([[c,s],[-s,c]])+np.array([ego.x,ego.y])
        inside=contains(region,world[...,0],world[...,1])
        self.last_full_pool_diagnostic=dict(diagnostic,iteration=current_input.iteration.index,
            selector_seconds=selector_seconds,total_planner_seconds=time.perf_counter()-begin,geometry=geometry,
            candidate_any_ool_fraction=float((~inside[:,1:]).any(1).mean()),
            selected_any_ool=bool((~inside[index,1:]).any()),
            executed_point_ool=not region.covers(Point(ego.x,ego.y)),
            OOL_definition='rear-axle sampled future points outside nearby lane union buffered0.3m; not official DAC',
            selected_sampled_polygon_good=bool(corridor.labels(paths[stage][index:index+1])[0]),
            selected_xy=selected.tolist(),future_GT_input=False)
        logging.info('DRIVEANCHOR_DIAGNOSTIC %s',json.dumps(self.last_full_pool_diagnostic))
        if self.diagnostic_path is not None:
            self.diagnostic_path.parent.mkdir(parents=True,exist_ok=True)
            with self.diagnostic_path.open('a') as stream:
                stream.write(json.dumps(self.last_full_pool_diagnostic)+'\n')
        return trajectory
