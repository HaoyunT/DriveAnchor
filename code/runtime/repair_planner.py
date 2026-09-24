"""A frozen EF/FM model with separately reported native/runtime corrections."""
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.vectorized import contains
import selection
from driveanchor_planner import local
from fast_planner import FastFullPoolPlanner
from full_pool_selector import generate_full_pool
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from geometry import route_corridor
from braking import braking_trajectory

GUARD = dict(version='lead_clearance_and_bounded_stop_v1', lead_longitudinal_margin_m=1.,
             learned_candidates=500, maximum_braking_candidates=1,
             brake_requires_all_original_constraints=True,
             activation='no feasible learned candidate with enlarged current lead footprint')
GUARD_SHA = hashlib.sha256(json.dumps(GUARD,sort_keys=True).encode()).hexdigest()


class RepairPlanner(FastFullPoolPlanner):
    def __init__(self, *args, enable_guard=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_guard=enable_guard
        self.pending_lane_id=None

    def initialize(self, initialization):
        super().initialize(initialization)
        self.pending_lane_id=None

    def name(self):
        return super().name()+('_geometry_guard_v5' if self.enable_guard else '_geometry_repair_v5')

    def _selection(self, xy, route, tracks, state, lights, lanes):
        inputs=dict(route=route,tracks=tracks,ego=state.rear_axle,
                    speed=state.dynamic_car_state.speed,lights=lights,lanes=lanes)
        original_id=self.choose(xy,**inputs)
        original=copy.deepcopy(self.selection_diagnostic)
        decision=dict(enabled=self.enable_guard,original_selected_id=int(original_id),
            original_diagnostic=original,selection_source='learned',margin_m=1.,
            guard_rules_sha256=GUARD_SHA,braking_candidate_added=False)
        if not self.enable_guard:
            return xy[original_id],original_id,original,decision
        inflated=[];lead_count=0
        for obj in tracks:
            pos=local([[obj.center.x,obj.center.y]],state.rear_axle)[0]
            yaw=obj.center.heading-state.rear_axle.heading
            lateral_extent=abs(np.sin(yaw))*obj.box.length/2+abs(np.cos(yaw))*obj.box.width/2
            if pos[0]>0 and abs(pos[1])<=selection.RULES.half_width+lateral_extent:
                attrs=dict(center=obj.center,box=SimpleNamespace(length=obj.box.length+2.,width=obj.box.width))
                if hasattr(obj,'velocity'):attrs['velocity']=obj.velocity
                inflated.append(SimpleNamespace(**attrs));lead_count+=1
            else: inflated.append(obj)
        decision['lead_count']=lead_count
        if not lead_count:
            return xy[original_id],original_id,original,decision
        robust_inputs=dict(inputs,tracks=inflated)
        robust_id=self.choose(xy,**robust_inputs)
        robust=copy.deepcopy(self.selection_diagnostic)
        decision['clearance_diagnostic']=robust
        if robust['feasible_count']:
            decision['selection_source']='learned_clearance'
            return xy[robust_id],robust_id,robust,decision
        dynamic=state.dynamic_car_state
        params=state.car_footprint.vehicle_parameters
        curvature=np.tan(state.tire_steering_angle)/params.wheel_base
        brake, profile=braking_trajectory(dynamic.rear_axle_velocity_2d.x,dynamic.rear_axle_acceleration_2d.x,curvature)
        decision['braking_profile']=profile
        if brake is not None:
            # The exact frozen scorer is shape-bound to500. Broadcast the one
            # candidate only for that API; it remains ONE deterministic candidate.
            self.choose(np.broadcast_to(brake,(500,40,2)),**inputs)
            checked=copy.deepcopy(self.selection_diagnostic)
            decision.update(braking_candidate_added=True,braking_original_constraint_check=checked)
            if checked['feasible_count']==500:
                decision['selection_source']='bounded_brake'
                diagnostic=dict(checked,candidate_count=500,feasible_count=original['feasible_count'],
                    selection_fallback=True)
                return brake,None,diagnostic,decision
        return xy[original_id],original_id,original,decision

    @torch.no_grad()
    def compute_planner_trajectory(self,current_input):
        begin=time.perf_counter();history=current_input.history
        state=history.ego_states[-1];ego=state.rear_axle
        features,mask,lanes,tracks=self.features(history)
        try:
            route=self.route_line(lanes,state)
        except ValueError as error:
            if str(error)!='Route centerline too short':raise
            route=None
        try:
            corridor,geometry=route_corridor(lanes,self.route_ids,state,state.car_footprint.vehicle_parameters.width/2,
                preferred_start_id=self.pending_lane_id)
        except ValueError as error:
            if self.diagnostic_path:
                snapshot=dict(error=str(error),ego=dict(x=ego.x,y=ego.y,heading=ego.heading),
                    speed=state.dynamic_car_state.speed,iteration=current_input.iteration.index,
                    route_ids=self.route_ids,lanes=[dict(id=x.id,roadblock=str(x.get_roadblock_id()),
                        baseline=list(x.baseline_path.linestring.coords),left=list(x.left_boundary.linestring.coords),
                        right=list(x.right_boundary.linestring.coords),polygon=list(x.polygon.exterior.coords),
                        outgoing=[y.id for y in x.outgoing_edges]) for x in lanes])
                Path(self.diagnostic_path).with_name('geometry_failure.json').write_text(json.dumps(snapshot))
            raise
        self.pending_lane_id=geometry['route_lane_ids'][0] if geometry.get('alternative_start') else None
        if geometry.get('alternative_start') or route is None:
            route=np.asarray(geometry['scoring_route_xy'])
        context,valid=self.model.encode(features,mask)
        paths=generate_full_pool(self.model,corridor,context,valid,self.fm_steps,self.chunk_size)
        stage='FM'+str(self.fm_steps);xy=paths[stage].cpu().numpy()
        tick=time.perf_counter()
        selected,index,diagnostic,guard=self._selection(xy,route,tracks,state,list(current_input.traffic_light_data or []),lanes)
        selector_seconds=time.perf_counter()-tick
        poses=np.column_stack([selected[1:],selection.stable_heading(selected[None])[0,1:]])
        trajectory=InterpolatedTrajectory(transform_predictions_to_states(poses,history.ego_states,3.9,.1))
        self.selection_diagnostic=diagnostic
        if self.enable_guard:
            diagnostic=dict(diagnostic,base_selector_rules_sha256=diagnostic.get('selector_rules_sha256'),
                selector_rules_sha256=GUARD_SHA)
        region=unary_union([lane.polygon for lane in lanes]).buffer(.3)
        c,s=np.cos(ego.heading),np.sin(ego.heading);rot=np.array([[c,s],[-s,c]])
        world=xy@rot+[ego.x,ego.y];inside=contains(region,world[...,0],world[...,1])
        selected_world=selected@rot+[ego.x,ego.y]
        selected_inside=contains(region,selected_world[:,0],selected_world[:,1])
        self.last_full_pool_diagnostic=dict(diagnostic,iteration=current_input.iteration.index,
            output_stage=stage,prefilter='none',selected_anchor_id=index,geometry=geometry,safety_guard=guard,
            inference_version=self.name(),selector_seconds=selector_seconds,
            total_planner_seconds=time.perf_counter()-begin,learned_candidate_count=500,
            extra_braking_candidates=int(guard['braking_candidate_added']),
            candidate_any_ool_fraction=float((~inside[:,1:]).any(1).mean()),
            selected_any_ool=bool((~selected_inside[1:]).any()),executed_point_ool=not region.covers(Point(ego.x,ego.y)),
            OOL_definition='unchanged buffered0.3m rear-axle point test; candidate fraction covers learned500 only',
            selected_xy=selected.tolist(),selected_sampled_polygon_good=bool(corridor.labels(selected[None])[0]),
            future_GT_input=False)
        if self.diagnostic_path:
            with Path(self.diagnostic_path).open('a') as f:f.write(json.dumps(self.last_full_pool_diagnostic)+'\n')
        return trajectory
