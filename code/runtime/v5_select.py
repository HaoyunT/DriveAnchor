"""Integrated CUDA selection with one explicit legacy candidate boundary.

The legacy boundary still includes road/red/direction/DTPP implementations.
It is not device resident; DTPP internal transfers remain separately reported.
"""
import numpy as np
import torch
from device_features import candidate_features
from frame_provider import FrameCollisionProvider
from static_track_history import StaticTrackHistory
from candidate_costs_gpu import comfort, route_progress
from selector_tensor import choose_tensor
from selection import RULES, RULES_SHA256


def make_v5_choose(legacy_factory, reference):
    legacy = legacy_factory(reference)
    local = reference.__globals__['local']

    @torch.inference_mode()
    def choose(self, xy, route, tracks, ego, speed, lights, lanes):
        key=xy.tobytes()
        reused=key==getattr(self,'_candidate_xy_key',None)
        gx=self._candidate_xy_cuda if reused else torch.as_tensor(xy,device='cuda')
        features=candidate_features(gx,dt=RULES.dt,heading_hold_speed=RULES.heading_hold_speed,center_offset=RULES.center_offset)
        # One explicit legacy boundary. xy is the model-output download already
        # produced by fm2_planner. No separate feature download is performed.
        old=legacy(self,xy,route,tracks,ego,speed,lights,lanes)
        packed=np.column_stack((old['outside'],old['red'],old['ttc_bad'],old['quality']))
        uploaded=torch.as_tensor(packed,device='cuda',dtype=torch.float64)
        outside,red,ttc_bad=uploaded[:,:3].bool().unbind(1)
        quality=uploaded[:,3]
        state=self._comfort_history[-1]
        frame_id=state.time_point.time_us
        if not hasattr(self,'_v5_collision_provider'):
            self._v5_collision_provider=FrameCollisionProvider()
            self._v5_static_history=StaticTrackHistory()
            self._v5_static_frame=None
        if self._v5_static_frame!=frame_id:
            import math
            observations=[]
            for obj in tracks:
                # Guard wrappers intentionally omit identity/type. Normal union
                # collision remains mandatory; do not invent braking identity.
                if not hasattr(obj,'track_token') or not hasattr(obj,'tracked_object_type'):continue
                vel=getattr(obj,'velocity',None)
                observations.append(dict(token=obj.track_token,kind=obj.tracked_object_type.name,x=obj.center.x,y=obj.center.y,speed=math.hypot(vel.x,vel.y) if vel is not None else 0.))
            self._v5_static_tokens=self._v5_static_history.update(state.time_point.time_s,observations)
            self._v5_static_frame=frame_id
        field=self._v5_collision_provider.prepare_tracks(tracks,ego,local,getattr(self,'_observed_decelerations',{}),self._v5_static_tokens,frame_id=frame_id,steps=gx.shape[1],dt=RULES.dt)
        cv=self._v5_collision_provider.collision_mask_tensor(features['box_centers'].double(),features['stable_heading'].double(),field,RULES)
        collision=cv['normal_rejected']
        vbad=(torch.linalg.vector_norm(features['segment_velocity'],dim=-1)>RULES.speed_limit).any(1)
        abad=(torch.linalg.vector_norm(features['segment_acceleration'],dim=-1)>RULES.acceleration_limit).any(1)
        violations=torch.stack((collision,outside,red,vbad,abad,ttc_bad),1)
        history=self._comfort_history;previous=history[-15:-1]
        tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float64)
        v=state.dynamic_car_state.rear_axle_velocity_2d;a=state.dynamic_car_state.center_acceleration_2d
        gv=tensor([v.x,v.y]);ga=tensor([a.x,a.y])
        gp=tensor([[s.dynamic_car_state.center_acceleration_2d.x,s.dynamic_car_state.center_acceleration_2d.y] for s in previous]).reshape(-1,2)
        gh=tensor([s.rear_axle.heading-state.rear_axle.heading for s in previous])
        good,stats,flags=comfort(gx.double(),features['stable_heading'].double(),gv,ga,gp,gh,center_offset=state.car_footprint.vehicle_parameters.rear_axle_to_center)
        progress=route_progress(gx.double(),tensor(route))
        ids=getattr(self,'_last_kept_ids',np.arange(len(xy)))
        if len(ids)!=len(xy):ids=np.arange(len(xy))
        none=torch.zeros(len(xy),device='cuda',dtype=torch.bool)
        picked=choose_tensor(gx,features['stable_heading'],None,~violations.any(1)&features['finite'],none,none,~none,torch.as_tensor(ids,device='cuda'),gv,ga,gp,gh,costs=(good,stats,flags,progress))
        counts=violations.sum(1);finite=torch.isfinite(quality)&features['finite']
        least=torch.where(finite,counts,torch.iinfo(counts.dtype).max).min()
        base=torch.argmax(torch.where(finite&(counts==least),quality,-torch.inf))
        index=torch.where(picked['found'],picked['index'],base)
        allviol=torch.cat((violations,~good[:,None]),1)
        statnames=list(stats)
        summary=torch.cat((torch.stack((index,base,picked['found'],picked['valid'].sum(),(~violations.any(1)).sum(),good.sum(),good[index],quality[index],progress[index],allviol[index].sum(),cv['invalid'].sum())).double(),allviol.sum(0).double(),allviol[index].double(),flags.sum(0).double(),torch.stack([stats[k][index] for k in statnames]).double()))
        values=summary.cpu().tolist();i=int(values[0]);found=bool(values[2]);nflags=flags.shape[1]
        diagnostic=dict(selector_version='v5_gpu_integrated_legacy_boundary',selector_rules_sha256=RULES_SHA256,candidate_count=len(xy),feasible_count=int(values[3]),safety_feasible_count=int(values[4]),comfort_candidate_count=int(values[5]),safety_and_comfort_candidate_count=int(values[3]),selected_comfortable=bool(values[6]),selected_quality=values[7],selected_route_progress_m=values[8],selected_violation_count=int(values[9]),selection_fallback=not found,baseline_selected_id=int(values[1]),selected_violations=[bool(v) for v in values[18:25]],rejected_by_constraint=[int(v) for v in values[11:18]],rejected_by_comfort_component=[int(v) for v in values[25:25+nflags]],selected_comfort_proxy=dict(zip(statnames,values[25+nflags:])),constraint_names=['predicted_collision','lane_footprint_exit','current_red_entry','speed','acceleration','dtpp_ttc','candidate_comfort'],constraint_mask_sha256=None,comfort_fallback_to_original_safety_order=not found,rank_definition='legacy safety + comfort; route progress max; original anchor ID tie',red_light_scope='assigned route roadblock connectors')
        diagnostic['dtpp_ttc']=dict(old['ttc_diag'],selected_min_ttc_s=old['ttc_diag']['min_ttc_capped_s'][i],selected_rejected=bool(old['ttc_bad'][i]))
        diagnostic['gpu_tensor_entry']=dict(input_device=str(gx.device),candidate_count=len(gx),model_output_reused=reused,found=found,executable_verified=False,selection_only=True,invalid_count=int(values[10]),legacy_mask_h2d_batches=1,decision_d2h_batches=1,derived_cpu_downloads=0,remaining_legacy='road/red/direction/DTPP and final speed/path checks; predictor internal transfers not included in counters')
        self.selection_diagnostic=diagnostic
        return i
    return choose
