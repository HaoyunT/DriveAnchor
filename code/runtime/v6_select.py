"""Integrated CUDA selection with PDM as the current decision boundary.

The legacy selector remains available by explicit opt-in.  PDM receives the
GPU model shortlist and, when available, the synchronized CPU IDM proposal.
"""
import numpy as np
import torch
import os
from device_features import candidate_features
from frame_provider import FrameCollisionProvider
from frame_checks import FrameChecks, direction_tensor
from static_track_history import StaticTrackHistory
from candidate_costs_gpu import comfort, route_progress
from selector_tensor import choose_tensor
from pdm_model_selector import select_vnorm_feasible
from selection import RULES, RULES_SHA256


def make_v6_choose(legacy_factory, reference):
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
        packed=np.column_stack((old['red'],old['ttc_bad'],old['quality']))
        uploaded=torch.as_tensor(packed,device='cuda',dtype=torch.float64)
        red,predicted_ttc_bad=uploaded[:,:2].bool().unbind(1)
        quality=uploaded[:,2]
        state=self._comfort_history[-1]
        frame_id=state.time_point.time_us
        if not hasattr(self,'_v6_checks'):self._v6_checks=FrameChecks()
        self._v6_checks.begin_frame(frame_id)
        outside=self._v6_checks.road_outside(features,lanes,ego,RULES)
        direction_bad,direction_meta=direction_tensor(features,xy,self._direction_context)
        # DTPP's TTC estimate is a soft ranking signal.  The shared GPU
        # collision/temporal checks below are the hard safety boundary.  The
        # predictor is intentionally conservative for crossing turns and can
        # reject every forward-moving candidate even when the actual rollout
        # remains collision-free; keeping it in the hard mask makes the
        # selector return a stationary fallback and fail progress.
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
        cv=self._v6_checks.collision(features,field,RULES)
        self._shared_cv_costmap=cv['shared_costmap']
        collision=cv['normal_rejected']
        vbad=(torch.linalg.vector_norm(features['segment_velocity'],dim=-1)>RULES.speed_limit).any(1)
        abad=(torch.linalg.vector_norm(features['segment_acceleration'],dim=-1)>RULES.acceleration_limit).any(1)
        # Do not hard-reject on DTPP TTC here.  Preserve it in diagnostics and
        # let final_cv_collision/final_temporal_risk plus the downstream speed
        # cap handle the selected trajectory.  Direction remains hard.
        violations=torch.stack((collision,outside,red,vbad,abad,direction_bad),1)
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
        anchor_ids=torch.as_tensor(ids,device='cuda')
        pdm_mode=os.environ.get('DRIVEANCHOR_SELECTOR_MODE','pdm_model')=='pdm_model'
        pdm=None
        if pdm_mode:
            vnorm=getattr(self,'_candidate_vnorm_cuda',None)
            if vnorm is None or len(vnorm)!=len(xy):
                vnorm=torch.linalg.vector_norm(gx.flatten(1),dim=1)
            # The production contract is deliberately simple and fully GPU
            # resident: hard feasibility first, then the smallest FM2 vnorm.
            # TTC remains available in diagnostics/final safety handling, but
            # it is not a selector score and cannot reorder candidates.
            pdm=select_vnorm_feasible(hard_violation=violations.any(1),
                                       vnorm=vnorm,anchor_ids=anchor_ids)
            picked=pdm
        else:
            picked=choose_tensor(gx,features['stable_heading'],None,~violations.any(1)&features['finite'],none,none,~none,anchor_ids,gv,ga,gp,gh,costs=(good,stats,flags,progress))
        counts=violations.sum(1);finite=torch.isfinite(quality)&features['finite']
        least=torch.where(finite,counts,torch.iinfo(counts.dtype).max).min()
        base=torch.argmax(torch.where(finite&(counts==least),quality,-torch.inf))
        index=torch.where(picked['found'],picked['index'],base)
        allviol=torch.cat((violations,~good[:,None]),1)
        statnames=list(stats)
        summary=torch.cat((torch.stack((index,base,picked['found'],picked['valid'].sum(),(~violations.any(1)).sum(),good.sum(),good[index],quality[index],progress[index],allviol[index].sum(),cv['invalid'].sum())).double(),allviol.sum(0).double(),allviol[index].double(),flags.sum(0).double(),torch.stack([stats[k][index] for k in statnames]).double()))
        summary_base=summary.numel()
        summary=torch.cat((summary,torch.stack((direction_bad.sum(),cv['map_candidate_hits'],cv['map_candidate_clear'],cv['map_invalid_queries'])).double()))
        values=summary.cpu().tolist();i=int(values[0]);found=bool(values[2]);nflags=flags.shape[1]
        diagnostic=dict(selector_version=('v8_pdm_vnorm_feasible_gpu' if pdm_mode else 'v6_gpu_road_direction_shared_cv'),selector_rules_sha256=RULES_SHA256,candidate_count=len(xy),feasible_count=int(values[3]),safety_feasible_count=int(values[4]),comfort_candidate_count=int(values[5]),safety_and_comfort_candidate_count=int(values[3]),selected_comfortable=bool(values[6]),selected_quality=values[7],selected_route_progress_m=values[8],selected_violation_count=int(values[9]),selection_fallback=not found,baseline_selected_id=int(values[1]),selected_violations=[bool(v) for v in values[18:25]],rejected_by_constraint=[int(v) for v in values[11:18]],rejected_by_comfort_component=[int(v) for v in values[25:25+nflags]],selected_comfort_proxy=dict(zip(statnames,values[25+nflags:summary_base])),constraint_names=['predicted_collision','lane_footprint_exit','current_red_entry','speed','acceleration','direction','candidate_comfort'],constraint_mask_sha256=None,comfort_fallback_to_original_safety_order=not found,rank_definition=('PDM hard feasibility; FM2 vnorm ascending; anchor-ID ties; TTC not used for ranking' if pdm_mode else 'legacy safety + comfort; route progress max; original anchor ID tie; DTPP TTC soft'),red_light_scope='assigned route roadblock connectors')
        if pdm_mode:
            diagnostic['pdm_model_terms']=dict(ranking='feasible_then_fm2_vnorm_ascending',selected_vnorm=float(vnorm[i].item()),ttc_used=False,progress_used=False,comfort_used=False,anchor_id_tie_break=True,probability_calibrated=False)
        diagnostic['dtpp_ttc']=dict(old['ttc_diag'],direction_rejected_count=int(values[summary_base]),selected_min_ttc_s=old['ttc_diag']['min_ttc_capped_s'][i],selected_rejected=bool(predicted_ttc_bad[i]),hard_filter=False)
        diagnostic['gpu_map']=dict(road='GPU inclusive world-space four corners',direction='GPU nearest baseline vertex; sparse reference tie recovery',**direction_meta)
        diagnostic['shared_costmap']=dict(layer_dt_s=.5,resolution_m=1.,layers=cv['map_layers'],frame_builds=cv['map_builds'],exact_candidates=int(values[summary_base+1]),clear_candidates=int(values[summary_base+2]),unsupported_queries=int(values[summary_base+3]),scope='CV broad phase; hits refined with unchanged strict SAT; not learned TTC or final time reserve')
        diagnostic['gpu_tensor_entry']=dict(input_device=str(gx.device),candidate_count=len(gx),model_output_reused=reused,found=found,executable_verified=False,selection_only=True,invalid_count=int(values[10]),legacy_mask_h2d_batches=1,decision_d2h_batches=1,derived_cpu_downloads=0,remaining_legacy='quality/red/DTPP and final speed/path checks; direction has sparse tie recovery; map preparation on CPU; predictor internal transfers excluded')
        self.selection_diagnostic=diagnostic
        return i
    return choose
