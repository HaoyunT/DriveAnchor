from concurrent.futures import ThreadPoolExecutor

from geometry_horizon import extend as extend_geometry_seed
from static_speed_envelope import build as build_static_speed_envelope
from static_track_history import StaticTrackHistory
from static_geometry import build as static_geometry
from final_speed_cap import apply as apply_final_speed_cap
from risk_reduction import order as risk_reduction_order, only_ttc_violation
from clearance_profiles import candidates as clearance_candidates
from cv_temporal_clearance import conflict_mask as temporal_conflict_mask
from curvature_check import evaluate as geometry_evaluate
from map_speed_limit import limits as map_limits
from corridor_union import road_union
from shapely.affinity import affine_transform
from three_circle_boundary import ThreeCircleBoundary
from observed_braking import estimate as observed_braking
from stopped_lead_wall import stop_station
from brake_profiles import candidates as brake_candidates
from direction_guard import prepare as direction_prepare
from frame_checks import bad_direction_numpy as direction_bad
from wait_resume import candidates as timing_candidates
from region_paths import generate as region_paths
from tracker_forecast import risk as tracker_forecast
from tracker_control import estimate as tracker_estimate
from signal_scope import controls_route
from reference_path import build as reference_path
from speed_dp import search as speed_search
from comfort_progress import evaluate_cached
from prediction8 import build as predict8
from point_road import attach as attach_point_road
from predicted_obstacles import build as predicted_obstacles
from route_sl_dp import solve, Reference
from retime import retime
from comfort_progress import route_progress
from repair_planner import *
from accelerated_predictor import Predictor
from choose_gpu import make_fast_choose
from driveanchor_planner import DriveAnchorPlanner
import fast_features
from indexed_features import IndexedFeaturesPlanner
from v6_select import make_v6_choose
from pdm_model_selector import vnorm_topk
from idm_fallback import IDMPlannerFallback
import os
_CHOOSE=make_v6_choose(make_fast_choose,DriveAnchorPlanner.choose)
# Reuse the selector's exact dimensions and coordinate transform.
import inspect
from collision_gpu import collision_mask
from device_obstacles import (temporal_reserve_masks_tensor,
                               temporal_clearance_tensor,
                               collision_masks_tensor)
from device_features import candidate_features
import torch
from selection import RULES as _COLLISION_RULES
_COLLISION_LOCAL=DriveAnchorPlanner.choose.__globals__['local']
def final_cv_collision(xy,heading,tracks,ego,decelerations):
    centers=xy+_COLLISION_RULES.center_offset*np.stack([np.cos(heading),np.sin(heading)],-1)
    return bool(collision_mask(centers[None],heading[None],tracks,ego,_COLLISION_LOCAL,_COLLISION_RULES,decelerations)[0])

def _final_cv_collision(planner, xy, heading, tracks, ego, field):
    """Run the already-uploaded strict CV SAT field without host re-upload."""
    if field is not None:
        try:
            device_xy=(xy.to(device='cuda',dtype=torch.float64)
                       if torch.is_tensor(xy) else
                       torch.as_tensor(np.asarray(xy,float),device='cuda',dtype=torch.float64))
            device_heading=(heading.to(device='cuda',dtype=torch.float64)
                            if torch.is_tensor(heading) else
                            torch.as_tensor(np.asarray(heading,float),device='cuda',dtype=torch.float64))
            forward=torch.stack((device_heading.cos(),device_heading.sin()),-1)
            centers=device_xy+_COLLISION_RULES.center_offset*forward
            out=collision_masks_tensor(centers,device_heading,field,
                                       half_length=_COLLISION_RULES.half_length,
                                       half_width=_COLLISION_RULES.half_width)
            return bool(out['normal_rejected'].any().item())
        except (RuntimeError,ValueError):
            pass
    return final_cv_collision(np.asarray(xy),np.asarray(heading),tracks,ego,
                              getattr(planner,'_observed_decelerations',{}))

def final_temporal_actors(tracks,ego):
    actors=[]
    for obj in tracks:
        position=_COLLISION_LOCAL([[obj.center.x,obj.center.y]],ego)[0]
        observed_velocity=getattr(obj,'velocity',None)
        velocity=_COLLISION_LOCAL([[ego.x+observed_velocity.x,ego.y+observed_velocity.y]],ego)[0] if observed_velocity is not None else np.zeros(2)
        actors.append(dict(position=position,velocity=velocity,heading=obj.center.heading-ego.heading,length=obj.box.length,width=obj.box.width))
    return actors

def final_temporal_risk(batch,actors):
    # After final cap, sample zero is t0; remaining samples are every 0.1 s.
    heading=selection.stable_heading(batch)
    return temporal_conflict_mask(batch,heading,actors,
        ego_length=2*_COLLISION_RULES.half_length,
        ego_width=2*_COLLISION_RULES.half_width,
        # Reserve a full second around each sampled ego pose so a crossing
        # actor is detected before the vehicle reaches the conflict point.
        # The previous 250 ms window allowed a trajectory to become
        # unrecoverable between control cycles.
        center_offset=_COLLISION_RULES.center_offset,dt=.1,reserve_s=1.0).any(axis=1)

def temporal_clearance_score(batch,actors):
    """Rank risky profiles by their minimum swept-volume clearance."""
    xy=np.asarray(batch,float)
    if xy.ndim==2:xy=xy[None]
    if not actors:return np.full(len(xy),np.inf)
    heading=selection.stable_heading(xy)
    f=np.stack([np.cos(heading),np.sin(heading)],-1)
    center=xy+_COLLISION_RULES.center_offset*f
    times=np.arange(xy.shape[1],dtype=float)*.1
    best=np.full(len(xy),np.inf)
    ego_radius=float(np.hypot(_COLLISION_RULES.half_length,_COLLISION_RULES.half_width))
    for actor in actors:
        pos=np.asarray(actor['position'],float)
        vel=np.asarray(actor['velocity'],float)
        actor_center=pos[None]+times[:,None]*vel[None]
        actor_radius=float(np.hypot(actor['length'],actor['width'])*.5)
        margin=np.linalg.norm(center-actor_center[None],axis=-1).min(axis=1)-(ego_radius+actor_radius)
        best=np.minimum(best,margin)
    return best

def _temporal_field(planner, tracks, ego, steps):
    """Reuse the frame actor upload for GPU reserve/ranking checks."""
    provider = getattr(planner, '_v5_collision_provider', None)
    if provider is None:
        return None
    try:
        state = planner._comfort_history[-1]
        return provider.prepare_tracks(
            tracks, ego, _COLLISION_LOCAL,
            getattr(planner, '_observed_decelerations', {}),
            getattr(planner, '_v5_static_tokens', ()),
            frame_id=state.time_point.time_us, steps=int(steps), dt=.1)
    except (AttributeError, TypeError, ValueError):
        # This is an optimization cache only.  The reference NumPy path below
        # remains the correctness fallback when a legacy runner lacks the
        # provider or gives an incomplete frame object.
        return None

def _temporal_reserve(planner, batch, headings, field, actors):
    """GPU continuous reserve mask with a NumPy fallback for old runners."""
    if field is not None:
        try:
            device_batch=(batch.to(device='cuda',dtype=torch.float64)
                          if torch.is_tensor(batch) else
                          torch.as_tensor(np.asarray(batch,float),device='cuda',dtype=torch.float64))
            device_heading=(headings.to(device='cuda',dtype=torch.float64)
                            if torch.is_tensor(headings) else
                            torch.as_tensor(np.asarray(headings,float),device='cuda',dtype=torch.float64))
            return temporal_reserve_masks_tensor(
                device_batch,device_heading,field,
                half_length=_COLLISION_RULES.half_length,
                half_width=_COLLISION_RULES.half_width,
                center_offset=_COLLISION_RULES.center_offset,
                reserve_s=1.0,dt=.1).any(1).cpu().numpy()
        except (RuntimeError,ValueError):
            pass
    return final_temporal_risk(np.asarray(batch), actors)

def _temporal_clearance(planner, batch, field, actors):
    """GPU clearance proxy for large-pool ranking; no policy change."""
    if field is not None:
        try:
            device_batch=(batch.to(device='cuda',dtype=torch.float64)
                          if torch.is_tensor(batch) else
                          torch.as_tensor(np.asarray(batch,float),device='cuda',dtype=torch.float64))
            headings=candidate_features(device_batch)['stable_heading']
            return temporal_clearance_tensor(
                device_batch,headings,field,
                half_length=_COLLISION_RULES.half_length,
                half_width=_COLLISION_RULES.half_width,
                center_offset=_COLLISION_RULES.center_offset,
                dt=.1).cpu().numpy()
        except (RuntimeError,ValueError):
            pass
    return temporal_clearance_score(np.asarray(batch), actors)

def route_centerline_profile(route,speed,count=81,dt=.1):
    """Time-parameterize the route centerline for a moving fallback."""
    r=np.asarray(route,float)
    if r.ndim!=2 or len(r)<2:return None
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(r,axis=0),axis=1))]
    # Keep a modest crawl floor so a left/right turn can clear a crossing
    # point, while the final speed cap enforces the actual vehicle envelope.
    # Entering a turn at walking speed can leave the ego in the crossing
    # vehicle's path.  Use a cautious but moving 3 m/s profile; the final
    # speed cap still honors acceleration and map speed limits.
    v=max(float(speed),3.0)
    stations=np.minimum(v*np.arange(count)*dt,arc[-1])
    out=np.column_stack([np.interp(stations,arc,r[:,j]) for j in range(2)])
    out[0]=0.
    return out

@torch.no_grad()
def generate_full_pool(model,corridor,context,valid,fm_steps=2,chunk_size=500):
    assert fm_steps==2 and len(model.anchors)==3000 and not any(x.training for x in model.modules())
    anchors=model.anchors
    feature=torch.as_tensor(corridor.feature,device=anchors.device,dtype=anchors.dtype)
    ef=anchors+model.ef_head(anchors,feature,context,valid)
    ef2=ef+model.ef_head(ef,feature,context,valid)
    paths={'anchor':anchors,'EF':ef,'EFx2':ef2};state=model.normalize(ef2)
    for step in (1,2):
        state=torch.cat([model.raw(x,context,valid) for x in state.split(chunk_size)])
        paths['FM'+str(step)]=model.denormalize(state)
    assert all(x.shape==(3000,40,2) and torch.isfinite(x).all() for x in paths.values())
    return paths

def generate_fm_only(model,corridor,context,valid,fm_steps,chunk_size):
    assert fm_steps==2 and not model.training
    state=model.normalize(model.anchors)
    paths={'anchor':model.anchors}
    for step in (1,2):
        state=torch.cat([model.raw(part,context,valid) for part in state.split(chunk_size)])
        paths['FM'+str(step)]=model.denormalize(state)
    assert all(x.shape==(3000,40,2) and torch.isfinite(x).all() for x in paths.values())
    return paths

def forbid_ef(*args):raise AssertionError('EF invoked in no-EF ablation')

def count_ef_call(module,inputs,output):
    module._probe_ef_calls+=1

class FM2Top250Planner(RepairPlanner):
    def __init__(self,*args,**kwargs):
        assert kwargs.get('use_ef') is True
        super().__init__(*args,**kwargs)
        self.model.ef_head._probe_ef_calls=0
        self.model.ef_head.register_forward_hook(count_ef_call)
        assert self.fm_steps==2
        fast_features.install()
        self.predictor=Predictor()
        self._risk_cache={}
        self._idm_fallback=IDMPlannerFallback()
        # IDM is CPU-bound while FM kernels run on CUDA.  A single persistent
        # worker overlaps the two computations without allowing concurrent IDM
        # calls on the stateful planner instance.
        self._idm_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='driveanchor-idm')
        self._idm_future=None
        self._idm_candidate=None
    def initialize(self,initialization):
        super().initialize(initialization);self._dtpp_init=initialization;self._risk_cache={}
        self._idm_fallback.initialize(initialization)
        import spatial_index
        spatial_index.install(self.map)
    features=IndexedFeaturesPlanner.features
    def name(self):
        topk=os.environ.get('DRIVEANCHOR_VNORM_TOPK','20')
        mode=os.environ.get('DRIVEANCHOR_SELECTOR_MODE','pdm_model')
        return super().name()+f'_EF2_FM2_vnorm{topk}_{mode}'
    def choose(self,xy,route,tracks,ego,speed,lights,lanes):
        return _CHOOSE(self,xy,route,tracks,ego,speed,lights,lanes)
    def dtpp_prediction_only(self,xy):
        # No road-direction result in this cache. Preserve n==1 conditioning.
        if not hasattr(self,'_prediction_only_cache'):self._prediction_only_cache={}
        key=(xy.shape,xy.dtype.str,xy.strides[0]==0,xy.tobytes())
        if key not in self._prediction_only_cache:
            if xy.strides[0]==0 and len(xy)>1:
                bad,d=self.predictor.evaluate(xy[:1]);bad=np.repeat(bad,len(xy))
                d=dict(d,min_ttc_capped_s=d['min_ttc_capped_s']*len(xy),predicted_collision=d['predicted_collision']*len(xy),rejected_count=int(bad.sum()))
            else:bad,d=self.predictor.evaluate(xy)
            self._prediction_only_cache[key]=(bad,d)
        return self._prediction_only_cache[key]
    def dtpp_evaluate(self,xy):
        key=xy.tobytes()
        if key not in self._risk_cache:
            if xy.strides[0]==0:
                bad,d=self.predictor.evaluate(xy[:1]);bad=np.repeat(bad,len(xy))
                d=dict(d,min_ttc_capped_s=d['min_ttc_capped_s']*len(xy),predicted_collision=d['predicted_collision']*len(xy),rejected_count=int(bad.sum()))
            else:bad,d=self.predictor.evaluate(xy)
            wrong=direction_bad(xy,self._direction_context)
            bad=bad|wrong
            d=dict(d,direction_rejected_count=int(wrong.sum()))
            self._risk_cache[key]=(bad,d)
        return self._risk_cache[key]
    @torch.no_grad()
    def compute_planner_trajectory(self,current_input):
        begin=time.perf_counter();self._prediction_only_cache={};self._risk_cache={};self._invariant_cache={};self._score_cache={};self._comfort_progress_cache={}
        self._comfort_history=list(current_input.history.ego_states)
        selector_mode=os.environ.get('DRIVEANCHOR_SELECTOR_MODE','pdm_model')
        # Start IDM before feature preparation and FM generation.  The future
        # is consumed immediately before selection, so PDM sees a synchronized
        # model/IDM candidate set while GPU and CPU work overlap.
        self._idm_future=None
        self._idm_candidate=None
        if selector_mode=='pdm_model' and self._idm_fallback.available:
            self._idm_future=self._idm_executor.submit(
                self._idm_fallback.compute_candidate,current_input,count=40,dt=.1)
        import spatial_index
        spatial_index.update_roi(self.map,current_input.history.ego_states[-1].rear_axle)
        self.predictor.prepare(current_input,self._dtpp_init)
        history=current_input.history
        self._observed_decelerations=observed_braking(history)
        self.predictor.observed_decelerations=self._observed_decelerations
        state=history.ego_states[-1];ego=state.rear_axle
        features,mask,lanes,tracks=self.features(history)
        self._direction_context=direction_prepare(lanes,ego,state.car_footprint.vehicle_parameters.rear_axle_to_center)
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
        before=self.model.ef_head._probe_ef_calls
        paths=generate_full_pool(self.model,corridor,context,valid,self.fm_steps,self.chunk_size)
        assert self.model.ef_head._probe_ef_calls==before+2
        assert self.fm_steps==2
        stage='FM2'
        norm=torch.linalg.vector_norm(paths['FM2'].flatten(1),dim=1)
        topk=int(os.environ.get('DRIVEANCHOR_VNORM_TOPK','20'))
        if topk < 1 or topk > 3000:
            raise ValueError('DRIVEANCHOR_VNORM_TOPK must be in [1,3000]')
        kept=torch.arange(3000,device=norm.device) if topk==3000 else vnorm_topk(norm,topk)
        ids=kept.cpu().numpy()
        self._candidate_xy_cuda=paths[stage][kept]
        self._candidate_vnorm_cuda=norm[kept]
        xy=self._candidate_xy_cuda.cpu().numpy()
        idm_trajectory=None
        idm_xy=None
        idm_error=None
        if self._idm_future is not None:
            try:
                idm_trajectory,idm_xy=self._idm_future.result()
                # PDM ranks learned and IDM proposals together.  The IDM path
                # is appended only after the model's CUDA vnorm shortlist; it
                # does not increase FM inference or the 3000-anchor budget.
                if idm_xy is not None and len(xy) < 3000:
                    xy=np.concatenate((xy,np.asarray(idm_xy,float)[None]),axis=0)
                    # Keep the selector's reusable CUDA buffer aligned with
                    # the appended CPU proposal; otherwise the cache key would
                    # hit while ``gx`` still contained only the model paths.
                    self._candidate_xy_cuda=torch.cat((
                        self._candidate_xy_cuda,
                        torch.as_tensor(np.asarray(idm_xy,float)[None],
                                        device=self._candidate_xy_cuda.device,
                                        dtype=self._candidate_xy_cuda.dtype),
                    ),dim=0)
                    # Give IDM a neutral model-prior rank.  It has no FM
                    # vnorm, so median shortlist norm prevents the proxy term
                    # from favoring or suppressing it by construction.
                    idm_vnorm=self._candidate_vnorm_cuda.median().reshape(1)
                    self._candidate_vnorm_cuda=torch.cat((self._candidate_vnorm_cuda,idm_vnorm))
                    self._idm_candidate=dict(index=len(xy)-1,anchor_id=3000,trajectory=idm_trajectory,xy=np.asarray(idm_xy,float))
            except Exception as exc:
                idm_error=f'{type(exc).__name__}: {exc}'
        self._candidate_xy_key=xy.tobytes()
        self._gpu_tensor_costs={}
        if idm_xy is None:
            self._last_kept_ids=ids
        else:
            # ``ids`` is the model-only shortlist; the appended candidate was
            # already assigned anchor id 3000 above.
            self._last_kept_ids=np.concatenate((ids,np.asarray([3000],dtype=np.int64)))
        candidate_ids=np.asarray(self._last_kept_ids,dtype=np.int64)
        self._last_norms=norm.detach().cpu().numpy()
        tick=time.perf_counter()
        selected,index,diagnostic,guard=self._selection(xy,route,tracks,state,list(current_input.traffic_light_data or []),lanes)
        idm_selected=bool(idm_xy is not None and index is not None and int(index)==len(xy)-1)
        diagnostic['pdm_candidate_sources']=['model']*int(len(xy)-(1 if idm_xy is not None else 0))+(['IDMPlanner'] if idm_xy is not None else [])
        diagnostic['pdm_idm_candidate_available']=bool(idm_xy is not None)
        diagnostic['pdm_idm_selected']=idm_selected
        if idm_selected:
            guard['selection_source']='pdm_idm'
            diagnostic['selected_source']='IDMPlanner'
        elif idm_error is not None:
            diagnostic['pdm_idm_error']=idm_error
        self._idm_candidate = self._idm_candidate if idm_xy is not None else None
        idm_fallback_used=idm_selected
        raw_selected=selected.copy()
        pp_start=time.perf_counter()
        sl_diag=dict(attempted=False,accepted=False,version='dp_region_v48',planning_horizon_s=8.,execution_changed=False,point_search=True,full_body_recheck=True)
        committed_geometry=None
        # Geometric choice only: no speed/comfort/TTC ranking in this DP.
        vp=state.car_footprint.vehicle_parameters
        c,s=np.cos(ego.heading),np.sin(ego.heading)
        local_transform=[c,s,-s,c,-c*ego.x-s*ego.y,s*ego.x-c*ego.y]
        region=road_union([affine_transform(lane.polygon,local_transform) for lane in lanes])
        if not hasattr(self,'_static_track_history'):
            self._static_track_history=StaticTrackHistory()
        static_obstacles,static_tokens=static_geometry(tracks,ego,state.time_point.time_s,self._static_track_history)
        sl_diag['static_obstacle_tokens']=static_tokens
        static_speed_envelope=build_static_speed_envelope(tracks,static_tokens,ego,vp)
        sl_diag['static_ttc_envelope_s']=1.6 if static_speed_envelope is not None else None
        free_region=region if static_obstacles is None else region.difference(static_obstacles)
        body=ThreeCircleBoundary(free_region,vp.length,vp.width,vp.rear_axle_to_center)
        physical_curvature=np.tan(np.pi/3)/vp.wheel_base
        geometric_ok=body.feasible(selected,selection.stable_heading(selected[None])[0]).all()
        geometric_ok=bool(geometric_ok and geometry_evaluate(selected,physical_curvature)[0])
        if not geometric_ok:
            curvature=np.tan(state.tire_steering_angle)/vp.wheel_base
            newpaths,info=region_paths(route,lanes,self.route_ids,ego,state.dynamic_car_state.speed,curvature,vp.width/2,tier=0,vehicle_length=vp.length,rear_to_center=vp.rear_axle_to_center,max_curvature=physical_curvature,static_obstacles=static_obstacles)
            attempts=[info]
            if not newpaths:
                newpaths,info=region_paths(route,lanes,self.route_ids,ego,state.dynamic_car_state.speed,curvature,vp.width/2,tier=1,vehicle_length=vp.length,rear_to_center=vp.rear_axle_to_center,max_curvature=physical_curvature,static_obstacles=static_obstacles)
                attempts.append(info)
            sl_diag.update(attempted=True,region_search=attempts,region_paths_xy=[p.tolist() for p in newpaths])
            if newpaths:
                path=newpaths[0]  # Already ranked by geometric lattice cost.
                committed_geometry=path
                arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
                desired=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(selected,axis=0),axis=1))]
                selected=np.column_stack([np.interp(desired,arc,path[:,j]) for j in [0,1]])
                index=None
                sl_diag.update(accepted=True,execution_changed=True,reason='geometric_path_repair',geometry_path_xy=path.tolist())
        final_direction_bad=bool(direction_bad(selected[None],self._direction_context)[0])
        sl_diag['final_direction_bad']=final_direction_bad
        if final_direction_bad:
            seed,extension=extend_geometry_seed(route,selected,committed_geometry)
            timings=timing_candidates(seed if seed is not None else selected,state.dynamic_car_state.speed,state.dynamic_car_state.rear_axle_acceleration_2d.x)
            safe_timings=[r for r in timings if not direction_bad(r['xy'][None],self._direction_context)[0]]
            sl_diag['direction_timing_candidates']=len(safe_timings)
            safe_timings.sort(key=lambda r:-r['station'][-1])
            for option in safe_timings:
                alternative=option['xy'][:40]
                checked,cid,cd,cg=self._selection(np.broadcast_to(alternative,(500,40,2)),route,tracks,state,list(current_input.traffic_light_data or []),lanes)
                if cid is not None and cd.get('selected_violation_count',99)==0 and cd.get('selected_comfortable',False) and np.allclose(checked,alternative,rtol=0,atol=1e-6):
                    selected=alternative;diagnostic=cd;guard=cg;index=None
                    sl_diag.update(direction_timing_adopted=True,direction_timing_stop_s=option['stop_s'],direction_timing_wait_s=option['wait_s'])
                    break
        _,current_risk=self.predictor.evaluate(selected[None])
        current_ttc=float(current_risk['min_ttc_capped_s'][0])
        sl_diag['speed_shield_initial_ttc']=current_ttc
        if current_ttc<1.5:
            seed,extension=extend_geometry_seed(route,selected,committed_geometry)
            options=brake_candidates(seed if seed is not None else selected,state.dynamic_car_state.speed,state.dynamic_car_state.rear_axle_acceleration_2d.x)
            options=[o for o in options if not direction_bad(o['xy'][None],self._direction_context)[0]]
            sl_diag['speed_shield_candidates']=len(options)
            if options:
                batch=np.asarray([o['xy'] for o in options])
                risks,rd=self.predictor.evaluate(batch)
                # Prefer the least lost progress among fully checked profiles
                # retaining an anticipatory 1.1-second TTC margin.
                progress=route_progress(batch,route)
                for q in np.argsort(-progress,kind='stable'):
                    if risks[q] or rd['min_ttc_capped_s'][q]<1.1:continue
                    option=options[q];alternative=option['xy'][:40]
                    checked,cid,cd,cg=self._selection(np.broadcast_to(alternative,(500,40,2)),route,tracks,state,list(current_input.traffic_light_data or []),lanes)
                    if cid is None or cd.get('selected_violation_count',99)>0 or not cd.get('selected_comfortable',False) or not np.allclose(checked,alternative,rtol=0,atol=1e-6):continue
                    selected=alternative;diagnostic=cd;guard=cg;index=None
                    sl_diag.update(speed_shield_adopted=True,speed_shield_target=option['target'],speed_shield_duration=option['duration'],speed_shield_ttc=rd['min_ttc_capped_s'][q])
                    break
        seed,extension=extend_geometry_seed(route,selected,committed_geometry)
        wall=stop_station(seed if seed is not None else selected,tracks,ego,state.car_footprint.vehicle_parameters)
        sl_diag['stopped_lead_wall']=wall
        if wall is not None and np.linalg.norm(np.diff(selected,axis=0),axis=1).sum()>max(0.,wall['station']):
            path=seed if seed is not None else selected
            options=brake_candidates(path,state.dynamic_car_state.speed,state.dynamic_car_state.rear_axle_acceleration_2d.x)
            options+=timing_candidates(path,state.dynamic_car_state.speed,state.dynamic_car_state.rear_axle_acceleration_2d.x)
            options=[o for o in options if np.linalg.norm(np.diff(o['xy'],axis=0),axis=1).sum()<=max(0.,wall['station'])+1e-6 and not direction_bad(o['xy'][None],self._direction_context)[0]]
            sl_diag['stopped_lead_candidates']=len(options)
            if options:
                batch=np.asarray([o['xy'] for o in options]);risk,rd=self.predictor.evaluate(batch)
                progress=route_progress(batch,route)
                for q in np.argsort(-progress,kind='stable'):
                    if risk[q]:continue
                    alternative=batch[q,:40]
                    checked,cid,cd,cg=self._selection(np.broadcast_to(alternative,(500,40,2)),route,tracks,state,list(current_input.traffic_light_data or []),lanes)
                    if cid is None or cd.get('selected_violation_count',99)>0 or not cd.get('selected_comfortable',False) or not np.allclose(checked,alternative,rtol=0,atol=1e-6):continue
                    selected=alternative;diagnostic=cd;guard=cg;index=None
                    sl_diag.update(stopped_lead_adopted=True,stopped_lead_planned_station=float(progress[q]))
                    break
        # Final road shield: brake along the current geometry before exiting.
        vp=state.car_footprint.vehicle_parameters
        c,s=np.cos(ego.heading),np.sin(ego.heading)
        local_transform=[c,s,-s,c,-c*ego.x-s*ego.y,s*ego.x-c*ego.y]
        safe_region=road_union([affine_transform(lane.polygon,local_transform) for lane in lanes])
        final_free_region=safe_region if static_obstacles is None else safe_region.difference(static_obstacles)
        final_body=ThreeCircleBoundary(final_free_region,vp.length,vp.width,vp.rear_axle_to_center)
        road_bad=not final_body.feasible(selected,selection.stable_heading(selected[None])[0]).all()
        sl_diag['final_road_bad']=bool(road_bad)
        if road_bad:
            seed,extension=extend_geometry_seed(route,selected,committed_geometry)
            path=seed if seed is not None else selected
            options=brake_candidates(path,state.dynamic_car_state.speed,state.dynamic_car_state.rear_axle_acceleration_2d.x)
            options+=timing_candidates(path,state.dynamic_car_state.speed,state.dynamic_car_state.rear_axle_acceleration_2d.x)
            options=[o for o in options if final_body.feasible(o['xy'],selection.stable_heading(o['xy'][None])[0]).all() and not direction_bad(o['xy'][None],self._direction_context)[0]]
            sl_diag['road_shield_candidates']=len(options)
            if options:
                batch=np.asarray([o['xy'] for o in options]);risk,rd=self.predictor.evaluate(batch)
                progress=route_progress(batch,route)
                for q in np.argsort(-progress,kind='stable'):
                    if risk[q]:continue
                    alternative=batch[q,:40]
                    checked,cid,cd,cg=self._selection(np.broadcast_to(alternative,(500,40,2)),route,tracks,state,list(current_input.traffic_light_data or []),lanes)
                    if cid is None or cd.get('selected_violation_count',99)>0 or not cd.get('selected_comfortable',False) or not np.allclose(checked,alternative,rtol=0,atol=1e-6):continue
                    selected=alternative;diagnostic=cd;guard=cg;index=None
                    sl_diag.update(road_shield_adopted=True,road_shield_station=float(progress[q]))
                    break
        # Preserve a checked plan across frames instead of reverting to an
        # unsafe model proposal when the new optimization has no solution.
        now=state.time_point.time_s
        heading=selection.stable_heading(selected[None])[0]
        outgoing_safe=bool(final_body.feasible(selected,heading).all())
        prior=getattr(self,'_last_road_safe_plan',None)
        if not outgoing_safe and prior is not None:
            age=now-prior['time']
            if 0.<age<1.:
                query=np.arange(40)*.1+age
                world=np.column_stack([np.interp(query,prior['times'],prior['world'][:,j]) for j in [0,1]])
                # Extend the tiny uncovered tail at terminal velocity; holding
                # the endpoint would manufacture a one-frame emergency stop.
                tail_velocity=(prior['world'][-1]-prior['world'][-2])/(prior['times'][-1]-prior['times'][-2])
                world+=np.maximum(query-prior['times'][-1],0.)[:,None]*tail_velocity
                alternative=(world-[ego.x,ego.y])@np.array([[c,-s],[s,c]])
                alternative[0]=0.
                heading=selection.stable_heading(alternative[None])[0]
                if final_body.feasible(alternative,heading).all() and not direction_bad(alternative[None],self._direction_context)[0]:
                    risk,_=self.predictor.evaluate(alternative[None])
                    sl_diag['previous_safe_plan_risk']=bool(risk[0])
                    if not risk[0]:
                        checked,cid,cd,cg=self._selection(np.broadcast_to(alternative,(500,40,2)),route,tracks,state,list(current_input.traffic_light_data or []),lanes)
                        sl_diag['previous_safe_plan_check']=dict(violations=cd.get('selected_violations'),comfortable=cd.get('selected_comfortable'),cid=cid)
                        if cid is not None and cd.get('selected_violation_count',99)==0 and cd.get('selected_comfortable',False) and np.allclose(checked,alternative,rtol=0,atol=1e-6):
                            selected=alternative;diagnostic=cd;guard=cg;index=None
                            sl_diag['previous_safe_plan_adopted']=True
                            outgoing_safe=True
        sl_diag['final_road_unresolved']=not outgoing_safe
        if outgoing_safe:
            self._last_road_safe_plan=dict(time=now,times=np.arange(len(selected))*.1,world=selected@np.array([[c,s],[-s,c]])+[ego.x,ego.y])
        # One 8 s physical rollout is checked; its exact first 3.9 s is executed.
        # Do not evaluate an unrelated constant-speed tail or recap the prefix.
        seed,extension=extend_geometry_seed(route,selected,committed_geometry)
        speed_path=seed if seed is not None else selected
        final_wall=stop_station(speed_path,tracks,ego,vp)
        sl_diag['final_stopped_lead_wall']=final_wall
        sl_diag['final_geometry_extension']=extension
        full_selected,cap_info=apply_final_speed_cap(speed_path,state.dynamic_car_state.speed,state.dynamic_car_state.rear_axle_acceleration_2d.x,lambda points:map_limits(points,lanes,ego,vp.rear_axle_to_center),stop_distance=None if final_wall is None else final_wall['station'],count=81,extra_speed_envelope=static_speed_envelope)
        selected=full_selected[:40].copy();index=None
        full_heading=selection.stable_heading(full_selected[None])[0]
        final_risk,final_rd=self.predictor.evaluate(full_selected[None])
        temporal_actors=final_temporal_actors(tracks,ego)
        # Build the 8 s actor field once and keep the continuous reserve query
        # on CUDA.  The NumPy implementation remains a parity fallback for
        # legacy runners that do not install the shared provider.
        temporal_field=_temporal_field(self,tracks,ego,81)
        final_collision=_final_cv_collision(
            self,full_selected,full_heading,tracks,ego,temporal_field)
        temporal_risk=bool(_temporal_reserve(
            self,full_selected[None],full_heading,temporal_field,temporal_actors)[0])
        full_geometry_bad=not bool(final_body.feasible(full_selected,full_heading).all()) or bool(direction_bad(full_selected[None],self._direction_context)[0])
        sl_diag.update(final_collision_before_repair=final_collision,final_temporal_reserve_before_repair=temporal_risk,temporal_reserve_s=1.0,final_full_geometry_before_repair=full_geometry_bad,safety_horizon_s=8.,execution_horizon_s=3.9)
        audit=dict(count=0,risk_rejected=0,road_direction_rejected=0,
                   selection_rejected=0,adopted=False,safety_horizon_s=8.,
                   timed_candidates=0,timed_endpoint_rejected=0,
                   candidate_option_indices=[])
        # Give an interaction stop priority over a moving risk-reduction
        # profile.  Once the ego has entered a crossing conflict, choosing a
        # slower moving candidate can still carry it into the conflict point;
        # waiting at the current pose preserves the opportunity for the other
        # actor to clear on the next frame.
        # DTPP TTC is a conservative advisory signal.  A predicted TTC hit
        # alone must not freeze the ego vehicle: the shared collision kernel
        # and temporal-reserve check are the hard safety boundary, while the
        # selected trajectory is subsequently speed-capped.  This prevents
        # false-positive crossing predictions from destroying route progress.
        # A temporal-reserve hit is advisory: stopping at the current pose
        # can leave the ego vehicle in the crossing actor's swept path.  Use
        # the strict oriented-footprint collision result as the stop trigger;
        # the temporal result remains diagnostic and informs speed capping.
        interaction_stop_required=bool(final_collision)
        if interaction_stop_required:
            hold=np.repeat(np.asarray(speed_path[0:1],dtype=float),81,axis=0)
            hold_heading=selection.stable_heading(hold[None])[0]
            hold_geometry=bool(final_body.feasible(hold,hold_heading).all())
            hold_direction=bool(direction_bad(hold[None],self._direction_context)[0])
            hold_collision=_final_cv_collision(
                self,hold,hold_heading,tracks,ego,temporal_field)
            audit.update(emergency_hold_checked=True,
                         emergency_hold_geometry_ok=hold_geometry,
                         emergency_hold_direction_bad=hold_direction,
                         emergency_hold_collision=hold_collision)
            if hold_geometry and not hold_direction and not hold_collision:
                full_selected=hold
                selected=hold[:40].copy()
                full_heading=hold_heading
                cap_info=dict(reason='emergency_hold',initial_speed=0.0,
                              min_acceleration=0.0,max_acceleration=0.0)
                final_risk=np.asarray([False])
                diagnostic=dict(diagnostic,selected_emergency_hold=True)
                audit.update(adopted=True,emergency_hold_adopted=True,
                             progress=float(route_progress(hold[None],route)[0]))
        # Temporal reserve hits also enter the candidate repair pass.  Unlike
        # the emergency hold, this searches for a moving profile that clears
        # the swept path; a stop is only used when the current footprint is
        # already in strict collision.
        if final_collision or temporal_risk or full_geometry_bad:
            options=brake_candidates(speed_path,state.dynamic_car_state.speed,state.dynamic_car_state.rear_axle_acceleration_2d.x)
            options+=timing_candidates(speed_path,state.dynamic_car_state.speed,state.dynamic_car_state.rear_axle_acceleration_2d.x)
            options+=clearance_candidates(speed_path,state.dynamic_car_state.speed,state.dynamic_car_state.rear_axle_acceleration_2d.x,horizon=8.)
            capped=[];cap_records=[];option_indices=[];timed_rejected=0;timed_count=0
            for option_index,option in enumerate(options):
                timed=option if 'station' in option and 'speed' in option else None
                candidate,ci=apply_final_speed_cap(speed_path if timed is not None else option['xy'],state.dynamic_car_state.speed,state.dynamic_car_state.rear_axle_acceleration_2d.x,lambda points:map_limits(points,lanes,ego,vp.rear_axle_to_center),stop_distance=None if final_wall is None else final_wall['station'],count=81,extra_speed_envelope=static_speed_envelope,time_reference=timed)
                if timed is not None:
                    timed_count+=1
                    if ci.get('endpoint_exhausted_with_motion',False):
                        timed_rejected+=1;continue
                capped.append(candidate);cap_records.append(ci);option_indices.append(option_index)
            audit.update(count=len(capped),timed_candidates=timed_count,
                         timed_endpoint_rejected=timed_rejected,
                         candidate_option_indices=option_indices)
            if capped and not audit['adopted']:
                batch=np.asarray(capped)
                bad,rd=self.predictor.evaluate(batch)
                audit['risk_rejected']=int(bad.sum())
                reserve_bad=_temporal_reserve(
                    self,batch,selection.stable_heading(batch),temporal_field,temporal_actors)
                audit['temporal_reserve_rejected']=int(reserve_bad.sum())
                progress=route_progress(batch,route)
                for q in np.argsort(-progress,kind='stable'):
                    if bad[q] or reserve_bad[q]:continue
                    candidate=batch[q]
                    headings=selection.stable_heading(candidate[None])[0]
                    if not final_body.feasible(candidate,headings).all() or direction_bad(candidate[None],self._direction_context)[0]:
                        audit['road_direction_rejected']+=1;continue
                    if _final_cv_collision(self,candidate,headings,tracks,ego,temporal_field):
                        audit['final_cv_rejected']=audit.get('final_cv_rejected',0)+1;continue
                    # The legacy selector is fixed at 40 samples. Full-horizon
                    # road/direction/CV/DTPP checks are completed above; retain
                    # its red-light/comfort/etc. checks on the executed prefix.
                    prefix=candidate[:40]
                    checked,cid,cd,cg=self._selection(np.broadcast_to(prefix,(500,40,2)),route,tracks,state,list(current_input.traffic_light_data or []),lanes)
                    if cid is None or cd.get('selected_violation_count',99)>0 or not np.allclose(checked,prefix,rtol=0,atol=1e-6):
                        audit['selection_rejected']+=1;audit['last_rejection']=cd.get('selected_violations');continue
                    full_selected=candidate;selected=prefix.copy();cap_info=cap_records[q];diagnostic=cd;guard=cg;index=None
                    full_heading=headings;final_risk=np.array([False])
                    audit.update(adopted=True,progress=float(progress[q]),selected=int(q),selected_option_index=int(option_indices[q]));break
                # If every option is marked by the conservative temporal
                # boolean, still choose the profile with the best clearance
                # margin.  DTPP risk is advisory here; strict footprint
                # collision and road checks above remain mandatory.
                if not audit['adopted'] and temporal_risk:
                    clearance=_temporal_clearance(
                        self,batch,temporal_field,temporal_actors)
                    audit['temporal_clearance_best_m']=float(np.max(clearance))
                    for q in np.argsort(-clearance,kind='stable'):
                        # A negative margin means this profile remains inside
                        # the conservative swept-volume envelope.  Prefer a
                        # different lateral FM mode before accepting it.
                        if clearance[q] < 0.0:continue
                        candidate=batch[q]
                        headings=selection.stable_heading(candidate[None])[0]
                        if not final_body.feasible(candidate,headings).all() or direction_bad(candidate[None],self._direction_context)[0]:
                            continue
                        if _final_cv_collision(self,candidate,headings,tracks,ego,temporal_field):
                            audit['final_cv_rejected']=audit.get('final_cv_rejected',0)+1;continue
                        prefix=candidate[:40]
                        checked,cid,cd,cg=self._selection(np.broadcast_to(prefix,(500,40,2)),route,tracks,state,list(current_input.traffic_light_data or []),lanes)
                        if cid is None or not np.allclose(checked,prefix,rtol=0,atol=1e-6):
                            continue
                        full_selected=candidate;selected=prefix.copy();full_heading=headings
                        cap_info=cap_records[q];diagnostic=cd;guard=cg;index=None
                        final_risk=np.asarray([False])
                        audit.update(adopted=True,temporal_clearance_adopted=True,
                                     temporal_clearance_selected=float(clearance[q]),
                                     progress=float(progress[q]),selected=int(q),
                                     selected_option_index=int(option_indices[q]))
                        break
                # If the learned pool has no positive-clearance mode, use the
                # already constructed route centerline as a deterministic
                # moving fallback.  This supplies the missing turn geometry;
                # it is still checked by the same footprint, direction and
                # speed guards before emission.
                if not audit['adopted']:
                    route_profile=route_centerline_profile(route,state.dynamic_car_state.speed,81,.1)
                    if route_profile is not None:
                        candidate,ci=apply_final_speed_cap(route_profile,state.dynamic_car_state.speed,state.dynamic_car_state.rear_axle_acceleration_2d.x,lambda points:map_limits(points,lanes,ego,vp.rear_axle_to_center),stop_distance=None if final_wall is None else final_wall['station'],count=81,extra_speed_envelope=static_speed_envelope)
                        headings=selection.stable_heading(candidate[None])[0]
                        if final_body.feasible(candidate,headings).all() and not direction_bad(candidate[None],self._direction_context)[0] and not _final_cv_collision(self,candidate,headings,tracks,ego,temporal_field):
                            prefix=candidate[:40]
                            checked,cid,cd,cg=self._selection(np.broadcast_to(prefix,(500,40,2)),route,tracks,state,list(current_input.traffic_light_data or []),lanes)
                            if cid is not None and np.allclose(checked,prefix,rtol=0,atol=1e-6):
                                full_selected=candidate;selected=prefix.copy();full_heading=headings
                                cap_info=ci;diagnostic=cd;guard=cg;index=None;final_risk=np.asarray([False])
                                audit.update(adopted=True,route_centerline_fallback=True,
                                             progress=float(route_progress(candidate[None],route)[0]))
                # The speed/timing profiles above preserve one lateral path.
                # If that path has no clearance, search the original FM2 pool
                # for a different lateral mode before giving up.  Rank all
                # 3,000 paths on the inexpensive clearance proxy, then run
                # the full speed cap and unchanged geometry/selector checks
                # only on the best 64 candidates.
                if not audit['adopted']:
                    raw_pool=np.asarray(xy,float)
                    pool_field=_temporal_field(self,tracks,ego,raw_pool.shape[1])
                    raw_clearance=_temporal_clearance(
                        self,self._candidate_xy_cuda.detach(),pool_field,temporal_actors)
                    raw_progress=route_progress(raw_pool,route)
                    top=np.argsort(-(raw_clearance+1e-3*raw_progress),kind='stable')[:min(64,len(raw_pool))]
                    audit['pool_clearance_candidates']=int(len(top))
                    audit['pool_clearance_best_m']=float(raw_clearance[top[0]]) if len(top) else None
                    for q in top:
                        candidate,ci=apply_final_speed_cap(raw_pool[q],state.dynamic_car_state.speed,state.dynamic_car_state.rear_axle_acceleration_2d.x,lambda points:map_limits(points,lanes,ego,vp.rear_axle_to_center),stop_distance=None if final_wall is None else final_wall['station'],count=81,extra_speed_envelope=static_speed_envelope)
                        headings=selection.stable_heading(candidate[None])[0]
                        if not final_body.feasible(candidate,headings).all() or direction_bad(candidate[None],self._direction_context)[0]:continue
                        if _final_cv_collision(self,candidate,headings,tracks,ego,temporal_field):
                            audit['pool_final_cv_rejected']=audit.get('pool_final_cv_rejected',0)+1;continue
                        prefix=candidate[:40]
                        checked,cid,cd,cg=self._selection(np.broadcast_to(prefix,(500,40,2)),route,tracks,state,list(current_input.traffic_light_data or []),lanes)
                        if cid is None or not np.allclose(checked,prefix,rtol=0,atol=1e-6):continue
                        full_selected=candidate;selected=prefix.copy();full_heading=headings
                        cap_info=ci;diagnostic=cd;guard=cg;index=int(q);final_risk=np.asarray([False])
                        audit.update(adopted=True,pool_clearance_adopted=True,pool_clearance_selected=float(raw_clearance[q]),progress=float(raw_progress[q]),selected=int(q))
                        break
                if not audit['adopted']:
                    original_ttc=float(final_rd['min_ttc_capped_s'][0])
                    audit['risk_reduction_adopted']=False
                    audit['risk_reduction_original_ttc_s']=original_ttc
                    for q in risk_reduction_order(bool(final_risk[0] or final_collision or temporal_risk or full_geometry_bad),original_ttc,rd['min_ttc_capped_s'],progress):
                        if bool(rd['predicted_collision'][q]):continue
                        candidate=batch[q];headings=selection.stable_heading(candidate[None])[0]
                        if not final_body.feasible(candidate,headings).all() or direction_bad(candidate[None],self._direction_context)[0]:continue
                        if _final_cv_collision(self,candidate,headings,tracks,ego,temporal_field):continue
                        prefix=candidate[:40]
                        checked,cid,cd,cg=self._selection(np.broadcast_to(prefix,(500,40,2)),route,tracks,state,list(current_input.traffic_light_data or []),lanes)
                        # Retain every non-TTC selector constraint and refuse
                        # a hidden guard replacement or unidentified candidate.
                        if cid is None or not np.allclose(checked,prefix,rtol=0,atol=1e-6) or not only_ttc_violation(cd):continue
                        full_selected=candidate;selected=prefix.copy();full_heading=headings
                        cap_info=cap_records[q];diagnostic=cd;guard=cg;index=None
                        final_risk=np.asarray([bool(bad[q])])
                        audit.update(risk_reduction_adopted=True,risk_reduction_selected=int(q),risk_reduction_option_index=int(option_indices[q]),risk_reduction_candidate_ttc_s=float(rd['min_ttc_capped_s'][q]),risk_reduction_progress=float(progress[q]),risk_reduction_reserve_unresolved=bool(reserve_bad[q]))
                        sl_diag['risk_reduction_not_safe']=True
                        self._last_road_safe_plan=None
                        break
            # If every moving profile remains in the predicted conflict zone,
            # keep the ego vehicle at the current pose for one control horizon.
            # This is a last-resort interaction action for crossing/turning
            # scenes: it lets the dynamic actor clear instead of committing to
            # a moving profile that the predictor marks as colliding.  The
            # candidate still has to pass the final vehicle-footprint and
            # direction checks before it can be emitted.
            if not audit['adopted'] and interaction_stop_required:
                hold=np.repeat(np.asarray(speed_path[0:1],dtype=float),81,axis=0)
                hold_heading=selection.stable_heading(hold[None])[0]
                hold_geometry=bool(final_body.feasible(hold,hold_heading).all())
                hold_direction=bool(direction_bad(hold[None],self._direction_context)[0])
                hold_collision=_final_cv_collision(
                    self,hold,hold_heading,tracks,ego,temporal_field)
                audit.update(emergency_hold_checked=True,
                             emergency_hold_geometry_ok=hold_geometry,
                             emergency_hold_direction_bad=hold_direction,
                             emergency_hold_collision=hold_collision)
                if hold_geometry and not hold_direction and not hold_collision:
                    full_selected=hold
                    selected=hold[:40].copy()
                    full_heading=hold_heading
                    cap_info=dict(reason='emergency_hold',initial_speed=0.0,
                                  min_acceleration=0.0,max_acceleration=0.0)
                    final_risk=np.asarray([False])
                    diagnostic=dict(diagnostic,selected_emergency_hold=True)
                    audit.update(adopted=True,emergency_hold_adopted=True,
                                 progress=float(route_progress(hold[None],route)[0]))
        sl_diag['final_dynamic_repair']=audit
        final_collision=_final_cv_collision(
            self,full_selected,full_heading,tracks,ego,temporal_field)
        sl_diag['final_collision_unresolved']=final_collision
        sl_diag['final_temporal_reserve_unresolved']=bool(_temporal_reserve(
            self,full_selected[None],full_heading,temporal_field,temporal_actors)[0])
        sl_diag['final_speed_cap']=cap_info
        sl_diag['final_speed_cap_ttc_risk']=bool(final_risk[0])
        sl_diag['final_road_unresolved']=not bool(final_body.feasible(full_selected,full_heading).all())
        sl_diag['final_direction_unresolved']=bool(direction_bad(full_selected[None],self._direction_context)[0])
        sl_diag['full_selected_xy']=full_selected.tolist()
        assert np.array_equal(selected,full_selected[:40]),'risk/execution prefix mismatch'
        if not sl_diag.get('risk_reduction_not_safe',False) and not sl_diag['final_road_unresolved'] and not sl_diag['final_direction_unresolved'] and not final_risk[0] and not final_collision and not sl_diag['final_temporal_reserve_unresolved']:
            self._last_road_safe_plan=dict(time=state.time_point.time_s,times=np.arange(len(selected))*.1,world=selected@np.array([[c,s],[-s,c]])+[ego.x,ego.y])
        sl_diag['observed_braking_tracks']=self._observed_decelerations
        sl_diag['seconds']=time.perf_counter()-pp_start
        selector_seconds=time.perf_counter()-tick
        index=None if index is None or not 0 <= index < len(candidate_ids) else int(candidate_ids[index])
        unresolved=(bool(sl_diag['final_collision_unresolved'])
                    or bool(sl_diag['final_temporal_reserve_unresolved'])
                    or bool(sl_diag['final_road_unresolved'])
                    or bool(sl_diag['final_direction_unresolved'])
                    or bool(sl_diag['final_speed_cap_ttc_risk']))
        if unresolved and idm_trajectory is None and self._idm_fallback.available:
            try:
                idm_trajectory=self._idm_fallback.compute_planner_trajectory(current_input)
                idm_fallback_used=True
                sl_diag['idm_fallback']=dict(attempted=True,used=True,source='nuplan.IDMPlanner',
                    reason='learned_and_repair_checks_unresolved')
                diagnostic=dict(diagnostic,fallback_source='IDMPlanner',fallback_used=True)
            except Exception as exc:
                sl_diag['idm_fallback']=dict(attempted=True,used=False,source='nuplan.IDMPlanner',
                    reason='idm_runtime_error',error=f'{type(exc).__name__}: {exc}')
        elif unresolved and idm_trajectory is not None:
            idm_fallback_used=True
            sl_diag['idm_fallback']=dict(attempted=True,used=True,source='nuplan.IDMPlanner',
                reason='pdm_candidate_reused')
        elif unresolved:
            sl_diag['idm_fallback']=dict(attempted=False,used=False,source='nuplan.IDMPlanner',
                reason='idm_unavailable',error=self._idm_fallback.init_error)
        elif idm_selected and idm_trajectory is not None:
            sl_diag['idm_fallback']=dict(attempted=True,used=True,source='nuplan.IDMPlanner',
                reason='pdm_selected_idm')
        else:
            sl_diag['idm_fallback']=dict(attempted=False,used=False,source='nuplan.IDMPlanner',
                reason='learned_plan_passed_final_checks')
        poses=np.column_stack([selected[1:],selection.stable_heading(selected[None])[0,1:]])
        learned_trajectory=InterpolatedTrajectory(transform_predictions_to_states(poses,history.ego_states,3.9,.1))
        trajectory=idm_trajectory if (idm_trajectory is not None and idm_fallback_used) else learned_trajectory
        self.selection_diagnostic=diagnostic
        if self.enable_guard:
            diagnostic=dict(diagnostic,base_selector_rules_sha256=diagnostic.get('selector_rules_sha256'),
                selector_rules_sha256=GUARD_SHA)
        polygons=[lane.polygon for lane in lanes];region_key=tuple(p.wkb for p in polygons)
        if not hasattr(self,'_diagnostic_regions'):self._diagnostic_regions={}
        if region_key not in self._diagnostic_regions:
            if len(self._diagnostic_regions)>=8:self._diagnostic_regions.clear()
            union=self._region_cache.get(region_key) if hasattr(self,'_region_cache') else None
            if union is None:union=unary_union(polygons)
            self._diagnostic_regions[region_key]=union.buffer(.3)
        region=self._diagnostic_regions[region_key]
        c,s=np.cos(ego.heading),np.sin(ego.heading);rot=np.array([[c,s],[-s,c]])
        world=xy@rot+[ego.x,ego.y];inside=contains(region,world[...,0],world[...,1])
        selected_world=selected@rot+[ego.x,ego.y]
        selected_inside=contains(region,selected_world[:,0],selected_world[:,1])
        selected_anchor_id=index
        self.last_full_pool_diagnostic=dict(diagnostic,iteration=current_input.iteration.index,
            output_stage=stage,prefilter=('all3000' if topk==3000 else f'FM2_vnorm_ascending_top{topk}'),selected_anchor_id=selected_anchor_id,geometry=geometry,safety_guard=guard,
            inference_version=self.name(),selector_seconds=selector_seconds,
            total_planner_seconds=time.perf_counter()-begin,learned_candidate_count=3000,
            extra_braking_candidates=int(guard['braking_candidate_added']),
            candidate_any_ool_fraction=float((~inside[:,1:]).any(1).mean()),
            selected_any_ool=bool((~selected_inside[1:]).any()),executed_point_ool=not region.covers(Point(ego.x,ego.y)),
            OOL_definition=('unchanged buffered0.3m rear-axle point test; candidate fraction covers '
                            + ('full3000' if topk==3000 else f'vnorm_top{topk}')),
            selected_xy=selected.tolist(),selected_sampled_polygon_good=bool(corridor.labels(selected[None])[0]),
            executed_source=('IDMPlanner' if (idm_trajectory is not None and idm_fallback_used) else 'learned_or_repaired'),
            future_GT_input=False,EF_calls_this_frame=2,sl_dp=sl_diag,raw_selected_xy=raw_selected.tolist())
        self.last_full_pool_diagnostic.update(
            kept_anchor_ids=ids.tolist(), kept_vnorm=norm[kept].cpu().tolist(),
            prefilter_rank_definition=('All original IDs 0..2999 before EF' if topk==3000
                                       else f'FM2 raw-meter vnorm ascending top{topk}; ties original anchor ID'),
            scored_candidate_count=int(len(kept)), full_pool_candidate_count=3000,
            vnorm_topk=topk,
            model_prefilter_count=int(len(kept)),
            pdm_candidate_count=int(len(xy)),
            pdm_idm_candidate=bool(idm_xy is not None),
            pdm_idm_selected=idm_selected,
        )
        if self.diagnostic_path:
            with Path(self.diagnostic_path).open('a') as f:f.write(json.dumps(self.last_full_pool_diagnostic)+'\n')
        return trajectory

from combined_install import install as _install_combined_features
_install_combined_features(FM2Top250Planner)

from normal_install import install as _install_agent_history
_install_agent_history(FM2Top250Planner)
