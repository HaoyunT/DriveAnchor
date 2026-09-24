"""Short-circuit: scorehead topk==1 skips full _selection stack (simplified insert)."""
from pathlib import Path

P = Path('/tmp/driveanchor_selector_v8_full_runtime/fm2_planner.py')
src = P.read_text()

old = """        selected,index,diagnostic,guard=self._selection(xy,route,tracks,state,list(current_input.traffic_light_data or []),lanes)
        idm_selected=bool(idm_xy is not None and index is not None and int(index)==len(xy)-1)
"""
assert src.count(old) == 1

new = """        if prefilter_mode=='scorehead' and topk==1:
            heading=selection.stable_heading(xy)[0]
            direction_bad_final=bool(direction_bad(xy,self._direction_context)[0])
            collision_final=bool(_final_cv_collision(self,xy,heading,tracks,ego,None))
            feasible=int(not(direction_bad_final or collision_final))
            diagnostic=dict(selector_version='scorehead_top1_shortcircuit',
                candidate_count=1,feasible_count=feasible,safety_feasible_count=feasible,
                comfort_candidate_count=1,safety_and_comfort_candidate_count=1,
                selected_comfortable=True,selected_quality=0.0,
                selected_route_progress_m=float(route_progress(xy[None],route)[0]),
                selected_violation_count=int(direction_bad_final)+int(collision_final),
                selection_fallback=False,baseline_selected_id=0,
                selected_violations=[collision_final,False,False,False,False,direction_bad_final,False],
                rejected_by_constraint=[0,0,0,0,0,int(direction_bad_final),0],
                rejected_by_comfort_component=[],
                constraint_names=['predicted_collision','lane_footprint_exit','current_red_entry','speed','acceleration','direction','candidate_comfort'],
                comfort_fallback_to_original_safety_order=False,
                rank_definition='score head FAG top-1; short-circuit, no downstream ranking',
                prefilter='scorehead_top1')
            guard=dict(enabled=False,original_selected_id=0,selection_source='scorehead_top1',
                margin_m=1.,braking_candidate_added=False)
            selected=xy[0];index=0
            self.selection_diagnostic=diagnostic
            idm_selected=False
            idm_xy=None
        else:
            selected,index,diagnostic,guard=self._selection(xy,route,tracks,state,list(current_input.traffic_light_data or []),lanes)
            idm_selected=bool(idm_xy is not None and index is not None and int(index)==len(xy)-1)
"""
src = src.replace(old, new)
P.write_text(src)
print('patched')
