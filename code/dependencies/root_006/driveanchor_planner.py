"""Pilot native-observation nuPlan adapter; no logged-future/GT inputs."""
import json,logging
from shapely.ops import unary_union
from shapely.vectorized import contains,touches
from typing import Type
import numpy as np
import torch
from scipy.spatial import cKDTree
from shapely.geometry import Point,LineString,Polygon
from selection import constrained_choice,stable_heading,RULES
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer,TrafficLightStatusType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks,Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from rebase.model import AnchorFM
from rebase.ef import Corridor


def local(points,ego):
    points=np.asarray(points,dtype=float)
    c,s=np.cos(ego.heading),np.sin(ego.heading)
    return (points-np.array([ego.x,ego.y]))@np.array([[c,-s],[s,c]])

def sample_line(line,n):
    return np.array([line.interpolate(t,normalized=True).coords[0][:2] for t in np.linspace(0,1,n)])

class DriveAnchorPlanner(AbstractPlanner):
    def __init__(self,checkpoint,device='cuda',use_ef=False):
        torch.set_num_threads(2)
        self.device=device;self.use_ef=use_ef;self.checkpoint=checkpoint
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if ckpt['architecture']!=AnchorFM.architecture:raise ValueError('Incompatible checkpoint')
        self.model=AnchorFM(ckpt['model']['anchors'],ckpt['config'])
        self.model.load_state_dict(ckpt['model'],strict=True);self.model.to(device).eval()
    def name(self):return 'DriveAnchor_EF_FMRL_pilot' if self.use_ef else 'DriveAnchor_FMRL_pilot'
    def observation_type(self)->Type[Observation]:return DetectionsTracks
    def initialize(self,initialization):
        self.map=initialization.map_api;self.route_ids=list(map(str,initialization.route_roadblock_ids));self.route=set(self.route_ids)

    def features(self,history):
        states=list(history.ego_states);ego=states[-1].rear_axle;now=states[-1].time_point.time_s
        times=np.array([x.time_point.time_s for x in states]);xy=np.array([[x.rear_axle.x,x.rear_axle.y] for x in states])
        # Cache uses 40 history positions at .1s; unavailable past is left-held and logged as an adapter difference.
        query=now+np.arange(-39,1)*.1
        past=np.stack([np.interp(query,times,xy[:,i]) for i in range(2)],1)
        ef=np.zeros((1,1,40,8),np.float32);ef[0,0,:,0]=1;ef[0,0,:,1:3]=local(past,ego)
        ef[0,0,:,3]=states[-1].dynamic_car_state.speed;ef[0,0,:,4]=1;ef[0,0,:,6:8]=[4.5,2.]
        agents=np.zeros((1,8,40,22),np.float32)
        tracks=list(history.observations[-1].tracked_objects.tracked_objects)
        moving=[x for x in tracks if hasattr(x,'velocity') and np.hypot(x.velocity.x,x.velocity.y)>.5 and np.hypot(x.center.x-ego.x,x.center.y-ego.y)<80]
        moving.sort(key=lambda x:np.hypot(x.center.x-ego.x,x.center.y-ego.y))
        for i,obj in enumerate(moving[:8]):
            seen=[]
            for t,obs in zip(times,history.observations):
                match=next((o for o in obs.tracked_objects.tracked_objects if o.track_token==obj.track_token),None)
                if match is not None:seen.append((t,match.center.x,match.center.y))
            if not seen:continue
            a=np.array(seen);h=np.stack([np.interp(query,a[:,0],a[:,j]) for j in (1,2)],1)
            agents[0,i,:,:2]=local(h,ego)
            vel=local([[ego.x+obj.velocity.x,ego.y+obj.velocity.y]],ego)[0]
            agents[0,i,:,2:4]=vel;agents[0,i,:,4:7]=[obj.box.width,obj.box.length,obj.box.height]
            agents[0,i,:,7]=(obj.center.heading-ego.heading+np.pi)%(2*np.pi)-np.pi
        layers=[SemanticMapLayer.LANE,SemanticMapLayer.LANE_CONNECTOR,SemanticMapLayer.INTERSECTION,SemanticMapLayer.CROSSWALK,SemanticMapLayer.STOP_LINE,SemanticMapLayer.WALKWAYS,SemanticMapLayer.CARPARK_AREA]
        near=self.map.get_proximal_map_objects(Point2D(ego.x,ego.y),100,layers)
        lanes=near[SemanticMapLayer.LANE]+near[SemanticMapLayer.LANE_CONNECTOR]
        lanes.sort(key=lambda x:x.polygon.distance(Point(ego.x,ego.y)))
        li=np.zeros((1,100,5),np.float32);lp=np.zeros((1,100,50,8),np.float32);valid=[]
        for lane in lanes:
            if len(valid)==100:break
            line=lane.baseline_path.linestring
            if line.is_empty or line.length<.01:continue
            i=len(valid);center=local(sample_line(line,50),ego)
            left=local(sample_line(lane.left_boundary.linestring,50),ego);right=local(sample_line(lane.right_boundary.linestring,50),ego)
            # Align independently sampled boundary orientation to the lane baseline.
            if np.linalg.norm(left[0]-center[0])>np.linalg.norm(left[-1]-center[0]):left=left[::-1]
            if np.linalg.norm(right[0]-center[0])>np.linalg.norm(right[-1]-center[0]):right=right[::-1]
            li[0,i]=[1,0,np.linalg.norm(left-center,axis=1).mean(),np.linalg.norm(right-center,axis=1).mean(),1]
            lp[0,i,:,:2]=1;lp[0,i,:,2:4]=center;lp[0,i,:,4:6]=left;lp[0,i,:,6:8]=right;valid.append(lane)
        if not valid:raise ValueError('No native map lanes; refusing fabricated features')
        pi=np.zeros((1,100,2),np.float32);pp=np.zeros((1,100,20,2),np.float32);poly=[]
        for layer,kind in [(SemanticMapLayer.INTERSECTION,1),(SemanticMapLayer.CROSSWALK,3),(SemanticMapLayer.STOP_LINE,5),(SemanticMapLayer.WALKWAYS,10),(SemanticMapLayer.CARPARK_AREA,6)]:
            for obj in near[layer]:poly.append((obj,kind))
        poly.sort(key=lambda pair:pair[0].polygon.distance(Point(ego.x,ego.y)))
        for i,(obj,kind) in enumerate(poly[:100]):
            pi[0,i]=[kind,1];pp[0,i]=local(sample_line(obj.polygon.simplify(.1,preserve_topology=True).exterior,20),ego)
        mask=np.zeros((1,208),bool);mask[:,:8]=True;mask[:,8:8+len(valid)]=True;mask[:,108:108+min(len(poly),100)]=True
        feats={k:torch.from_numpy(v).to(self.device) for k,v in dict(ego=ef,interact_obstacle=agents,lane_instance_wise=li,lane_instance_points=lp,polygon_instance_wise=pi,polygon_instance_points=pp).items()}
        return feats,torch.from_numpy(mask).to(self.device),valid,tracks

    def route_line(self,lanes,state):
        ego=state.rear_axle
        eligible=[l for l in lanes if str(l.get_roadblock_id()) in self.route]
        if not eligible:
            from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
            self.route_ids=route_roadblock_correction(state,self.map,self.route_ids)
            self.route=set(self.route_ids)
            eligible=[l for l in lanes if str(l.get_roadblock_id()) in self.route]
            logging.info('DriveAnchor corrected off-route start: %s',self.route_ids)
            if not eligible:raise ValueError('No route lane after topology correction')
        def rank(l):
            line=l.baseline_path.linestring;d=line.project(Point(ego.x,ego.y));a=line.interpolate(d);b=line.interpolate(min(line.length,d+1))
            h=np.arctan2(b.y-a.y,b.x-a.x)
            return line.distance(Point(ego.x,ego.y))+2*(1-np.cos(h-ego.heading))
        lane=min(eligible,key=rank);points=[];seen=set()
        for step in range(12):
            if lane.id in seen:break
            seen.add(lane.id);line=lane.baseline_path.linestring
            start=line.project(Point(ego.x,ego.y)) if step==0 else 0
            points.extend([line.interpolate(t).coords[0] for t in np.arange(start,line.length,.5)])
            if len(points)>200:break
            next_lanes=[l for l in lane.outgoing_edges if str(l.get_roadblock_id()) in self.route and l.id not in seen]
            if not next_lanes:break
            lane=next_lanes[0] # Route-constrained deterministic branch choice; no expert future.
        if len(points)<2:raise ValueError('Route centerline too short')
        return local(points,ego)

    def choose(self,xy,route,tracks,ego,speed,lights,lanes):
        distance=cKDTree(route).query(xy.reshape(-1,2))[0].reshape(len(xy),40)
        velocity=np.diff(xy,axis=1)/RULES.dt;acceleration=np.diff(velocity,axis=1)/RULES.dt
        score=RULES.route_weight*(distance**2).mean(1)
        score+=RULES.speed_match_weight*(np.linalg.norm(velocity[:,0],axis=-1)-speed)**2+RULES.acceleration_weight*(acceleration**2).mean((1,2))-RULES.endpoint_distance_weight*np.linalg.norm(xy[:,-1],axis=-1)
        heading=stable_heading(xy)
        collision=np.zeros(len(xy),bool);red=np.zeros(len(xy),bool)
        centers=xy+RULES.center_offset*np.stack([np.cos(heading),np.sin(heading)],-1)
        for obj in tracks:
            pos=local([[obj.center.x,obj.center.y]],ego)[0]
            vel=local([[ego.x+obj.velocity.x,ego.y+obj.velocity.y]],ego)[0] if hasattr(obj,'velocity') else np.zeros(2)
            pred=pos+np.arange(40)[:,None]*RULES.dt*vel
            yaw=obj.center.heading-ego.heading;c,s=np.cos(yaw),np.sin(yaw)
            relative=(centers-pred)@np.array([[c,-s],[s,c]])
            angle=heading-yaw
            half_l=obj.box.length/2+np.abs(np.cos(angle))*RULES.half_length+np.abs(np.sin(angle))*RULES.half_width
            half_w=obj.box.width/2+np.abs(np.sin(angle))*RULES.half_length+np.abs(np.cos(angle))*RULES.half_width
            hit=(np.abs(relative[...,0])<half_l)&(np.abs(relative[...,1])<half_w)
            collision |= hit.any(1)
        for light in lights:
            if light.status!=TrafficLightStatusType.RED:continue
            connector=self.map.get_map_object(str(light.lane_connector_id),SemanticMapLayer.LANE_CONNECTOR)
            if connector is None:continue
            # Already inside a red connector is not a new entry.
            if connector.polygon.covers(Point(ego.x,ego.y)):continue
            polygon=Polygon(local(np.array(connector.polygon.exterior.coords),ego))
            red |= (contains(polygon,xy[...,0],xy[...,1])|touches(polygon,xy[...,0],xy[...,1])).any(1)
        forward=np.stack([np.cos(heading),np.sin(heading)],-1)
        side=np.stack([-np.sin(heading),np.cos(heading)],-1)
        corners=np.stack([centers+l*RULES.half_length*forward+w*RULES.half_width*side for l,w in [(1,1),(1,-1),(-1,-1),(-1,1)]],axis=2)
        c,s=np.cos(ego.heading),np.sin(ego.heading)
        world=corners@np.array([[c,s],[-s,c]])+np.array([ego.x,ego.y])
        # Conservative lane-union footprint screen, distinct from official DAC.
        region=unary_union([lane.polygon for lane in lanes])
        inside=contains(region,world[...,0],world[...,1])|touches(region,world[...,0],world[...,1])
        outside=~inside.all((1,2))
        vbad=(np.linalg.norm(velocity,axis=-1)>RULES.speed_limit).any(1)
        abad=(np.linalg.norm(acceleration,axis=-1)>RULES.acceleration_limit).any(1)
        violations=np.column_stack([collision,outside,red,vbad,abad])
        index,diagnostic=constrained_choice(-score,violations)
        diagnostic['constraint_names']=['predicted_collision','lane_footprint_exit','current_red_entry','speed','acceleration']
        diagnostic['selected_violations']=violations[index].tolist()
        diagnostic['rejected_by_constraint']=violations.sum(0).tolist()
        self.selection_diagnostic=diagnostic
        return index

    @torch.no_grad()
    def compute_planner_trajectory(self,current_input):
        history=current_input.history;ego=history.ego_states[-1].rear_axle
        feats,mask,lanes,tracks=self.features(history);route=self.route_line(lanes,history.ego_states[-1])
        corridor=None
        if self.use_ef:
            ribbon=LineString(route[:60]).buffer(3.,cap_style=2)
            vertices=np.array(ribbon.exterior.coords)[:-1];tip=route[min(59,len(route)-1)]
            edge=int(np.linalg.norm((vertices+np.roll(vertices,-1,axis=0))/2-tip,axis=1).argmin())
            corridor=Corridor(dict(polygon=vertices.tolist(),exit_edge=edge))
        xy=self.model.generate(feats,mask,corridor=corridor).reshape(-1,40,2).cpu().numpy()
        if not np.isfinite(xy).all():raise ValueError('Nonfinite model trajectory')
        choice=self.choose(xy,route,tracks,ego,history.ego_states[-1].dynamic_car_state.speed,list(current_input.traffic_light_data or []),lanes)
        selected=xy[choice]
        # Diagnostic only: rear-axle point within native nearby lane polygons + .3m tolerance.
        # Separate from the official full-vehicle drivable-area and collision metrics.
        region=unary_union([l.polygon for l in lanes]).buffer(.3)
        c,s=np.cos(ego.heading),np.sin(ego.heading)
        world=xy@np.array([[c,s],[-s,c]])+np.array([ego.x,ego.y])
        inside=contains(region,world[...,0],world[...,1])
        logging.info('DRIVEANCHOR_DIAGNOSTIC %s',json.dumps(dict(iteration=current_input.iteration.index,
            candidate_any_ool_fraction=float((~inside[:,1:]).any(1).mean()),
            selected_any_ool=bool((~inside[choice,1:]).any()),
            executed_point_ool=not region.covers(Point(ego.x,ego.y)),selected_index=choice,**self.selection_diagnostic)))
        yaw=stable_heading(selected[None])[0,1:]
        # Model index 0 is the current timestamp: retain future indices 1..39.
        poses=np.column_stack([selected[1:],yaw])
        states=transform_predictions_to_states(poses,history.ego_states,3.9,.1)
        return InterpolatedTrajectory(states)
