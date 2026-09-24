"""Independent adapter: bounded content-keyed world-geometry sampling cache.
All map proximity queries, ego transforms, actor history, masks refresh each call.
No changes to frozen baseline source. Cache clears when map identity changes.
"""
from collections import OrderedDict
from driveanchor_planner import *
from driveanchor_planner import DriveAnchorPlanner as BaselinePlanner

class CachedGeometryPlanner(BaselinePlanner):
    def sample_line_cached(self, line, n):
        identity = (id(self.map), getattr(self.map, 'map_name', None))
        if getattr(self, '_geometry_map_identity', None) != identity:
            self._geometry_map_identity = identity
            self._geometry_samples = OrderedDict()
        key = (line.wkb, n)
        if key in self._geometry_samples:
            value = self._geometry_samples.pop(key)
            self._geometry_samples[key] = value
            return value
        value = sample_line(line, n)
        value.setflags(write=False)
        self._geometry_samples[key] = value
        if len(self._geometry_samples) > 4096:
            self._geometry_samples.popitem(last=False)
        return value

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
            i=len(valid);center=local(self.sample_line_cached(line,50),ego)
            left=local(self.sample_line_cached(lane.left_boundary.linestring,50),ego);right=local(self.sample_line_cached(lane.right_boundary.linestring,50),ego)
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
            pi[0,i]=[kind,1];pp[0,i]=local(self.sample_line_cached(obj.polygon.simplify(.1,preserve_topology=True).exterior,20),ego)
        mask=np.zeros((1,208),bool);mask[:,:8]=True;mask[:,8:8+len(valid)]=True;mask[:,108:108+min(len(poly),100)]=True
        feats={k:torch.from_numpy(v).to(self.device) for k,v in dict(ego=ef,interact_obstacle=agents,lane_instance_wise=li,lane_instance_points=lp,polygon_instance_wise=pi,polygon_instance_points=pp).items()}
        return feats,torch.from_numpy(mask).to(self.device),valid,tracks
