"""Indexed actor history and cached static polygon sampling; native feature semantics."""
from cached_planner import *

class IndexedFeaturesPlanner(CachedGeometryPlanner):
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
        if not hasattr(self, '_actor_frames'):self._actor_frames=OrderedDict()
        frames=[]
        for obs in history.observations:
            key=id(obs)
            if key not in self._actor_frames:
                by_token={}
                for actor in obs.tracked_objects.tracked_objects:
                    by_token.setdefault(actor.track_token,actor)
                self._actor_frames[key]=(obs,by_token)
                if len(self._actor_frames)>64:self._actor_frames.popitem(last=False)
            frames.append(self._actor_frames[key][1])
        for i,obj in enumerate(moving[:8]):
            seen=[]
            for t,frame in zip(times,frames):
                match=frame.get(obj.track_token)
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
        if not hasattr(self,'_polygon_samples'):self._polygon_samples={}
        for i,(obj,kind) in enumerate(poly[:100]):
            key=(id(self.map),type(obj).__name__,obj.id)
            if key not in self._polygon_samples:
                if len(self._polygon_samples)>4096:self._polygon_samples.clear()
                self._polygon_samples[key]=(self.map,self.sample_line_cached(obj.polygon.simplify(.1,preserve_topology=True).exterior,20))
            pi[0,i]=[kind,1];pp[0,i]=local(self._polygon_samples[key][1],ego)
        mask=np.zeros((1,208),bool);mask[:,:8]=True;mask[:,8:8+len(valid)]=True;mask[:,108:108+min(len(poly),100)]=True
        feats={k:torch.from_numpy(v).to(self.device) for k,v in dict(ego=ef,interact_obstacle=agents,lane_instance_wise=li,lane_instance_points=lp,polygon_instance_wise=pi,polygon_instance_points=pp).items()}
        return feats,torch.from_numpy(mask).to(self.device),valid,tracks
