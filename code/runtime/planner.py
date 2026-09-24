import numpy as np
from repair_planner import RepairPlanner
from driveanchor_planner import DriveAnchorPlanner
from choose_dtpp import make_fast_choose
from predictor import Predictor
_CHOOSE=make_fast_choose(DriveAnchorPlanner.choose)

class DTPPTTCPlanner(RepairPlanner):
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw);self.predictor=Predictor();self._risk_cache={}
 def initialize(self,initialization):
  super().initialize(initialization);self._dtpp_init=initialization;self._risk_cache={}
 def name(self):return super().name()+'_dtpp_nearest10_ttc_v1'
 def compute_planner_trajectory(self,current_input):
  self._risk_cache={};self.predictor.prepare(current_input,self._dtpp_init)
  return super().compute_planner_trajectory(current_input)
 def dtpp_evaluate(self,xy):
  key=xy.tobytes()
  if key not in self._risk_cache:
   # Guard broadcasts one braking proposal to500; predict that candidate only once.
   if xy.strides[0]==0:
    bad,diag=self.predictor.evaluate(xy[:1]);bad=np.repeat(bad,len(xy))
    diag=dict(diag,min_ttc_capped_s=diag['min_ttc_capped_s']*len(xy),predicted_collision=diag['predicted_collision']*len(xy),rejected_count=int(bad.sum()))
   else:bad,diag=self.predictor.evaluate(xy)
   self._risk_cache[key]=(bad,diag)
  return self._risk_cache[key]
 def choose(self,xy,route,tracks,ego,speed,lights,lanes):
  return _CHOOSE(self,xy,route,tracks,ego,speed,lights,lanes)
