"""Production binding: main features and per-instance DTPP prepare only."""
from planner_features import install as install_main
from dtpp_features import install as install_dtpp

def install(planner_class):
    if planner_class.__dict__.get('_combined_features_v1_installed',False):
        return
    install_main(planner_class)
    original_init=planner_class.__init__
    def initialize_instance(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        install_dtpp(self.predictor,map_mode='cuda')
    planner_class.__init__=initialize_instance
    planner_class._combined_features_v1_installed=True
