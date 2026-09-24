"""Install accepted history processing after combined GPU feature installation."""
from agent_history_v1.integration import install as install_predictor

def install(cls):
    original=cls.__init__
    def initialize(self,*args,**kwargs):
        original(self,*args,**kwargs)
        install_predictor(self.predictor)
    cls.__init__=initialize
