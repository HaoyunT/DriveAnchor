"""Only remove infrastructure dependencies; run the branch's dense PyTorch path."""
import copy
from types import SimpleNamespace
import torch
BaseModule = torch.nn.Module
class MatMulWrapper(torch.nn.Module):
    def forward(self, a, b): return torch.matmul(a,b)
class TritonAttention:
    @staticmethod
    def apply(*args, **kwargs):
        raise RuntimeError('Triton path is disabled in local adapter')
def ModuleConfig(owner, defaults, overrides):
    def merge(a,b):
        for k,v in b.items():
            if isinstance(v,dict) and isinstance(a.get(k),dict): merge(a[k],v)
            else: a[k]=v
        return a
    def namespace(x):
        return SimpleNamespace(**{k:namespace(v) if isinstance(v,dict) else v for k,v in x.items()})
    return namespace(merge(copy.deepcopy(defaults), overrides or {}))
