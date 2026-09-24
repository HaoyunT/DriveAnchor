"""Independent polygon-guided residual head and geometric supervision."""
import numpy as np
import torch
from shapely.geometry import Polygon, LineString, Point
from .branch_modules import MLP

class Corridor:
    def __init__(self, spec):
        vertices=np.asarray(spec['polygon'],dtype=float)
        if vertices.ndim!=2 or vertices.shape[1]!=2 or len(vertices)<3 or not np.isfinite(vertices).all():
            raise ValueError('Expected finite polygon vertices in ego-frame meters')
        self.polygon=Polygon(vertices)
        if not self.polygon.is_valid or self.polygon.area<=1e-6:raise ValueError('Invalid corridor polygon')
        edge=int(spec['exit_edge'])
        if not 0<=edge<len(vertices):raise ValueError('Exit edge out of range')
        self.exit=LineString([vertices[edge],vertices[(edge+1)%len(vertices)]])
        # First two coordinates encode the designated exit; remaining 14 sample perimeter.
        points=list(self.exit.coords)+[list(self.polygon.exterior.interpolate(t,normalized=True).coords)[0] for t in np.linspace(0,1,14,endpoint=False)]
        self.feature=np.concatenate([[float(spec.get('scene_type',0))],np.asarray(points).reshape(-1)]).astype(np.float32)
        if not np.isfinite(self.feature).all():raise ValueError('Invalid scene type')

    def good(self, trajectory):
        xy=np.asarray(trajectory,dtype=float)
        if not np.isfinite(xy).all():raise ValueError('Nonfinite trajectory')
        path=LineString(xy)
        if self.polygon.covers(path):return True
        if self.polygon.disjoint(path):return False
        entered=False;previous=None
        for a,b in zip(xy[:-1],xy[1:]):
            segment=LineString([a,b])
            if segment.length<1e-10:
                entered |= self.polygon.covers(Point(a));continue
            intersection=segment.intersection(self.polygon.boundary)
            def endpoints(g):
                if g.is_empty:return []
                if hasattr(g,'geoms'):return [p for part in g.geoms for p in endpoints(part)]
                return [Point(c) for c in g.coords]
            ts=sorted(set([0.,1.]+[min(1.,max(0.,segment.project(p,normalized=True))) for p in endpoints(intersection)]))
            for lo,hi in zip(ts[:-1],ts[1:]):
                if hi-lo<1e-10:continue
                inside=self.polygon.covers(segment.interpolate((lo+hi)/2,normalized=True))
                if previous is True and not inside:
                    if self.exit.distance(segment.interpolate(lo,normalized=True))>1e-6:return False
                entered |= inside;previous=inside
        return bool(entered)

    def labels(self, anchors):
        return torch.tensor([self.good(a) for a in anchors.detach().cpu().numpy()],device=anchors.device)

class EFHead(torch.nn.Module):
    def __init__(self,points=40):
        super().__init__()
        self.scene=MLP(1,128);self.polygon=MLP(32,128)
        self.fusion=MLP(512,256)
        self.attention=torch.nn.MultiheadAttention(256,2,batch_first=True)
        self.output=torch.nn.Sequential(MLP(256,256),torch.nn.Linear(256,points*2))
        torch.nn.init.zeros_(self.output[-1].weight);torch.nn.init.zeros_(self.output[-1].bias)
    def forward(self,anchor_embedding,feature,context,valid):
        scaled=feature.clone();scaled[1:]=scaled[1:]/scaled.new_tensor([40.,5.]).repeat(16)
        n=anchor_embedding.shape[0]
        query=self.fusion(torch.cat([self.scene(scaled[:1]).expand(n,-1),self.polygon(scaled[1:]).expand(n,-1),anchor_embedding],-1))[None]
        decoded=self.attention(query,context,context,key_padding_mask=~valid,need_weights=False)[0]
        return self.output(decoded+query)[0].reshape(n,-1,2)


def target_displacements(anchors,corridor):
    good=corridor.labels(anchors)
    if not good.any():raise ValueError('No good vocabulary anchor for this polygon; cannot fabricate EF targets')
    targets=torch.zeros_like(anchors)
    bad=~good
    if bad.any():
        pool=anchors[good]
        nearest=torch.cdist(anchors[bad].flatten(1),pool.flatten(1)).argmin(-1)
        targets[bad]=pool[nearest]-anchors[bad]
    return targets,good
