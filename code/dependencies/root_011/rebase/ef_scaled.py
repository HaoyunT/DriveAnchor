"""Metric-consistent EF residual head; scales are fitted on training data only."""
import torch
from .branch_modules import MLP

class ScaledEFHead(torch.nn.Module):
    version='scaled_absolute_v2'
    def __init__(self,points=40):
        super().__init__()
        self.register_buffer('position_scale',torch.ones(2))
        self.register_buffer('residual_scale',torch.ones(2))
        self.anchor=MLP(points*2,256)
        self.scene=MLP(1,128);self.polygon=MLP(32,128)
        self.fusion=MLP(512,256)
        self.context_norm=torch.nn.LayerNorm(256)
        self.query_norm=torch.nn.LayerNorm(256)
        self.attention=torch.nn.MultiheadAttention(256,2,batch_first=True)
        self.output=torch.nn.Sequential(MLP(256,256),torch.nn.Linear(256,points*2))
        torch.nn.init.zeros_(self.output[-1].weight);torch.nn.init.zeros_(self.output[-1].bias)
        self.points=points
    @torch.no_grad()
    def fit_scales(self,anchors,targets,good):
        bad_targets=targets[~good]
        if not len(bad_targets):raise ValueError('No correction targets to fit residual scale')
        self.position_scale.copy_(anchors.square().mean((0,1)).sqrt().clamp_min(1.))
        self.residual_scale.copy_(bad_targets[:,1:].square().mean((0,1)).sqrt().clamp_min(.1))
    def forward(self,anchors,feature,context,valid):
        feature=feature.to(device=anchors.device,dtype=anchors.dtype)
        emb=self.anchor((anchors/self.position_scale).flatten(1))
        poly=feature[1:]/self.position_scale.repeat(16)
        n=len(anchors)
        query=self.query_norm(self.fusion(torch.cat([self.scene(feature[:1]).expand(n,-1),self.polygon(poly).expand(n,-1),emb],-1)))[None]
        ctx=self.context_norm(context)
        decoded=self.attention(query,ctx,ctx,key_padding_mask=~valid.bool(),need_weights=False)[0]
        normalized=self.output(decoded+query)[0].reshape(n,self.points,2)
        # All anchors use the same current ego pose; never move time zero.
        normalized=torch.cat([torch.zeros_like(normalized[:,:1]),normalized[:,1:]],1)
        return normalized*self.residual_scale
    def loss(self,prediction,target,good):
        error=((prediction-target)/self.residual_scale)[:,1:].square().mean((1,2))
        pieces=[error[mask].mean() for mask in [good,~good] if mask.any()]
        return torch.stack(pieces).mean()
