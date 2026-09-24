"""Local nuPlan adapter for rebase_anchor_fm's no-EF, t-free DiT model.

Core network classes are extracted verbatim in branch_modules.py. Cache feature
widths/horizon are explicit adaptations; this is not a production checkpoint loader.
"""
from dataclasses import dataclass, asdict
import numpy as np
import torch
from .compat import ModuleConfig
from .branch_modules import PlanningSharedEncoderV6, GenerativeDecoderDenoiser, detached_integral

@dataclass
class Config:
    fm_revision: str = "legacy"
    ef_rematch: bool = True
    project_dim: int = 256
    num_heads: int = 8
    depth: int = 6
    mlp_ratio: float = 4.
    dropout: float = 0.
    num_trajectory_points: int = 40
    trajectory_tokenize_mode: str = 'whole'
    trajectory_patch_size: int = 10
    condition_inject_mode: str = 'cross_attn'
    enable_local_point_mixer: bool = True
    local_point_mixer_dim: int = 32
    local_point_mixer_num_heads: int = 2
    local_point_mixer_radius: int = 4
    time_condition_mode: str = 'constant'
    time_condition_constant: float = 1.
    enable_ef_head: bool = False
    ef_head_version: str = "legacy"
    enable_energy_condition: bool = False
    energy_position_scale: tuple = (30.,5.)
    energy_condition_weight: float = 1.
    padding_value: float = -300.

class AnchorFM(torch.nn.Module):
    architecture = 'rebase_anchor_fm_nuplan_v1'
    def __init__(self, anchors, config=None):
        super().__init__()
        self.config = Config(**config) if config else Config()
        a = torch.as_tensor(np.load(anchors) if isinstance(anchors,str) else anchors,dtype=torch.float32)
        if a.ndim != 3 or a.shape[-1] != 2 or a.shape[1] < self.config.num_trajectory_points or not torch.isfinite(a).all():
            raise ValueError('Expected finite metric XY anchor trajectories')
        self.register_buffer('anchors',a[:,:self.config.num_trajectory_points].clone())
        self.register_buffer('delta_scale',torch.tensor([2.,.15]))
        if self.config.fm_revision not in ('legacy','refine_v2'):raise ValueError('Unknown FM revision')
        if self.config.fm_revision=='refine_v2':self.register_buffer('refinement_scale',torch.ones(2))
        # Cache uses 22 obstacle channels and 8 lane channels; do not silently
        # reinterpret these as production's 12/4-channel feature definitions.
        self.encoder=PlanningSharedEncoderV6({'feature_config':{
            'interact_obstacle_feature_dim':22,'lane_point_wise_feature_dim':8}})
        self.denoiser=GenerativeDecoderDenoiser(self.config,2)
        from .ef import EFHead
        from .ef_scaled import ScaledEFHead
        from .ef_branch import BranchEFHead
        from .ef_delta import DeltaEFHead
        head_types={'legacy':EFHead,'scaled_absolute_v2':ScaledEFHead,'branch_guided_v3':BranchEFHead,
                    'delta_guided_v4':DeltaEFHead}
        if self.config.ef_head_version not in head_types:raise ValueError('Unknown EF head version')
        head_type=head_types[self.config.ef_head_version]
        self.ef_head=head_type(self.config.num_trajectory_points) if self.config.enable_ef_head else None

    def normalize(self, xy):
        previous=torch.cat([torch.zeros_like(xy[...,:1,:]),xy[...,:-1,:]],dim=-2)
        return (xy-previous)/self.delta_scale

    def denormalize(self, delta): return (delta*self.delta_scale).cumsum(-2)

    @torch.no_grad()
    def fit_scale(self, gt):
        previous=torch.cat([torch.zeros_like(gt[:,:1]),gt[:,:-1]],dim=1)
        self.delta_scale.copy_((gt-previous).square().mean((0,1)).sqrt().clamp_min(1e-6))

    def encode(self, feats, valid_mask):
        ego=feats['ego'].clone()
        if self.training:
            drop=torch.rand(ego.shape[0],device=ego.device)<.25
            frame=torch.arange(ego.shape[2],device=ego.device)>=3
            ego=torch.where(drop[:,None,None,None]&frame[None,None,:,None],-299.8,ego)
        def packed(inst,points,valid):
            x=torch.cat([inst,points.flatten(-2)],-1).unsqueeze(2)
            return torch.where(valid[:,:,None,None],x,-300.)
        n_ego=ego[:,:,::5].shape[2]; nl=feats['lane_instance_wise'].shape[1]
        lanes=packed(feats['lane_instance_wise'],feats['lane_instance_points'],valid_mask[:,n_ego:n_ego+nl].bool())
        polys=packed(feats['polygon_instance_wise'],feats['polygon_instance_points'],valid_mask[:,n_ego+nl:].bool())
        moving=feats['interact_obstacle']
        present=(moving[...,4:6]>0).all(-1).any(-1)
        moving=torch.where(present[:,:,None,None],moving,-300.)
        # Missing modalities are explicitly masked, not treated as observed empty.
        static=ego.new_full((ego.shape[0],1,1,60),-300.)
        lights=ego.new_full((ego.shape[0],1,40,10),-300.)
        context,invalid=self.encoder(ego,moving,static,lanes,polys,lights)
        return context.squeeze(2),~invalid.bool()

    def raw(self,state,context,mask,time=None):
        # Preserve branch scene K/V caching, including its sample broadcast.
        cache=self.denoiser.prepare_context_cache(context,mask)[:2]
        if time is None: time=state.new_ones(state.shape[0])
        out=self.denoiser(state,time,context,mask,context_cache=cache)[0]
        if self.config.fm_revision=='refine_v2':
            out=state+out
            out=torch.cat([torch.zeros_like(out[:,:1]),out[:,1:]],1)
        return out

    def generate(self,feats,mask,iterations=1,starts=None,chunk_size=64,corridor=None):
        if iterations < 1: raise ValueError('iterations must be positive')
        context,valid=self.encode(feats,mask)
        if context.shape[0] != 1: raise ValueError('Local cache adapter expects one scene per call')
        source=self.anchors if starts is None else starts.reshape(-1,self.config.num_trajectory_points,2)
        if corridor is not None:
            proposal=source+self.ef_displacement(feats,mask,source,corridor)
            if self.config.ef_rematch:
                ids=torch.cdist(proposal.flatten(1),self.anchors.flatten(1)).argmin(-1)
                source=self.anchors[ids]
            else:source=proposal
        outputs=[]
        for chunk in source.split(chunk_size):
            state=self.normalize(chunk)
            for _ in range(iterations):
                # Branch predict_type=x: raw is clean state, not a displacement.
                state=self.raw(state,context,valid)
            outputs.append(self.denormalize(state))
        return torch.cat(outputs).flatten(1)

    def supervised_loss(self,feats,mask,gt,k=50):
        context,valid=self.encode(feats,mask)
        if self.config.fm_revision=='refine_v2':
            from .fm_refine import loss
            return loss(self,context,valid,gt,k)
        target=self.normalize(gt.reshape(1,self.config.num_trajectory_points,2))
        table=self.normalize(self.anchors)
        ids=(table-target).flatten(1).norm(dim=-1).topk(min(k,len(table)),largest=False).indices
        source=table[ids]; target=target.expand_as(source)
        t=(torch.randn(len(ids),device=gt.device)*.6+1.3).sigmoid()
        omt=1-t[:,None,None]
        state=t[:,None,None]*target+omt*source
        raw=self.raw(state,context,valid,t)
        # predict=x, loss=v, weight=x, t_eps=.05 from branch config.
        residual=(raw-target)/omt.clamp_min(.05)
        flow=residual.square().sum(-1).mean()
        predicted_waypoints=detached_integral(raw*self.delta_scale,10)
        hybrid=(predicted_waypoints-gt.reshape(1,-1,2)).square().sum(-1).mean()
        return flow+.01*hybrid,{'flow_loss':float(flow.detach()),'waypoint_loss':float(hybrid.detach())}

    def ef_displacement(self,feats,mask,anchors,corridor):
        if self.ef_head is None:raise ValueError("EF head is not enabled")
        if self.config.ef_head_version in ('branch_guided_v3','delta_guided_v4'):
            from .ef_geometry import BranchCorridor
            if not isinstance(corridor,BranchCorridor):
                raise ValueError('EF v3/v4 requires ordered native BranchCorridor; old 33D geometry is incompatible')
            context,valid=self.encode(feats,mask)
            return self.ef_head(anchors,torch.as_tensor(corridor.feature,device=anchors.device),context,valid)
        static=dict(feats)
        static['ego']=torch.full_like(feats['ego'],-300.)
        static['interact_obstacle']=torch.full_like(feats['interact_obstacle'],-300.)
        # Suppress ego augmentation during the static-only encoder pass.
        training=self.training
        self.training=False
        try:context,valid=self.encode(static,mask)
        finally:self.training=training
        from .ef_scaled import ScaledEFHead
        if isinstance(self.ef_head,ScaledEFHead):
            return self.ef_head(anchors,torch.as_tensor(corridor.feature,device=anchors.device),context,valid)
        embedding=self.denoiser._input_projector(self.normalize(anchors).flatten(1))
        feature=torch.as_tensor(corridor.feature,device=anchors.device)
        return self.ef_head(embedding,feature,context,valid)

    def ef_loss(self,feats,mask,corridor):
        if self.config.ef_head_version in ('branch_guided_v3','delta_guided_v4'):
            raise ValueError('Use the versioned ef_branch_training trainer for EF v3/v4 supervision')
        from .ef import target_displacements
        with torch.no_grad():target,good=target_displacements(self.anchors,corridor)
        prediction=self.ef_displacement(feats,mask,self.anchors,corridor)
        from .ef_scaled import ScaledEFHead
        loss=self.ef_head.loss(prediction,target,good) if isinstance(self.ef_head,ScaledEFHead) else torch.nn.functional.smooth_l1_loss(prediction,target)
        return loss,{'ef_loss':float(loss.detach()),'ef_good_anchor_fraction':float(good.float().mean())}
