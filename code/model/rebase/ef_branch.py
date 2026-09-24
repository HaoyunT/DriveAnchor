"""Scaled EF head with the guide_flow_training_v6.0_rebased feature contract.

The reference branch uses 50 values: scene, sixteen (x, y, enterable)
triples, and an expanded flag.  The output is an additive displacement in
meters, not a time derivative.  Coordinate/residual scaling and the fixed
current ego pose are retained from ScaledEFHead for the 40-point nuPlan model.
"""

import torch

from .branch_modules import MLP
from .ef_scaled import ScaledEFHead


class BranchEFHead(ScaledEFHead):
    version = 'branch_guided_v3'
    feature_dim = 50
    num_polygon_points = 16
    num_scenes = 5

    def __init__(self, points=40):
        if not isinstance(points, int) or points < 2:
            raise ValueError('EF requires at least two trajectory points')
        super().__init__(points=points)
        self.polygon = MLP(49, 128)
        self.position_embedding = torch.nn.Parameter(torch.randn(49) * .02)
        self.post_attention_norm = torch.nn.LayerNorm(256)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(256, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),
        )
        self.output_norm = torch.nn.LayerNorm(256)

    def normalized_feature(self, feature, anchors):
        """Validate the v3 contract and scale only polygon coordinates."""
        feature = torch.as_tensor(feature, device=anchors.device, dtype=anchors.dtype)
        if feature.ndim != 1 or feature.shape[0] != self.feature_dim:
            raise ValueError('Branch EF requires a one-dimensional 50-value feature')
        if not torch.isfinite(feature).all():
            raise ValueError('Branch EF feature must be finite')
        scene = feature[0]
        if scene != scene.round() or not 0 <= scene < self.num_scenes:
            raise ValueError('Branch EF scene must be an integer in [0, 4]')
        vertices = feature[1:49].reshape(self.num_polygon_points, 3)
        flags = torch.cat((vertices[:, 2], feature[49:50]))
        if ((flags < 0) | (flags > 1)).any():
            raise ValueError('Enterable and expanded flags must lie in [0, 1]')
        if not torch.isfinite(self.position_scale).all() or (self.position_scale <= 0).any():
            raise ValueError('EF position scales must be finite and positive')
        if not torch.isfinite(self.residual_scale).all() or (self.residual_scale <= 0).any():
            raise ValueError('EF residual scales must be finite and positive')
        normalized_vertices = torch.cat(
            (vertices[:, :2] / self.position_scale, vertices[:, 2:3]), dim=-1,
        )
        return torch.cat((feature[:1], normalized_vertices.flatten(), feature[49:50]))

    def forward(self, anchors, feature, context, valid):
        if anchors.ndim != 3 or anchors.shape[1:] != (self.points, 2) or not len(anchors):
            raise ValueError('Expected a nonempty [anchors, points, 2] trajectory tensor')
        if not torch.isfinite(anchors).all():
            raise ValueError('EF input trajectories must be finite')
        if context.ndim != 3 or context.shape[0] != 1 or context.shape[-1] != 256:
            raise ValueError('Branch EF expects one scene context with shape [1, tokens, 256]')
        if valid.shape != context.shape[:2] or not valid.bool().any():
            raise ValueError('EF requires a matching context mask with at least one valid token')
        if not torch.isfinite(context).all():
            raise ValueError('EF context must be finite, including padded tokens')
        normalized_feature = self.normalized_feature(feature, anchors)
        n = len(anchors)
        anchor_embedding = self.anchor((anchors / self.position_scale).flatten(1))
        polygon_embedding = self.polygon(normalized_feature[1:] + self.position_embedding)
        query = self.query_norm(self.fusion(torch.cat((
            self.scene(normalized_feature[:1]).expand(n, -1),
            polygon_embedding.expand(n, -1),
            anchor_embedding,
        ), dim=-1)))[None]
        ctx = self.context_norm(context)
        attention = self.attention(
            query, ctx, ctx, key_padding_mask=~valid.bool(), need_weights=False,
        )[0]
        decoded = self.post_attention_norm(attention + query)
        decoded = self.output_norm(self.ffn(decoded) + decoded)
        correction = self.output(decoded)[0].reshape(n, self.points, 2)
        correction = torch.cat((torch.zeros_like(correction[:, :1]), correction[:, 1:]), dim=1)
        return correction * self.residual_scale
