"""Batched EF with normalized step corrections and integrated metric output.

The delta representation and local point mixer follow ``rebase_anchor_fm``.
They are useful inductive biases, not hard velocity/acceleration/jerk bounds.
All scales must be fitted using training records only.  The public EF output
remains an additive XY trajectory displacement, preserving the planner API.
"""

import torch

from .branch_modules import GenerativeDecoderLocalPointMixer, MLP, _onnx_multihead_self_attention


class _DeltaLocalPointMixer(GenerativeDecoderLocalPointMixer):
    """Same branch mixer, keeping its embedding dtype aligned under AMP.

    Without this cast, fp32 positional embeddings promote bf16 projected
    points back to fp32. PyTorch 2.0 then builds an fp32 MHA mask before AMP
    projects the query to bf16, which its SDPA implementation rejects.
    """

    def forward(self, trajectory):
        projected = self._input_projector(trajectory)
        point_feature = projected + self._position_embedding.weight.to(projected.dtype).unsqueeze(0)
        normed = self._norm(point_feature)
        if torch.onnx.is_in_onnx_export():
            attended = _onnx_multihead_self_attention(normed, self._attention, self._attention_mask)
        else:
            attended = self._attention(
                normed, normed, normed, attn_mask=self._attention_mask, need_weights=False,
            )[0]
        return self._output_projector((point_feature + attended).mean(dim=1)).unsqueeze(1)


class DeltaEFHead(torch.nn.Module):
    version = 'delta_guided_v4'
    feature_dim = 50
    num_polygon_points = 16
    num_scenes = 5

    def __init__(self, points=40, validate_inputs=True):
        super().__init__()
        if not isinstance(points, int) or isinstance(points, bool) or points < 2:
            raise ValueError('EF requires at least two trajectory points')
        self.points = points
        self.validate_inputs = bool(validate_inputs)
        for name in ('position_scale', 'input_delta_scale', 'step_scale', 'waypoint_scale'):
            self.register_buffer(name, torch.ones(2))
        self.anchor = MLP(points * 2, 256)
        self.absolute_anchor = MLP(points * 2, 256)
        self.local_point_mixer = _DeltaLocalPointMixer(
            num_points=points, trajectory_dim=2, local_dim=32,
            num_heads=2, radius=4, output_dim=256, dropout=0.,
        )
        self.scene = MLP(1, 128)
        self.polygon = MLP(49, 128)
        self.position_embedding = torch.nn.Parameter(torch.randn(49) * .02)
        self.fusion = MLP(512, 256)
        self.query_norm = torch.nn.LayerNorm(256)
        self.context_norm = torch.nn.LayerNorm(256)
        self.attention = torch.nn.MultiheadAttention(256, 2, batch_first=True)
        self.post_attention_norm = torch.nn.LayerNorm(256)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(256, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 256),
        )
        self.output_norm = torch.nn.LayerNorm(256)
        self.output = torch.nn.Sequential(MLP(256, 256), torch.nn.Linear(256, points * 2))
        torch.nn.init.zeros_(self.output[-1].weight)
        torch.nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def _steps(xy):
        previous = torch.cat((torch.zeros_like(xy[..., :1, :]), xy[..., :-1, :]), dim=-2)
        return xy - previous

    def _trajectory_shape(self, value, name):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(name + ' must be a floating-point tensor')
        if value.ndim not in (3, 4) or value.shape[-2:] != (self.points, 2):
            raise ValueError(name + ' must have shape [N,T,2] or [B,N,T,2]')
        if any(size == 0 for size in value.shape):
            raise ValueError(name + ' must be nonempty')

    @torch.no_grad()
    def fit_scales(self, anchors, target_displacements, good):
        """Fit train-only per-axis RMS; ``good`` masks identity targets.

        Anchor and target counts may differ. ``good`` must have the target's
        leading shape, e.g. [M] or [B,M]; at least one repair target is required.
        The zero-time output is fixed and excluded from correction statistics.
        """
        self._trajectory_shape(anchors, 'anchors')
        self._trajectory_shape(target_displacements, 'target_displacements')
        if not torch.isfinite(anchors).all() or not torch.isfinite(target_displacements).all():
            raise ValueError('Scale fitting requires finite training trajectories')
        good = torch.as_tensor(good, device=target_displacements.device)
        if good.shape != target_displacements.shape[:-2]:
            raise ValueError('good must match the leading dimensions of the targets')
        if good.dtype != torch.bool:
            raise ValueError('good must be a Boolean mask')
        repairs = target_displacements.reshape(-1, self.points, 2)[~good.reshape(-1)].float()
        if not len(repairs):
            raise ValueError('No repair targets to fit EF scales')
        source = anchors.reshape(-1, self.points, 2).float()
        rms = lambda value: value.square().mean(dim=(0, 1)).sqrt()
        scales = {
            'position_scale': rms(source).clamp_min(1.),
            'input_delta_scale': rms(self._steps(source)[:, 1:]).clamp_min(.01),
            'step_scale': rms(self._steps(repairs)[:, 1:]).clamp_min(.01),
            'waypoint_scale': rms(repairs[:, 1:]).clamp_min(.1),
        }
        for name, scale in scales.items():
            if not torch.isfinite(scale).all():
                raise ValueError('EF scale overflow in ' + name)
            getattr(self, name).copy_(scale)

    def _pack(self, anchors, feature, context, valid, validate_inputs):
        self._trajectory_shape(anchors, 'anchors')
        unbatched = anchors.ndim == 3
        if unbatched:
            anchors = anchors.unsqueeze(0)
        batch = anchors.shape[0]
        feature = torch.as_tensor(feature, device=anchors.device, dtype=anchors.dtype)
        if feature.ndim == 1 and feature.shape[0] == self.feature_dim:
            feature = feature.unsqueeze(0).expand(batch, -1)
        if feature.shape != (batch, self.feature_dim):
            raise ValueError('EF feature must be [50] or [B,50]')
        if not isinstance(context, torch.Tensor) or context.ndim != 3:
            raise ValueError('context must have shape [B,L,256]')
        if context.shape[0] != batch or context.shape[-1] != 256 or not context.shape[1]:
            raise ValueError('context must have shape [B,L,256] with nonempty L')
        if context.device != anchors.device or not context.is_floating_point():
            raise ValueError('context and anchors must be floating point on the same device')
        if not isinstance(valid, torch.Tensor) or valid.shape != context.shape[:2]:
            raise ValueError('valid must have shape [B,L]')
        if valid.device != anchors.device:
            raise ValueError('valid and anchors must be on the same device')
        check = self.validate_inputs if validate_inputs is None else bool(validate_inputs)
        if check:
            if not torch.isfinite(anchors).all() or not torch.isfinite(feature).all():
                raise ValueError('EF anchors and features must be finite')
            if not torch.isfinite(context).all():
                raise ValueError('EF context must be finite, including padded tokens')
            if not ((valid == 0) | (valid == 1)).all() or not valid.bool().any(-1).all():
                raise ValueError('Each context must contain at least one valid token')
            scene = feature[:, 0]
            if ((scene != scene.round()) | (scene < 0) | (scene >= self.num_scenes)).any():
                raise ValueError('EF scene must be an integer in [0,4]')
            flags = torch.cat((feature[:, 1:49].reshape(batch, 16, 3)[:, :, 2], feature[:, 49:50]), -1)
            if ((flags < 0) | (flags > 1)).any():
                raise ValueError('EF enterable/expanded flags must lie in [0,1]')
            for name in ('position_scale', 'input_delta_scale', 'step_scale', 'waypoint_scale'):
                scale = getattr(self, name)
                if not torch.isfinite(scale).all() or (scale <= 0).any():
                    raise ValueError('EF scales must be finite and positive: ' + name)
        return anchors, feature, context, valid.bool(), unbatched

    def normalized_feature(self, feature):
        """Scale polygon XY only; input may be [50] or [B,50]."""
        vertices = feature[..., 1:49].reshape(*feature.shape[:-1], 16, 3)
        vertices = torch.cat((vertices[..., :2] / self.position_scale, vertices[..., 2:3]), -1)
        return torch.cat((feature[..., :1], vertices.flatten(-2), feature[..., 49:50]), -1)

    def forward_steps(self, anchors, feature, context, valid, validate_inputs=None):
        """Return ``(metric_step_correction, cumulative_XY_correction)``.

        Batched inputs produce [B,N,T,2]; legacy inputs produce [N,T,2].
        Set validate_inputs=False only for trusted, prevalidated training data
        to avoid value-check synchronization on the accelerator. Shape checks
        always run. No scene/context cache is kept across calls.
        """
        anchors, feature, context, valid, unbatched = self._pack(
            anchors, feature, context, valid, validate_inputs,
        )
        batch, count = anchors.shape[:2]
        normalized_steps = self._steps(anchors) / self.input_delta_scale
        flat_steps = normalized_steps.reshape(batch * count, self.points, 2)
        anchor_embedding = self.anchor(normalized_steps.flatten(-2))
        anchor_embedding = anchor_embedding + self.absolute_anchor((anchors / self.position_scale).flatten(-2))
        anchor_embedding = anchor_embedding + self.local_point_mixer(flat_steps).reshape(batch, count, 256)
        normalized_feature = self.normalized_feature(feature)
        scene_embedding = self.scene(normalized_feature[:, :1]).unsqueeze(1).expand(-1, count, -1)
        polygon_embedding = self.polygon(normalized_feature[:, 1:] + self.position_embedding)
        polygon_embedding = polygon_embedding.unsqueeze(1).expand(-1, count, -1)
        query = self.query_norm(self.fusion(torch.cat((scene_embedding, polygon_embedding, anchor_embedding), -1)))
        ctx = self.context_norm(context)
        attended = self.attention(query, ctx, ctx, key_padding_mask=~valid, need_weights=False)[0]
        decoded = self.post_attention_norm(query + attended)
        decoded = self.output_norm(decoded + self.ffn(decoded))
        raw_steps = self.output(decoded).reshape(batch, count, self.points, 2)
        # Keep metric integration in fp32 under mixed precision, avoiding
        # cumulative bf16/fp16 rounding while preserving the autograd graph.
        metric_dtype = torch.float32 if anchors.dtype in (torch.float16, torch.bfloat16) else anchors.dtype
        raw_steps = raw_steps.to(metric_dtype)
        steps = raw_steps * self.step_scale.to(metric_dtype)
        steps = torch.cat((torch.zeros_like(steps[..., :1, :]), steps[..., 1:, :]), dim=-2)
        correction = steps.cumsum(dim=-2)
        if unbatched:
            return steps[0], correction[0]
        return steps, correction

    def forward(self, anchors, feature, context, valid, validate_inputs=None):
        return self.forward_steps(anchors, feature, context, valid, validate_inputs)[1]
