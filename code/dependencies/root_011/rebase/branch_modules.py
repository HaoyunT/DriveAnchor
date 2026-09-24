# Copyright 2025 Anonymous Rights Holder. All Rights Reserved.
# Extracted from rebase_anchor_fm; see source_manifest.json for attribution/locations.
from __future__ import annotations
import math
import copy
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn.functional as F
from torch import Tensor
from .compat import BaseModule, ModuleConfig, MatMulWrapper, TritonAttention
INVALID_VALUE = -299.8


class MLP(BaseModule):

    def __init__(self, input_channels, output_channels, dropout_rate=0.0, intermediate_channels=None):
        super().__init__()
        self._output_channels = output_channels
        if intermediate_channels is None:
            intermediate_channels = output_channels
        self._mlp = torch.nn.Sequential(torch.nn.Linear(input_channels, intermediate_channels),
                                        torch.nn.LayerNorm(intermediate_channels), torch.nn.GELU(),
                                        torch.nn.Linear(intermediate_channels, output_channels),
                                        torch.nn.Dropout(dropout_rate))

    def forward(self, input_data):
        dims = len(input_data.shape)
        if dims < 4:
            return self._mlp(input_data)
        elif dims == 4:
            b, h, w, c = input_data.shape
            out = self._mlp(input_data.reshape(b, -1, c))
            return out.reshape(b, h, w, self._output_channels)
        elif dims == 5:
            b, h, w, z, c = input_data.shape
            out = self._mlp(input_data.reshape(b, -1, c))
            return out.reshape(b, h, w, z, self._output_channels)
        else:
            raise NotImplementedError('MLP does NOT support input data with 6 or more dimensions')

class MultiHeadAttention(BaseModule):
    _enable_flex_attn = False
    _enable_triton_flash = False
    _flash_attn_head_dim = -1
    _enable_auto_padding = False
    _padding_base = 128
    _block_mask = None

    def __init__(self, num_heads, q_input_channels, kv_input_channels, qk_output_channels, v_output_channels,
                 num_output_channels, dropout_rate=0.1, bias=True, position_embed_q_input_channels=None,
                 position_embed_k_input_channels=None, position_embed_bias=True, add_bias_su=False, add_bias_v=False):
        super().__init__()
        self._num_heads = num_heads
        self._qk_output_channels = qk_output_channels
        self._v_output_channels = v_output_channels
        self._normalize_factor = qk_output_channels**0.5
        self._query_func = torch.nn.Linear(q_input_channels, num_heads * qk_output_channels, bias=bias)
        self._key_func = torch.nn.Linear(kv_input_channels, num_heads * qk_output_channels, bias=bias)
        self._value_func = torch.nn.Linear(kv_input_channels, num_heads * v_output_channels, bias=add_bias_v)
        self._summary_func = torch.nn.Linear(num_heads * v_output_channels, num_output_channels, bias=add_bias_su)
        self._position_embed_q_func = self._build_position_embed_func(position_embed_q_input_channels,
                                                                      qk_output_channels, position_embed_bias)
        self._position_embed_k_func = self._build_position_embed_func(position_embed_k_input_channels,
                                                                      qk_output_channels, position_embed_bias)
        self._attention_matmul = MatMulWrapper()
        self._value_matmul = MatMulWrapper()
        self._attention_dropout = torch.nn.Dropout(dropout_rate)
        self._query_dropout = torch.nn.Dropout(dropout_rate)
        self._flash_attn = TritonAttention.apply

    def _build_position_embed_func(self, position_embed_input_channels, output_channels, bias):
        if position_embed_input_channels is None:
            return None
        return torch.nn.Linear(position_embed_input_channels, self._num_heads * output_channels, bias=bias)

    def _attention(self, query, key, value, attn_mask=None, key_padding_mask=None):
        attention = self._attention_matmul(query / self._normalize_factor, key.transpose(2, 3))
        dtype = query.dtype
        eps = torch.finfo(dtype).eps
        min_float = torch.finfo(attention.dtype).min
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            if self.training:
                attention = attention.masked_fill(attn_mask < eps, min_float)
            else:
                attention += (attn_mask < eps) * (-100.0)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            if self.training:
                attention = attention.masked_fill(key_padding_mask < eps, min_float)
            else:
                attention += (key_padding_mask < eps) * (-100.0)
        softmax = self._attention_dropout(F.softmax(attention, dim=-1))
        output = self._value_matmul(softmax, value)
        return output, attention

    def forward(self, input_q, input_kv, attn_mask=None, average_attn_weight=False, position_embed_q=None,
                position_embed_k=None, key_padding_mask=None):
        batch_size = input_q.size(0)
        len_query = input_q.size(1)
        len_key = input_kv.size(1)
        len_value = input_kv.size(1)
        attn_heads = self._num_heads
        attn_dim = self._qk_output_channels
        padding_base = MultiHeadAttention._padding_base
        # set head dim = flex_attn_head_dim for faster compute
        if len_query > padding_base and len_key > padding_base and MultiHeadAttention._flash_attn_head_dim != -1:
            attn_dim = MultiHeadAttention._flash_attn_head_dim
            attn_heads = attn_heads * (self._qk_output_channels // attn_dim)      
        # Seperate different layers.
        query_linear = self._query_func(input_q)
        # position_embed_q_func and position_embed_q should both be None or not None
        assert not ((self._position_embed_q_func is None) ^ (position_embed_q is None))
        # position_embed_k_func and position_embed_k should both be None or not None
        assert not ((self._position_embed_k_func is None) ^ (position_embed_k is None))
        if self._position_embed_q_func is not None and position_embed_q is not None:
            position_embed_q_linear = self._position_embed_q_func(position_embed_q)
            query_linear = query_linear + position_embed_q_linear
        query_linear = query_linear.view(batch_size, len_query, attn_heads, attn_dim)
        key_linear = self._key_func(input_kv)
        if self._position_embed_k_func is not None and position_embed_k is not None:
            position_embed_k_linear = self._position_embed_k_func(position_embed_k)
            key_linear = key_linear + position_embed_k_linear
        key_linear = key_linear.view(batch_size, len_key, attn_heads, attn_dim)
        value_linear = self._value_func(input_kv).view(batch_size, len_value, attn_heads, attn_dim)
        query_trans = query_linear.transpose(1, 2)
        key_trans = key_linear.transpose(1, 2)
        value_trans = value_linear.transpose(1, 2)

        query_trans, key_trans, value_trans, attn_mask, key_padding_mask, enable_padding, q_padding_size, kv_padding_size = \
            self._do_auto_padding(query_trans, key_trans, value_trans, attn_mask, key_padding_mask)
        # TODO(yangyuanhao02): check whether need to add head_dim == 128 support
        use_flex_attn = self.training and (MultiHeadAttention._block_mask is not None) and query_trans.size(-2) % 128 == 0 and key_trans.size(-2) % 128 == 0 and value_trans.size(-2) % 128 == 0 and attn_dim == 64
        use_flash_attn = self.training and MultiHeadAttention._enable_triton_flash and query_trans.size() == key_trans.size() and query_trans.size() == value_trans.size() \
            and query_trans.size(-2) % 128 == 0 and attn_dim == 64
        if use_flex_attn:
            block_mask = MultiHeadAttention._block_mask
            query_attention = flex_attention(query_trans, key_trans, value_trans, block_mask=block_mask, scale=1.0 / self._normalize_factor)
            attention = None
        elif use_flash_attn:
            query_attention = self._flash_attn(query_trans, key_trans, value_trans, attn_mask, key_padding_mask,
                                               1 / self._normalize_factor)
            attention = None
        else:
            query_attention, attention = self._attention(query_trans, key_trans, value_trans, attn_mask=attn_mask,
                                                         key_padding_mask=key_padding_mask)

        query_attention, attention = self._get_silced_padding_results(query_attention, attention, enable_padding,
                                                                      q_padding_size, kv_padding_size)
        query_attention_trans = query_attention.transpose(1, 2).contiguous().view(batch_size, len_query, -1)
        query_summary = self._query_dropout(self._summary_func(query_attention_trans))

        if use_flex_attn or use_flash_attn:
            # flex / flash attention mode doesn't return attention matrix, using invalid tensor torch.empty(1) instead.
            return query_summary, torch.empty(1)
        else:
            if average_attn_weight:
                attention = attention.sum(dim=1) / self._num_heads
            return query_summary, attention

    def _do_auto_padding(self, query_trans, key_trans, value_trans, attn_mask, key_padding_mask):
        padding_base = MultiHeadAttention._padding_base
        len_q_padding_to = ((query_trans.size(-2) - 1) // padding_base + 1) * padding_base
        q_padding_size = len_q_padding_to - query_trans.size(-2)
        len_kv_padding_to = ((key_trans.size(-2) - 1) // padding_base + 1) * padding_base
        kv_padding_size = len_kv_padding_to - key_trans.size(-2)
        # Now only do padding when seqlen > 128 and head dim == 64
        # TODO(yangyuanhao02): check whether need to add head_dim == 128 support
        enable_padding = len_q_padding_to >= 256 and len_kv_padding_to >= 256 and query_trans.size(-1) == 64

        if MultiHeadAttention._enable_auto_padding and enable_padding:
            if q_padding_size != 0 or kv_padding_size != 0:
                query_padding_tensor = torch.zeros(1, device=query_trans.device, dtype=query_trans.dtype,
                                                   requires_grad=False)
                query_padding_tensor = query_padding_tensor.broadcast_to(query_trans.size(0), query_trans.size(1),
                                                                         q_padding_size, query_trans.size(-1))
                key_value_padding_tensor = torch.zeros(1, device=key_trans.device, dtype=key_trans.dtype,
                                                       requires_grad=False)
                key_value_padding_tensor = key_value_padding_tensor.broadcast_to(key_trans.size(0), key_trans.size(1),
                                                                                 kv_padding_size, key_trans.size(-1))

                if key_padding_mask is not None:
                    key_mask_padding_tensor = torch.zeros(1, device=key_padding_mask.device, requires_grad=False)
                    key_mask_padding_tensor = key_mask_padding_tensor.broadcast_to(key_padding_mask.size(0),
                                                                                   kv_padding_size)
                    key_padding_mask = torch.cat([key_padding_mask, key_mask_padding_tensor], dim=-1)

                if attn_mask is not None:
                    attn_mask_padding_tensor = torch.zeros(1, device=attn_mask.device, requires_grad=False)
                    attn_mask_padding_tensor_kv = attn_mask_padding_tensor.broadcast_to(
                        attn_mask.size(0), query_trans.size(-2), kv_padding_size)
                    attn_mask_padding_tensor_query = attn_mask_padding_tensor.broadcast_to(
                        attn_mask.size(0), q_padding_size, len_kv_padding_to)
                    attn_mask = torch.cat(
                        [torch.cat([attn_mask, attn_mask_padding_tensor_kv], dim=-1), attn_mask_padding_tensor_query],
                        dim=-2)

                if attn_mask is None and key_padding_mask is None:
                    unmask_tensor = torch.ones(1, device=query_trans.device,
                                               requires_grad=False).broadcast_to(query_trans.size(0),
                                                                                 query_trans.size(2), key_trans.size(2))
                    attn_mask_padding_tensor = torch.zeros(1, device=query_trans.device, requires_grad=False)
                    attn_mask_padding_tensor_kv = attn_mask_padding_tensor.broadcast_to(
                        query_trans.size(0), query_trans.size(-2), kv_padding_size)
                    attn_mask_padding_tensor_query = attn_mask_padding_tensor.broadcast_to(
                        query_trans.size(0), q_padding_size, len_kv_padding_to)
                    attn_mask = torch.cat([
                        torch.cat([unmask_tensor, attn_mask_padding_tensor_kv], dim=-1), attn_mask_padding_tensor_query
                    ], dim=-2)

                query_trans = torch.cat([query_trans, query_padding_tensor], dim=-2)
                key_trans = torch.cat([key_trans, key_value_padding_tensor], dim=-2)
                value_trans = torch.cat([value_trans, key_value_padding_tensor], dim=-2)

            return query_trans, key_trans, value_trans, attn_mask, key_padding_mask, enable_padding, q_padding_size, kv_padding_size
        else:
            return query_trans, key_trans, value_trans, attn_mask, key_padding_mask, enable_padding, 0, 0

    def _get_silced_padding_results(self, query_attention, attention=None, enable_padding=False, q_padding_size=0,
                                    kv_padding_size=0):
        if MultiHeadAttention._enable_auto_padding and enable_padding:
            if q_padding_size != 0:
                query_attention = query_attention[:, :, :(-1) * q_padding_size, :]
                if attention is not None:
                    attention = attention[:, :, :(-1) * q_padding_size, :]
            if kv_padding_size != 0 and attention is not None:
                attention = attention[:, :, :, :(-1) * kv_padding_size]
        return query_attention, attention

    @staticmethod
    def set_block_mask(valid_token_cnts, batch, q_len, kv_len, device):
        def pnc_mask(b, h, q_idx, kv_idx):
            return kv_idx < valid_token_cnts[b]
        padding_base = MultiHeadAttention._padding_base
        q_len = (q_len + padding_base - 1) // padding_base * padding_base
        kv_len = (kv_len + padding_base - 1) // padding_base * padding_base
        MultiHeadAttention._block_mask= create_block_mask(
            pnc_mask, batch, None, q_len, kv_len, device=device, _compile=True)

    @staticmethod
    def unset_block_mask():
        MultiHeadAttention._block_mask = None

    @staticmethod
    def set_attention_mode(enable_flex_attn=False, flash_attn_head_dim=64, enable_triton_flash=False, enable_auto_padding=False, padding_base=128):
        if enable_auto_padding:
            print(f"Enable auto padding, auto pad seq_len dimension to multiples of {padding_base}.")
        if enable_flex_attn:
            print(
                    "Enable flex attention. If possible, flex attention will be prioritized for use."
                    +
                    (f"Make sure that seq_len dims of q, k, v are multiples of {padding_base} for use flex attention." if not enable_auto_padding else "")
                )
        elif enable_triton_flash:
                print("Enable triton flash attention." 
                    + 
                    (f"Make sure that seq_len dims of q, k, v are multiples of {padding_base}." if not enable_auto_padding else "")
                )

        MultiHeadAttention._enable_flex_attn = enable_flex_attn
        MultiHeadAttention._enable_triton_flash = enable_triton_flash
        MultiHeadAttention._flash_attn_head_dim = flash_attn_head_dim
        MultiHeadAttention._enable_auto_padding = enable_auto_padding
        MultiHeadAttention._padding_base = padding_base

class TransformerEncoder(BaseModule):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x, src_key_padding_mask):
        output = x
        for layer in self.layers:
            output = layer(output, src_key_padding_mask)
        return output

class FFN(BaseModule):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, d_model)
        self.activation = torch.nn.functional.relu
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(x)

class TransformerLayer(BaseModule):

    def __init__(self, d_model: int, num_heads: int, d_feedforward: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self._attention = MultiHeadAttention(num_heads=num_heads, q_input_channels=d_model,
                                             kv_input_channels=d_model,
                                             qk_output_channels=d_model // num_heads,
                                             v_output_channels=d_model // num_heads,
                                             num_output_channels=d_model)
        self._ffn = FFN(d_model, d_feedforward)

    def forward(self, x, src_key_padding_mask):
        x_norm = self.norm1(x)
        x = x + self._attention(x_norm, x_norm, key_padding_mask=~src_key_padding_mask)[0]
        x = x + self._ffn(self.norm2(x))
        return x

class TimeDimensionFeaturePyramid(BaseModule):

    def __init__(self, input_dim, input_seq_len, output_dim, output_seq_len):
        super().__init__()
        self._output_dim = output_dim
        ratio = input_seq_len // output_seq_len
        # TODO(tangwei52): use nn.ModuleList.
        feature_level1_dim = input_dim * ratio // (2**3)
        feature_level2_dim = feature_level1_dim * 2
        feature_level3_dim = feature_level2_dim * 2
        feature_level4_dim = feature_level3_dim * 2

        self._time_conv1 = torch.nn.Conv1d(
            input_dim, feature_level1_dim, kernel_size=5, stride=2, padding=1
        )
        self._time_conv2 = torch.nn.Conv1d(
            feature_level1_dim, feature_level2_dim, kernel_size=5, stride=2
        )
        self._time_conv3 = torch.nn.Conv1d(
            feature_level2_dim, feature_level3_dim, kernel_size=5, stride=2, padding=1
        )
        self._time_conv4 = torch.nn.Conv1d(
            feature_level3_dim, feature_level4_dim, kernel_size=3, stride=1
        )

        self._feature_mlp1 = MLP(feature_level1_dim, output_dim)
        self._feature_mlp2 = MLP(feature_level2_dim, output_dim)
        self._feature_mlp3 = MLP(feature_level3_dim, output_dim)
        self._feature_mlp4 = MLP(feature_level4_dim, output_dim)

    def forward(self, feature):
        batch_size, num_of_object, seq_len, input_dim = feature.shape
        feature_level1 = self._time_conv1(
            feature.reshape(-1, seq_len, input_dim).transpose(1, 2)
        )
        feature_level2 = self._time_conv2(feature_level1)
        feature_level3 = self._time_conv3(feature_level2)
        feature_level4 = self._time_conv4(feature_level3)

        feature_level1 = self._feature_mlp1(feature_level1.transpose(1, 2))
        feature_level2 = self._feature_mlp2(feature_level2.transpose(1, 2))
        feature_level3 = self._feature_mlp3(feature_level3.transpose(1, 2))
        feature_level4 = self._feature_mlp4(feature_level4.transpose(1, 2))

        feature_fusing = (
            feature_level1.max(1)[0]
            + feature_level2.max(1)[0]
            + feature_level3.max(1)[0]
            + feature_level4.max(1)[0]
        )
        feature_fusing = (
            (feature_fusing / 4)
            .reshape(batch_size, num_of_object, self._output_dim)
            .unsqueeze(2)
        )
        return feature_fusing

class RoadGraphInstanceProjector(BaseModule):
    def __init__(
        self,
        d_model: int,
        instance_wise_feature_dim: int,
        point_wise_feature_dim: int,
        point_num: int,
        instance_global_token_ratio: float,
        point_wise_feature_hidden_dim: int,
    ):
        super().__init__()
        self._instance_wise_feature_dim = instance_wise_feature_dim
        self._point_wise_feature_dim = point_wise_feature_dim
        self._point_num = point_num
        self._instance_global_dim_in_token = int(d_model * instance_global_token_ratio)
        self._instance_point_dim_in_token = d_model - self._instance_global_dim_in_token
        self._point_wise_feature_hidden_dim = point_wise_feature_hidden_dim
        self._instance_wise_feature_projector = MLP(
            self._instance_wise_feature_dim, self._instance_global_dim_in_token
        )
        self._single_point_feature_projector = MLP(
            self._point_wise_feature_dim, self._point_wise_feature_hidden_dim
        )
        self._multiple_points_feature_projector = MLP(
            self._point_wise_feature_hidden_dim * self._point_num, self._instance_point_dim_in_token
        )

    def forward(self, road_graph_instance_feature: torch.Tensor) -> torch.Tensor:
        batch_size, num_of_instances = road_graph_instance_feature.shape[:2]
        # [B, N, 1, instance_wise_feature_dim]
        instance_wise_feature = road_graph_instance_feature[..., : self._instance_wise_feature_dim]
        # [B, N, 1, instance_wise_feature_dim] => [B, N, instance_wise_feature_dim]
        instance_wise_feature = instance_wise_feature.squeeze(-2)
        instance_global_embedding = self._instance_wise_feature_projector(instance_wise_feature)
        # [B, N, 1, point_wise_feature_dim * point_num]
        multiple_points_feature = road_graph_instance_feature[
            ..., self._instance_wise_feature_dim:
        ]
        # [B, N, 1, point_wise_feature_dim * point_num] => [B, N, point_num, point_wise_feature_dim]
        multiple_points_feature = multiple_points_feature.squeeze(-2).reshape(
            batch_size, num_of_instances, self._point_num, self._point_wise_feature_dim
        )
        multiple_points_embedding = self._single_point_feature_projector(
            multiple_points_feature
        ).flatten(-2)
        multiple_points_embedding = self._multiple_points_feature_projector(
            multiple_points_embedding
        )
        instance_embedding = torch.concat(
            [instance_global_embedding, multiple_points_embedding], dim=-1
        )
        return instance_embedding

def is_tensor_abs_close(a: torch.Tensor, b: float, atol=1e-6):
    return torch.abs(a - b) < atol

class PlanningSharedEncoderV6(BaseModule):
    DEFAULT_CONFIG = {
        "d_model": 384,
        "num_heads": 6,
        "num_layers": 6,
        "project_dim": 256,
        # TODO(tangwei52): move feature_config in dataset config.
        "feature_config": {
            "ego_feature_dim": 8,
            "interact_obstacle_feature_dim": 12,
            "static_obstacle_feature_dim": 60,
            # road graph instance config
            "lane_instance_traj_point_num": 50,
            "lane_instance_wise_feature_dim": 5,
            "lane_point_wise_feature_dim": 4,
            "polygon_instance_corner_point_num": 20,
            "polygon_instance_wise_feature_dim": 2,
            "polygon_point_wise_feature_dim": 2,
            "instance_global_token_ratio": 0.2,
            "point_wise_feature_hidden_dim": 64,
            "traffic_light_feature_dim": 10,
            "num_temporal": 40,
            "num_token_per_interact_obstacle": 1,
            "time_skip_interval": 5,
        }
    }

    def __init__(self, encoder_config):
        super().__init__()
        self._encoder_config = ModuleConfig(self, self.DEFAULT_CONFIG, encoder_config)
        self._d_model = self._encoder_config.d_model
        self._ego_feature_dim = self._encoder_config.feature_config.ego_feature_dim
        self._interact_obstacle_feature_dim = self._encoder_config.feature_config.interact_obstacle_feature_dim
        self._static_obstacle_feature_dim = self._encoder_config.feature_config.static_obstacle_feature_dim
        self._traffic_light_feature_dim = self._encoder_config.feature_config.traffic_light_feature_dim
        self._num_temporal = self._encoder_config.feature_config.num_temporal
        self._num_token_per_interact_obstacle = self._encoder_config.feature_config.num_token_per_interact_obstacle
        self._time_skip_interval = self._encoder_config.feature_config.time_skip_interval

        # road graph instance config
        self._instance_global_token_ratio = self._encoder_config.feature_config.instance_global_token_ratio
        self._point_wise_feature_hidden_dim = self._encoder_config.feature_config.point_wise_feature_hidden_dim
        # lane
        self._lane_instance_wise_feature_dim = self._encoder_config.feature_config.lane_instance_wise_feature_dim
        self._lane_point_wise_feature_dim = self._encoder_config.feature_config.lane_point_wise_feature_dim
        self._lane_instance_traj_point_num = self._encoder_config.feature_config.lane_instance_traj_point_num
        # polygon
        self._polygon_instance_wise_feature_dim = self._encoder_config.feature_config.polygon_instance_wise_feature_dim
        self._polygon_point_wise_feature_dim = self._encoder_config.feature_config.polygon_point_wise_feature_dim
        self._polygon_instance_corner_point_num = self._encoder_config.feature_config.polygon_instance_corner_point_num

        self._ego_projector = MLP(self._ego_feature_dim, self._d_model)
        # reduce interact obstacle time dimension, see [internal reference redacted]
        self._interact_obstacle_projector = TimeDimensionFeaturePyramid(
            self._interact_obstacle_feature_dim,
            self._num_temporal,
            self._d_model,
            self._num_token_per_interact_obstacle,
        )
        self._traffic_light_projector = MLP(self._traffic_light_feature_dim, self._d_model)
        self._static_obstacle_projector = MLP(self._static_obstacle_feature_dim, self._d_model)
        self._lane_instance_projector = RoadGraphInstanceProjector(
            self._d_model,
            self._lane_instance_wise_feature_dim,
            self._lane_point_wise_feature_dim,
            self._lane_instance_traj_point_num,
            self._instance_global_token_ratio,
            self._point_wise_feature_hidden_dim,
        )
        self._polygon_instance_projector = RoadGraphInstanceProjector(
            self._d_model,
            self._polygon_instance_wise_feature_dim,
            self._polygon_point_wise_feature_dim,
            self._polygon_instance_corner_point_num,
            self._instance_global_token_ratio,
            self._point_wise_feature_hidden_dim,
        )

        layer = TransformerLayer(self._d_model, self._encoder_config.num_heads, self._d_model * 4)
        self._transformer_encoder = TransformerEncoder(layer, self._encoder_config.num_layers)
        self._output_projector = MLP(self._d_model, self._encoder_config.project_dim)

    def forward(self, ego_feature: Tensor, interact_obs_feature: Tensor, static_obs_feature: Tensor,
                lane_instance_feature: Tensor, polygon_instance_feature: Tensor, traffic_light_feature: Tensor):
        """
        Forward pass for the PlanningSharedEncoderV6.

        Args:
            ego_feature (torch.FloatTensor): 
                A tensor of shape [B, 1, num_time, 8] representing the ego feature.
            interact_obs_feature (torch.FloatTensor): 
                A tensor of shape [B, num_interact_obs, num_time, 12] representing the interact obstacle feature.
            static_obs_feature (torch.FloatTensor): 
                A tensor of shape [B, num_static_obs, 1, 60] representing the static obstacle feature.
            lane_instance_feature (torch.FloatTensor): 
                A tensor of shape [B, num_road_graph, 1, 205] representing the lane instance feature.
            polygon_instance_feature (torch.FloatTensor): 
                A tensor of shape [B, num_road_graph, 1, 42] representing the polygon instance feature.
            traffic_light_feature (torch.FloatTensor): 
                A tensor of shape [B, num_traffic_light, num_time, 10] representing the traffic light feature.

        Returns:
            encoded_feature (torch.FloatTensor): 
                A tensor of shape [B, num_tokens, 1, D] representing the encoded features.
            encoded_feature_mask (torch.FloatTensor): 
                A tensor of shape [B, num_tokens] representing the mask for the encoded features.
        """

        batch_size = ego_feature.shape[0]
        # [batch, 1, time_steps, feature_dim] => [batch, time_steps/skip_interval, d_model]
        ego_feature = ego_feature[:, :, ::self._time_skip_interval, :]
        ego_embedding = self._ego_projector(ego_feature).view(batch_size, -1, self._d_model)
        
        # use TimeDimensionFeaturePyramid for time reduce, do not downsample.
        # [batch, num_agents, time_steps, feature_dim]
        interact_obs_embedding = self._interact_obstacle_projector(interact_obs_feature).view(
            batch_size, -1, self._d_model)

        # [batch, num_obs, 1, feature_dim] => [batch, num_obs, d_model]
        static_obs_embedding = self._static_obstacle_projector(static_obs_feature).view(
            batch_size, -1, self._d_model)

        # [B, N, 1, D] => [B, N, d_model]
        lane_instance_embedding = self._lane_instance_projector(lane_instance_feature)
        polygon_instance_embedding = self._polygon_instance_projector(polygon_instance_feature)

        # [batch, num_segments, 1, feature_dim] => [batch, num_segments, d_model]
        # road_graph_embedding = self._road_graph_segment_projector(road_graph_feature).view(
        #     batch_size, -1, self._d_model)

        # [batch, num_lights, time_steps, feature_dim] => [batch, num_lights*time_steps/skip_interval, d_model]
        traffic_light_feature = traffic_light_feature[:, :, ::self._time_skip_interval, :]
        traffic_light_embedding = self._traffic_light_projector(traffic_light_feature).view(
            batch_size, -1, self._d_model)

        # [batch, num_tokens, d_model]
        flat_embedding = torch.concat([
            ego_embedding, interact_obs_embedding, static_obs_embedding, lane_instance_embedding,
            polygon_instance_embedding, traffic_light_embedding
        ], dim=1)

        invalid_lane_instance_feature_mask = is_tensor_abs_close(
            lane_instance_feature[..., self._lane_instance_wise_feature_dim].squeeze(
                -2
            ),
            -300.0,
            1e-2,
        )
        invalid_polygon_instance_feature_mask = is_tensor_abs_close(
            polygon_instance_feature[
                ..., self._polygon_instance_wise_feature_dim
            ].squeeze(-2),
            -300.0,
            1e-2,
        )

        invalid_masks = [
            (ego_feature.view(batch_size, -1, ego_feature.shape[-1]) < INVALID_VALUE).any(dim=-1),
            ((interact_obs_feature < INVALID_VALUE).any(dim=-1)).all(dim=-1),
            (static_obs_feature.view(batch_size, -1, static_obs_feature.shape[-1]) <
             INVALID_VALUE).any(dim=-1),
            # (road_graph_feature.view(batch_size, -1, road_graph_feature.shape[-1]) <
            #  INVALID_VALUE).any(dim=-1),
            invalid_lane_instance_feature_mask.any(dim=-1),
            invalid_polygon_instance_feature_mask.any(dim=-1),
            (traffic_light_feature.view(batch_size, -1, traffic_light_feature.shape[-1]) <
             INVALID_VALUE).any(dim=-1)
        ]
        # [batch, num_tokens]
        src_key_padding_mask = torch.concat(invalid_masks, dim=1).detach()
        # flat_embedding = self._pos_encoder(flat_embedding)
        # [batch, num_tokens, 1, d_model]
        if MultiHeadAttention._enable_flex_attn and self.training:
            flat_embedding, valid_token_cnts, reverse_indices = rearrange_x_by_mask(flat_embedding, ~src_key_padding_mask)
            batch, seqlen = src_key_padding_mask.shape
            MultiHeadAttention.set_block_mask(valid_token_cnts, batch, q_len=seqlen, kv_len=seqlen, device=flat_embedding.device)
            encoded_feature = self._transformer_encoder(flat_embedding, src_key_padding_mask)
            encoded_feature = restore_x_by_reverse_indices(encoded_feature, reverse_indices)
            MultiHeadAttention.unset_block_mask()
        else:
            encoded_feature = self._transformer_encoder(flat_embedding, src_key_padding_mask)
        encoded_feature = encoded_feature.unsqueeze(2)

        # [batch, num_tokens, 1, project_dim]
        encoded_feature = self._output_projector(encoded_feature)
        return encoded_feature, src_key_padding_mask

_LOGGER_PREFIX = "generative_decoder"

_PADDING_TOLERANCE = 1.0

_EPSILON = 1e-6

_MIN_NORM_STD = 1e-6

_FM_VORONOI_NOISE_STD = 1.0

_FM_VORONOI_INSCRIBED_BALL_RADIUS_RATIO = 0.49

_FM_VORONOI_PROPOSAL_BALL_RADIUS_RATIO = 1.0

_FM_VORONOI_SAMPLING_MAX_ATTEMPTS = 8

_FM_VORONOI_NEIGHBOR_SEARCH_CHUNK_SIZE = 8

_TIME_MONITOR_BINS = 10

_POSITION_CHANNELS = 2

_HEADING_CHANNELS = 2

_HEADING_COS_CENTER = 1.0

_ENERGY_FEATURE_DIM = 50

_ENERGY_SCENE_TYPE_INDEX = 0

_ENERGY_POINT_OFFSET = 1

_ENERGY_NUM_POINTS = 16

_ENERGY_POINT_CHANNELS = 3

_ENERGY_IS_EXPANDED_INDEX = 49

_ENERGY_NUM_SCENE_TYPES = 5

PREDICT_X = "x"

PREDICT_V = "v"

CONDITION_CROSS_ATTN = "cross_attn"

CONDITION_ADALN = "adaln"

CONDITION_BOTH = "both"

PARAMETERIZATION_DELTA = "delta"

PARAMETERIZATION_ABSOLUTE = "absolute"

PARAMETERIZATION_MODES = (PARAMETERIZATION_DELTA, PARAMETERIZATION_ABSOLUTE)

TOKENIZE_WHOLE = "whole"

TOKENIZE_PATCH = "patch"

TOKENIZE_PER_POINT = "per_point"

TOKENIZE_MODES = (TOKENIZE_WHOLE, TOKENIZE_PATCH, TOKENIZE_PER_POINT)

SOURCE_MODE_GAUSSIAN = "gaussian"

SOURCE_MODE_ANCHOR = "anchor"

SOURCE_MODES = (SOURCE_MODE_GAUSSIAN, SOURCE_MODE_ANCHOR)

EVAL_SOURCE_GT_TOPK = "gt_topk"

EVAL_SOURCE_ALL_ANCHOR = "all_anchor"

EVAL_SOURCE_GAUSSIAN = "gaussian"

EVAL_SOURCE_GUIDE_FLOW = "guide_flow"

EVAL_SOURCE_MODES = (
    EVAL_SOURCE_GT_TOPK,
    EVAL_SOURCE_ALL_ANCHOR,
    EVAL_SOURCE_GAUSSIAN,
    EVAL_SOURCE_GUIDE_FLOW,
)

TIME_CONDITION_FULL = "full"

TIME_CONDITION_CONSTANT = "constant"

TIME_CONDITION_MODES = (TIME_CONDITION_FULL, TIME_CONDITION_CONSTANT)

GENERATION_MODE_LEGACY_LINEAR_V1 = "legacy_linear_v1"

GENERATION_MODE_DIT_VP_SDE_X0_DPMPP_V1 = "dit_vp_sde_x0_dpmpp_v1"

GENERATION_MODES = (
    GENERATION_MODE_LEGACY_LINEAR_V1,
    GENERATION_MODE_DIT_VP_SDE_X0_DPMPP_V1,
)

_GENERATION_MODE_IDS = {
    GENERATION_MODE_LEGACY_LINEAR_V1: 0,
    GENERATION_MODE_DIT_VP_SDE_X0_DPMPP_V1: 1,
}

_VP_BETA_MIN = 0.1

_VP_BETA_MAX = 20.0

_VP_TIME_MIN = 1e-3

TRAJECTORY_MODE_CUSTOM = "custom"

TRAJECTORY_MODE_ABSOLUTE_WHOLE = "absolute_whole"

TRAJECTORY_MODE_DELTA_PER_POINT = "delta_per_point"

TRAJECTORY_MODE_SPECS = {
    TRAJECTORY_MODE_ABSOLUTE_WHOLE: (
        PARAMETERIZATION_ABSOLUTE,
        TOKENIZE_WHOLE,
    ),
    TRAJECTORY_MODE_DELTA_PER_POINT: (
        PARAMETERIZATION_DELTA,
        TOKENIZE_PER_POINT,
    ),
}

TRAJECTORY_MODES = (TRAJECTORY_MODE_CUSTOM, *TRAJECTORY_MODE_SPECS)

_PARAMETERIZATION_IDS = {
    PARAMETERIZATION_ABSOLUTE: 0,
    PARAMETERIZATION_DELTA: 1,
}

_TOKENIZE_MODE_IDS = {
    TOKENIZE_WHOLE: 0,
    TOKENIZE_PATCH: 1,
    TOKENIZE_PER_POINT: 2,
}

HEADING_SOURCE_DERIVED = "derived"

HEADING_SOURCE_LABEL = "label"

HEADING_SOURCES = (HEADING_SOURCE_DERIVED, HEADING_SOURCE_LABEL)

_HEADING_LABEL_POINT_DIM = 4

_HEADING_LABEL_OFFSET = 2

LOSS_WEIGHT_X = "x"

LOSS_WEIGHT_V_EQUIVALENT = "v_equivalent"

LOSS_WEIGHT_MODES = (
    LOSS_WEIGHT_X,
    LOSS_WEIGHT_V_EQUIVALENT,
)

_NAMED_REPRESENTATIONS = {
    (PARAMETERIZATION_DELTA, TOKENIZE_PER_POINT, True):
        "variant 1: per-point token, differential (dx, dy, cos, sin)",
    (PARAMETERIZATION_DELTA, TOKENIZE_WHOLE, True):
        "variant 2: whole-trajectory token, differential (dx, dy, cos, sin)",
    (PARAMETERIZATION_ABSOLUTE, TOKENIZE_WHOLE, False):
        "variant 3: whole-trajectory token, absolute waypoints (x, y)",
    (PARAMETERIZATION_ABSOLUTE, TOKENIZE_PATCH, False):
        "variant 4: patch tokens, absolute waypoints (x, y)",
}

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(-2)) + shift.unsqueeze(-2)

def _onnx_multihead_self_attention(
    query: torch.Tensor,
    attention: torch.nn.MultiheadAttention,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run self-attention without PyTorch's unsupported ONNX fastpath."""
    query, key, value = torch.nn.functional.linear(
        query, attention.in_proj_weight, attention.in_proj_bias
    ).chunk(3, dim=-1)
    batch_size, num_tokens, _ = query.shape

    def split_heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(
            batch_size,
            num_tokens,
            attention.num_heads,
            attention.head_dim,
        ).transpose(1, 2)

    query = split_heads(query)
    key = split_heads(key)
    value = split_heads(value)
    logits = torch.matmul(query, key.transpose(-2, -1))
    logits = logits * (attention.head_dim ** -0.5)
    if attention_mask is not None:
        logits = logits.masked_fill(
            attention_mask.unsqueeze(0).unsqueeze(0),
            torch.finfo(logits.dtype).min,
        )
    weights = torch.softmax(logits, dim=-1)
    weights = torch.nn.functional.dropout(
        weights, p=attention.dropout, training=attention.training
    )
    attended = torch.matmul(weights, value).transpose(1, 2).reshape(
        batch_size, num_tokens, attention.embed_dim
    )
    return attention.out_proj(attended)

def detached_integral(velocity: torch.Tensor, detach_window_size: int) -> torch.Tensor:
    """Integrate over time while truncating gradients to the last W steps."""
    cumulative = torch.cumsum(velocity, dim=-2)
    num_points = velocity.shape[-2]
    if detach_window_size is None or detach_window_size <= 0 or detach_window_size >= num_points:
        return cumulative
    window = detach_window_size
    zeros = torch.zeros_like(cumulative[..., :window, :])
    prefix = torch.cat([zeros, cumulative[..., :-window, :]], dim=-2)
    return prefix.detach() + (cumulative - prefix)

def _tensor_diagnostics(tensor: torch.Tensor) -> Tuple[float, float, float]:
    """Return detached maximum, RMS, and non-finite count."""
    with torch.no_grad():
        finite = torch.isfinite(tensor)
        nonfinite = float((~finite).sum().item())
        float_tensor = tensor.float()
        safe = torch.where(finite, float_tensor, torch.zeros_like(float_tensor))
        finite_count = finite.sum().clamp_min(1)
        rms = (safe.square().sum() / finite_count).sqrt()
        return float(safe.abs().max().item()), float(rms.item()), nonfinite

def _add_tensor_diagnostics(
    metrics: Dict[str, float], prefix: str, tensor: torch.Tensor
) -> None:
    absmax, rms, nonfinite = _tensor_diagnostics(tensor)
    metrics[f"monitor/{prefix}_absmax"] = absmax
    metrics[f"monitor/{prefix}_rms"] = rms
    metrics[f"monitor/{prefix}_nonfinite"] = nonfinite

class TimestepEmbedder(BaseModule):
    """Sinusoidal embedding of the flow time."""

    def __init__(self, hidden_dim: int, frequency_embedding_dim: int = 256, max_period: float = 100.0):
        super().__init__()
        self._frequency_embedding_dim = frequency_embedding_dim
        self._max_period = max_period
        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

    def forward(self, flow_time: torch.Tensor) -> torch.Tensor:
        half = self._frequency_embedding_dim // 2
        frequencies = torch.exp(
            -math.log(self._max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=flow_time.device)
            / half
        )
        args = flow_time.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self._frequency_embedding_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self._mlp(embedding.to(dtype=self._mlp[0].weight.dtype))

class ContextConditionEmbedder(BaseModule):
    """Build an adaLN condition with masked pooling and LayerNorm."""

    def __init__(self, context_dim: int, hidden_dim: int):
        super().__init__()
        self._proj = MLP(context_dim, hidden_dim)
        self._norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, context_feature: torch.Tensor, context_valid_mask: torch.Tensor) -> torch.Tensor:
        """Pool context [B, tokens, D] with a valid-token mask [B, tokens]."""
        weight = context_valid_mask.unsqueeze(-1).to(dtype=context_feature.dtype)
        pooled = (context_feature * weight).sum(dim=-2) / weight.sum(dim=-2).clamp_min(1.0)
        return self._norm(self._proj(pooled))

class GuideFlowEnergyConditionEmbedder(BaseModule):
    """Embed the typed corridor feature into a normalized adaLN condition."""

    # (x, y), enterable_mask and validity for each of the 16 points, laid out as
    # the left boundary followed by the right one.
    _BOUNDARY_CHANNELS = _ENERGY_NUM_POINTS * (_POSITION_CHANNELS + 2)
    # is_expanded and the whole-feature validity flag.
    _SCALAR_CHANNELS = 2

    def __init__(
        self,
        hidden_dim: int,
        position_scale: List[float],
        padding_value: float,
    ):
        super().__init__()
        self._scene_type_embedding = torch.nn.Embedding(
            _ENERGY_NUM_SCENE_TYPES, hidden_dim
        )
        self._boundary_proj = MLP(self._BOUNDARY_CHANNELS, hidden_dim)
        self._scalar_proj = MLP(self._SCALAR_CHANNELS, hidden_dim)
        self._norm = torch.nn.LayerNorm(hidden_dim)
        self._padding_threshold = padding_value + _PADDING_TOLERANCE
        self.register_buffer(
            "_position_scale",
            torch.tensor(position_scale, dtype=torch.float32).view(1, 1, -1),
            persistent=False,
        )
        # Match the initial scale of the normalized projection branches.
        torch.nn.init.normal_(self._scene_type_embedding.weight, std=0.02)

    def forward(
        self, energy_feature: torch.Tensor, collect_monitor: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return the embedded condition and, when asked, scalar diagnostics."""
        flat = energy_feature.reshape(energy_feature.shape[0], -1)
        if flat.shape[-1] != _ENERGY_FEATURE_DIM:
            raise ValueError(
                f"guide flow energy feature must hold {_ENERGY_FEATURE_DIM} channels, "
                f"got {tuple(energy_feature.shape)}"
            )
        dtype = self._scene_type_embedding.weight.dtype
        flat = flat.to(dtype=dtype)

        # The scene-type slot identifies an entirely padded feature.
        feature_valid = flat[:, _ENERGY_SCENE_TYPE_INDEX] > self._padding_threshold

        # [batch, 16, 3] of (x, y, enterable_mask); the writer interleaves them.
        point_block = flat[
            :,
            _ENERGY_POINT_OFFSET : _ENERGY_POINT_OFFSET
            + _ENERGY_NUM_POINTS * _ENERGY_POINT_CHANNELS,
        ].reshape(-1, _ENERGY_NUM_POINTS, _ENERGY_POINT_CHANNELS)
        points = point_block[..., :_POSITION_CHANNELS]
        enterable = point_block[..., _POSITION_CHANNELS]
        point_valid = (points > self._padding_threshold).all(dim=-1) & feature_valid.unsqueeze(-1)
        # Remove padding before coordinate scaling.
        scale = self._position_scale.to(dtype=dtype)
        points = torch.where(point_valid.unsqueeze(-1), points, torch.zeros_like(points)) / scale
        enterable = torch.where(point_valid, enterable, torch.zeros_like(enterable))

        # Preserve the exporter's fixed point ordering in the dense projection.
        boundary = torch.cat(
            [
                points.reshape(points.shape[0], -1),
                enterable,
                point_valid.to(dtype=dtype),
            ],
            dim=-1,
        )

        is_expanded = flat[:, _ENERGY_IS_EXPANDED_INDEX]
        is_expanded = torch.where(feature_valid, is_expanded, torch.zeros_like(is_expanded))
        scalars = torch.cat(
            [
                is_expanded.unsqueeze(-1),
                # Distinguish missing features from real all-zero corridors.
                (~feature_valid).to(dtype=dtype).unsqueeze(-1),
            ],
            dim=-1,
        )

        # Clamp before embedding lookup; missing features use LANE_KEEP.
        scene_type = (
            flat[:, _ENERGY_SCENE_TYPE_INDEX].round().long().clamp(0, _ENERGY_NUM_SCENE_TYPES - 1)
        )
        scene_type = torch.where(feature_valid, scene_type, torch.zeros_like(scene_type))
        condition = self._norm(
            self._boundary_proj(boundary)
            + self._scalar_proj(scalars)
            + self._scene_type_embedding(scene_type)
        )

        monitor: Dict[str, float] = {}
        # Each entry below forces a GPU->CPU sync, so skip the whole block on
        # steps that discard it.
        if collect_monitor:
            _add_tensor_diagnostics(monitor, "energy_condition", condition)
            _add_tensor_diagnostics(monitor, "energy_boundary", boundary)
            with torch.no_grad():
                # Emit additive counts for distributed reduction.
                monitor["monitor/energy_valid_count_mean"] = float(feature_valid.sum().item())
                monitor["monitor/energy_batch_mean"] = float(flat.shape[0])
                monitor["monitor/energy_expanded_count_mean"] = float((is_expanded > 0.5).sum().item())
                monitor["monitor/energy_blocked_point_count_mean"] = float(
                    (point_valid & (enterable < 0.5)).sum().item()
                )
                monitor["monitor/energy_point_valid_count_mean"] = float(point_valid.sum().item())
        return condition, monitor

class GenerativeDecoderDiTBlock(BaseModule):
    """DiT block with adaLN-zero and optional self/cross attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        enable_self_attn: bool = True,
        enable_cross_attn: bool = True,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self._enable_self_attn = enable_self_attn
        self._enable_cross_attn = enable_cross_attn

        if enable_self_attn:
            self._self_attn_norm = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            self._self_attn = torch.nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
        if enable_cross_attn:
            self._cross_attn_norm = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            self._cross_attn = torch.nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
        self._ffn_norm = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self._ffn = MLP(
            hidden_dim, hidden_dim, dropout_rate=dropout, intermediate_channels=int(hidden_dim * mlp_ratio)
        )
        self._ada_ln_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 9 * hidden_dim, bias=True),
        )

    def prepare_cross_attention_cache(
        self,
        context_feature: torch.Tensor,
        context_valid_mask: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Project scene K/V once, retaining the unexpanded scene batch."""
        if not self._enable_cross_attn:
            return None
        attention = self._cross_attn
        embed_dim = attention.embed_dim
        kv_weight = attention.in_proj_weight[embed_dim:]
        kv_bias = (
            attention.in_proj_bias[embed_dim:]
            if attention.in_proj_bias is not None
            else None
        )
        key, value = torch.nn.functional.linear(
            context_feature, kv_weight, kv_bias
        ).chunk(2, dim=-1)
        batch_size, num_tokens, _ = key.shape
        key = key.reshape(
            batch_size, num_tokens, attention.num_heads, attention.head_dim
        ).transpose(1, 2)
        value = value.reshape(
            batch_size, num_tokens, attention.num_heads, attention.head_dim
        ).transpose(1, 2)
        return key, value, context_valid_mask.bool()

    def _cached_cross_attention(
        self,
        query: torch.Tensor,
        cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Run ONNX-compatible attention against cached scene K/V."""
        key, value, context_valid_mask = cache
        attention = self._cross_attn
        embed_dim = attention.embed_dim
        query_bias = (
            attention.in_proj_bias[:embed_dim]
            if attention.in_proj_bias is not None
            else None
        )
        query = torch.nn.functional.linear(
            query, attention.in_proj_weight[:embed_dim], query_bias
        )

        scene_batch = key.shape[0]
        trajectory_batch, num_queries, _ = query.shape
        if (
            not torch.jit.is_tracing()
            and trajectory_batch % scene_batch != 0
        ):
            raise ValueError(
                "trajectory batch must be a multiple of scene batch for cached "
                f"cross attention, got trajectory={trajectory_batch}, scene={scene_batch}"
            )
        num_samples = trajectory_batch // scene_batch
        query = query.reshape(
            scene_batch,
            num_samples,
            num_queries,
            attention.num_heads,
            attention.head_dim,
        ).permute(0, 1, 3, 2, 4)

        # K/V broadcast over the sample dimension.
        logits = torch.matmul(query, key.unsqueeze(1).transpose(-2, -1))
        logits = logits * (attention.head_dim ** -0.5)
        valid = context_valid_mask[:, None, None, None, :]
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        # Make the all-masked edge case finite and semantically empty.
        has_valid_context = valid.any(dim=-1, keepdim=True)
        # Avoid exporting a full attention-shaped zero constant.
        weights = weights * has_valid_context.to(dtype=weights.dtype)
        weights = torch.nn.functional.dropout(
            weights, p=attention.dropout, training=self.training
        )
        attended = torch.matmul(weights, value.unsqueeze(1))
        attended = attended.permute(0, 1, 3, 2, 4).reshape(
            trajectory_batch, num_queries, embed_dim
        )
        return attention.out_proj(attended)

    def _onnx_self_attention(self, query: torch.Tensor) -> torch.Tensor:
        return _onnx_multihead_self_attention(query, self._self_attn)

    def forward(
        self,
        trajectory_feature: torch.Tensor,
        context_feature: Optional[torch.Tensor],
        context_valid_mask: Optional[torch.Tensor],
        condition: torch.Tensor,
        cross_attention_cache: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the updated tokens and the raw adaLN modulation for monitoring."""
        modulation = self._ada_ln_modulation(condition)
        (
            shift_self, scale_self, gate_self,
            shift_cross, scale_cross, gate_cross,
            shift_ffn, scale_ffn, gate_ffn,
        ) = modulation.chunk(9, dim=-1)

        if self._enable_self_attn:
            normed = modulate(
                self._self_attn_norm(trajectory_feature), shift_self, scale_self
            )
            if torch.onnx.is_in_onnx_export():
                attended = self._onnx_self_attention(normed)
            else:
                attended = self._self_attn(
                    normed, normed, normed, need_weights=False
                )[0]
            trajectory_feature = trajectory_feature + gate_self.unsqueeze(-2) * attended

        if self._enable_cross_attn:
            normed = modulate(
                self._cross_attn_norm(trajectory_feature), shift_cross, scale_cross
            )
            if cross_attention_cache is None:
                # nn.MultiheadAttention masks out entries that are True.
                attended = self._cross_attn(
                    normed, context_feature, context_feature,
                    key_padding_mask=~context_valid_mask.bool(), need_weights=False,
                )[0]
            else:
                attended = self._cached_cross_attention(
                    normed, cross_attention_cache
                )
            trajectory_feature = trajectory_feature + gate_cross.unsqueeze(-2) * attended

        normed = modulate(self._ffn_norm(trajectory_feature), shift_ffn, scale_ffn)
        trajectory_feature = trajectory_feature + gate_ffn.unsqueeze(-2) * self._ffn(normed)
        return trajectory_feature, modulation

class GenerativeDecoderFinalLayer(BaseModule):
    """adaLN-zero output head."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self._norm = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self._proj = torch.nn.Linear(hidden_dim, output_dim, bias=True)
        self._ada_ln_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )

    def forward(self, trajectory_feature: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self._ada_ln_modulation(condition).chunk(2, dim=-1)
        return self._proj(modulate(self._norm(trajectory_feature), shift, scale))

class GenerativeDecoderLocalPointMixer(BaseModule):
    """Mix neighbouring trajectory points into a whole-token residual."""

    def __init__(
        self,
        num_points: int,
        trajectory_dim: int,
        local_dim: int,
        num_heads: int,
        radius: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._input_projector = torch.nn.Linear(trajectory_dim, local_dim)
        self._position_embedding = torch.nn.Embedding(num_points, local_dim)
        self._norm = torch.nn.LayerNorm(
            local_dim, elementwise_affine=False, eps=1e-6
        )
        self._attention = torch.nn.MultiheadAttention(
            local_dim, num_heads, dropout=dropout, batch_first=True
        )
        self._output_projector = torch.nn.Linear(local_dim, output_dim)

        point_index = torch.arange(num_points)
        attention_mask = (
            point_index.unsqueeze(0) - point_index.unsqueeze(1)
        ).abs() > radius
        self.register_buffer(
            "_attention_mask", attention_mask, persistent=False
        )

        torch.nn.init.normal_(self._position_embedding.weight, std=0.02)
        torch.nn.init.constant_(self._output_projector.weight, 0.0)
        torch.nn.init.constant_(self._output_projector.bias, 0.0)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        point_feature = (
            self._input_projector(trajectory)
            + self._position_embedding.weight.unsqueeze(0)
        )
        normed = self._norm(point_feature)
        if torch.onnx.is_in_onnx_export():
            attended = _onnx_multihead_self_attention(
                normed, self._attention, self._attention_mask
            )
        else:
            attended = self._attention(
                normed,
                normed,
                normed,
                attn_mask=self._attention_mask,
                need_weights=False,
            )[0]
        point_feature = point_feature + attended
        return self._output_projector(point_feature.mean(dim=1)).unsqueeze(1)

class GenerativeDecoderDenoiser(BaseModule):
    """Predict clean trajectories or flow velocity over configurable tokens."""

    def __init__(self, config, trajectory_dim: int):
        super().__init__()
        hidden_dim = config.project_dim
        self._num_trajectory_points = config.num_trajectory_points
        self._trajectory_dim = trajectory_dim
        self._tokenize_mode = config.trajectory_tokenize_mode
        self._condition_mode = config.condition_inject_mode
        if self._condition_mode not in (CONDITION_CROSS_ATTN, CONDITION_ADALN, CONDITION_BOTH):
            raise ValueError(f"Unknown condition_inject_mode: {self._condition_mode}")
        self._patch_size = self._resolve_patch_size(config)
        self._num_tokens = self._num_trajectory_points // self._patch_size
        single_token = self._num_tokens == 1
        use_cross_attn = self._condition_mode in (CONDITION_CROSS_ATTN, CONDITION_BOTH)
        self._use_context_condition = self._condition_mode in (CONDITION_ADALN, CONDITION_BOTH)

        token_dim = self._patch_size * self._trajectory_dim
        self._input_projector = MLP(token_dim, hidden_dim)
        self._local_point_mixer = (
            GenerativeDecoderLocalPointMixer(
                num_points=self._num_trajectory_points,
                trajectory_dim=self._trajectory_dim,
                local_dim=config.local_point_mixer_dim,
                num_heads=config.local_point_mixer_num_heads,
                radius=config.local_point_mixer_radius,
                output_dim=hidden_dim,
                dropout=config.dropout,
            )
            if config.enable_local_point_mixer
            else None
        )
        self._point_embedding = (
            None if single_token else torch.nn.Embedding(self._num_tokens, hidden_dim)
        )
        self._time_embedder = TimestepEmbedder(hidden_dim)
        # T-free mode: replace the incoming flow time with a constant before
        # embedding, so the denoiser output does not depend on t.
        self._time_condition_mode = config.time_condition_mode
        self._time_condition_constant = float(config.time_condition_constant)
        self._context_projector = MLP(config.project_dim, hidden_dim) if use_cross_attn else None
        self._context_condition_embedder = (
            ContextConditionEmbedder(config.project_dim, hidden_dim)
            if self._use_context_condition
            else None
        )
        # The corridor always enters through adaLN.
        self._enable_energy_condition = bool(config.enable_energy_condition)
        self._energy_condition_embedder = (
            GuideFlowEnergyConditionEmbedder(
                hidden_dim,
                list(config.energy_position_scale),
                float(config.padding_value),
            )
            if self._enable_energy_condition
            else None
        )
        # Scale the corridor branch before combining it with time conditioning.
        self._energy_condition_weight = float(config.energy_condition_weight)
        self._blocks = torch.nn.ModuleList(
            [
                GenerativeDecoderDiTBlock(
                    hidden_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.dropout,
                    enable_self_attn=not single_token,
                    enable_cross_attn=use_cross_attn,
                )
                for _ in range(config.depth)
            ]
        )
        self._final_layer = GenerativeDecoderFinalLayer(hidden_dim, token_dim)
        self._initialize_weights()

    def _resolve_patch_size(self, config) -> int:
        mode = self._tokenize_mode
        num_points = self._num_trajectory_points
        if mode == TOKENIZE_WHOLE:
            return num_points
        if mode == TOKENIZE_PER_POINT:
            return 1
        if mode == TOKENIZE_PATCH:
            patch_size = int(config.trajectory_patch_size)
            if patch_size <= 0 or num_points % patch_size != 0:
                raise ValueError(
                    f"trajectory_patch_size={patch_size} must be a positive divisor of "
                    f"num_trajectory_points={num_points}"
                )
            return patch_size
        raise ValueError(
            f"Unknown trajectory_tokenize_mode: {mode}, expected one of {TOKENIZE_MODES}"
        )

    def _initialize_weights(self) -> None:
        if self._point_embedding is not None:
            torch.nn.init.normal_(self._point_embedding.weight, std=0.02)
        torch.nn.init.normal_(self._time_embedder._mlp[0].weight, std=0.02)
        torch.nn.init.normal_(self._time_embedder._mlp[2].weight, std=0.02)
        for block in self._blocks:
            torch.nn.init.constant_(block._ada_ln_modulation[-1].weight, 0.0)
            torch.nn.init.constant_(block._ada_ln_modulation[-1].bias, 0.0)
        torch.nn.init.constant_(self._final_layer._ada_ln_modulation[-1].weight, 0.0)
        torch.nn.init.constant_(self._final_layer._ada_ln_modulation[-1].bias, 0.0)
        torch.nn.init.constant_(self._final_layer._proj.weight, 0.0)
        torch.nn.init.constant_(self._final_layer._proj.bias, 0.0)

    def _tokenize(self, trajectory: torch.Tensor) -> torch.Tensor:
        return trajectory.reshape(trajectory.shape[0], self._num_tokens, -1)

    def _detokenize(self, token_output: torch.Tensor) -> torch.Tensor:
        return token_output.reshape(
            token_output.shape[0], self._num_trajectory_points, self._trajectory_dim
        )

    def prepare_context_cache(
        self,
        context_feature: torch.Tensor,
        context_valid_mask: torch.Tensor,
        energy_feature: Optional[torch.Tensor] = None,
        collect_monitor: bool = False,
    ) -> Tuple[
        Optional[torch.Tensor],
        List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        Dict[str, float],
    ]:
        """Cache scene-only projections shared by every trajectory sample.

        Also returns the energy diagnostics, because the cached path is the only
        place the corridor condition is embedded: dropping them here would make
        the emitted metric keys differ from the uncached path and desynchronise
        the distributed reduction.
        """
        context_condition = (
            self._context_condition_embedder(context_feature, context_valid_mask)
            if self._context_condition_embedder is not None
            else None
        )
        energy_condition, energy_monitor = self._embed_energy_condition(
            energy_feature, collect_monitor
        )
        if energy_condition is not None:
            context_condition = (
                energy_condition
                if context_condition is None
                else context_condition + energy_condition
            )
        projected_context = (
            self._context_projector(context_feature)
            if self._context_projector is not None
            else None
        )
        block_caches = (
            [
                block.prepare_cross_attention_cache(
                    projected_context, context_valid_mask
                )
                for block in self._blocks
            ]
            if projected_context is not None
            else [None] * len(self._blocks)
        )
        return context_condition, block_caches, energy_monitor

    @staticmethod
    def _repeat_scene_dim(scene_tensor: torch.Tensor, num_samples) -> torch.Tensor:
        """Repeat scenes with trace-safe expand/reshape semantics."""
        scene_batch = scene_tensor.shape[0]
        trailing = scene_tensor.shape[1:]
        return (
            scene_tensor.unsqueeze(1)
            .expand(scene_batch, num_samples, *trailing)
            .reshape(-1, *trailing)
        )

    @classmethod
    def _expand_cached_condition(
        cls, context_condition: torch.Tensor, target_batch: int
    ) -> torch.Tensor:
        """Broadcast a per-scene condition over the trajectory samples."""
        scene_batch = context_condition.shape[0]
        # Avoid tracing a shape-dependent Python branch.
        if not torch.jit.is_tracing() and target_batch % scene_batch != 0:
            raise ValueError(
                "trajectory batch must be a multiple of scene batch for cached "
                f"conditioning, got trajectory={target_batch}, scene={scene_batch}"
            )
        return cls._repeat_scene_dim(context_condition, target_batch // scene_batch)

    def _embed_energy_condition(
        self, energy_feature: Optional[torch.Tensor], collect_monitor: bool = False
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Return the weighted energy condition, or None when it is disabled."""
        if self._energy_condition_embedder is None:
            return None, {}
        if energy_feature is None:
            raise ValueError(
                "enable_energy_condition is on but GUIDE_FLOW_ENERGY_FEATURE was not "
                "provided; add guide_flow_energy_config_v2.pb.txt to "
                "`dataset.enabled_features` or turn the condition off"
            )
        condition, monitor = self._energy_condition_embedder(
            energy_feature, collect_monitor
        )
        return condition * self._energy_condition_weight, monitor

    def forward(
        self,
        noised_trajectory: torch.Tensor,
        flow_time: torch.Tensor,
        context_feature: torch.Tensor,
        context_valid_mask: torch.Tensor,
        collect_monitor: bool = False,
        context_cache: Optional[
            Tuple[
                Optional[torch.Tensor],
                List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
            ]
        ] = None,
        energy_feature: Optional[torch.Tensor] = None,
        cached_energy_monitor: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Denoise trajectories and optionally collect scalar diagnostics."""
        batch_size = noised_trajectory.shape[0]
        if self._time_condition_mode == TIME_CONDITION_CONSTANT:
            # T-free mode: the caller's schedule time never reaches the network.
            flow_time = torch.full_like(flow_time, self._time_condition_constant)
        trajectory_feature = self._input_projector(self._tokenize(noised_trajectory))
        if self._local_point_mixer is not None:
            trajectory_feature = (
                trajectory_feature + self._local_point_mixer(noised_trajectory)
            )
        if self._point_embedding is not None:
            trajectory_feature = trajectory_feature + self._point_embedding.weight.unsqueeze(0).expand(
                batch_size, -1, -1
            )

        time_condition = self._time_embedder(flow_time)
        condition = time_condition
        # On the cached path the corridor condition was embedded by
        # `prepare_context_cache`, so its diagnostics arrive from the caller.
        energy_monitor: Dict[str, float] = cached_energy_monitor or {}
        if context_cache is None:
            context_condition = (
                self._context_condition_embedder(context_feature, context_valid_mask)
                if self._context_condition_embedder is not None
                else None
            )
            energy_condition, energy_monitor = self._embed_energy_condition(
                energy_feature, collect_monitor
            )
            if energy_condition is not None:
                context_condition = (
                    energy_condition
                    if context_condition is None
                    else context_condition + energy_condition
                )
            projected_context = (
                self._context_projector(context_feature)
                if self._context_projector is not None
                else None
            )
            block_caches = [None] * len(self._blocks)
        else:
            context_condition, block_caches = context_cache
            projected_context = None
        if context_condition is not None:
            # Cached conditions retain scene batch and must expand over samples.
            if context_cache is not None:
                context_condition = self._expand_cached_condition(
                    context_condition, batch_size
                )
            condition = condition + context_condition

        monitor = {}
        if collect_monitor:
            monitor.update(energy_monitor)
            _add_tensor_diagnostics(monitor, "time_condition", time_condition)
            _add_tensor_diagnostics(monitor, "condition", condition)

        gate_absmax = 0.0
        gate_rms = 0.0
        gate_nonfinite = 0.0
        # Track the worst modulation across blocks.
        part_absmax = {"shift": 0.0, "scale": 0.0, "gate": 0.0}
        part_rms = {"shift": 0.0, "scale": 0.0, "gate": 0.0}
        for block_index, (block, block_cache) in enumerate(
            zip(self._blocks, block_caches)
        ):
            trajectory_feature, modulation = block(
                trajectory_feature,
                projected_context,
                context_valid_mask,
                condition,
                cross_attention_cache=block_cache,
            )
            if collect_monitor:
                block_absmax, block_rms, block_nonfinite = _tensor_diagnostics(
                    modulation
                )
                gate_absmax = max(gate_absmax, block_absmax)
                gate_rms = max(gate_rms, block_rms)
                gate_nonfinite += block_nonfinite
                monitor[f"monitor/adaln_block_{block_index}_absmax"] = block_absmax
                monitor[f"monitor/adaln_block_{block_index}_rms"] = block_rms
                monitor[f"monitor/adaln_block_{block_index}_nonfinite"] = block_nonfinite
                chunks = modulation.chunk(9, dim=-1)
                # Each attention/FFN group uses (shift, scale, gate).
                for offset, name in enumerate(("shift", "scale", "gate")):
                    part_chunks = chunks[offset::3]
                    for chunk in part_chunks:
                        absmax, rms, _ = _tensor_diagnostics(chunk)
                        part_absmax[name] = max(part_absmax[name], absmax)
                        part_rms[name] = max(part_rms[name], rms)
                    _add_tensor_diagnostics(
                        monitor,
                        f"adaln_block_{block_index}_{name}",
                        torch.cat(part_chunks, dim=-1),
                    )
        if collect_monitor:
            monitor["monitor/adaln_absmax"] = gate_absmax
            monitor["monitor/adaln_rms"] = gate_rms
            monitor["monitor/adaln_nonfinite"] = gate_nonfinite
            for name in ("shift", "scale", "gate"):
                monitor[f"monitor/adaln_{name}_absmax"] = part_absmax[name]
                monitor[f"monitor/adaln_{name}_rms"] = part_rms[name]

        return self._detokenize(self._final_layer(trajectory_feature, condition)), monitor
