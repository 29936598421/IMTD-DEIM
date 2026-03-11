import torch
import torch.nn as nn
import torch.nn.init as init

from engine.deim.utils import get_activation, bias_init_with_prob
import math
import copy
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List


from engine.deim.utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from engine.deim.utils import bias_init_with_prob

class MSDeformableAttention(nn.Module):
    """
    多尺度可变形注意力模块，自适应地从不同尺度特征图上采样关键点
    """

    def __init__(
            self,
            embed_dim=256,  # 嵌入维度
            num_heads=8,  # 注意力头数
            num_levels=4,  # 特征层级数
            num_points=4,  # 每个层级的采样点数
            method='default',  # 采样方法
            offset_scale=0.5,  # 偏移量缩放因子
    ):
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        # 处理每个层级的采样点数量
        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        # 注册采样点缩放因子为缓冲区
        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim必须能被num_heads整除"

        # 线性层预测采样偏移量和注意力权重
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        # 绑定可变形注意力核心函数
        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)

        self._reset_parameters()  # 初始化参数

        # 如果使用离散方法，固定采样偏移量参数
        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # 初始化采样偏移量参数
        init.constant_(self.sampling_offsets.weight, 0)
        # 为每个注意力头设置初始方向
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # 初始化注意力权重参数
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(self,
                query: torch.Tensor,  # 查询特征 [bs, query_length, C]
                reference_points: torch.Tensor,  # 参考点 [bs, query_length, n_levels, 2/4]
                value: torch.Tensor,  # 键值特征 [bs, value_length, C]
                value_spatial_shapes: List[int]):  # 每个层级的空间形状 [(H_0, W_0), ...]
        """
        计算多尺度可变形注意力
        """
        bs, Len_q = query.shape[:2]

        # 计算采样偏移量和注意力权重
        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)  # 对注意力权重进行softmax归一化

        # 根据参考点计算采样位置
        if reference_points.shape[-1] == 2:  # 参考点为2D坐标(x,y)
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1,2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:  # 参考点为4D坐标(x,y,w,h)
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "reference_points的最后一维必须是2或4，但得到{}".format(reference_points.shape[-1]))

        # 调用核心函数计算可变形注意力
        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights,
                                              self.num_points_list)

        return output

class Gate(nn.Module):
    """
    门控机制，用于自适应融合两个输入张量
    """

    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)  # 使用特定概率初始化偏置
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        # 计算门控值并融合输入
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)

class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层，包含自注意力、可变形交叉注意力和前馈网络
    """

    def __init__(self,
                 d_model=256,  # 模型维度
                 n_head=8,  # 注意力头数
                 dim_feedforward=1024,  # 前馈网络隐藏维度
                 dropout=0.,  # dropout率
                 activation='relu',  # 激活函数
                 n_levels=4,  # 特征层级数
                 n_points=4,  # 每个层级的采样点数
                 cross_attn_method='default',  # 交叉注意力方法
                 layer_scale=None):  # 层缩放因子
        super(TransformerDecoderLayer, self).__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # 自注意力模块
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 交叉注意力模块，使用多尺度可变形注意力
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)

        self.dropout2 = nn.Dropout(dropout)

        # 门控机制，用于融合不同来源的特征
        self.gateway = Gate(d_model)

        # 前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()  # 初始化参数

    def _reset_parameters(self):
        # 使用Xavier初始化线性层权重
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        # 将位置编码添加到输入张量
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        # 前馈网络的前向传播
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                target,  # 目标特征
                reference_points,  # 参考点
                value,  # 键值特征
                spatial_shapes,  # 空间形状
                attn_mask=None,  # 注意力掩码
                query_pos_embed=None):  # 查询位置编码
        """
        Transformer解码器层的前向传播
        """
        # 自注意力计算
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)  # 残差连接
        target = self.norm1(target)  # 层归一化

        # 交叉注意力计算，使用多尺度可变形注意力

        # target2 = self.cross_attn(
        #     self.with_pos_embed(target, query_pos_embed),
        #     reference_points,
        #     value,
        #     spatial_shapes)

        # 通过门控机制融合特征
        target = self.gateway(target, self.dropout2(target2))

        # 前馈网络
        target2 = self.forward_ffn(target)


        target = target + self.dropout4(target2)  # 残差连接
        target = self.norm3(target.clamp(min=-65504, max=65504))  # 层归一化并限制范围

        return target



# 假设输入参数
batch_size = 2
num_queries = 100  # 目标序列长度
d_model = 256      # 模型维度
n_head = 8         # 注意力头数
n_levels = 4       # 特征层级数
value_length = 300 # 值序列长度

# 1. 准备输入数据 ------------------------------------------------------------
# 目标特征 (batch_size, num_queries, d_model)
target = torch.rand(batch_size, num_queries, d_model)

# 参考点 (batch_size, num_queries, n_levels, 4)
# 格式为(x,y,w,h)，其中x,y是中心坐标，w,h是宽高
reference_points = torch.rand(batch_size, num_queries, n_levels, 4)

# 值特征 (batch_size, value_length, d_model)
value = torch.rand(batch_size, value_length, d_model)

# 各层级的空间形状 [(H1,W1), (H2,W2), (H3,W3), (H4,W4)]
spatial_shapes = [
    (60, 60),  # 第一层级特征图大小
    (30, 30),  # 第二层级
    (15, 15),  # 第三层级
    (8, 8)     # 第四层级
]

# 查询位置编码 (batch_size, num_queries, d_model)
query_pos_embed = torch.rand(batch_size, num_queries, d_model)

# 2. 初始化Transformer解码器层 ----------------------------------------------
decoder_layer = TransformerDecoderLayer(
    d_model=d_model,
    n_head=n_head,
    dim_feedforward=1024,
    n_levels=n_levels,
    n_points=4,
    cross_attn_method='default'
)

# 3. 前向传播 --------------------------------------------------------------
output = decoder_layer(
    target=target,
    reference_points=reference_points,
    value=value,
    spatial_shapes=spatial_shapes,
    query_pos_embed=query_pos_embed
)

# 4. 检查输出 --------------------------------------------------------------
print("输入特征形状:", target.shape)          # [2, 100, 256]
print("输出特征形状:", output.shape)           # [2, 100, 256]
print("输出范围:", output.min().item(), "~", output.max().item())