"""
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from engine.backbone.common import FrozenBatchNorm2d  # 导入冻结BN层
from engine.backbone.hgnetv2_org import HG_Stage, HGNetv2_test
from engine.core import register  # 注册模块
import logging  # 日志记录

from engine.newaddmodules.add_block import EBlock

# 初始化函数别名
kaiming_normal_ = nn.init.kaiming_normal_  #  kaiming正态分布初始化
zeros_ = nn.init.zeros_  # 全零初始化
ones_ = nn.init.ones_  # 全一初始化
# print("Ai缝合怪-带你冲顶会、顶刊--使用了")
__all__ = ['HGNetv2_EBlock']  # 导出HGNetv2类

class HG_Stage_EBlock(HG_Stage):
    def __init__(
            self,
            in_chs,  # 输入通道数
            mid_chs,  # 中间通道数
            out_chs,  # 输出通道数
            block_num,  # 块数
            layer_num,  # 每块的卷积层数
            downsample=True,  # 是否下采样（阶段起始是否降维）
            light_block=False,  # 是否使用轻量级块
            kernel_size=3,  # 卷积核大小
            use_lab=False,  # 是否使用仿射块
            agg='se',  # 聚合方式
            drop_path=0.,  # 随机失活率（可为列表，每块独立设置）
    ):
        super().__init__( in_chs,  # 输入通道数
                        mid_chs,  # 中间通道数
                        out_chs,  # 输出通道数
                        block_num,  # 块数
                        layer_num,  # 每块的卷积层数
                        downsample,  # 是否下采样（阶段起始是否降维）
                        light_block,  # 是否使用轻量级块
                        kernel_size,  # 卷积核大小
                        use_lab,  # 是否使用仿射块
                        agg,  # 聚合方式
                        drop_path,  # 随机失活率（可为列表，每块独立设置）
                         )

        # 构建多个HG_Block
        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                EBlock(
                    in_chs if i == 0 else out_chs,  # 第一块输入为in_chs，后续为out_chs（残差连接）
                    out_chs
                )
            )
        self.blocks = nn.Sequential(*blocks_list)  # 组合块为序列

@register()
class HGNetv2_EBlock(HGNetv2_test):
    """
    HGNetV2 主干网络
    Args:
        name: 模型版本（如'B0', 'B1'等，对应不同配置）
        use_lab: 是否在激活函数后添加可学习仿射块
        return_idx: 输出哪些阶段的特征（索引从0开始）
        freeze_stem_only: 是否仅冻结Stem块参数
        freeze_at: 冻结至第几个阶段（-1表示不冻结）
        freeze_norm: 是否冻结BN层参数
        pretrained: 是否加载预训练权重
        local_model_dir: 预训练权重本地路径
    """

    def __init__(self,
                 name,
                 use_lab=False,
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True,
                 agg= 'ese',
                 local_model_dir='weight/hgnetv2/'):
        super().__init__(  name,
                 use_lab,
                 return_idx,
                 freeze_stem_only,
                 freeze_at,
                 freeze_norm,
                 pretrained,
                 agg,
                 local_model_dir)

        stage_config = self.arch_configs[name]['stage_config']

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            print("Ai缝合怪-带你冲顶会、顶刊--使用了EBlock")
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num \
                = stage_config[k]
            self.stages.append(
                HG_Stage_EBlock(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                    agg))

