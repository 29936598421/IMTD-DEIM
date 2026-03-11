import torch
import torch.nn as nn
# 论文地址;https://arxiv.org/abs/2412.20066

__all__ = ['ShuffleAttn']
'''
来自CVPR2025顶会
即插即用模块：ShuffleAttn（SSA） 序列打乱注意力

SSA（Sequence Shuffle Attention）模块的作用是融合来自不同扫描方向的序列特征，
以增强模型对图像中复杂结构和细节的恢复能力。其原理是在不同方向提取的序列之间建立通道级的注意力机制。
具体地，SSA首先对每个方向的序列进行平均池化并拼接，随后通过序列打乱和分组卷积实现不同方向间的特征交互，
再将卷积后的权重恢复至原始顺序，最后对输入序列进行加权融合。该模块不仅有效整合多方向信息，还能在保持结构一致性的同时提升图像复原质量。
SSA作用总结:
    1.跨序列的信息交互：充分利用从不同扫描方向获得的互补信息；
    2.保持通道一致性：避免简单像素加和导致的语义混淆；
    3.增强特征表达能力：通过注意力机制强化有用信息，抑制冗余噪声；
    4.提升图像复原效果：尤其是在纹理、边缘和细节保持上更出色。

SSA模块适合：图像恢复，目标检测，图像分割，语义分割，图像增强，图像去噪，遥感语义分割，图像分类等所有CV任务通用的即插即用模块
'''
'''来自Ai缝合怪创新改进-带你冲顶会、顶刊--'''
class ShuffleAttn(nn.Module):
    def __init__(self, in_features,  group=4):
        super().__init__()
        self.group = group
        out_features = in_features
        self.in_features = in_features
        self.out_features = out_features

        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, out_features, groups=self.group, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def channel_rearrange(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, self.group, group_channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)
        return x

    def forward(self, x):
        m = x
        x = self.channel_shuffle(x)  # 1. 打乱通道顺序
        x = self.gating(x)  # 2. 加权注意力（通道加权）
        x = self.channel_rearrange(x)  # 3. 通道重组恢复
        return m * x
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = ShuffleAttn(64)
    input = torch.rand(1, 64, 64, 64)
    output = block(input)
    print(input.size(), output.size())
