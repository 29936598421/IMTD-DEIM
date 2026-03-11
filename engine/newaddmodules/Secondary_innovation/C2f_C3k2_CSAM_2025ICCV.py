import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ['C2f_CSAM','C3k2_CSAM']

from ultralytics.nn.modules import C3


class Shift_channel_mix(nn.Module):
    def __init__(self, shift_size=1):
        super(Shift_channel_mix, self).__init__()
        self.shift_size = shift_size

    def forward(self, x):  # x的张量 [B,C,H,W]
        x1, x2, x3, x4 = x.chunk(4, dim=1)

        x1 = torch.roll(x1, self.shift_size, dims=2)  # [:,:,1:,:]

        x2 = torch.roll(x2, -self.shift_size, dims=2)  # [:,:,:-1,:]

        x3 = torch.roll(x3, self.shift_size, dims=3)  # [:,:,:,1:]

        x4 = torch.roll(x4, -self.shift_size, dims=3)  # [:,:,:,:-1]

        x = torch.cat([x1, x2, x3, x4], 1)

        return x
'''
来自CVPR 2025 顶会
即插即用模块： SCM 特征位移混合模块

主要内容：
模型二值化在实现卷积神经网络（CNN）的实时高效计算方面取得了显著进展，为视觉Transformer（ViT）在边缘设备上的部署挑战提供了潜在解决方案。
然而，由于CNN和Transformer架构的结构差异，直接将二值化CNN策略应用于ViT模型会导致性能显著下降。
为解决这一问题，我们提出了BHViT——一种适合二值化的混合ViT架构及其全二值化模型，其设计基于以下三个重要观察：

1.局部信息交互与分层特征聚合：BHViT利用从粗到细的分层特征聚合技术，减少因冗余token带来的计算开销。
2.基于移位操作的新型模块：提出一种基于移位操作的模块（SCM），在不显著增加计算负担的情况下提升二值化多层感知机（MLP）的性能。
3.量化分解的注意力矩阵二值化方法：提出一种基于量化分解的创新方法，用于评估二值化注意力矩阵中各token的重要性。

该Shift_channel_mix（SCM）模块是论文中提出的一个轻量化模块，用于增强二进制多层感知器（MLP）在二进制视觉变换器（BViT）中的表现。
它通过对输入特征图进行不同的移位操作，帮助缓解信息丢失和梯度消失的问题，从而提高网络的性能，同时避免增加过多的计算开销。
SCM模块的主要操作包括：
1.水平移位（Horizontal Shift）：通过torch.roll函数将特征图的列按指定的大小进行右/左移操作。这种操作模拟了在处理二进制向量时的特征循环，增强了表示能力。
2.垂直移位（Vertical Shift）：类似于水平移位，垂直移位会使特征图的行发生上下移动。这有助于捕获跨行的信息，同时适应不同的特征维度。
在代码实现中，torch.chunk将输入特征图沿着通道维度分成四个部分，之后通过不同的移位操作处理每一部分，最后将处理后的四个部分通过torch.cat拼接起来，形成最终的输出。
'''
class CSAM(nn.Module):
    def __init__(self, in_channels, att_channels=16, lk_size=13, sk_size=3, reduction=2):
        """
        :param in_channels: 输入特征图通道数
        :param att_channels: 用于注意力通道数，默认为16
        :param lk_size: 静态大核卷积核尺寸（如图中13）
        :param sk_size: 动态卷积核尺寸（如图中3）
        :param reduction: 动态卷积中间层压缩因子
        """
        super().__init__()
        self.in_channels = in_channels
        self.att_channels = att_channels
        self.idt_channels = in_channels - att_channels
        self.lk_size = lk_size
        self.sk_size = sk_size

        # 动态卷积核生成器
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(att_channels, att_channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(att_channels // reduction, att_channels * sk_size * sk_size, 1)
        )
        nn.init.zeros_(self.kernel_gen[-1].weight)
        nn.init.zeros_(self.kernel_gen[-1].bias)

        # 共享静态大核卷积核：定义为参数，非卷积层
        self.lk_filter = nn.Parameter(torch.randn(att_channels, att_channels, lk_size, lk_size))
        nn.init.kaiming_normal_(self.lk_filter, mode='fan_out', nonlinearity='relu')

        # 融合层
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scm =  Shift_channel_mix()
    def forward(self, x):

        B, C, H, W = x.shape
        assert C == self.att_channels + self.idt_channels, f"Input channel {C} must match att + idt ({self.att_channels} + {self.idt_channels})"

        # 通道拆分
        F_att, F_idt = torch.split(x, [self.att_channels, self.idt_channels], dim=1)

        # 生成动态卷积核 [B * att, 1, 3, 3]
        kernel = self.kernel_gen(F_att).reshape(B * self.att_channels, 1, self.sk_size, self.sk_size)

        # 动态卷积操作
        F_att_re = rearrange(F_att, 'b c h w -> 1 (b c) h w')
        out_dk = F.conv2d(F_att_re, kernel, padding=self.sk_size // 2, groups=B * self.att_channels)
        out_dk = rearrange(out_dk, '1 (b c) h w -> b c h w', b=B, c=self.att_channels)

        # 静态大核卷积
        out_lk = F.conv2d(F_att, self.lk_filter, padding=self.lk_size // 2)

        # 融合（两个卷积结果加和）
        out_att = out_lk + out_dk

        # 拼接 F_idt（保留通道）
        out = torch.cat([out_att, F_idt], dim=1)
        out = self.scm(out)
        # 对通道拼接后的特征图，使用SCM 特征位移混合模块，这有助于捕获跨行的信息。
        # 它通过对输入特征图进行不同的移位操作，帮助缓解信息丢失和梯度消失的问题，从而提高网络的性能，同时避免增加过多的计算开销。
        # 1x1 融合
        out = self.fusion(out)
        return out

#------------
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Bottleneck_CSAM(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = CSAM(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))



class C2f_CSAM(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_CSAM(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
class C3k_CSAM(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_CSAM(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

class C3k2_CSAM(C2f_CSAM):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_CSAM(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_CSAM(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )


if __name__ == "__main__":

    input = torch.randn(1,64,128,128)
    CSAM = CSAM(in_channels=64)
    output= CSAM(input)
    print("Ai缝合怪整理的CSAM_输入张量形状:", input.shape)  # (1, 64, 32, 32)
    print("Ai缝合怪整理的CSAM_输出张量形状:", output.shape)  # (1, 64, 32, 32)

