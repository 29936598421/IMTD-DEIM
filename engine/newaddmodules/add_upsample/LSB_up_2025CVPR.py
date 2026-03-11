import torch
import torch.nn as nn
from timm.models.layers import DropPath
from ultralytics.nn.modules import C3k2, Conv

from ultralytics.nn.modules.block import C3k

BNNorm2d = nn.BatchNorm2d
LNNorm = nn.LayerNorm
Activation = nn.GELU
#CVPR 2025 医学图像分割
#论文： nnWNet: Rethinking the Use of Transformers in Biomedical Image Segmentation and Calling for a Unified Evaluation Benchmark
__all__ = ['LSB_up']
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            BNNorm2d(ch_out),
            Activation()
        )

    def forward(self, x):
        x = self.up(x)
        return x

class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(down_conv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=False),
            BNNorm2d(ch_out),
            Activation()
        )

    def forward(self, x):
        x = self.down(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, groups=1):
        super(ResBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=1, padding=1)
        self.bn1 = BNNorm2d(planes)
        self.act = Activation()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, groups=groups, padding=1)
        self.bn2 = BNNorm2d(planes)

        if self.inplanes != self.planes:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride=1),
                BNNorm2d(planes)
            )
    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)

        if self.inplanes != self.planes:
            identity = self.down(x)

        out = self.bn2(out) + identity
        out = self.act(out)

        return out


class LSB_up(nn.Module):
    def __init__(self, inplanes,  outplanes,down_or_up='up', groups=4):
        super(LSB_up, self).__init__()
        hidden_planes= 2*inplanes
        if down_or_up is None:
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
                # ResBlock(inplanes=planes, planes=planes, groups=groups),
            )

        elif down_or_up == 'down':
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
                # ResBlock(inplanes=planes, planes=planes, groups=groups),
                down_conv(hidden_planes, outplanes)
            )
        elif down_or_up == 'up':
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
                # ResBlock(inplanes=planes, planes=planes, groups=groups),
                up_conv(hidden_planes, outplanes),
            )

    def forward(self, x):
        out = self.BasicBlock(x)
        return out


# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input= torch.randn(1, 32, 64, 64)
    # 创建 LSB 模块
    LSB_up = LSB_up(inplanes=32,outplanes=32,down_or_up='up')
    # 将输入图像传入LSB_up 模块进行上采样处理
    output = LSB_up(input)
    # 打印输入和输出的形状
    print('Ai缝合即插即用模块永久更新-LSB_up_input_size:', input.size())
    print('Ai缝合即插即用模块永久更新-LSB_up_output_size:', output.size())
