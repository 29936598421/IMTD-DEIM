
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from ultralytics.nn.modules.conv import autopad
#论文：Simone Cammarasana and Giuseppe Patanè. Optimal Density Functions for Weighted Convolution in Learning
#代码地址：https://github.com/cammarasana123/weightedConvolution2.0
# wConv2d(in_channels, out_channels, kernel_size = 1, den = [])
# wConv2d(in_channels, out_channels, kernel_size = 3, den = [0.7])
# wConv2d(in_channels, out_channels, kernel_size = 5, den= [0.2, 0.8])
__all__ = ['wConv2D']
class wConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, den=[0.7], stride=1, padding=1, groups=1, dilation=1, bias=False):
        super(wConv2D, self).__init__()
        self.stride = _pair(stride)
        self.kernel_size = _pair(kernel_size)
        # self.padding = autopad(self.kernel_size, d=dilation)
        self.padding = 1
        self.groups = groups
        self.dilation = _pair(dilation)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        device = torch.device('cpu')
        self.register_buffer('alfa', torch.cat([torch.tensor(den, device=device),torch.tensor([1.0], device=device),torch.flip(torch.tensor(den, device=device), dims=[0])]))
        self.register_buffer('Phi', torch.outer(self.alfa, self.alfa))

        if self.Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size {self.kernel_size}")

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return F.conv2d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block =wConv2D(32,64).cuda()
    input = torch.rand(1, 32, 64, 64).cuda()
    output = block(input)
    print("input.shape:", input.shape)
    print("output.shape:",output.shape)
