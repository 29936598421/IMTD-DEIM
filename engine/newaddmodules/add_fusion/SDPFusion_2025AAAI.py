# Code Structure of HS-FPN (https://arxiv.org/abs/2412.10116)
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange

__all__ = ['SDPFusion']

from ultralytics.nn.modules import Conv


#论文地址：HS-FPN (https://arxiv.org/abs/2412.10116)
# ------------------------------------------------------------------#
# Spatial Dependency Perception Module SDP
# ------------------------------------------------------------------#
class SDPFusion(nn.Module):
    def __init__(self,
                 dim=256,
                 inter_dim=None,
                 patch=8
               ):
        super(SDPFusion, self).__init__()
        self.dim = dim
        self.inter_dim=inter_dim
        if self.inter_dim == None:
            self.inter_dim = dim
        self.conv_q = nn.Sequential(*[nn.Conv2d(dim, self.inter_dim, 1, padding=0, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.conv_k = nn.Sequential(*[nn.Conv2d(dim, self.inter_dim, 1, padding=0, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.softmax = nn.Softmax(dim=-1)
        self.patch_size = (patch,patch)
        self.conv1x1 = Conv(self.dim,self.inter_dim,1)
    def forward(self, data):
        x_low, x_high = data
        b_, _, h_, w_ = x_low.size()
        q = rearrange(self.conv_q(x_low), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=self.patch_size[0],p2=self.patch_size[1])
        q = q.transpose(1, 2)  # 1,4096,128
        k = rearrange(self.conv_k(x_high), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=self.patch_size[0],p2=self.patch_size[1])
        attn = torch.matmul(q, k)  # 1, 4096, 1024
        attn = attn / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)
        v = k.transpose(1, 2)  # 1, 1024, 128
        output = torch.matmul(attn, v)  # 1, 4096, 128
        output = rearrange(output.transpose(1, 2).contiguous(), '(b h w) c (p1 p2) -> b c (h p1) (w p2)',p1=self.patch_size[0], p2=self.patch_size[1], h=h_ // self.patch_size[0],w=w_ // self.patch_size[1])
        if self.dim != self.inter_dim:
            x_low = self.conv1x1(x_low)
        return output + x_low

if __name__ == '__main__':

    # 定义输入张量的形状为 B, C, H, W
    input1= torch.randn(1, 64, 128, 128)
    input2 = torch.randn(1, 64, 128, 128)
    # 创建 SDP 模块
    sdp= SDPFusion(64,64)  #第二个模块
    # 将输入图像传入 SDP 模块进行处理
    output = sdp([input1,input2])
    # 打印输入和输出的形状
    print('Ai缝合即插即用模块永久更新-SDP_input_size:', input1.size())
    print('Ai缝合即插即用模块永久更新-SDP_output_size:', output.size())
