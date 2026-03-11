import torch
import torch.nn as nn
from einops import rearrange
from math import sqrt
__all__ =['MSCAFusion']
# 论文：Multiscale Sparse Cross-Attention Network for Remote Sensing Scene Classification
#第一即插即用模块是 MSCA模块
class MSCAFusion(nn.Module):
    def __init__(self, dim, num_heads=8, topk=True, kernel=[3, 5, 7], s=[1, 1, 1], pad=[1, 2, 3],
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0., k1=2, k2=3):
        super(MSCAFusion, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.k1 = k1
        self.k2 = k2

        self.attn1 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        # self.attn3 = torch.nn.Parameter(torch.tensor([0.3]), requires_grad=True)

        self.avgpool1 = nn.AvgPool2d(kernel_size=kernel[0], stride=s[0], padding=pad[0])
        self.avgpool2 = nn.AvgPool2d(kernel_size=kernel[1], stride=s[1], padding=pad[1])
        self.avgpool3 = nn.AvgPool2d(kernel_size=kernel[2], stride=s[2], padding=pad[2])

        self.layer_norm = nn.LayerNorm(dim)

        self.topk = topk  # False True

    def forward(self,data) :

        x, y=data
        y1 = self.avgpool1(y)
        y2 = self.avgpool2(y)
        y3 = self.avgpool3(y)
        # y = torch.cat([y1.flatten(-2,-1),y2.flatten(-2,-1),y3.flatten(-2,-1)],dim = -1)
        y = y1 + y2 + y3
        y = y.flatten(-2, -1)

        y = y.transpose(1, 2)
        y = self.layer_norm(y)
        x = rearrange(x, 'b c h w -> b (h w) c')
        # y = rearrange(y,'b c h w -> b (h w) c')
        B, N1, C = y.shape
        # print(y.shape)
        kv = self.kv(y).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # print(self.k1,self.k2)
        mask1 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=int(N1 / self.k1), dim=-1, largest=True)[1]
        # print(index[0,:,48])
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = (attn1 @ v)

        mask2 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=int(N1 / self.k2), dim=-1, largest=True)[1]
        # print(index[0,:,48])
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        out2 = (attn2 @ v)

        out = out1 * self.attn1 + out2 * self.attn2  # + out3 * self.attn3
        # out = out1 * self.attn1 + out2 * self.attn2

        x = out.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        hw = int(sqrt(N))
        x = rearrange(x, 'b (h w) c -> b c h w', h=hw, w=hw)
        # x = x + x0
        return x

# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input1= torch.randn(1, 32, 64, 64)
    input2 = torch.randn(1, 32, 64, 64)
    # 创建 MSCA 模块
    MSCA = MSCAFusion(dim=32)
    # 将输入图像传入 MSCA 模块进行处理
    output = MSCA([input1,input2])
    # 打印输入和输出的形状
    print('Ai缝合即插即用模块永久更新-MSCA_input_size:', input1.size())
    print('Ai缝合即插即用模块永久更新-MSCA_output_size:', output.size())
