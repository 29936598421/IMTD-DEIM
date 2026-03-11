import torch
import torch.nn as nn
from einops import rearrange
from timm.models.efficientvit_mit import DSConv
__all__ = ['MSAM']

class IndentityBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters, rate=1):
        super(IndentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            SeparableConvBNReLU(F1, F2, kernel_size, dilation=rate),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.SiLU(),
            # nn.Dropout(0.1)
        )

class SelfAttention(nn.Module):
    def __init__(self, dim_out):
        super(SelfAttention, self).__init__()
        self.conv = DSConv(dim_out * 2, (dim_out // 2) * 3, 3)
        self.att_dim = dim_out // 2

    def forward(self, x, y):
        b, c, h, w = x.shape
        fm = self.conv(torch.concat([x, y], dim=1))

        Q, K, V = rearrange(fm, 'b (qkv c) h w -> qkv b h c w', qkv=3, b=b, c=self.att_dim, h=h, w=w)

        dots = (Q @ K.transpose(-2, -1))
        attn = dots.softmax(dim=-1)
        attn = attn @ V
        attn = attn.view(b, -1, h, w)
        return attn


class MSAM(nn.Module):
    def __init__(self, dim_in,dim_out):
        super(MSAM, self).__init__()
        self.dim_out = dim_out

        self.branch3 = nn.Sequential(
            SeparableConvBNReLU(dim_in, dim_out, kernel_size=1)
        )
        self.branch4 = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(dim_in, dim_out, kernel_size=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU6()
        )
        self.merge = nn.Sequential(
            nn.Conv2d(2 * dim_out, dim_out, kernel_size=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU6()
        )
        self.resblock = nn.Sequential(
            IndentityBlock(in_channel=dim_out, kernel_size=3, filters=[dim_out, dim_out, dim_out]),
            IndentityBlock(in_channel=dim_out, kernel_size=3, filters=[dim_out, dim_out, dim_out])
        )
        self.transformer = SelfAttention(dim_out)
        self.conv = nn.Conv2d(dim_out // 2 * 3, dim_out, 1)

    def forward(self, skip_list):
        x = skip_list[-1]
        b, c, h, w = x.shape
        list1 = []
        list2 = []

        x1 = self.branch3(skip_list[0])
        x = self.branch4(x)

        # CNN
        merge = self.merge(torch.cat([x, x1], dim=1))
        merge = self.resblock(merge)

        # Transformer
        list1.append(x)
        list1.append(x1)

        for i in range(len(list1)):
            for j in range(len(list1)):
                if i <= j:
                    att = self.transformer(list1[i], list1[j])
                    list2.append(att)

        out = self.conv(torch.concat(list2, dim=1))

        return out + merge
if __name__ == "__main__":
    # 设置输入的 batch size 和空间尺寸
    B, H, W = 1, 16, 16  # 空间尺寸要和 MSAM 中下采样后的分支相匹配
    dim_out = 128        # MSAM 输出通道数

    # 模拟4个分支输入，按 MSAM 模块需要的维度创建
    skip_1 = torch.randn(B, 64, H*8, W*8)     # 分支1输入
    skip_2 = torch.randn(B, 128, H*4, W*4)    # 分支2输入
    skip_3 = torch.randn(B, 128, H*4, W*4)    # 分支3输入


    # 初始化 MSAM 模块
    msam = MSAM(dim_in=128,dim_out=128)

    # 前向传播
    output = msam([skip_2,skip_3])

    # 打印输出形状
    print("MSAM 输出张量形状:", output.shape)