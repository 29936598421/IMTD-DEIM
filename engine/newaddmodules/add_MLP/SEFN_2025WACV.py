import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers, math
from einops import rearrange

__all__ = ['SEFN']
'''来自Ai缝合怪创新改进-带你冲顶会、顶刊--'''
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class SEFN(nn.Module):
    def __init__(self, dim,hidd_dim=1024, ffn_expansion_factor=3, bias=False):
        super(SEFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.fusion = nn.Conv2d(hidden_features + dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv_afterfusion = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                            groups=hidden_features, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        x = to_4d(x, h, w)

        spatial = x
        x = self.project_in(x)

        #### Spatial branch
        y = self.avg_pool(spatial)
        y = self.conv(y)
        y = self.upsample(y)
        ####

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.fusion(torch.cat((x1, y), dim=1))
        x1 = self.dwconv_afterfusion(x1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return to_3d(x)
if __name__ == '__main__':
    models = SEFN(768)
    # 2.如何输入的3维数据 . NLP方向的小伙伴都可以拿去使用
    B, N, C = 2, 196, 768  # 批量大小、序列长度、特征维度
    input = torch.randn(B, N, C)
    output = models(input)
    print('SEFN_size:',input.size())
    print('SEFN_size:',output.size())
