import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from einops import rearrange
__all__ = ['RSA']

class RSA(nn.Module):
    def __init__(self, channels, shifts=1, window_sizes=4, bias=False):
        super(RSA, self).__init__()
        self.channels = channels
        self.shifts = shifts
        self.window_sizes = window_sizes

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.act = nn.ReLU()

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        wsize = self.window_sizes
        x_ = x
        if self.shifts > 0:
            x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
        qkv = self.qkv_dwconv(self.qkv(x_))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b c (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)
        k = rearrange(k, 'b c (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)
        v = rearrange(v, 'b c (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q.transpose(-2, -1) @ k) * self.temperature  # b (h w) (dh dw) (dh dw)
        attn = self.act(attn)
        out = (v @ attn)
        out = rearrange(out, 'b (h w) (dh dw) c-> b (c) (h dh) (w dw)', h=h // wsize, w=w // wsize, dh=wsize, dw=wsize)
        if self.shifts > 0:
            out = torch.roll(out, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
        y = self.project_out(out)
        return y

