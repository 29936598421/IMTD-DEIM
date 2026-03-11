import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv
#https://github.com/takeyoutime/UMFormer/blob/main/geoseg/models/My/IIA.py
__all__ = ['IIA_Fusion']
class AttentionWeight(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(AttentionWeight, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size, padding=padding, groups=channel, bias=False)
        self.bn = nn.BatchNorm1d(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, w, c, h = x.size()
        x_weight = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_weight = self.conv1(x_weight).view(b, c, h)
        x_weight = self.sigmoid(self.bn(self.conv2(x_weight)))
        x_weight = x_weight.view(b, 1, c, h)

        return x * x_weight


class IIA_Fusion(nn.Module):
    def __init__(self, channel):
        super(IIA_Fusion, self).__init__()
        self.attention = AttentionWeight(channel)
        self.conv = Conv(2*channel,channel,1)
    def forward(self, data):

        x = self.conv(torch.cat(data,dim=1))
        # b, w, c, h
        x_h = x.permute(0, 3, 1, 2).contiguous()
        x_h = self.attention(x_h).permute(0, 2, 3, 1)
        # b, h, c, w
        x_w = x.permute(0, 2, 1, 3).contiguous()
        x_w = self.attention(x_w).permute(0, 2, 1, 3)
        # b, c, h, w
        # x_c = self.attention(x)

        # return x + 1 / 2 * (x_h + x_w)  # 89.8	92.5	81.9
        return x + x_h + x_w
if __name__ == '__main__':
    block = IIA_Fusion(256)
    sar = torch.randn(2, 256, 64, 64)
    opt = torch.randn(2, 256, 64, 64)
    print("input:", sar.shape, opt.shape)
    print("output:", block([sar, opt]).shape)