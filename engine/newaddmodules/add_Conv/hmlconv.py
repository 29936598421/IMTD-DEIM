import torch
import torch.nn as nn
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


class HMLConv(nn.Module):
    def __init__(self, dim, k_size):
        super().__init__()
        self.k_size = k_size
        if k_size == 7:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=((3 - 1) // 2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, groups=dim, dilation=2)
        elif k_size == 11:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=((3 - 1) // 2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=4, groups=dim, dilation=2)
        elif k_size == 23:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=((5 - 1) // 2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        elif k_size == 35:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=((5 - 1) // 2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=11, stride=1, padding=15, groups=dim, dilation=3)
        elif k_size == 41:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=((5 - 1) // 2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=13, stride=1, padding=18, groups=dim, dilation=3)
        elif k_size == 53:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=((5 - 1) // 2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=17, stride=1, padding=24, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim * 2, dim, 1)
        self.scm = Shift_channel_mix()
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(self.scm(torch.cat([x, self.conv_spatial(x)], dim=1)))
        return x


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    ch = 32  # 输入通道数
    # 实例化模型对象
    model = HSLConv(ch,k_size=7)
    input = torch.randn(1, ch , 32, 32)
    # 将输入传入模型进行前向传播
    output = model(input)
    # 打印输入和输出的形状
    print('HMLConv_input_size:', input.size())
    print('HMLConv_output_size:', output.size())
    # HSLConv是TGRS 2025 HLKConv的二次创新模块,，在二次创新交流群！
    '''
    HMLConv 分层移动大核卷积模块，通过结合深度卷积和扩张卷积来提取多尺度特征，
    同时使用 Shift_channel_mix 对通道进行空间偏移和混合，以增强空间感知能力。
    该模块通过这些操作提高了网络的特征提取能力，尤其在处理复杂空间结构时表现更好。
    最终输出与输入形状相同，适用于需要多尺度上下文信息的任务。
    '''