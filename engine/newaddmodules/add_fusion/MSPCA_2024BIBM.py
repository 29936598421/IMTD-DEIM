import math

import torch
import torch.nn as nn
#代码：https://github.com/nkicsl/SF-UNet
__all__ = ['MSPCA']
'''来自Ai缝合怪创新改进-带你冲顶会、顶刊--'''
class MSPCA(nn.Module):
    def __init__(self, input_channel, gamma=2, bias=1):
        super(MSPCA, self).__init__()
        input_channel1, input_channel2 = input_channel,input_channel
        self.input_channel1 = input_channel1
        self.input_channel2 = input_channel2

        self.avg1 = nn.AdaptiveAvgPool2d(1)
        self.avg2 = nn.AdaptiveAvgPool2d(1)

        kernel_size1 = int(abs((math.log(input_channel1, 2) + bias) / gamma))
        kernel_size1 = kernel_size1 if kernel_size1 % 2 else kernel_size1 + 1

        kernel_size2 = int(abs((math.log(input_channel2, 2) + bias) / gamma))
        kernel_size2 = kernel_size2 if kernel_size2 % 2 else kernel_size2 + 1

        kernel_size3 = int(abs((math.log(input_channel1 + input_channel2, 2) + bias) / gamma))
        kernel_size3 = kernel_size3 if kernel_size3 % 2 else kernel_size3 + 1

        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size1, padding=(kernel_size1 - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size2, padding=(kernel_size2 - 1) // 2, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=kernel_size3, padding=(kernel_size3 - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.up = nn.ConvTranspose2d(in_channels=input_channel2, out_channels=input_channel1, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)

    def forward(self, x):
        x1, x2 = x
        x1_ = self.avg1(x1)
        x2_ = self.avg2(x2)

        x1_ = self.conv1(x1_.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x2_ = self.conv2(x2_.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        x_middle = torch.cat((x1_, x2_), dim=1)
        x_middle = self.conv3(x_middle.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x_middle = self.sigmoid(x_middle)

        x_1, x_2 = torch.split(x_middle, [self.input_channel1, self.input_channel2], dim=1)

        x1_out = x1 * x_1
        x2_out = x2 * x_2


        result = x1_out + x2_out
        return result

# 双分支特征融合
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = MSPCA(32)
    input1 = torch.rand(3, 32, 64, 64)
    input2 = torch.rand(3, 32, 64, 64)
    output = block([input1, input2])
    print(output.size())
