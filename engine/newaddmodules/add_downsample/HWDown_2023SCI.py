import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
# 按照这个第三方库需要安装pip install pytorch_wavelets==1.3.0
# 如果提示缺少pywt库则安装 pip install PyWavelets
__all__ =['HWDown']

class HWDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWDown, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = HWDown(in_ch=32, out_ch=32)  # 输入通道数，输出通道数
    input = torch.rand(1, 32, 64, 64)
    output = block(input)
    print('input :',input.size())
    print('output :', output.size())

