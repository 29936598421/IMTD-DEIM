import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
# https://github.com/yysdck/SFFNet/blob/main/models/FMS.py
# https://arxiv.org/pdf/2405.01992
__all__ = ['WTFDown']
class WTFD(nn.Module): #小波变化高低频分解模块
    def __init__(self, in_ch, out_ch):
        super(WTFD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)
        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)
        return yL,yH


class WTFDown(nn.Module):#小波变化高低频分解下采样模块
    def __init__(self, in_ch, out_ch):
        super(WTFDown, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)
        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)
        # return yL , yH
        return yL + yH #小波变化高低频分解下采样模块

if __name__ == "__main__":
    # 创建一个简单的输入特征图
    input = torch.randn(1,32, 64, 64)


    WTFDown = WTFDown(32,64)  #小波变化高低频分解下采样模块
    output = WTFDown(input)
    print(f"input  shape: {input.shape}")
    print(f"output shape: {output.shape}")



