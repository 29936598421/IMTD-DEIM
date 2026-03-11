import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv
__all__ = ['CARAFE']
'''来自Ai缝合怪创新改进-带你冲顶会、顶刊--'''
class CARAFE(nn.Module):
    def __init__(self, c,c1=0 ,k_enc=3, k_up=5, c_mid=64, scale=2):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.reshape(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        # X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_  利用下面两行等价替换这一行

        # 扩展 W 的维度以匹配 X
        W_expanded = W.unsqueeze(1)  # Shape: [B, 1, K, H, W]
        # 逐元素相乘后对 K 维度求和
        X = (W_expanded * X).sum(dim=2)  # Sum over K, output shape: [B, C, H, W]
        return X
if __name__ == '__main__':
    input = torch.rand(1, 64, 4, 4)
    # in_channels=64, scale=4, style='lp'/‘pl’,
    CARAFE_UP = CARAFE(64,64)
    output = CARAFE_UP(input)
    print('input_size:', input.size())
    print('output_size:', output.size())