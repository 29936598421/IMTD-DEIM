import torch
from torch import nn
# 论文：Depth Information Assisted Collaborative Mutual Promotion Network for Single Image Dehazing
__all__ = ['MFM']
'''来自Ai缝合怪创新改进-带你冲顶会、顶刊--'''
class MFM(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(MFM, self).__init__()

        self.height = height
        d = max(int(dim/reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim*height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)
    def forward(self, data):
        in_feats1, in_feats2 =data
        in_feats = [in_feats1,in_feats2]
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats*attn, dim=1)
        return out

# 输入 B H W C , 输出 B  H W C
if __name__ == "__main__":
    # 使用MFM模块
    input1 = torch.randn(1, 64, 32, 32)
    input2 = torch.randn(1, 64, 32, 32)
    mfm = MFM(64) # 创建 MFM模块实例，输入通道数为 64，
    output = mfm ([input1,input2])
    print("MFM_输入张量形状:", input1.shape)  # (1, 64, 32, 32)
    print("MFM_输出张量形状:", output.shape)  # (1, 64, 32, 32)