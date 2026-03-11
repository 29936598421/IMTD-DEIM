# 1. 实例化模型（以DFine版本为例）
import torch

from engine.deim import HybridEncoder

model = HybridEncoder(
    in_channels=[512, 1024],   # 输入特征通道（假设来自骨干网络的C2-C4）
    feat_strides=[16, 32],        # 特征步长（对应输入特征的下采样倍数）
    hidden_dim=128,                  # 隐藏层维度（统一通道数）
    nhead=8,                         # Transformer头数
    dim_feedforward=512,            # Transformer前馈网络维度
    use_encoder_idx=[1],             # 对第三个特征层（索引2，2048通道）应用Transformer编码器
    num_encoder_layers=1,            # Transformer编码器层数
    eval_spatial_size=None,     # 评估时的空间尺寸（用于预计算位置编码）
    version='dfine',                  # 使用DFine轻量级版本
)

# 2. 打印模型结构（可选）
print("HybridEncoder Model Structure:")
print(model)

# 3. 构造输入数据（模拟骨干网络输出的多尺度特征）
batch_size = 2
# 假设输入特征为3个层级，尺寸分别为80x80（C2）、40x40（C3）、20x20（C4）
feats = [
    torch.randn(batch_size, 512, 80, 80),   # 特征1: [B, 512, 80, 80]
    torch.randn(batch_size, 1024, 40, 40),  # 特征2: [B, 1024, 40, 40]
    # torch.randn(batch_size, 2048, 20, 20),  # 特征3: [B, 2048, 20, 20]
]

outputs = model(feats)

# 5. 验证输出结果
print("\nOutput Features Shape:")
for i, feat in enumerate(outputs):
    print(f"Output {i+1}: {feat.shape}")

# 预期输出（以DFine版本为例）：
# 经过FPN+PAN后，输出3个层级的特征，尺寸与输入对应，通道均为256
# Output 1: (2, 256, 80, 80)   （最浅层，融合后尺寸不变）
# Output 2: (2, 256, 40, 40)   （中间层，下采样一次）
# Output 3: (2, 256, 20, 20)   （最深层，下采样两次）