# 1. 实例化模型（以B0版本为例）
import torch

from engine import HGNetv2

model = HGNetv2(
    name='B0',                # 模型版本，对应不同配置
    use_lab=False,            # 不使用可学习仿射块
    return_idx=[0, 1, 2, 3],  # 输出所有阶段的特征（stage1-stage4，索引0-3）
    freeze_stem_only=True,    # 仅冻结Stem块参数
    freeze_at=-1,             # 不冻结任何阶段（-1表示禁用冻结）
    freeze_norm=False,        # 不冻结BN层
    pretrained=False,         # 不加载预训练权重（因示例中无有效权重路径）
    local_model_dir='weight/hgnetv2/'  # 本地权重路径（可自定义）
)

# 2. 打印模型结构（可选）
print("HGNetv2-B0 Model Structure:")
print(model)

# 3. 构造输入数据（假设输入为3通道512x512图像，批次大小为2）
batch_size = 2
input_tensor = torch.randn(batch_size, 3, 512, 512)  # 输入形状: (B, C, H, W) = (2, 3, 512, 512)

# 4. 前向传播
outputs = model(input_tensor)
# 5. 验证输出结果
print("\nOutputs from each stage (stage index, shape):")
for i, feat in enumerate(outputs):
    print(f"Stage {i+1}: {feat.shape}")  # stage1对应索引0，输出尺寸依次为128x128, 128x128, 64x64, 32x32

# 预期输出形状（以B0为例，输入512x512）:
# Stage 1: (2, 64, 128, 128)   （Stem后尺寸缩小4倍，stage1输出通道64）
# Stage 2: (2, 256, 128, 128)  （stage2不下采样，通道256）
# Stage 3: (2, 512, 64, 64)    （stage3下采样，尺寸减半，通道512）
# Stage 4: (2, 1024, 32, 32)   （stage4下采样，尺寸再减半，通道1024）