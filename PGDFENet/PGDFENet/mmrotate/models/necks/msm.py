import torch
import torch.nn as nn
import torch.nn.functional as F

class MSM(nn.Module):
    """
    Multigranularity Self-Attention Module (MSM)

    【功能】
    1. 对齐后的局部特征(aligned_local_feats)经过空间池化(max & avg)和1×1卷积后，产生注意力掩码。
    2. 掩码通过sigmoid激活后，与全局特征(global_feats)逐元素相乘，得到加权全局-局部特征。
    3. 通过concat([加权特征, 原全局特征])并用1×1卷积融合，保持输出通道一致。

    """
    def __init__(self, channels: int):
        super(MSM, self).__init__()
        self.channels = channels
        self.conv_attn = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_fuse = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def forward(self, aligned_local_feats, global_feats):
        # 池化分支
        avg_pool = F.adaptive_avg_pool2d(aligned_local_feats, 1)  # [B, C, 1, 1]
        max_pool = F.adaptive_max_pool2d(aligned_local_feats, 1)  # [B, C, 1, 1]
        # 卷积处理
        avg_out = self.conv_attn(avg_pool)
        max_out = self.conv_attn(max_pool)
        # 相加并激活
        attn_mask = torch.sigmoid(avg_out + max_out)             # [B, C, 1, 1]
        # 扩展到空间尺寸并与全局特征逐元素相乘
        weighted = global_feats * attn_mask                      # 自动广播到 [B, C, H, W]
        # 拼接并融合
        fused = torch.cat([weighted, global_feats], dim=1)       # [B, 2C, H, W]
        fused_feats = self.conv_fuse(fused)                      # [B, C, H, W]
        return fused_feats

if __name__ == "__main__":
    # 模拟输入
    aligned_local_feats = torch.rand(2, 256, 128, 128)
    global_feats = torch.rand(2, 256, 128, 128)
    # 模块实例
    msm = MSM(channels=256)
    output = msm(aligned_local_feats, global_feats)
    print("Output shape:", output.shape)
