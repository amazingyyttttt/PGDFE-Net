import torch
import torch.nn as nn
import torch.nn.functional as F


class Involution(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels

        # 使用卷积来生成involution权重
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=in_channels)

    def forward(self, x):
        return self.conv(x)


class LFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LFE, self).__init__()

        # 保持输出尺寸不变的1x1卷积
        self.branch1_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # Involution操作，保持输出尺寸不变
        self.involution = Involution(out_channels, kernel_size=7, stride=1, padding=3)  # 填充为3以保持尺寸

        self.branch1_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # 第二个分支：1x1卷积，步长为1以保持尺寸不变
        self.branch2_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # 残差结构：使用1x1卷积，stride=1以保持尺寸不变
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 第一个分支
        branch1 = self.branch1_conv1(x)
        branch1 = self.involution(branch1)
        branch1 = self.branch1_conv2(branch1)

        # 第二个分支
        branch2 = self.branch2_conv1(x)

        # 合并两个分支
        out = branch1 + branch2

        # 加入残差
        residual = self.residual_conv(x)
        out += residual

        return out


# 测试代码
if __name__ == "__main__":
    # 设置输入张量的尺寸：batch_size=2, channels=256, height=128, width=128
    x = torch.randn(2, 256, 128, 128)

    # 初始化LFE模块，仅传入输入输出通道数
    lfe = LFE(in_channels=256, out_channels=256)

    # 前向传播
    output = lfe(x)

    # 打印输出尺寸
    print(f"Output shape: {output.shape}")