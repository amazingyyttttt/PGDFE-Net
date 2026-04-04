import torch
import torch.nn as nn
import torch.nn.functional as F
from .lfe import LFE

class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, shift_size=0):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        # self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.scale = dim ** -0.5
        self.attn_drop = nn.Dropout(0)
        self.at_conv = nn.Conv2d(dim, dim, 3, padding=1)
        # self.fsbm = FSBM(in_channel=dim)
        self.lem = LFE(in_channels=dim,out_channels=dim)
        self.instance_norm = nn.InstanceNorm2d(dim,affine=False)

    def attention_map(self,x):
        return torch.sigmoid(x)

    def forward(self, x):
        # x = x.permute(0, 2, 3, 1)
        x_size_ori = x.shape[-2:]

        ###########Resize###################
        x = RESIZE_padding(x, window_size=self.window_size)
        x_size = x.shape[-2:]
        ####################################
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x = self.window_partition(x, self.window_size)

        ########### 确保每张图的patch特征独立处理 ##################
        # 将 x 分成两部分（分别为 batch 中的每张图），然后独立处理
        B, C, H, W = x.shape
        # 分别提取每个 batch 中的特征
        x = self.lem(x)
        # x = self.instance_norm(x)
        # atten = self.attention_map(x)
        # out = atten * x
        ##########################################################

        out = self.window_reverse(x, self.window_size, x_size[0], x_size[1])

        if self.shift_size > 0:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        ###########Recover###################
        out = RESIZE_recover(out, x_size_ori[0], x_size_ori[1])
        ####################################

        out = self.at_conv(out)
        return out

    def window_partition(self, x, window_size):
        b, c, h, w = x.shape
        x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
        windows = windows.permute(0, 3, 1, 2)
        return windows

    def window_reverse(self, windows, window_size, h, w):
        windows = windows.permute(0, 2, 3, 1)
        b = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
        x = x.permute(0, 3, 1, 2)
        return x


def RESIZE_padding(x, window_size):
    h, w = x.shape[-2:]
    if h % window_size != 0:
        padd_h = (h // window_size + 1) * window_size - h
    else:
        padd_h = 0

    if w % window_size != 0:
        padd_w = (w // window_size + 1) * window_size - w
    else:
        padd_w = 0

    padding = (0, padd_w, 0, padd_h)

    out = F.pad(x, padding, mode='constant', value=0)
    return out


def RESIZE_recover(x, h_ori, w_ori):
    x_size = x.shape[-2:]
    dh = x_size[0] - h_ori
    dw = x_size[1] - w_ori
    if dh != 0:
        x = x[:, :, : - dh, :]
    if dw != 0:
        x = x[:, :, :, : - dw]
    out = x
    return out

class FRE(nn.Module):
    def __init__(self, in_channels):
        super(FRE, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.branch2 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0))
        self.branch3 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3))

        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        shutcut = x
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        add = out1 + out2 + out3 + shutcut

        pooled = self.adaptive_pool(x)  # [B, C, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [B, C]
        fc_out = self.fc(pooled)  # [B, C]
        fc_out = fc_out.view(fc_out.size(0), fc_out.size(1), 1, 1)  # [B, C, 1, 1]

        out = add * fc_out

        return out

if __name__ == '__main__':
    input_shape = [2, 256, 128, 128]
    input_tensor = torch.randn(input_shape)

    www = WindowAttention(dim=256, window_size=7)
    output_tensor = www(input_tensor)

    print(f"输入形状：{input_tensor.shape}")
    print(f"输出形状：{output_tensor.shape}")

##################################################################################################
class PEC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PEC, self).__init__()

        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)

        self.output_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x, y):
        # 第一条分支
        branch1 = self.conv3(x)
        branch1 = F.relu(branch1)
        branch1 = self.conv5(branch1)
        branch1 = F.relu(branch1)

        # 第二条分支
        branch2 = self.conv7(y)
        branch2 = F.relu(branch2)



        #合并分支
        combined = torch.cat([branch1, branch2], dim=1)  # (batch_size, in_channels * 2, height, width)

        # 输出卷积
        out = self.output_conv(combined)
        # out += x  # 残差连接

        return out

class FSBM(nn.Module):
    def __init__(self, in_channel):
        super(FSBM, self).__init__()
        self.pec = PEC(in_channels=in_channel, out_channels=in_channel)  # PEC模块

        # 定义卷积、池化操作
        self.stripconv = nn.Conv2d(in_channel, 1, 1, 1, 0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, fm):
        # b, c, w, h = fm.shape

        # 对每个patch进行处理
        fms_conv = self.stripconv(fm)  # 1*1卷积层，转换成1通道
        fms_pool = self.avgpool(fms_conv)  # 对每个区域做池化

        # softmax 权重计算，计算每个位置的激活权重
        fms_softmax = torch.softmax(fms_pool, dim=2)  # 每个部分都有一个得分 [B*C*K*1]

        # boost操作，放大特征
        alpha = 0.5
        fms_boost = fm + alpha * (fm * fms_softmax)

        # suppress操作，抑制不重要的特征
        fms_max = torch.max(fms_softmax, dim=2, keepdim=True)[0]
        fms_softmax_suppress = torch.clamp((fms_softmax < fms_max).float(), min=0.5)  # 抑制不重要的部分

        fms_suppress = fm * fms_softmax_suppress  # 对抑制部分进行处理

        # 使用PEC模块进行处理
        out = self.pec(fms_boost, fms_suppress)  # 将增强和抑制后的特征送入PEC

        return out

# 示例使用
# if __name__ == "__main__":
#     input_tensor = torch.randn(722, 256, 7, 7)  # 示例输入（batch_size, channels, patch_height, patch_width）
#     fsbm = FSBM(in_channel=256)
#
#     output = fsbm(input_tensor)
#     print("输出形状:", output.shape)