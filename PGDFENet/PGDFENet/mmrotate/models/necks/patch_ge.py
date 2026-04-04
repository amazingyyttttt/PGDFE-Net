import torch
import torch.nn as nn
import torch.nn.functional as F


def RESIZE_padding(x, k=7):
    h, w = x.shape[-2:]
    if h % k != 0:
        padd_h = (h // k + 1) * k - h
    else:
        padd_h = 0

    if w % k != 0:
        padd_w = (w // k + 1) * k - w
    else:
        padd_w = 0

    padding = (0, padd_w, 0, padd_h)   #padding = (0, padd_w, 0, padd_h)  (0, 0, 0, padd_h)

    out = F.pad(x, padding, mode='constant', value=0)
    return out


def RESIZE_recover(x, h_ori, w_ori):  #def RESIZE_recover(x, h_ori, w_ori):
    x_size = x.shape[-2:]
    dh = x_size[0] - h_ori
    dw = x_size[1] - w_ori
    if dh != 0:
        x = x[:, :, : - dh, :]
    if dw != 0:
        x = x[:, :, :, : - dw]
    out = x
    return out


# Calculate pixel-wise local attention. Costs more computations.
class Patch_AttentionV2(nn.Module):
    def __init__(self, in_channels, reduction=16, k=7, add_input=False):
        super(Patch_AttentionV2, self).__init__()
        self.pool_window = k
        self.add_input = add_input
        self.SA = nn.Sequential(
            nn.AvgPool2d(kernel_size=k + 1, stride=1, padding=k // 2),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        A = self.SA(x)

        A = F.interpolate(A, (h, w), mode='bilinear')
        output = x * A
        if self.add_input:
            output += x

        return output


# Attention embedding module. Parameters: reduction is the rate of channel reduction. pool_window should be set according to the scaling rate.
class Attention_Embedding(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, k=3, add_input=False):
        super(Attention_Embedding, self).__init__()
        self.add_input = add_input
        self.SE = nn.Sequential(
            nn.AvgPool2d(kernel_size=k + 1, stride=1, padding=k // 2),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.Sigmoid())
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_input = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.CA = ChannelAttention(in_channels=in_channels)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),  # 大核捕捉船舶部件
            nn.Sigmoid()
        )

    def forward(self, high_feat, low_feat):
        b, c, h, w = low_feat.size()
        A = self.SE(high_feat)
        A = F.interpolate(A, (h, w), mode='bilinear')
        output = low_feat * A

        if self.add_input:
            output += low_feat

        return output

class MS(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(MS, self).__init__()
        self.reduce_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=3, padding=1)
        self.dilate_conv_3x3 = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=3, padding=2, dilation=2)
        self.dilate_conv_1x3 = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=(1, 3), padding=(0, 2), dilation=2)
        self.dilate_conv_3x1 = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=(3, 1), padding=(2, 0), dilation=2)
        self.restore_conv = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        reduced = self.reduce_conv(x)
        conv3x3_out = self.conv3x3(reduced)

        dilate3x3_out = self.dilate_conv_3x3(reduced)
        dilate1x3_out = self.dilate_conv_1x3(reduced)
        dilate3x1_out = self.dilate_conv_3x1(reduced)

        fused = conv3x3_out + dilate3x3_out + dilate1x3_out + dilate3x1_out  #fused = conv3x3_out + dilate3x3_out + dilate1x3_out + dilate3x1_out

        restored = self.restore_conv(fused)

        attention = torch.sigmoid(restored)

        out = x * attention
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_pool = F.adaptive_avg_pool2d(x, 1)
        out = self.fc1(avg_pool)
        out = self.relu(out)
        out = self.fc2(out)
        attention = self.sigmoid(out)
        out = x * attention

        return out

class Patch_Attention(nn.Module):
    def __init__(self, in_channels, reduction=8, k=7, add_input=False):
        super(Patch_Attention, self).__init__()
        self.pool_window = k
        self.add_input = add_input
        self.SA = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        self.MS = MS(in_channels=in_channels)
        # self.PA = ParallelAttention(in_channels=in_channels)
        # self.CA = ChannelAttention(in_channels=in_channels)

    def forward(self, x):
        b, c, h, w = x.size()

        pool_h = h // self.pool_window
        pool_w = w // self.pool_window

        A = F.adaptive_avg_pool2d(x, (pool_h, pool_w))

        A = self.SA(A)
        # B = self.MS(A)
        # A = A + B

        A = F.interpolate(A, (h, w), mode='bilinear')
        output = x * A
        if self.add_input:
            output += x

        return output

class CustomModule(nn.Module):
    def __init__(self, in_channels):
        super(CustomModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels*2, out_channels=2, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_input = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    def forward(self, x, y):
        cat = torch.cat((x, y), dim=1)
        # add = x + y
        conv_out = self.conv(cat)

        softmax_out = F.softmax(conv_out, dim=1)


        x_weighted = x * softmax_out[:, 0:1, :, :]
        y_weighted = y * softmax_out[:, 1:2, :, :]
        x_weighted = self.conv_input(x_weighted)

        out = x_weighted + y_weighted

        return out

class PEC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PEC, self).__init__()

        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)

        self.output_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x, y):

        branch1 = self.conv3(x)
        branch1 = F.relu(branch1)
        branch1 = self.conv5(branch1)
        branch1 = F.relu(branch1)


        branch2 = self.conv7(y)
        branch2 = F.relu(branch2)


        combined = torch.cat([x, y], dim=1)  # (batch_size, in_channels * 2, height, width)


        out = self.output_conv(combined)


        return out


class FSBM(nn.Module):
    def __init__(self, in_channel, k):
        super(FSBM, self).__init__()
        self.k = k
        self.stripconv = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, 1, 0),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cat = CustomModule(in_channels=in_channel)
        self.safm = Patch_Attention(in_channels=in_channel)


    def forward(self, fm):
        fm_size_ori = fm.shape[-2:]
        fm = RESIZE_padding(fm, k=self.k)

        b, c, h, w = fm.shape
        shotlow = fm
        fm = self.safm(fm) + shotlow
        fms = torch.split(fm, h // self.k, dim=2)
        fms_conv = map(self.stripconv, fms)
        fms_pool = list(map(self.avgpool, fms_conv))
        fms_pool = torch.cat(fms_pool, dim=2)
        fms_softmax = torch.softmax(fms_pool, dim=2)  # every parts has one score [B*C*K*1]
        fms_softmax_boost = torch.repeat_interleave(fms_softmax, h // self.k, dim=2)
        alpha = 0.5
        fms_boost = fm + alpha * (fm * fms_softmax_boost)


        beta = 0.5 * torch.where(fms_softmax < 0.5, fms_softmax, 1 - fms_softmax)
        fms_max = torch.max(fms_softmax, dim=2, keepdim=True)[0]
        fms_softmax_suppress = torch.clamp((fms_softmax < fms_max).float(), min=beta)
        fms_softmax_suppress = torch.repeat_interleave(fms_softmax_suppress, h // self.k, dim=2)
        fms_suppress = fm * fms_softmax_suppress

        out = self.cat(fms_boost, fms_suppress)
        out = RESIZE_recover(out, fm_size_ori[0], fm_size_ori[1])

        return out



