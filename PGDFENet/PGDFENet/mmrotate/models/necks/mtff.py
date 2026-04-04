import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Channel Shuffle
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    def forward(self, x):
        B, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, 'channels must be divisible by groups'
        x = x.view(B, g, C//g, H, W)
        x = x.permute(0,2,1,3,4).contiguous()
        return x.view(B, C, H, W)

# ----------------------
# MTFF Module
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# 轻量化 Channel Shuffle
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, 'channels must be divisible by groups'
        x = x.view(B, g, C // g, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(B, C, H, W)

# ----------------------
# 轻量化 MTFF 模块
class MTFF(nn.Module):
    def __init__(self, channels, groups=2, reduction=4):
        super().__init__()
        mid = channels // reduction
        # 1x1 减维 + Shuffle
        self.reduce = nn.Conv2d(channels * 2, mid, 1, bias=False)
        self.shuffle = ChannelShuffle(groups)
        # 深度可分离卷积替代标准卷积
        self.dw = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        # 简化全局分支：全局池化 + 1x1 投影
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(channels * 2, mid, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        # 融合
        self.sigmoid = nn.Sigmoid()
        self.fuse = nn.Conv2d(channels, channels, 1, bias=False)
        self.act = nn.SiLU()

    def forward(self, low_feat, high_feat):
        B, C, H, W = low_feat.size()
        hf = F.interpolate(high_feat, size=(H, W), mode='nearest')
        cat = torch.cat([low_feat, hf], dim=1)

        # 局部分支
        x = self.reduce(cat)
        x = self.shuffle(x)
        local = self.dw(x)

        # 全局分支
        g = self.global_pool(cat)
        global_feat = self.global_conv(g).expand_as(local)

        # 注意力融合
        w = self.sigmoid(local + global_feat)
        out = local * w + global_feat * (1 - w)
        out = self.fuse(out)
        return self.act(out)

class FMM(nn.Module):
    def __init__(self, channels, focal_levels=3):
        """
        Args:
            channels (int): input/output channel dimension.
            focal_levels (int): number of local context levels L (depthwise conv sizes = [3,5,7] by default).
        """
        super(FMM, self).__init__()
        self.L = focal_levels
        # 1×1 projection to Q + (L+1) context + Gate
        self.proj = nn.Conv2d(channels, channels * (self.L + 2), kernel_size=1)
        # Depthwise convs for local contexts
        ks = [3, 5, 7]
        assert len(ks) == self.L, "kernel sizes must match focal_levels"
        self.dw_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, k, padding=k//2, groups=channels),
                nn.GELU()
            ) for k in ks
        ])
        # Final 1×1 mix and normalization
        self.mix = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns: [B, C, H, W]
        """
        B, C, H, W = x.shape
        # 1) Projection
        proj = self.proj(x)
        # Split into Q, CTX_0…CTX_L, Gate
        q, *ctx_gate = torch.chunk(proj, self.L + 2, dim=1)
        ctxs = ctx_gate[:-1]   # list of length L+1 (CTX_0…CTX_L)
        gate = ctx_gate[-1]    # gating tensor [B, C, H, W]

        # 2) Contextualization
        local_outs = []
        inp = ctxs[0]
        for l, dw in enumerate(self.dw_convs):
            inp = dw(inp)  # CTX_l = GELU(DWConv(CTX_{l-1}))
            local_outs.append(inp * gate[:, l:l+1])
        # Global context
        ctx_global = ctxs[-1]
        g = ctx_global.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        gated_global = g * gate[:, -1:].mean(dim=(2,3), keepdim=True)

        # 3) Gated aggregation
        ctx_all = sum(local_outs) + gated_global
        ctx_all = self.mix(ctx_all)
        # Modulate Q
        out = q * ctx_all
        # LayerNorm over channel dim
        out = out.permute(0, 2, 3, 1).contiguous()
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out

# ----------------------
# Test MTFF
if __name__ == '__main__':
    x_low = torch.randn(2, 256, 128, 128)
    x_high = torch.randn(2, 256, 64, 64)
    mtff = MTFF(256, groups=4)
    fmm = FMM(channels=256, focal_levels=3)
    output = mtff(x_low, x_high)
    output1 = fmm(x_low)
    print('MTFF output shape:', output.shape)  # expect [2,256,128,128]
    print('MTFF output shape:', output1.shape)