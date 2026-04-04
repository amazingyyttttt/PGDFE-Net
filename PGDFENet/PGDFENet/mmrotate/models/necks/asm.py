import torch
import torch.nn as nn
import torch.nn.functional as F
from .mtff import FMM


class ASM_FineEnhancement(nn.Module):
    def __init__(self, coarse_patch_size=16, fine_patch_size=8, alpha=0.25, channels=256):
        """
        coarse_patch_size: 粗粒度补丁尺寸（例如16×16）
        fine_patch_size: 细粒度补丁尺寸（例如8×8）
        alpha: 选择的粗粒度补丁比例（例如0.25表示选择25%的粗补丁）
        channels: 特征通道数
        """
        super(ASM_FineEnhancement, self).__init__()
        self.coarse_patch_size = coarse_patch_size
        self.fine_patch_size = fine_patch_size
        self.alpha = alpha
        self.channels = channels
        self.fmm = FMM(channels=channels, focal_levels=3)

        self.fine_enhancement = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        B, C, H_orig, W_orig = x.size()


        cp = self.coarse_patch_size
        fp = self.fine_patch_size
        pad_h = (cp - (H_orig % cp)) % cp
        pad_w = (cp - (W_orig % cp)) % cp
        if pad_h != 0 or pad_w != 0:

            x = F.pad(x, (0, pad_w, 0, pad_h))


        B, C, H, W = x.size()


        nH_coarse = H // cp
        nW_coarse = W // cp
        L_coarse = nH_coarse * nW_coarse


        coarse_patches = x.unfold(2, cp, cp).unfold(3, cp, cp)
        coarse_patches = coarse_patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, L_coarse, C, cp, cp)
        coarse_scores = coarse_patches.abs().mean(dim=[2, 3, 4])  # [B, L_coarse]
        top_k_coarse = max(1, int(self.alpha * L_coarse))
        _, top_coarse_idx = torch.topk(coarse_scores, top_k_coarse, dim=1, largest=True)  # [B, top_k_coarse]


        nH_fine = H // fp
        nW_fine = W // fp
        L_fine = nH_fine * nW_fine


        fine_patches = x.unfold(2, fp, fp).unfold(3, fp, fp)

        fine_patches = fine_patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, L_fine, C, fp, fp)

        r = torch.div(top_coarse_idx, nW_coarse, rounding_mode='trunc') # [B, top_k_coarse]
        c = top_coarse_idx % nW_coarse  # [B, top_k_coarse]
        fine_idx1 = (2 * r) * nW_fine + (2 * c)
        fine_idx2 = (2 * r) * nW_fine + (2 * c + 1)
        fine_idx3 = (2 * r + 1) * nW_fine + (2 * c)
        fine_idx4 = (2 * r + 1) * nW_fine + (2 * c + 1)

        fine_indices = torch.cat([fine_idx1, fine_idx2, fine_idx3, fine_idx4], dim=1)
        selected_fine_patches = batch_index_select(fine_patches, fine_indices)


        B_sel, num_sel, C_sel, fpH, fpW = selected_fine_patches.size()
        selected_fine_patches = selected_fine_patches.view(B_sel * num_sel, C_sel, fpH, fpW)
        enhanced = self.fine_enhancement(selected_fine_patches)
        enhanced = enhanced.view(B_sel, num_sel, C_sel, fpH, fpW)


        fine_patches_enhanced = fine_patches.clone()
        fine_patches_enhanced = replace_patches(fine_patches_enhanced, fine_indices, enhanced)
        output = reconstruct_from_patches(fine_patches_enhanced, nH_fine, nW_fine, fp, B, C, H, W)


        output = output[:, :, :H_orig, :W_orig]
        return output


def batch_index_select(x, idx):
    """
    x: [B, L, C, p, p]
    idx: [B, M]，在 L 维度的索引
    返回选中的补丁，形状为 [B, M, C, p, p]
    """
    B, L, C, pH, pW = x.size()
    M = idx.size(1)
    offset = torch.arange(B, device=x.device).view(B, 1) * L
    idx = idx + offset  # [B, M]
    idx = idx.view(-1)
    out = x.view(B * L, C, pH, pW)[idx]
    out = out.view(B, M, C, pH, pW)
    return out


def replace_patches(original, idx, enhanced):
    """
    original: [B, L, C, p, p] 原始细补丁集合
    idx: [B, M] 要替换的补丁索引
    enhanced: [B, M, C, p, p] 增强后的补丁
    返回替换后的细补丁集合
    """
    B, L, C, pH, pW = original.size()
    M = idx.size(1)
    offset = torch.arange(B, device=original.device).view(B, 1) * L
    idx = idx + offset  # [B, M]
    idx = idx.view(-1)
    original = original.view(B * L, C, pH, pW)
    original[idx] = enhanced.view(B * M, C, pH, pW)
    original = original.view(B, L, C, pH, pW)
    return original


def reconstruct_from_patches(patches, nH, nW, p, B, C, H, W):
    """
    patches: [B, L, C, p, p]，其中 L = nH * nW
    重构为原始图像尺寸 [B, C, H, W]
    """
    patches = patches.view(B, nH, nW, C, p, p)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    output = patches.view(B, C, nH * p, nW * p)
    return output

