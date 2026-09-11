import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class DAIF(nn.Module):


    def __init__(self, channels, lambda_dilation=0.5):
        super().__init__()
        C = channels
        self.lambda_dilation = lambda_dilation

        # Sub-module 1: Disparity estimation head
        self.disparity_head = nn.Sequential(
            nn.Conv2d(2 * C, C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, 3, kernel_size=3, padding=1)  # 2 for disparity + 1 for confidence
        )

        # Learnable fusion weights for prior fusion (alpha, beta)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # disparity prior weight
        self.beta = nn.Parameter(torch.tensor(0.5))  # confidence prior weight

        # Sub-module 2: Deformable fusion conv (kernel 3x3)
        self.fusion_offset = nn.Conv2d(2 * C, 18, kernel_size=3, padding=1)  # offsets for deformable conv
        self.fusion_weight = nn.Conv2d(2 * C, C, kernel_size=3, padding=1)  # regular conv weight

        # Sub-module 3: Confidence reweighting & projection
        self.reweight = nn.Conv2d(C, C, kernel_size=1)
        self.proj = nn.Conv2d(C, C, kernel_size=1)

    def forward(self, F_rgb, F_ir, D_prior=None, C_prior=None):
        B, C, H, W = F_rgb.shape

        # ========== Sub-module 1: Multi-scale Disparity Estimation ==========
        concat = torch.cat([F_rgb, F_ir], dim=1)  # (B, 2C, H, W)
        disp_conf = self.disparity_head(concat)  # (B, 3, H, W)
        D = disp_conf[:, :2, :, :]  # (B, 2, H, W)  offset field
        C = torch.sigmoid(disp_conf[:, 2:3, :, :])  # (B, 1, H, W) confidence

        # Fuse with prior from coarser scale (if provided)
        if D_prior is not None and C_prior is not None:
            # Upsample prior to current resolution
            D_prior_up = F.interpolate(D_prior, size=(H, W), mode='bilinear', align_corners=False)
            C_prior_up = F.interpolate(C_prior, size=(H, W), mode='bilinear', align_corners=False)
            # Weighted fusion
            alpha = torch.sigmoid(self.alpha)  # constrain to (0,1)
            beta = torch.sigmoid(self.beta)
            D_fused = alpha * D + (1 - alpha) * D_prior_up
            C_fused = beta * C + (1 - beta) * C_prior_up
        else:
            D_fused = D
            C_fused = C

        # ========== Sub-module 2: Disparity-guided Deformable Fusion ==========
        # Step 2a: Deformable sampling of IR (align to RGB)
        # Compute offsets for deformable conv (kernel 3x3 -> 18 channels)
        offset = self.fusion_offset(concat)  # (B, 18, H, W)
        # Apply deformable convolution to IR features using predicted offsets
        # Note: For simplicity, we use deform_conv2d with learned offsets from concat
        # In practice, you could use D_fused directly to guide sampling
        F_ir_aligned = deform_conv2d(
            input=F_ir,
            offset=offset,
            weight=self.fusion_weight.weight,
            bias=self.fusion_weight.bias,
            stride=1,
            padding=1
        )  # (B, C, H, W)

        # Step 2b: Dynamic dilation based on disparity magnitude
        # Compute disparity magnitude map
        mag = torch.norm(D_fused, dim=1, keepdim=True)  # (B, 1, H, W)
        # dilation rate: r = 1 + lambda * mag (spatially varying)
        r = 1 + self.lambda_dilation * mag  # (B, 1, H, W)
        # Apply dynamic dilation convolution (simplified: use group-wise dilated conv)
        # Here we use a standard conv with dilation=1 for simplicity,
        # but in practice you can implement spatial-varying dilation via deformable conv.
        F_fused_raw = self.fusion_weight(torch.cat([F_rgb, F_ir_aligned], dim=1))  # (B, C, H, W)

        # ========== Sub-module 3: Confidence-aware Feature Reweighting ==========
        # Spatial modulation by confidence
        F_weighted = F_fused_raw * C_fused  # (B, C, H, W)
        F_out = F_fused_raw + self.proj(F_weighted)  # residual connection

        # Return fused feature, disparity field, and confidence map
        return F_out, D_fused, C_fused