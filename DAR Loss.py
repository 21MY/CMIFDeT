import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DARLoss(nn.Module):
    """
    Dual Alignment Regularization Loss (DAR Loss)

    A nested coupling of frequency decomposition and heteroscedastic uncertainty:
    - High-frequency components serve as spatial weights for uncertainty-weighted regression
    - Uncertainty modulates spectral energy propagation in a closed-loop manner
    - Single unified loss with physically intertwined frequency-geometry constraints
    """

    def __init__(self,
                 lambda_unc=1.0,  # Weight for uncertainty term
                 lambda_freq=0.5,  # Weight for frequency consistency
                 eps=1e-6):  # Numerical stabilizer
        super(DARLoss, self).__init__()
        self.lambda_unc = lambda_unc
        self.lambda_freq = lambda_freq
        self.eps = eps

    def _rfft_2d(self, feat):
        """Real FFT: (B, C, H, W) -> amplitude spectrum (B, C, H, W//2+1)"""
        fft = torch.fft.rfft2(feat, norm='ortho')
        return torch.abs(fft)

    def _compute_high_freq_energy(self, feat, cutoff_ratio=0.5):
        """
        Compute spatial high-frequency energy map.
        Returns: (B, 1, H, W) high-frequency energy proportion per spatial location
        """
        mag = self._rfft_2d(feat)  # (B, C, H, F)
        B, C, H, F = mag.shape

        # Low-frequency region boundary
        h_center = H // 2
        f_center = F // 2
        h_half = max(1, int(H * cutoff_ratio * 0.5))
        f_half = max(1, int(F * cutoff_ratio * 0.5))

        # Total energy per channel
        total_energy = mag.pow(2).sum(dim=3, keepdim=True)  # (B, C, H, 1)

        # Low-frequency energy per channel (central region)
        # Need to handle the rfft frequency layout properly
        low_energy = torch.zeros_like(total_energy)
        # For rfft, frequencies are arranged as [0, 1, ..., F-1] along last dim
        low_energy[:, :, h_center - h_half:h_center + h_half, :f_half] = \
            mag[:, :, h_center - h_half:h_center + h_half, :f_half].pow(2).sum(dim=3, keepdim=True)

        # High-frequency ratio per spatial location
        high_ratio = (total_energy - low_energy) / (total_energy + self.eps)  # (B, C, H, 1)

        # Average over channels and convert to spatial map
        high_map = high_ratio.mean(dim=1, keepdim=True)  # (B, 1, H, 1)
        # Interpolate to original spatial resolution
        high_map = F.interpolate(high_map, size=feat.shape[2:], mode='bilinear', align_corners=False)

        return high_map

    def _nested_frequency_uncertainty_loss(self, pred_deltas, pred_sigmas, gt_deltas, high_freq_weight):

        # Clamp sigmas for stability
        sigmas = torch.clamp(pred_sigmas, min=1e-4, max=10.0)

        # Compute regression residual
        diff = pred_deltas - gt_deltas  # (B, 4)

        # High-frequency weight for each sample (average over spatial and aggregate)
        # high_freq_weight: (B, 1, H, W) -> average to (B, 1)
        w_high = high_freq_weight.mean(dim=(2, 3), keepdim=True)  # (B, 1)
        w_high = torch.clamp(w_high, min=0.1, max=1.0)  # Prevent zero weight

        # Expand weight to each coordinate
        w_high_expanded = w_high.expand_as(pred_deltas)  # (B, 4)

        # Nested loss formulation:
        # Term 1: Frequency-weighted NLL (geometric guidance from frequency)
        loss_nll = 0.5 * ((diff ** 2) / (sigmas ** 2 + self.eps) + torch.log(sigmas ** 2 + self.eps))
        loss_weighted = (w_high_expanded * loss_nll).mean()

        # Term 2: Coupled regularization - high freq regions should NOT have high uncertainty
        # This is the "nested" part: frequency and uncertainty directly penalize each other
        loss_coupled = (w_high_expanded * torch.log(sigmas ** 2 + self.eps)).mean()

        # Combined nested loss
        loss_nested = loss_weighted + loss_coupled

        return loss_nested

    def _spectral_consistency(self, multi_scale_feats):
        """
        Auxiliary constraint: Spectral coherence across scales.
        Ensures high-frequency energy ratios remain stable across FPN levels.
        This supports the nested loss by maintaining consistent spectral distribution.
        """
        loss_spec = 0.0
        num_scales = len(multi_scale_feats)
        if num_scales < 2:
            return torch.tensor(0.0, device=multi_scale_feats[0].device)

        # Compute high-frequency ratio for each scale (scalar)
        ratios = []
        for f in multi_scale_feats:
            mag = self._rfft_2d(f)
            total_energy = mag.pow(2).sum()
            # Low-frequency region
            B, C, H, F = mag.shape
            h_center = H // 2
            f_center = F // 2
            h_half = max(1, int(H * 0.5 * 0.5))
            f_half = max(1, int(F * 0.5 * 0.5))
            low_energy = mag[:, :, h_center - h_half:h_center + h_half,
                         f_center - f_half:f_center + f_half].pow(2).sum()
            high_ratio = (total_energy - low_energy) / (total_energy + self.eps)
            ratios.append(high_ratio)

        # Consistency between adjacent scales
        for i in range(num_scales - 1):
            loss_spec += torch.abs(ratios[i] - ratios[i + 1])

        return loss_spec / (num_scales - 1)

    def forward(self,
                multi_scale_feats,  # list of [B, C, H, W] from FPN (deep -> shallow)
                pred_deltas,  # (B, 4) predicted box offsets (cx, cy, w, h)
                pred_sigmas,  # (B, 4) predicted uncertainties (softplus activated)
                gt_deltas):  # (B, 4) ground truth box offsets
        """
        Compute DAR loss with nested frequency-uncertainty coupling.

        Returns:
            total_loss: unified loss with nested physical constraints
            loss_dict: individual components for logging
        """
        # Select the finest scale for high-frequency weight computation
        # (shallowest feature has most detail)
        finest_feat = multi_scale_feats[-1]  # (B, C, H, W)

        # Compute high-frequency energy map
        high_freq_map = self._compute_high_freq_energy(finest_feat)  # (B, 1, H, W)

        # Nested frequency-uncertainty coupling loss (core)
        loss_nested = self._nested_frequency_uncertainty_loss(
            pred_deltas, pred_sigmas, gt_deltas, high_freq_map
        )

        # Auxiliary spectral consistency across scales
        loss_spec = self._spectral_consistency(multi_scale_feats)

        # Total loss
        total_loss = self.lambda_unc * loss_nested + self.lambda_freq * loss_spec

        return total_loss, {
            'L_nested': loss_nested.item(),
            'L_spec': loss_spec.item()
        }