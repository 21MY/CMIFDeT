import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectralModulation(nn.Module):
    """
    Spectral Modulation Unit (SMU)

    Modulates high-frequency components of the feature map based on uncertainty gating.
    Implements the core "uncertainty-modulated spectral adaptation" mechanism.
    """

    def __init__(self, in_channels, reduction=16):
        super(SpectralModulation, self).__init__()
        # Gating network: from pooled uncertainty to spectral mask
        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=1),
            nn.Sigmoid()  # gate in [0,1]
        )
        self.in_channels = in_channels

    def forward(self, x, uncertainty):
        """
        Args:
            x: input feature (B, C, H, W)
            uncertainty: predicted sigma (B, 4) or aggregated confidence (B, 1)
        Returns:
            modulated feature (B, C, H, W)
        """
        # Generate gating signal from uncertainty
        # uncertainty aggregated to scalar per sample (average over coordinates)
        if uncertainty.dim() == 2 and uncertainty.shape[1] == 4:
            gate_scalar = uncertainty.mean(dim=1, keepdim=True)  # (B,1)
        else:
            gate_scalar = uncertainty  # assume (B,1)

        # Normalize gate to [0,1] using sigmoid if not already
        # Here we assume gate is already in [0,1] from the predictor, but we'll re-map
        gate = torch.sigmoid(1.0 - gate_scalar)  # higher uncertainty -> gate close to 0
        gate = gate.view(gate.size(0), 1, 1, 1)  # (B,1,1,1)

        # Compute FFT of input
        fft = torch.fft.rfft2(x, norm='ortho')
        mag = torch.abs(fft)
        phase = torch.angle(fft)

        # High-frequency mask: outer region (70% of spectrum)
        B, C, H, F = mag.shape
        h_center = H // 2
        f_center = F // 2
        # Create a high-pass mask (central low frequencies excluded)
        mask = torch.ones_like(mag)
        h_half = max(1, int(H * 0.3 * 0.5))  # keep 30% low frequencies
        f_half = max(1, int(F * 0.3 * 0.5))
        mask[:, :, h_center - h_half:h_center + h_half, f_center - f_half:f_center + f_half] = 0

        # Apply gate: high-frequency components are multiplied by gate
        # gate is (B,1,1,1) -> broadcast to (B,1,H,F)
        modulated_mag = mag * (1 - gate * mask)  # suppress high freq when gate is small (high uncertainty)

        # Reconstruct complex spectrum
        fft_mod = modulated_mag * torch.exp(1j * phase)

        # Inverse FFT
        x_mod = torch.fft.irfft2(fft_mod, s=x.shape[2:], norm='ortho')

        return x_mod


class UMSADecoupledHead(nn.Module):
    """
    Uncertainty-Modulated Spectral-Adaptive Decoupled Head (UMSA-Head)

    Designed to work with DAR Loss for coarse registration scenarios.
    Contains:
    - Classification branch (scale-invariant semantics)
    - Regression branch with spectral modulation driven by uncertainty
    """

    def __init__(self,
                 in_channels,  # input feature channels from FPN
                 num_classes,  # number of object classes
                 num_convs=2,  # number of conv layers per branch
                 norm='BN',  # normalization type
                 act='ReLU'):
        super(UMSADecoupledHead, self).__init__()
        self.in_channels = in_channels

        # Shared feature extraction (optional) - we keep separate for decoupling
        # Classification branch (pure spatial features)
        cls_layers = []
        for i in range(num_convs):
            if i == 0:
                ch_in = in_channels
            else:
                ch_in = in_channels
            cls_layers.append(nn.Conv2d(ch_in, in_channels, kernel_size=3, padding=1))
            if norm == 'BN':
                cls_layers.append(nn.BatchNorm2d(in_channels))
            if act == 'ReLU':
                cls_layers.append(nn.ReLU(inplace=True))
        self.cls_conv = nn.Sequential(*cls_layers)
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)

        # Regression branch with spectral modulation
        reg_layers = []
        for i in range(num_convs):
            if i == 0:
                ch_in = in_channels
            else:
                ch_in = in_channels
            reg_layers.append(nn.Conv2d(ch_in, in_channels, kernel_size=3, padding=1))
            if norm == 'BN':
                reg_layers.append(nn.BatchNorm2d(in_channels))
            if act == 'ReLU':
                reg_layers.append(nn.ReLU(inplace=True))
        self.reg_conv = nn.Sequential(*reg_layers)

        # Spectral modulation unit (placed after reg_conv or before?)
        # We place it after reg_conv to modulate features for final prediction
        self.smu = SpectralModulation(in_channels)

        # Regression outputs: deltas (4) and sigmas (4)
        self.reg_delta = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.reg_sigma = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # Classification path (untouched by spectral modulation)
        cls_feat = self.cls_conv(x)
        cls_logits = self.cls_pred(cls_feat)

        # Regression path: first extract features
        reg_feat = self.reg_conv(x)

        # Predict initial deltas and sigmas (used for gating)
        delta_raw = self.reg_delta(reg_feat)  # (B,4,H,W)
        sigma_raw = self.reg_sigma(reg_feat)  # (B,4,H,W)
        sigmas = F.softplus(sigma_raw) + 1e-4  # ensure positive

        # Aggregate uncertainty over spatial and coordinate dimensions for gating
        # We use the mean sigma across spatial locations and coordinates
        uncertainty_scalar = sigmas.mean(dim=(1, 2, 3), keepdim=True)  # (B,1,1,1)

        # Apply spectral modulation to the reg_feat using uncertainty
        reg_feat_mod = self.smu(reg_feat, uncertainty_scalar)

        # Final predictions from modulated features
        deltas = self.reg_delta(reg_feat_mod)  # (B,4,H,W)

        # For training, we output both deltas and sigmas (sigmas from unmodulated branch)
        return cls_logits, deltas, sigmas