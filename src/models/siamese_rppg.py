"""
Siamese-rPPG Network Architecture.
Reference: "Siamese-rPPG Network: Remote Photoplethysmography Signal Estimation from Face Videos"
(Yun-Yun Tsou et al., ACM SAC 2020)
"""

from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class RPPGBranch(nn.Module):
    """
    Single branch of 3D-CNN feature extractor.
    Processes video tensor of shape: (B, C, T, H, W) where C=3, H=40, W=140.
    Outputs 1D rPPG signal of shape: (B, T)
    """
    def __init__(self, negative_slope: float = 0.2):
        super(RPPGBranch, self).__init__()

        # Six 3D CNN stages
        self.stage1 = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (B, 8, T, 20, 70)
        )

        self.stage2 = nn.Sequential(
            nn.Conv3d(8, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (B, 32, T, 10, 35)
        )

        self.stage3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (B, 64, T, 5, 17)
        )

        self.stage4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (B, 64, T, 2, 8)
        )

        self.stage5 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (B, 128, T, 1, 4)
        )

        self.stage6 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)  # (B, 256, T, 1, 4)
        )

        # Global Average Pooling on spatial dimensions (H, W) -> (B, 256, T, 1, 1)
        self.gap = nn.AdaptiveAvgPool3d((None, 1, 1))

        # 1x1x1 Conv3D to project channels from 256 to 1
        self.head = nn.Conv3d(256, 1, kernel_size=(1, 1, 1), stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            signal: (B, T)
        """
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.stage6(out)
        out = self.gap(out)
        out = self.head(out)  # Shape: (B, 1, T, 1, 1)
        out = out.squeeze(1).squeeze(-1).squeeze(-1)  # Shape: (B, T)
        return out


class SiameseRPPGNet(nn.Module):
    """
    Siamese-rPPG Network with shared weights.
    Reference: "Siamese-rPPG Network: Remote Photoplethysmography Signal Estimation from Face Videos"
    (Yun-Yun Tsou et al., ACM SAC 2020)

    Accepts either:
      - Two video streams: v_forehead (B, C, T, H, W) and v_cheek (B, C, T, H, W)
      - OR a single face tensor: x (B, C, T, H, W) where Forehead and Cheek are extracted.
    Fuses branch outputs using element-wise addition.
    """
    def __init__(self, target_roi_h: int = 40, target_roi_w: int = 140, negative_slope: float = 0.2):
        super(SiameseRPPGNet, self).__init__()
        self.target_roi_h = target_roi_h
        self.target_roi_w = target_roi_w
        # Shared branch instance
        self.shared_branch = RPPGBranch(negative_slope=negative_slope)

    def extract_rois(self, face_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits face tensor (B, C, T, H, W) into Forehead (top 1/3) and Cheek (middle 1/3).
        """
        B, C, T, H, W = face_tensor.shape
        one_third_h = H // 3
        w_inset = int(W * 0.1)
        w_start, w_end = w_inset, max(w_inset + 1, W - w_inset)

        # Forehead: top 1/3
        forehead = face_tensor[:, :, :, 0:one_third_h, w_start:w_end]
        # Cheek: middle 1/3
        cheek = face_tensor[:, :, :, one_third_h:2 * one_third_h, w_start:w_end]

        # Reshape to 4D to interpolate spatial dims: (B*T, C, H_crop, W_crop)
        f_flat = forehead.permute(0, 2, 1, 3, 4).reshape(-1, C, forehead.shape[3], forehead.shape[4])
        c_flat = cheek.permute(0, 2, 1, 3, 4).reshape(-1, C, cheek.shape[3], cheek.shape[4])

        f_resized = F.interpolate(f_flat, size=(self.target_roi_h, self.target_roi_w), mode='bilinear', align_corners=False)
        c_resized = F.interpolate(c_flat, size=(self.target_roi_h, self.target_roi_w), mode='bilinear', align_corners=False)

        # Reshape back to (B, C, T, H, W)
        f_roi = f_resized.reshape(B, T, C, self.target_roi_h, self.target_roi_w).permute(0, 2, 1, 3, 4)
        c_roi = c_resized.reshape(B, T, C, self.target_roi_h, self.target_roi_w).permute(0, 2, 1, 3, 4)
        return f_roi, c_roi

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        v_cheek: Optional[torch.Tensor] = None,
        return_branches: bool = False,
        v_forehead: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: If v_cheek is provided, this is Forehead ROI tensor (B, 3, T, 40, 140).
               If v_cheek is None, this is a full Face video tensor (B, 3, T, H, W).
            v_cheek: Optional Cheek ROI video tensor (B, 3, T, 40, 140)
            return_branches: If True, returns (fused_signal, x_forehead, x_cheek)
            v_forehead: Optional alias for x for backward compatibility with keyword callers.
        Returns:
            x_fused: Estimated rPPG signal (B, T)
        """
        if x is None and v_forehead is not None:
            x = v_forehead
        if x is None:
            raise ValueError("Expected an input tensor (x or v_forehead) to forward()")

        v_f, v_c = (x, v_cheek) if v_cheek is not None else self.extract_rois(x)
        x_forehead = self.shared_branch(v_f)
        x_cheek = self.shared_branch(v_c)

        # Fusion: Addition
        x_fused = x_forehead + x_cheek

        if return_branches:
            return x_fused, x_forehead, x_cheek
        return x_fused


# Alias for backward and drop-in compatibility
SiameseRPPG = SiameseRPPGNet

__all__ = ["RPPGBranch", "SiameseRPPGNet", "SiameseRPPG"]
