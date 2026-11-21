import torch
import torch.nn as nn

from .siamese_rppg import RPPGBranch


class BiArterialDeepfakeNet(nn.Module):
    """
    Dual-Stream Bi-Arterial Deepfake Detection & rPPG Network.
    Processes:
        v_f: Forehead DiffNormalized video tensor (B, 3, T, 40, 140)
        v_c: Cheek DiffNormalized video tensor    (B, 3, T, 40, 140)
    Outputs:
        logits: (B, 2) where class 0 = FAKE, class 1 = REAL
        sig_f:  (B, T) Forehead 1D rPPG pulse
        sig_c:  (B, T) Cheek 1D rPPG pulse
    """
    def __init__(self, negative_slope: float = 0.2, dropout_rate: float = 0.35):
        super(BiArterialDeepfakeNet, self).__init__()
        # Shared 3D-CNN feature extraction branch
        self.branch = RPPGBranch(negative_slope=negative_slope)

        # Spatiotemporal Classification Head
        # Concatenates pooled forehead embedding (256) + cheek embedding (256) + difference (|f - c| = 256)
        in_dim = 256 * 3  # 768
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)  # [logit_fake, logit_real]
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branch.stage1(x)
        out = self.branch.stage2(out)
        out = self.branch.stage3(out)
        out = self.branch.stage4(out)
        out = self.branch.stage5(out)
        out = self.branch.stage6(out)  # Shape: (B, 256, T, 1, 4)
        return out

    def forward(self, v_f: torch.Tensor, v_c: torch.Tensor):
        feat_f = self.extract_features(v_f)
        feat_c = self.extract_features(v_c)

        # 1. Classification Branch
        pool_f = torch.mean(feat_f, dim=(2, 3, 4))   # (B, 256)
        pool_c = torch.mean(feat_c, dim=(2, 3, 4))   # (B, 256)
        diff_fc = torch.abs(pool_f - pool_c)         # (B, 256)

        combined = torch.cat([pool_f, pool_c, diff_fc], dim=1)  # (B, 768)
        logits = self.classifier(combined)                       # (B, 2)

        # 2. rPPG Pulse Reconstruction Branch
        sig_f = self.branch.head(self.branch.gap(feat_f)).squeeze(1).squeeze(-1).squeeze(-1)  # (B, T)
        sig_c = self.branch.head(self.branch.gap(feat_c)).squeeze(1).squeeze(-1).squeeze(-1)  # (B, T)

        return logits, sig_f, sig_c
