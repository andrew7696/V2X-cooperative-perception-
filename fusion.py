import torch
import torch.nn as nn
from typing import Optional


class FusionNeck(nn.Module):
    """
    Fuse Vehicle B's BEV features with Vehicle A's aligned BEV features.

    Fusion: concatenate along channel dim → 3-layer Conv2D neck.

    Solo mode: when feat_a is None, a zero tensor is substituted for A's features
    so the same network path is used for both solo and cooperative inference.
    """

    def __init__(self, in_channels_each: int = 64, out_channels: int = 64):
        super().__init__()
        self.in_channels_each = in_channels_each
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels_each * 2, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        feat_b: torch.Tensor,
        feat_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            feat_b: (1, C, H, W) Vehicle B's own BEV features
            feat_a: (1, C, H, W) Vehicle A's aligned BEV features, or None for solo mode
        Returns:
            (1, out_channels, H, W) fused feature map
        """
        if feat_a is None:
            feat_a = torch.zeros_like(feat_b)
        x = torch.cat([feat_b, feat_a], dim=1)  # (1, 2C, H, W)
        return self.neck(x)
