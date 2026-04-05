import numpy as np
import torch
import torch.nn as nn

# Shared BEV constants — imported by other modules
BEV_RANGE = 50.0        # metres in each direction from vehicle origin
BEV_RESOLUTION = 0.4    # metres per pixel
BEV_SIZE = 250          # = int(2 * BEV_RANGE / BEV_RESOLUTION)
BEV_IN_CHANNELS = 4     # max_z, min_z, log_density, mean_intensity
BEV_OUT_CHANNELS = 64


class BEVEncoder(nn.Module):
    """
    Lightweight BEV encoder.
    1. Project LiDAR point cloud to a 4-channel BEV grid (points_to_bev).
    2. Apply a 2D CNN backbone to produce a BEV_OUT_CHANNELS feature map.
    """

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(BEV_IN_CHANNELS, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, BEV_OUT_CHANNELS, 3, padding=1),
            nn.BatchNorm2d(BEV_OUT_CHANNELS),
            nn.ReLU(inplace=True),
        )

    def points_to_bev(self, points: np.ndarray) -> torch.Tensor:
        """
        Convert (N, 4) point cloud [x, y, z, intensity] to
        (BEV_IN_CHANNELS, BEV_SIZE, BEV_SIZE) BEV grid.

        Channels:
          0: max_z        — height of tallest return in each cell
          1: min_z        — height of lowest return
          2: log_density  — log1p(point count per cell)
          3: mean_intensity
        """
        bev = np.zeros((BEV_IN_CHANNELS, BEV_SIZE, BEV_SIZE), dtype=np.float32)

        if len(points) == 0:
            return torch.from_numpy(bev)

        mask = (
            (points[:, 0] >= -BEV_RANGE) & (points[:, 0] < BEV_RANGE) &
            (points[:, 1] >= -BEV_RANGE) & (points[:, 1] < BEV_RANGE)
        )
        pts = points[mask]

        if len(pts) == 0:
            return torch.from_numpy(bev)

        xi = ((pts[:, 0] + BEV_RANGE) / BEV_RESOLUTION).astype(int).clip(0, BEV_SIZE - 1)
        yi = ((pts[:, 1] + BEV_RANGE) / BEV_RESOLUTION).astype(int).clip(0, BEV_SIZE - 1)
        flat = yi * BEV_SIZE + xi

        sz = BEV_SIZE * BEV_SIZE

        max_z = np.full(sz, -np.inf, dtype=np.float32)
        np.maximum.at(max_z, flat, pts[:, 2])
        max_z[max_z == -np.inf] = 0.0
        bev[0] = max_z.reshape(BEV_SIZE, BEV_SIZE)

        min_z = np.full(sz, np.inf, dtype=np.float32)
        np.minimum.at(min_z, flat, pts[:, 2])
        min_z[min_z == np.inf] = 0.0
        bev[1] = min_z.reshape(BEV_SIZE, BEV_SIZE)

        density = np.zeros(sz, dtype=np.float32)
        np.add.at(density, flat, 1.0)
        bev[2] = np.log1p(density).reshape(BEV_SIZE, BEV_SIZE)

        intensity_sum = np.zeros(sz, dtype=np.float32)
        np.add.at(intensity_sum, flat, pts[:, 3])
        count = density
        valid = count > 0
        intensity_sum[valid] /= count[valid]
        bev[3] = intensity_sum.reshape(BEV_SIZE, BEV_SIZE)

        return torch.from_numpy(bev)

    def forward(self, points: np.ndarray) -> torch.Tensor:
        """
        Args:
            points: (N, 4) numpy array — x, y, z, intensity in vehicle frame
        Returns:
            (1, BEV_OUT_CHANNELS, BEV_SIZE, BEV_SIZE) float32 tensor
        """
        bev = self.points_to_bev(points).unsqueeze(0)   # (1, 4, H, W)
        return self.cnn(bev)                              # (1, 64, H, W)
