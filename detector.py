import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


def _make_head(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, out_ch, 1),
    )


class DetectionHead(nn.Module):
    """
    CenterPoint-style anchor-free detection head.

    Produces four output maps from the fused BEV feature map:
      heatmap       — (1, num_classes, H, W) sigmoid class confidence
      center_offset — (1, 2, H, W) sub-pixel x/y refinement
      height        — (1, 1, H, W) object centre z coordinate
      size          — (1, 3, H, W) width, length, height of bounding box
    """

    def __init__(self, in_channels: int = 64, num_classes: int = 3):
        super().__init__()
        self.heatmap_head = _make_head(in_channels, num_classes)
        self.offset_head = _make_head(in_channels, 2)
        self.height_head = _make_head(in_channels, 1)
        self.size_head = _make_head(in_channels, 3)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            feat: (1, in_channels, H, W)
        Returns:
            dict with keys: heatmap, center_offset, height, size
        """
        return {
            'heatmap': torch.sigmoid(self.heatmap_head(feat)),
            'center_offset': self.offset_head(feat),
            'height': self.height_head(feat),
            'size': self.size_head(feat),
        }


def decode_detections(
    preds: Dict[str, torch.Tensor],
    bev_range: float,
    bev_size: int,
    bev_resolution: float,
    score_threshold: float = 0.3,
    max_detections: int = 100,
) -> List[Dict]:
    """
    Convert DetectionHead outputs to a list of 3D bounding boxes.

    Uses max-pooling NMS (peak detection) to suppress non-maximum heatmap values.

    Returns:
        List of dicts, each with keys:
          class (int), x, y, z, w, l, h (float, metres in BEV frame), score (float)
    """
    heatmap = preds['heatmap'][0]        # (num_classes, H, W)
    offset = preds['center_offset'][0]   # (2, H, W)
    height = preds['height'][0]          # (1, H, W)
    size = preds['size'][0]              # (3, H, W)

    # Peak detection via max-pool NMS
    hmap_pooled = F.max_pool2d(
        heatmap.unsqueeze(0), kernel_size=3, stride=1, padding=1
    )[0]
    peaks = (heatmap == hmap_pooled) & (heatmap > score_threshold)

    boxes = []
    for cls_idx in range(heatmap.shape[0]):
        ys, xs = torch.where(peaks[cls_idx])
        for y_t, x_t in zip(ys.tolist(), xs.tolist()):
            score = heatmap[cls_idx, y_t, x_t].item()
            ox = offset[0, y_t, x_t].item()
            oy = offset[1, y_t, x_t].item()
            # Convert pixel + sub-pixel offset to metric BEV coordinates
            cx = (x_t + ox - bev_size / 2) * bev_resolution
            cy = (y_t + oy - bev_size / 2) * bev_resolution
            cz = height[0, y_t, x_t].item()
            w = abs(size[0, y_t, x_t].item())
            l = abs(size[1, y_t, x_t].item())
            h = abs(size[2, y_t, x_t].item())
            boxes.append({
                'class': cls_idx, 'x': cx, 'y': cy, 'z': cz,
                'w': w, 'l': l, 'h': h, 'score': score,
            })
            if len(boxes) >= max_detections:
                break

    return boxes
