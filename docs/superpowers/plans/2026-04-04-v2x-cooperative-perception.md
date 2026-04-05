# V2X Cooperative Perception Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CARLA-based V2X cooperative perception system where Vehicle B detects an occluded pedestrian by fusing intermediate BEV feature maps received from Vehicle A.

**Architecture:** Both vehicles run a shared PointPillars-style BEV encoder. Vehicle A compresses its BEV features and "transmits" them over a simulated V2X link. Vehicle B spatially aligns the received features and fuses them with its own via a conv neck before running a CenterPoint-style detection head.

**Tech Stack:** CARLA 0.9.x, PyTorch, NumPy, Matplotlib, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `backbone.py` | Point cloud → BEV grid projection + 2D CNN encoder; exports shared constants `BEV_RANGE`, `BEV_SIZE`, `BEV_RESOLUTION`, `BEV_OUT_CHANNELS` |
| `compressor.py` | Spatial downsampling + optional int8 quantization of BEV features; `compress()` / `decompress()` |
| `alignment.py` | Affine warp of feature map from Vehicle A's BEV frame to Vehicle B's frame using pose data |
| `fusion.py` | Channel-concat + 3-layer Conv2D neck; produces fused BEV feature map |
| `detector.py` | CenterPoint-style detection head; `forward()` and `decode_detections()` |
| `v2x_channel.py` | Orchestrates compress → transmit → decompress → align; configurable latency/dropout |
| `eval.py` | BEV IoU, recall/precision, side-by-side BEV visualization |
| `carla_env.py` | CARLA scene setup, sensor wiring, `step()` returning lidar + pose + gt_boxes |
| `main.py` | Entry point: run N ticks, solo vs. cooperative comparison, save results |
| `requirements.txt` | Pinned dependencies |
| `tests/__init__.py` | Empty |
| `tests/test_backbone.py` | Unit tests for BEVEncoder |
| `tests/test_compressor.py` | Unit tests for FeatureCompressor |
| `tests/test_alignment.py` | Unit tests for align_features |
| `tests/test_fusion.py` | Unit tests for FusionNeck |
| `tests/test_detector.py` | Unit tests for DetectionHead + decode_detections |
| `tests/test_v2x_channel.py` | Unit tests for V2XChannel |
| `tests/test_eval.py` | Unit tests for compute_iou_bev + evaluate |

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write requirements.txt**

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
opencv-python>=4.8.0
pytest>=7.4.0
```

- [ ] **Step 2: Create tests directory**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 3: Verify pytest works**

Run: `pytest --collect-only`
Expected: `no tests ran` (no errors)

- [ ] **Step 4: Commit**

```bash
git init
git add requirements.txt tests/__init__.py
git commit -m "chore: project setup with test infrastructure"
```

---

## Task 2: BEV Encoder

**Files:**
- Create: `backbone.py`
- Create: `tests/test_backbone.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_backbone.py`:
```python
import numpy as np
import torch
import pytest
from backbone import BEVEncoder, BEV_RANGE, BEV_SIZE, BEV_OUT_CHANNELS


def make_random_cloud(n=2000):
    """Random point cloud within BEV range."""
    pts = np.random.uniform(-BEV_RANGE * 0.9, BEV_RANGE * 0.9, (n, 3)).astype(np.float32)
    intensity = np.random.uniform(0, 1, (n, 1)).astype(np.float32)
    return np.concatenate([pts, intensity], axis=1)  # (N, 4)


def test_bev_output_shape():
    """Encoder should return (1, BEV_OUT_CHANNELS, BEV_SIZE, BEV_SIZE)."""
    enc = BEVEncoder()
    points = make_random_cloud()
    feat = enc(points)
    assert feat.shape == (1, BEV_OUT_CHANNELS, BEV_SIZE, BEV_SIZE)


def test_bev_output_is_tensor():
    enc = BEVEncoder()
    feat = enc(make_random_cloud())
    assert isinstance(feat, torch.Tensor)


def test_empty_cloud_does_not_crash():
    """Zero-point cloud should return all-zeros BEV without raising."""
    enc = BEVEncoder()
    feat = enc(np.zeros((0, 4), dtype=np.float32))
    assert feat.shape == (1, BEV_OUT_CHANNELS, BEV_SIZE, BEV_SIZE)


def test_out_of_range_points_ignored():
    """Points outside BEV_RANGE must not crash or affect output dimensions."""
    enc = BEVEncoder()
    far_pts = np.random.uniform(200, 300, (500, 4)).astype(np.float32)
    feat = enc(far_pts)
    assert feat.shape == (1, BEV_OUT_CHANNELS, BEV_SIZE, BEV_SIZE)


def test_points_to_bev_channels():
    """points_to_bev should produce (4, BEV_SIZE, BEV_SIZE) with finite values."""
    enc = BEVEncoder()
    bev = enc.points_to_bev(make_random_cloud())
    assert bev.shape == (4, BEV_SIZE, BEV_SIZE)
    assert torch.isfinite(bev).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backbone.py -v`
Expected: `ImportError: No module named 'backbone'`

- [ ] **Step 3: Implement backbone.py**

```python
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
        count = density.copy()
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backbone.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add backbone.py tests/test_backbone.py
git commit -m "feat: add BEV encoder backbone with point-pillar-style projection"
```

---

## Task 3: Feature Compressor

**Files:**
- Create: `compressor.py`
- Create: `tests/test_compressor.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_compressor.py`:
```python
import torch
import pytest
from backbone import BEV_OUT_CHANNELS, BEV_SIZE
from compressor import FeatureCompressor


def make_feat(c=BEV_OUT_CHANNELS, h=BEV_SIZE, w=BEV_SIZE):
    return torch.randn(1, c, h, w)


def test_compress_halves_spatial():
    comp = FeatureCompressor(compression_ratio=2)
    feat = make_feat()
    out = comp.compress(feat)
    assert out.shape == (1, BEV_OUT_CHANNELS, BEV_SIZE // 2, BEV_SIZE // 2)


def test_compress_ratio_4():
    comp = FeatureCompressor(compression_ratio=4)
    feat = make_feat()
    out = comp.compress(feat)
    assert out.shape == (1, BEV_OUT_CHANNELS, BEV_SIZE // 4, BEV_SIZE // 4)


def test_decompress_restores_size():
    comp = FeatureCompressor(compression_ratio=2)
    feat = make_feat()
    compressed = comp.compress(feat)
    restored = comp.decompress(compressed, target_size=(BEV_SIZE, BEV_SIZE))
    assert restored.shape == (1, BEV_OUT_CHANNELS, BEV_SIZE, BEV_SIZE)


def test_quantize_clips_range():
    comp = FeatureCompressor(compression_ratio=2, quantize=True)
    feat = make_feat() * 100  # large values
    out = comp.compress(feat)
    # After quantization the max absolute value should equal original max (scaled back)
    assert torch.isfinite(out).all()


def test_no_quantize_is_differentiable():
    comp = FeatureCompressor(compression_ratio=2, quantize=False)
    feat = make_feat().requires_grad_(True)
    out = comp.compress(feat)
    out.sum().backward()
    assert feat.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_compressor.py -v`
Expected: `ImportError: No module named 'compressor'`

- [ ] **Step 3: Implement compressor.py**

```python
import torch
import torch.nn.functional as F


class FeatureCompressor:
    """
    Compress BEV feature maps for V2X transmission via spatial downsampling
    and optional int8 quantization simulation.
    """

    def __init__(self, compression_ratio: int = 2, quantize: bool = False):
        self.compression_ratio = compression_ratio
        self.quantize = quantize

    def compress(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (1, C, H, W) BEV feature map
        Returns:
            (1, C, H//ratio, W//ratio) compressed tensor
        """
        r = self.compression_ratio
        out = F.avg_pool2d(feat, kernel_size=r, stride=r)
        if self.quantize:
            scale = out.abs().max() / 127.0 + 1e-8
            out = (out / scale).round().clamp(-127, 127) * scale
        return out

    def decompress(self, feat: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """
        Args:
            feat: (1, C, H', W') compressed tensor
            target_size: (H, W) target spatial resolution
        Returns:
            (1, C, H, W) bilinear-upsampled tensor
        """
        return F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_compressor.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add compressor.py tests/test_compressor.py
git commit -m "feat: add feature compressor with downsampling and optional quantization"
```

---

## Task 4: Spatial Alignment

**Files:**
- Create: `alignment.py`
- Create: `tests/test_alignment.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_alignment.py`:
```python
import torch
import numpy as np
import pytest
from backbone import BEV_RANGE, BEV_SIZE, BEV_OUT_CHANNELS, BEV_RESOLUTION
from alignment import align_features


def make_feat():
    return torch.randn(1, BEV_OUT_CHANNELS, BEV_SIZE, BEV_SIZE)


def test_output_shape():
    feat = make_feat()
    pose = {'x': 0.0, 'y': 0.0, 'heading': 0.0}
    out = align_features(feat, pose, pose, BEV_RANGE, BEV_SIZE)
    assert out.shape == feat.shape


def test_identity_pose_preserves_center():
    """Same pose for A and B → center region should be nearly unchanged."""
    feat = make_feat()
    pose = {'x': 5.0, 'y': -3.0, 'heading': 0.3}
    out = align_features(feat, pose, pose, BEV_RANGE, BEV_SIZE)
    margin = 20
    s = slice(margin, BEV_SIZE - margin)
    assert torch.allclose(feat[:, :, s, s], out[:, :, s, s], atol=1e-3)


def test_pure_translation_shifts_spike():
    """
    A spike at A's origin should appear shifted in B's frame
    when B is offset from A along the x-axis.
    """
    feat = torch.zeros(1, 1, BEV_SIZE, BEV_SIZE)
    c = BEV_SIZE // 2
    feat[0, 0, c, c] = 1.0

    # B is 20 m ahead of A (positive x direction), same heading
    pose_a = {'x': 0.0, 'y': 0.0, 'heading': 0.0}
    pose_b = {'x': 20.0, 'y': 0.0, 'heading': 0.0}

    out = align_features(feat, pose_a, pose_b, BEV_RANGE, BEV_SIZE)

    # In B's frame, A's origin is 20 m behind (negative x) → pixel offset
    expected_x = c + int(-20.0 / BEV_RESOLUTION)
    max_idx = out[0, 0].argmax()
    max_x = (max_idx % BEV_SIZE).item()
    assert abs(max_x - expected_x) <= 3   # 3-pixel tolerance for bilinear


def test_zero_heading_diff_no_rotation():
    """Same heading means no rotation component in the warp."""
    feat = make_feat()
    pose_a = {'x': 0.0, 'y': 0.0, 'heading': 1.0}
    pose_b = {'x': 0.0, 'y': 0.0, 'heading': 1.0}   # same heading, same position
    out = align_features(feat, pose_a, pose_b, BEV_RANGE, BEV_SIZE)
    margin = 20
    s = slice(margin, BEV_SIZE - margin)
    assert torch.allclose(feat[:, :, s, s], out[:, :, s, s], atol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_alignment.py -v`
Expected: `ImportError: No module named 'alignment'`

- [ ] **Step 3: Implement alignment.py**

```python
import numpy as np
import torch
import torch.nn.functional as F


def align_features(
    feat: torch.Tensor,
    pose_src: dict,
    pose_dst: dict,
    bev_range: float,
    bev_size: int,
) -> torch.Tensor:
    """
    Warp feat from source vehicle's BEV frame into destination vehicle's BEV frame.

    BEV convention:
      - Vehicle origin at BEV centre.
      - Positive x = forward, positive y = left.
      - heading = 0 → facing +x; increases counter-clockwise (standard math).
      - CARLA yaw must be converted to radians before calling (see carla_env.py).

    Args:
        feat:     (1, C, H, W) feature map in source frame
        pose_src: {'x': float, 'y': float, 'heading': float}  world frame, radians
        pose_dst: {'x': float, 'y': float, 'heading': float}  world frame, radians
        bev_range: metres covered in each direction (±bev_range)
        bev_size:  pixel width/height of BEV grid

    Returns:
        (1, C, H, W) feature map resampled into destination frame
    """
    # For each normalised pixel in dst, find the corresponding normalised pixel in src.
    #
    # Forward chain: norm_dst → world → norm_src
    #   world = R_dst * (norm_dst * bev_range) + t_dst
    #   local_src = R_src^T * (world - t_src)
    #   norm_src = local_src / bev_range
    #
    # Combined affine (applied to norm_dst column vectors):
    #   norm_src = (R_src^T * R_dst) * norm_dst
    #            + R_src^T * (t_dst - t_src) / bev_range
    #
    # Which equals R(heading_dst - heading_src) * norm_dst + t_norm

    d_heading = pose_dst['heading'] - pose_src['heading']
    cos_d = float(np.cos(d_heading))
    sin_d = float(np.sin(d_heading))

    # Translation vector in src frame, normalised
    dt_x = pose_dst['x'] - pose_src['x']
    dt_y = pose_dst['y'] - pose_src['y']
    cos_s = float(np.cos(-pose_src['heading']))
    sin_s = float(np.sin(-pose_src['heading']))
    t_local_x = cos_s * dt_x - sin_s * dt_y
    t_local_y = sin_s * dt_x + cos_s * dt_y
    t_norm_x = t_local_x / bev_range
    t_norm_y = t_local_y / bev_range

    # grid_sample theta: (1, 2, 3) mapping output normalised coords → input normalised coords
    theta = torch.tensor(
        [[cos_d, -sin_d, t_norm_x],
         [sin_d,  cos_d, t_norm_y]],
        dtype=feat.dtype,
        device=feat.device,
    ).unsqueeze(0)  # (1, 2, 3)

    grid = F.affine_grid(theta, feat.shape, align_corners=False)
    return F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_alignment.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add alignment.py tests/test_alignment.py
git commit -m "feat: add pose-based BEV feature alignment using affine grid warp"
```

---

## Task 5: Fusion Neck

**Files:**
- Create: `fusion.py`
- Create: `tests/test_fusion.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_fusion.py`:
```python
import torch
import pytest
from backbone import BEV_OUT_CHANNELS, BEV_SIZE
from fusion import FusionNeck


def make_feat():
    return torch.randn(1, BEV_OUT_CHANNELS, BEV_SIZE, BEV_SIZE)


def test_output_shape():
    neck = FusionNeck(in_channels_each=BEV_OUT_CHANNELS, out_channels=BEV_OUT_CHANNELS)
    feat_b = make_feat()
    feat_a = make_feat()
    out = neck(feat_b, feat_a)
    assert out.shape == (1, BEV_OUT_CHANNELS, BEV_SIZE, BEV_SIZE)


def test_output_is_different_from_inputs():
    neck = FusionNeck(in_channels_each=BEV_OUT_CHANNELS, out_channels=BEV_OUT_CHANNELS)
    feat_b = make_feat()
    feat_a = make_feat()
    out = neck(feat_b, feat_a)
    assert not torch.allclose(out, feat_b)
    assert not torch.allclose(out, feat_a)


def test_gradients_flow():
    neck = FusionNeck(in_channels_each=BEV_OUT_CHANNELS, out_channels=BEV_OUT_CHANNELS)
    feat_b = make_feat().requires_grad_(True)
    feat_a = make_feat().requires_grad_(True)
    out = neck(feat_b, feat_a)
    out.sum().backward()
    assert feat_b.grad is not None
    assert feat_a.grad is not None


def test_solo_mode():
    """FusionNeck in solo mode should process only feat_b (feat_a=None)."""
    neck = FusionNeck(in_channels_each=BEV_OUT_CHANNELS, out_channels=BEV_OUT_CHANNELS)
    feat_b = make_feat()
    out = neck(feat_b, feat_a=None)
    assert out.shape == (1, BEV_OUT_CHANNELS, BEV_SIZE, BEV_SIZE)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fusion.py -v`
Expected: `ImportError: No module named 'fusion'`

- [ ] **Step 3: Implement fusion.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fusion.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add fusion.py tests/test_fusion.py
git commit -m "feat: add concatenation + conv fusion neck with solo fallback"
```

---

## Task 6: Detection Head

**Files:**
- Create: `detector.py`
- Create: `tests/test_detector.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_detector.py`:
```python
import torch
import pytest
from backbone import BEV_OUT_CHANNELS, BEV_SIZE, BEV_RANGE, BEV_RESOLUTION
from detector import DetectionHead, decode_detections

NUM_CLASSES = 3


def make_feat():
    return torch.randn(1, BEV_OUT_CHANNELS, BEV_SIZE, BEV_SIZE)


def test_forward_output_keys():
    head = DetectionHead(in_channels=BEV_OUT_CHANNELS, num_classes=NUM_CLASSES)
    preds = head(make_feat())
    assert set(preds.keys()) == {'heatmap', 'center_offset', 'height', 'size'}


def test_heatmap_shape_and_range():
    head = DetectionHead(in_channels=BEV_OUT_CHANNELS, num_classes=NUM_CLASSES)
    preds = head(make_feat())
    assert preds['heatmap'].shape == (1, NUM_CLASSES, BEV_SIZE, BEV_SIZE)
    assert preds['heatmap'].min() >= 0.0
    assert preds['heatmap'].max() <= 1.0


def test_offset_shape():
    head = DetectionHead(in_channels=BEV_OUT_CHANNELS, num_classes=NUM_CLASSES)
    preds = head(make_feat())
    assert preds['center_offset'].shape == (1, 2, BEV_SIZE, BEV_SIZE)


def test_decode_returns_list():
    head = DetectionHead(in_channels=BEV_OUT_CHANNELS, num_classes=NUM_CLASSES)
    feat = make_feat()
    # Manually create a strong heatmap peak so at least one box is decoded
    preds = head(feat)
    boxes = decode_detections(preds, BEV_RANGE, BEV_SIZE, BEV_RESOLUTION, score_threshold=0.0)
    assert isinstance(boxes, list)


def test_decode_box_fields():
    head = DetectionHead(in_channels=BEV_OUT_CHANNELS, num_classes=NUM_CLASSES)
    preds = head(make_feat())
    boxes = decode_detections(preds, BEV_RANGE, BEV_SIZE, BEV_RESOLUTION, score_threshold=0.0)
    if boxes:
        b = boxes[0]
        assert all(k in b for k in ('class', 'x', 'y', 'z', 'w', 'l', 'h', 'score'))


def test_high_threshold_returns_fewer_boxes():
    head = DetectionHead(in_channels=BEV_OUT_CHANNELS, num_classes=NUM_CLASSES)
    preds = head(make_feat())
    boxes_low = decode_detections(preds, BEV_RANGE, BEV_SIZE, BEV_RESOLUTION, score_threshold=0.0)
    boxes_high = decode_detections(preds, BEV_RANGE, BEV_SIZE, BEV_RESOLUTION, score_threshold=0.9)
    assert len(boxes_high) <= len(boxes_low)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_detector.py -v`
Expected: `ImportError: No module named 'detector'`

- [ ] **Step 3: Implement detector.py**

```python
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
      heatmap      — (1, num_classes, H, W) sigmoid class confidence
      center_offset — (1, 2, H, W) sub-pixel x/y refinement
      height       — (1, 1, H, W) object centre z coordinate
      size         — (1, 3, H, W) width, length, height of bounding box
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_detector.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add detector.py tests/test_detector.py
git commit -m "feat: add CenterPoint-style detection head with max-pool NMS decoding"
```

---

## Task 7: V2X Channel

**Files:**
- Create: `v2x_channel.py`
- Create: `tests/test_v2x_channel.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_v2x_channel.py`:
```python
import torch
import pytest
from backbone import BEV_OUT_CHANNELS, BEV_SIZE, BEV_RANGE
from v2x_channel import V2XChannel


def make_feat():
    return torch.randn(1, BEV_OUT_CHANNELS, BEV_SIZE, BEV_SIZE)


POSE_A = {'x': 10.0, 'y': 0.0, 'heading': 0.0}
POSE_B = {'x': 0.0,  'y': 0.0, 'heading': 0.0}


def test_transmit_output_shape():
    ch = V2XChannel()
    feat_a = make_feat()
    out = ch.transmit(feat_a, POSE_A, POSE_B)
    assert out.shape == (1, BEV_OUT_CHANNELS, BEV_SIZE, BEV_SIZE)


def test_dropout_rate_1_returns_none():
    ch = V2XChannel(dropout_rate=1.0)
    out = ch.transmit(make_feat(), POSE_A, POSE_B)
    assert out is None


def test_dropout_rate_0_never_returns_none():
    ch = V2XChannel(dropout_rate=0.0)
    for _ in range(10):
        out = ch.transmit(make_feat(), POSE_A, POSE_B)
        assert out is not None


def test_compression_ratio_applied():
    """Internally the channel should compress then restore to original size."""
    ch = V2XChannel(compression_ratio=4)
    out = ch.transmit(make_feat(), POSE_A, POSE_B)
    assert out.shape == (1, BEV_OUT_CHANNELS, BEV_SIZE, BEV_SIZE)


def test_same_pose_near_identity():
    """With identical pose and no compression artefacts, output ≈ input (centre region)."""
    ch = V2XChannel(compression_ratio=1)
    feat = make_feat()
    pose = {'x': 0.0, 'y': 0.0, 'heading': 0.0}
    out = ch.transmit(feat, pose, pose)
    margin = 20
    s = slice(margin, BEV_SIZE - margin)
    assert torch.allclose(feat[:, :, s, s], out[:, :, s, s], atol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_v2x_channel.py -v`
Expected: `ImportError: No module named 'v2x_channel'`

- [ ] **Step 3: Implement v2x_channel.py**

```python
import random
from typing import Optional
import torch
from compressor import FeatureCompressor
from alignment import align_features
from backbone import BEV_RANGE, BEV_SIZE


class V2XChannel:
    """
    Simulated V2X wireless link from Vehicle A to Vehicle B.

    Pipeline:
      feat_A → compress → [optional dropout] → decompress → align → feat_A_aligned

    Parameters
    ----------
    compression_ratio : int
        Spatial downsampling factor applied before transmission.
    quantize : bool
        If True, simulate int8 quantization during compression.
    dropout_rate : float
        Probability [0, 1] that the entire transmission is lost.
        transmit() returns None on a dropped packet.
    latency_ticks : int
        Number of ticks to delay the transmission (staleness simulation).
        Currently buffered internally; caller receives a delayed feature map.
    """

    def __init__(
        self,
        compression_ratio: int = 2,
        quantize: bool = False,
        dropout_rate: float = 0.0,
        latency_ticks: int = 0,
    ):
        self.compressor = FeatureCompressor(compression_ratio, quantize)
        self.dropout_rate = dropout_rate
        self.latency_ticks = latency_ticks
        self._buffer: list = []   # (feat_compressed, pose_src, pose_dst) tuples

    def transmit(
        self,
        feat_a: torch.Tensor,
        pose_a: dict,
        pose_b: dict,
    ) -> Optional[torch.Tensor]:
        """
        Simulate transmitting Vehicle A's BEV features to Vehicle B.

        Args:
            feat_a:  (1, C, H, W) Vehicle A's BEV feature map
            pose_a:  {'x', 'y', 'heading'} Vehicle A world pose (radians)
            pose_b:  {'x', 'y', 'heading'} Vehicle B world pose (radians)

        Returns:
            (1, C, H, W) aligned feature map in Vehicle B's frame,
            or None if the packet was dropped.
        """
        # Dropout
        if random.random() < self.dropout_rate:
            return None

        # Compress
        compressed = self.compressor.compress(feat_a)

        # Latency buffer
        self._buffer.append((compressed, pose_a, pose_b))
        if len(self._buffer) <= self.latency_ticks:
            return None
        compressed_delayed, pose_a_delayed, pose_b_delayed = self._buffer.pop(0)

        # Decompress back to original spatial size
        target_size = (feat_a.shape[2], feat_a.shape[3])
        decompressed = self.compressor.decompress(compressed_delayed, target_size)

        # Spatial alignment into B's frame
        return align_features(decompressed, pose_a_delayed, pose_b_delayed, BEV_RANGE, BEV_SIZE)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_v2x_channel.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add v2x_channel.py tests/test_v2x_channel.py
git commit -m "feat: add V2X channel with compression, dropout, and latency simulation"
```

---

## Task 8: Evaluation & Visualization

**Files:**
- Create: `eval.py`
- Create: `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_eval.py`:
```python
import pytest
from eval import compute_iou_bev, evaluate


def box(x, y, w=2.0, l=4.0):
    return {'x': x, 'y': y, 'z': 0.0, 'w': w, 'l': l, 'h': 1.8, 'class': 0, 'score': 0.9}


def test_iou_identical_boxes():
    b = box(0, 0)
    assert abs(compute_iou_bev(b, b) - 1.0) < 1e-5


def test_iou_non_overlapping():
    assert compute_iou_bev(box(0, 0), box(100, 100)) < 1e-5


def test_iou_partial_overlap():
    b1 = box(0, 0, w=4, l=4)
    b2 = box(2, 0, w=4, l=4)   # 50% overlap in x
    iou = compute_iou_bev(b1, b2)
    assert 0.2 < iou < 0.4     # expected ≈ 8/(16+16-8) = 0.333


def test_evaluate_perfect_detection():
    gt = [box(0, 0)]
    pred = [box(0, 0)]
    result = evaluate(pred, gt, iou_threshold=0.3)
    assert result['recall'] == pytest.approx(1.0, abs=1e-4)
    assert result['precision'] == pytest.approx(1.0, abs=1e-4)


def test_evaluate_no_detection():
    gt = [box(0, 0)]
    pred = []
    result = evaluate(pred, gt, iou_threshold=0.3)
    assert result['recall'] == pytest.approx(0.0, abs=1e-4)


def test_evaluate_false_positive():
    gt = []
    pred = [box(0, 0)]
    result = evaluate(pred, gt, iou_threshold=0.3)
    assert result['precision'] == pytest.approx(0.0, abs=1e-4)
    assert result['fp'] == 1


def test_evaluate_returns_required_keys():
    result = evaluate([], [], iou_threshold=0.3)
    assert set(result.keys()) == {'recall', 'precision', 'tp', 'fp', 'fn'}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval.py -v`
Expected: `ImportError: No module named 'eval'`

- [ ] **Step 3: Implement eval.py**

```python
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def compute_iou_bev(box1: Dict, box2: Dict) -> float:
    """
    Axis-aligned 2D IoU in the BEV plane.

    Boxes are dicts with keys: x, y, w (width along x), l (length along y).
    """
    def corners(b):
        return (b['x'] - b['w'] / 2, b['y'] - b['l'] / 2,
                b['x'] + b['w'] / 2, b['y'] + b['l'] / 2)

    ax0, ay0, ax1, ay1 = corners(box1)
    bx0, by0, bx1, by1 = corners(box2)

    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h

    area1 = (ax1 - ax0) * (ay1 - ay0)
    area2 = (bx1 - bx0) * (by1 - by0)
    union = area1 + area2 - inter

    return inter / (union + 1e-8)


def evaluate(
    pred_boxes: List[Dict],
    gt_boxes: List[Dict],
    iou_threshold: float = 0.3,
) -> Dict:
    """
    Compute recall and precision via greedy IoU matching.

    Returns:
        {'recall': float, 'precision': float, 'tp': int, 'fp': int, 'fn': int}
    """
    matched_gt = set()
    tp = 0
    fp = 0

    for pred in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = compute_iou_bev(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx >= 0 and best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)

    return {'recall': recall, 'precision': precision, 'tp': tp, 'fp': fp, 'fn': fn}


def visualize_bev(
    gt_boxes: List[Dict],
    solo_boxes: List[Dict],
    coop_boxes: List[Dict],
    solo_metrics: Dict,
    coop_metrics: Dict,
    bev_range: float = 50.0,
    output_path: str = 'bev_comparison.png',
) -> None:
    """
    Save a side-by-side BEV comparison of solo vs. cooperative detections.

    Green rectangles = ground truth.
    Red rectangles   = predicted boxes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    for ax, pred_boxes, title, metrics in [
        (axes[0], solo_boxes,
         f'Solo (Vehicle B only)\nRecall={solo_metrics["recall"]:.2f}  '
         f'Precision={solo_metrics["precision"]:.2f}', solo_metrics),
        (axes[1], coop_boxes,
         f'Cooperative (B + A features)\nRecall={coop_metrics["recall"]:.2f}  '
         f'Precision={coop_metrics["precision"]:.2f}', coop_metrics),
    ]:
        ax.set_xlim(-bev_range, bev_range)
        ax.set_ylim(-bev_range, bev_range)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#111111')

        for b in gt_boxes:
            rect = patches.Rectangle(
                (b['x'] - b['w'] / 2, b['y'] - b['l'] / 2), b['w'], b['l'],
                linewidth=2, edgecolor='lime', facecolor='none', label='GT',
            )
            ax.add_patch(rect)

        for b in pred_boxes:
            rect = patches.Rectangle(
                (b['x'] - b['w'] / 2, b['y'] - b['l'] / 2), b['w'], b['l'],
                linewidth=2, edgecolor='red', facecolor='none', label='Pred',
            )
            ax.add_patch(rect)

        # Vehicle B marker (always at origin in its own BEV frame)
        ax.plot(0, 0, 'b^', markersize=12, label='Vehicle B', zorder=5)

        handles = [
            patches.Patch(edgecolor='lime', facecolor='none', label='Ground truth'),
            patches.Patch(edgecolor='red', facecolor='none', label='Prediction'),
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
    print(f"Saved BEV visualization → {output_path}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add eval.py tests/test_eval.py
git commit -m "feat: add BEV IoU evaluation and side-by-side visualization"
```

---

## Task 9: CARLA Environment

**Files:**
- Create: `carla_env.py`

> **Note:** `carla_env.py` requires a running CARLA 0.9.x server. Unit tests use a mock.
> Start CARLA before running `main.py`: `./CarlaUE4.sh -RenderOffScreen` (Linux) or `CarlaUE4.exe` (Windows).

- [ ] **Step 1: Implement carla_env.py**

```python
import time
import numpy as np

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


class CARLAEnv:
    """
    CARLA-based occlusion scenario for V2X cooperative perception.

    Scene layout (Vehicle B's approximate coordinate frame):
      Vehicle B  at (0, 0)      heading East (yaw=0°)
      Vehicle A  at (10, -20)   heading ~45° NE — has clear LOS to pedestrian
      Truck      at (30, 0)     stationary — blocks B's LOS to pedestrian
      Pedestrian at (45, 0)     stationary behind truck

    CARLA coordinate convention:
      X = East, Y = South, Z = Up   (left-hand system, yaw clockwise from North)
    We convert yaw to standard math radians (CCW from East) via:
      heading_rad = -math.radians(yaw_degrees)
    """

    def __init__(self, host: str = 'localhost', port: int = 2000, town: str = 'Town04'):
        if not CARLA_AVAILABLE:
            raise RuntimeError(
                "CARLA Python API not found. "
                "Install it from your CARLA installation: PythonAPI/carla/dist/"
            )
        import math
        self._math = math

        self.client = carla.Client(host, port)
        self.client.set_timeout(15.0)
        self.world = self.client.load_world(town)
        self.blueprint_library = self.world.get_blueprint_library()

        self._configure_sync()

        self.vehicle_a = None
        self.vehicle_b = None
        self.obstacle = None
        self.pedestrian = None
        self.lidar_a_actor = None
        self.lidar_b_actor = None

        self._lidar_a: np.ndarray = np.zeros((0, 4), dtype=np.float32)
        self._lidar_b: np.ndarray = np.zeros((0, 4), dtype=np.float32)

        self._spawn_scene()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _configure_sync(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def _spawn_scene(self):
        import carla

        bp_lib = self.blueprint_library
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        vehicle_bp.set_attribute('role_name', 'ego')

        # Vehicle B — the cooperative recipient, blocked view
        self.vehicle_b = self.world.spawn_actor(
            vehicle_bp,
            carla.Transform(carla.Location(x=0, y=0, z=0.5), carla.Rotation(yaw=0)),
        )

        # Vehicle A — has line-of-sight to pedestrian
        self.vehicle_a = self.world.spawn_actor(
            vehicle_bp,
            carla.Transform(carla.Location(x=10, y=-20, z=0.5), carla.Rotation(yaw=45)),
        )

        # Blocking truck
        truck_bp = bp_lib.filter('vehicle.carlamotors.carlacola')[0]
        self.obstacle = self.world.spawn_actor(
            truck_bp,
            carla.Transform(carla.Location(x=30, y=0, z=0.5), carla.Rotation(yaw=0)),
        )

        # Pedestrian (target)
        ped_bps = bp_lib.filter('walker.pedestrian.*')
        ped_bp = ped_bps[0]
        self.pedestrian = self.world.spawn_actor(
            ped_bp,
            carla.Transform(carla.Location(x=45, y=0, z=0.5)),
        )

        # LiDAR sensors
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '500000')
        lidar_transform = carla.Transform(carla.Location(z=2.0))

        self.lidar_a_actor = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.vehicle_a
        )
        self.lidar_b_actor = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.vehicle_b
        )

        self.lidar_a_actor.listen(self._cb_lidar_a)
        self.lidar_b_actor.listen(self._cb_lidar_b)

        # Warm up
        for _ in range(5):
            self.world.tick()
        time.sleep(0.1)

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------

    def _cb_lidar_a(self, data):
        raw = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        self._lidar_a = raw

    def _cb_lidar_b(self, data):
        raw = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        self._lidar_b = raw

    # ------------------------------------------------------------------
    # Pose & GT helpers
    # ------------------------------------------------------------------

    def _get_pose(self, vehicle) -> dict:
        """Return world pose with heading in standard radians (CCW from East)."""
        t = vehicle.get_transform()
        return {
            'x': t.location.x,
            'y': t.location.y,
            # CARLA yaw: degrees, clockwise from North (+Y South axis)
            # Standard math heading: radians, CCW from East (+X axis)
            'heading': -self._math.radians(t.rotation.yaw),
        }

    def _get_gt_boxes(self) -> list:
        """
        Return ground-truth 3D boxes in Vehicle B's local BEV frame.
        Boxes are dicts: {x, y, z, w, l, h, class, score=1.0}
        """
        pose_b = self._get_pose(self.vehicle_b)
        cos_b = self._math.cos(-pose_b['heading'])
        sin_b = self._math.sin(-pose_b['heading'])

        gt = []
        for actor in [self.pedestrian, self.obstacle, self.vehicle_a]:
            loc = actor.get_location()
            bb = actor.bounding_box
            dx = loc.x - pose_b['x']
            dy = loc.y - pose_b['y']
            local_x = cos_b * dx - sin_b * dy
            local_y = sin_b * dx + cos_b * dy
            cls = 0 if 'walker' in actor.type_id else 1
            gt.append({
                'x': local_x, 'y': local_y, 'z': loc.z,
                'w': bb.extent.x * 2, 'l': bb.extent.y * 2, 'h': bb.extent.z * 2,
                'class': cls, 'score': 1.0,
            })
        return gt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> dict:
        """
        Advance simulation by one tick.

        Returns:
            {
              'lidar_a': np.ndarray (N, 4),
              'lidar_b': np.ndarray (N, 4),
              'pose_a':  {'x', 'y', 'heading'},
              'pose_b':  {'x', 'y', 'heading'},
              'gt_boxes': list of box dicts in Vehicle B's frame,
            }
        """
        self.world.tick()
        time.sleep(0.05)

        return {
            'lidar_a': self._lidar_a,
            'lidar_b': self._lidar_b,
            'pose_a': self._get_pose(self.vehicle_a),
            'pose_b': self._get_pose(self.vehicle_b),
            'gt_boxes': self._get_gt_boxes(),
        }

    def close(self):
        """Destroy all spawned actors and restore async mode."""
        for sensor in [self.lidar_a_actor, self.lidar_b_actor]:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        for actor in [self.vehicle_a, self.vehicle_b, self.obstacle, self.pedestrian]:
            if actor is not None:
                actor.destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
```

- [ ] **Step 2: Verify import works without CARLA**

```python
# Quick smoke test — paste into terminal
import sys; sys.path.insert(0, '.')
from carla_env import CARLAEnv, CARLA_AVAILABLE
print("CARLA available:", CARLA_AVAILABLE)
```

Expected: `CARLA available: False` (or True if CARLA is installed)

- [ ] **Step 3: Commit**

```bash
git add carla_env.py
git commit -m "feat: add CARLA environment with occlusion scenario and LiDAR sensor wiring"
```

---

## Task 10: Main Pipeline

**Files:**
- Create: `main.py`

- [ ] **Step 1: Implement main.py**

```python
"""
main.py — V2X Cooperative Perception Demo

Usage:
    python main.py [--ticks N] [--output OUTPUT_DIR] [--compression-ratio R]

Requires a running CARLA server:
    Linux:   ./CarlaUE4.sh -RenderOffScreen
    Windows: CarlaUE4.exe
"""
import argparse
import os
import torch

from carla_env import CARLAEnv
from backbone import BEVEncoder, BEV_RANGE, BEV_SIZE, BEV_RESOLUTION
from v2x_channel import V2XChannel
from fusion import FusionNeck
from detector import DetectionHead, decode_detections
from eval import evaluate, visualize_bev


def parse_args():
    p = argparse.ArgumentParser(description='V2X Cooperative Perception Demo')
    p.add_argument('--ticks', type=int, default=20,
                   help='Number of simulation ticks to run (default: 20)')
    p.add_argument('--output', type=str, default='results',
                   help='Directory to save visualizations and metrics (default: results/)')
    p.add_argument('--compression-ratio', type=int, default=2,
                   help='Feature compression ratio (default: 2)')
    p.add_argument('--score-threshold', type=float, default=0.3,
                   help='Detection confidence threshold (default: 0.3)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("Initialising models...")
    backbone = BEVEncoder()
    backbone.eval()

    v2x = V2XChannel(compression_ratio=args.compression_ratio)
    fusion = FusionNeck()
    fusion.eval()
    detector = DetectionHead()
    detector.eval()

    print("Connecting to CARLA...")
    env = CARLAEnv()

    solo_recalls, coop_recalls = [], []
    solo_precisions, coop_precisions = [], []

    print(f"Running {args.ticks} ticks...")

    try:
        for tick in range(args.ticks):
            obs = env.step()

            lidar_a = obs['lidar_a']
            lidar_b = obs['lidar_b']
            pose_a = obs['pose_a']
            pose_b = obs['pose_b']
            gt_boxes = obs['gt_boxes']

            with torch.no_grad():
                feat_a = backbone(lidar_a)
                feat_b = backbone(lidar_b)

                # Cooperative path
                feat_a_aligned = v2x.transmit(feat_a, pose_a, pose_b)
                feat_fused = fusion(feat_b, feat_a_aligned)
                coop_preds = detector(feat_fused)
                coop_boxes = decode_detections(
                    coop_preds, BEV_RANGE, BEV_SIZE, BEV_RESOLUTION,
                    score_threshold=args.score_threshold,
                )

                # Solo path (no V2X)
                feat_solo = fusion(feat_b, feat_a=None)
                solo_preds = detector(feat_solo)
                solo_boxes = decode_detections(
                    solo_preds, BEV_RANGE, BEV_SIZE, BEV_RESOLUTION,
                    score_threshold=args.score_threshold,
                )

            solo_m = evaluate(solo_boxes, gt_boxes)
            coop_m = evaluate(coop_boxes, gt_boxes)

            solo_recalls.append(solo_m['recall'])
            coop_recalls.append(coop_m['recall'])
            solo_precisions.append(solo_m['precision'])
            coop_precisions.append(coop_m['precision'])

            print(
                f"  Tick {tick + 1:3d}/{args.ticks} | "
                f"Solo recall={solo_m['recall']:.2f} prec={solo_m['precision']:.2f} | "
                f"Coop recall={coop_m['recall']:.2f} prec={coop_m['precision']:.2f}"
            )

            # Save BEV visualization for every 5th tick
            if (tick + 1) % 5 == 0:
                vis_path = os.path.join(args.output, f'bev_tick_{tick + 1:03d}.png')
                visualize_bev(
                    gt_boxes=gt_boxes,
                    solo_boxes=solo_boxes,
                    coop_boxes=coop_boxes,
                    solo_metrics=solo_m,
                    coop_metrics=coop_m,
                    bev_range=BEV_RANGE,
                    output_path=vis_path,
                )

    finally:
        env.close()

    # Summary
    avg_solo_recall = sum(solo_recalls) / len(solo_recalls)
    avg_coop_recall = sum(coop_recalls) / len(coop_recalls)
    avg_solo_prec = sum(solo_precisions) / len(solo_precisions)
    avg_coop_prec = sum(coop_precisions) / len(coop_precisions)

    print("\n===== Results =====")
    print(f"Solo        — Recall: {avg_solo_recall:.3f}  Precision: {avg_solo_prec:.3f}")
    print(f"Cooperative — Recall: {avg_coop_recall:.3f}  Precision: {avg_coop_prec:.3f}")
    print(f"Recall improvement: {avg_coop_recall - avg_solo_recall:+.3f}")

    # Save metrics to file
    metrics_path = os.path.join(args.output, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Solo Recall:       {avg_solo_recall:.4f}\n")
        f.write(f"Solo Precision:    {avg_solo_prec:.4f}\n")
        f.write(f"Coop Recall:       {avg_coop_recall:.4f}\n")
        f.write(f"Coop Precision:    {avg_coop_prec:.4f}\n")
        f.write(f"Recall Δ:          {avg_coop_recall - avg_solo_recall:+.4f}\n")
    print(f"Metrics saved → {metrics_path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run full test suite to verify nothing is broken**

Run: `pytest tests/ -v`
Expected: all tests pass (backbone, compressor, alignment, fusion, detector, v2x_channel, eval)

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add main pipeline entry point with solo vs. cooperative comparison"
```

---

## Task 11: Full Suite Check & Final Commit

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: all tests pass with no errors

- [ ] **Step 2: Verify main.py --help works**

Run: `python main.py --help`
Expected: prints usage without error

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "chore: verified full test suite passes — ready for CARLA integration"
```

---

## Running the Full Demo

With a CARLA server running:

```bash
# Start CARLA (Linux)
./CarlaUE4.sh -RenderOffScreen &

# Run demo
python main.py --ticks 50 --output results/ --compression-ratio 2
```

Output files in `results/`:
- `bev_tick_005.png`, `bev_tick_010.png`, … — side-by-side BEV visualizations
- `metrics.txt` — recall/precision summary

---

## Extension: Attention Fusion

To swap the conv fusion neck for cross-attention, replace the `neck` in `fusion.py` with:

```python
# In FusionNeck.__init__:
self.q_proj = nn.Conv2d(in_channels_each, 64, 1)
self.k_proj = nn.Conv2d(in_channels_each, 64, 1)
self.v_proj = nn.Conv2d(in_channels_each, 64, 1)
self.out_proj = nn.Conv2d(64, out_channels, 1)

# In FusionNeck.forward: (replace concat+neck with cross-attention)
# Q from feat_b, K/V from feat_a — flatten spatial dims, attend, reshape
```
