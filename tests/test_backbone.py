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


def test_single_point_lands_in_correct_cell():
    """A single known point should land in the correct BEV cell with correct values."""
    enc = BEVEncoder()
    pts = np.array([[0.0, 0.0, 1.5, 0.8]], dtype=np.float32)
    bev = enc.points_to_bev(pts)
    center = BEV_SIZE // 2  # 125
    assert abs(bev[0, center, center].item() - 1.5) < 1e-5, "max_z wrong"
    assert abs(bev[1, center, center].item() - 1.5) < 1e-5, "min_z wrong"
    assert abs(bev[3, center, center].item() - 0.8) < 1e-5, "mean_intensity wrong"


def test_empty_cloud_does_not_crash():
    """Zero-point cloud should return all-zeros BEV without raising."""
    enc = BEVEncoder()
    feat = enc(np.zeros((0, 4), dtype=np.float32))
    assert feat.shape == (1, BEV_OUT_CHANNELS, BEV_SIZE, BEV_SIZE)


def test_out_of_range_points_ignored():
    """Points outside BEV_RANGE must produce an all-zero BEV grid."""
    enc = BEVEncoder()
    far_pts = np.random.uniform(200, 300, (500, 4)).astype(np.float32)
    bev = enc.points_to_bev(far_pts)
    assert bev.shape == (4, BEV_SIZE, BEV_SIZE)
    assert (bev == 0).all(), "out-of-range points should produce a zero BEV grid"
    feat = enc(far_pts)
    assert feat.shape == (1, BEV_OUT_CHANNELS, BEV_SIZE, BEV_SIZE)


def test_points_to_bev_channels():
    """points_to_bev should produce (4, BEV_SIZE, BEV_SIZE) with finite values."""
    enc = BEVEncoder()
    bev = enc.points_to_bev(make_random_cloud())
    assert bev.shape == (4, BEV_SIZE, BEV_SIZE)
    assert torch.isfinite(bev).all()


def test_boundary_float32_precision():
    """Float32 values just below BEV_RANGE must not cause wrong-cell scatter or crash."""
    enc = BEVEncoder()
    x32 = np.nextafter(np.float32(BEV_RANGE), np.float32(0.0))
    pts = np.array([[x32, 0.0, 1.0, 0.5]], dtype=np.float32)
    bev = enc.points_to_bev(pts)
    assert torch.isfinite(bev).all()
    assert bev[2].sum().item() > 0, "boundary point should register in density channel"
