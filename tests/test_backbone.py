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
