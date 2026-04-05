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
