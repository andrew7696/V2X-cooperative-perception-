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


def test_intermediate_dropout_is_stochastic():
    """With dropout_rate=0.5, over 30 trials some transmissions should succeed and some fail."""
    ch = V2XChannel(dropout_rate=0.5)
    results = [ch.transmit(make_feat(), POSE_A, POSE_B) for _ in range(30)]
    nones = sum(1 for r in results if r is None)
    non_nones = sum(1 for r in results if r is not None)
    # With p=0.5 and 30 trials, P(all same) < 2e-9 — safe to assert both occurred
    assert nones > 0, "expected some dropped packets with dropout_rate=0.5"
    assert non_nones > 0, "expected some successful transmissions with dropout_rate=0.5"
