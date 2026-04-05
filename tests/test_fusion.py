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
