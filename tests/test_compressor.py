import torch
import torch.nn.functional as F
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
    """Quantization should clamp values and the max absolute value should not exceed the input range."""
    comp = FeatureCompressor(compression_ratio=2, quantize=True)
    feat = make_feat() * 100  # large values
    out = comp.compress(feat)
    assert torch.isfinite(out).all()
    # Max absolute value of output must not exceed max absolute value of the compressed (pre-quant) input
    # (quantization never amplifies values)
    unquantized = F.avg_pool2d(feat, kernel_size=2, stride=2)
    assert out.abs().max().item() <= unquantized.abs().max().item() + 1e-4


def test_no_quantize_is_differentiable():
    comp = FeatureCompressor(compression_ratio=2, quantize=False)
    feat = make_feat().requires_grad_(True)
    out = comp.compress(feat)
    out.sum().backward()
    assert feat.grad is not None
