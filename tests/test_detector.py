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
    preds = head(feat)
    boxes = decode_detections(preds, BEV_RANGE, BEV_SIZE, BEV_RESOLUTION, score_threshold=0.0)
    assert isinstance(boxes, list)


def test_decode_box_fields():
    """Decoded box must contain all required fields with valid values."""
    head = DetectionHead(in_channels=BEV_OUT_CHANNELS, num_classes=NUM_CLASSES)
    preds = head(make_feat())
    # Force a strong peak at centre so at least one box is always decoded
    preds['heatmap'][0, 0, BEV_SIZE // 2, BEV_SIZE // 2] = 1.0
    boxes = decode_detections(preds, BEV_RANGE, BEV_SIZE, BEV_RESOLUTION, score_threshold=0.5)
    assert len(boxes) >= 1, "should detect at least one box with injected peak"
    b = boxes[0]
    assert all(k in b for k in ('class', 'x', 'y', 'z', 'w', 'l', 'h', 'score'))
    assert b['score'] > 0.5
    assert b['w'] >= 0 and b['l'] >= 0 and b['h'] >= 0


def test_decode_coordinate_correctness():
    """A peak at BEV centre should decode to approximately (0, 0) world coordinates."""
    head = DetectionHead(in_channels=BEV_OUT_CHANNELS, num_classes=NUM_CLASSES)
    preds = head(make_feat())
    # Zero out offset so sub-pixel correction does not shift the result
    preds['center_offset'] = torch.zeros_like(preds['center_offset'])
    # Inject peak at grid centre (pixel BEV_SIZE//2, BEV_SIZE//2)
    preds['heatmap'] = torch.zeros_like(preds['heatmap'])
    preds['heatmap'][0, 0, BEV_SIZE // 2, BEV_SIZE // 2] = 1.0
    boxes = decode_detections(preds, BEV_RANGE, BEV_SIZE, BEV_RESOLUTION, score_threshold=0.5)
    assert len(boxes) == 1
    assert abs(boxes[0]['x']) < BEV_RESOLUTION, f"centre peak x should be ~0, got {boxes[0]['x']}"
    assert abs(boxes[0]['y']) < BEV_RESOLUTION, f"centre peak y should be ~0, got {boxes[0]['y']}"


def test_high_threshold_returns_fewer_boxes():
    head = DetectionHead(in_channels=BEV_OUT_CHANNELS, num_classes=NUM_CLASSES)
    preds = head(make_feat())
    boxes_low = decode_detections(preds, BEV_RANGE, BEV_SIZE, BEV_RESOLUTION, score_threshold=0.0)
    boxes_high = decode_detections(preds, BEV_RANGE, BEV_SIZE, BEV_RESOLUTION, score_threshold=0.9)
    assert len(boxes_high) <= len(boxes_low)
