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
