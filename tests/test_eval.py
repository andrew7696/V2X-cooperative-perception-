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
