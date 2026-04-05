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
    print(f"Saved BEV visualization -> {output_path}")
