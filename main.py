"""
main.py — V2X Cooperative Perception Demo

Usage:
    python main.py [--ticks N] [--output OUTPUT_DIR] [--compression-ratio R]

Requires a running CARLA server:
    Linux:   ./CarlaUE4.sh -RenderOffScreen
    Windows: CarlaUE4.exe
"""
import argparse
import os
import torch

from carla_env import CARLAEnv
from backbone import BEVEncoder, BEV_RANGE, BEV_SIZE, BEV_RESOLUTION
from v2x_channel import V2XChannel
from fusion import FusionNeck
from detector import DetectionHead, decode_detections
from eval import evaluate, visualize_bev


def parse_args():
    p = argparse.ArgumentParser(description='V2X Cooperative Perception Demo')
    p.add_argument('--ticks', type=int, default=20,
                   help='Number of simulation ticks to run (default: 20)')
    p.add_argument('--output', type=str, default='results',
                   help='Directory to save visualizations and metrics (default: results/)')
    p.add_argument('--compression-ratio', type=int, default=2,
                   help='Feature compression ratio (default: 2)')
    p.add_argument('--score-threshold', type=float, default=0.3,
                   help='Detection confidence threshold (default: 0.3)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("Initialising models...")
    backbone = BEVEncoder()
    backbone.eval()

    v2x = V2XChannel(compression_ratio=args.compression_ratio)
    fusion = FusionNeck()
    fusion.eval()
    detector = DetectionHead()
    detector.eval()

    print("Connecting to CARLA...")
    env = CARLAEnv()

    solo_recalls, coop_recalls = [], []
    solo_precisions, coop_precisions = [], []

    print(f"Running {args.ticks} ticks...")

    try:
        for tick in range(args.ticks):
            obs = env.step()

            lidar_a = obs['lidar_a']
            lidar_b = obs['lidar_b']
            pose_a = obs['pose_a']
            pose_b = obs['pose_b']
            gt_boxes = obs['gt_boxes']

            with torch.no_grad():
                feat_a = backbone(lidar_a)
                feat_b = backbone(lidar_b)

                # Cooperative path
                feat_a_aligned = v2x.transmit(feat_a, pose_a, pose_b)
                feat_fused = fusion(feat_b, feat_a_aligned)
                coop_preds = detector(feat_fused)
                coop_boxes = decode_detections(
                    coop_preds, BEV_RANGE, BEV_SIZE, BEV_RESOLUTION,
                    score_threshold=args.score_threshold,
                )

                # Solo path (no V2X)
                feat_solo = fusion(feat_b, feat_a=None)
                solo_preds = detector(feat_solo)
                solo_boxes = decode_detections(
                    solo_preds, BEV_RANGE, BEV_SIZE, BEV_RESOLUTION,
                    score_threshold=args.score_threshold,
                )

            solo_m = evaluate(solo_boxes, gt_boxes)
            coop_m = evaluate(coop_boxes, gt_boxes)

            solo_recalls.append(solo_m['recall'])
            coop_recalls.append(coop_m['recall'])
            solo_precisions.append(solo_m['precision'])
            coop_precisions.append(coop_m['precision'])

            print(
                f"  Tick {tick + 1:3d}/{args.ticks} | "
                f"Solo recall={solo_m['recall']:.2f} prec={solo_m['precision']:.2f} | "
                f"Coop recall={coop_m['recall']:.2f} prec={coop_m['precision']:.2f}"
            )

            # Save BEV visualization for every 5th tick
            if (tick + 1) % 5 == 0:
                vis_path = os.path.join(args.output, f'bev_tick_{tick + 1:03d}.png')
                visualize_bev(
                    gt_boxes=gt_boxes,
                    solo_boxes=solo_boxes,
                    coop_boxes=coop_boxes,
                    solo_metrics=solo_m,
                    coop_metrics=coop_m,
                    bev_range=BEV_RANGE,
                    output_path=vis_path,
                )

    finally:
        env.close()

    # Summary
    avg_solo_recall = sum(solo_recalls) / len(solo_recalls)
    avg_coop_recall = sum(coop_recalls) / len(coop_recalls)
    avg_solo_prec = sum(solo_precisions) / len(solo_precisions)
    avg_coop_prec = sum(coop_precisions) / len(coop_precisions)

    print("\n===== Results =====")
    print(f"Solo        — Recall: {avg_solo_recall:.3f}  Precision: {avg_solo_prec:.3f}")
    print(f"Cooperative — Recall: {avg_coop_recall:.3f}  Precision: {avg_coop_prec:.3f}")
    print(f"Recall improvement: {avg_coop_recall - avg_solo_recall:+.3f}")

    # Save metrics to file
    metrics_path = os.path.join(args.output, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Solo Recall:       {avg_solo_recall:.4f}\n")
        f.write(f"Solo Precision:    {avg_solo_prec:.4f}\n")
        f.write(f"Coop Recall:       {avg_coop_recall:.4f}\n")
        f.write(f"Coop Precision:    {avg_coop_prec:.4f}\n")
        f.write(f"Recall Delta:      {avg_coop_recall - avg_solo_recall:+.4f}\n")
    print(f"Metrics saved -> {metrics_path}")


if __name__ == '__main__':
    main()
