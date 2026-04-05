# V2X Cooperative Perception — Design Spec

**Date:** 2026-04-04  
**Project:** AV Final Test 1  
**Status:** Approved

---

## Overview

A simulation-based implementation of Cooperative Perception via V2X (Vehicle-to-Everything). Two vehicles in CARLA share intermediate BEV feature maps over a simulated wireless link, enabling Vehicle B to detect a pedestrian it cannot see due to occlusion by a truck. The primary goal is to demonstrate a measurable improvement in occluded object recall when cooperative perception is enabled.

---

## Goals

- Demonstrate occlusion handling: Vehicle B detects a pedestrian it cannot see alone
- Implement intermediate (feature-level) fusion using a shared PointPillars-style backbone
- Show a before/after comparison: solo detection vs. cooperative detection
- Produce a Python script pipeline with BEV visualization output

---

## Non-Goals

- Real hardware or radio communication
- Multi-vehicle (>2) coordination
- End-to-end training of the fusion network (pre-trained or lightweight weights)
- Raw sensor sharing (early fusion) or bounding-box-only sharing (late fusion)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CARLA Simulator                          │
│  ┌──────────────────┐          ┌──────────────────┐             │
│  │    Vehicle A      │          │    Vehicle B      │            │
│  │  (has LOS to ped) │          │  (ped occluded)   │            │
│  │  LiDAR sensor    │          │  LiDAR sensor    │             │
│  └────────┬─────────┘          └────────┬─────────┘             │
└───────────┼─────────────────────────────┼───────────────────────┘
            │                             │
            ▼                             ▼
     ┌─────────────┐               ┌─────────────┐
     │  Backbone   │               │  Backbone   │
     │(PointPillars│               │(PointPillars│
     │  shared wts)│               │  shared wts)│
     └──────┬──────┘               └──────┬──────┘
            │ BEV feature map             │ BEV feature map
            ▼                             │
     ┌─────────────┐                      │
     │ Compressor  │──── V2X "tx" ───────►│
     └─────────────┘                      │
                                          ▼
                                  ┌──────────────┐
                                  │  Spatial     │
                                  │  Alignment   │
                                  └──────┬───────┘
                                         │
                                  [concat along C dim]
                                         │
                                  ┌──────▼───────┐
                                  │  Fusion Neck │
                                  │ (Conv layers)│
                                  └──────┬───────┘
                                         │
                                  ┌──────▼───────┐
                                  │  Detection   │
                                  │    Head      │
                                  └──────┬───────┘
                                         │
                                  3D bounding boxes
```

Both vehicles use the same backbone weights. All fusion occurs in BEV space. The V2X link is a Python function call with configurable latency, dropout, and compression parameters.

---

## Modules

### `carla_env.py` — Scene Setup
Spawns two ego vehicles + LiDAR sensors in CARLA. Creates the occlusion scenario: a stationary truck positioned between Vehicle B and a pedestrian, while Vehicle A has a clear line of sight. Exposes a `step()` method returning:
- `lidar_A`, `lidar_B` — raw point clouds (N×4: x, y, z, intensity)
- `pose_A`, `pose_B` — position + heading from CARLA ground truth
- `gt_boxes` — ground-truth 3D bounding boxes for all actors

### `backbone.py` — Shared Perception Backbone
Lightweight PointPillars-style encoder. Converts a LiDAR point cloud into a dense BEV feature map (H×W×C). Shared weights for both vehicles. Input: point cloud. Output: BEV tensor.

### `compressor.py` — Feature Compression
Applies spatial downsampling (default 2×) and optional int8 quantization to Vehicle A's BEV feature map before transmission. Controls the bandwidth vs. accuracy trade-off. Input: BEV tensor. Output: compressed BEV tensor.

### `v2x_channel.py` — V2X Link Simulation
Thin wrapper around the compressor → alignment pipeline. Parameters:
- `latency_ms` — delays feature map by N ticks (simulates staleness)
- `dropout_rate` — randomly drops transmission (simulates packet loss)
- `compression_ratio` — spatial downsampling factor

Default demo config: zero latency, no dropout, 2× downsampling.

### `alignment.py` — Spatial Alignment
Warps Vehicle A's compressed feature map into Vehicle B's coordinate frame using a 2D affine transform on the BEV grid, derived from ground-truth pose data. Input: compressed BEV tensor + pose_A + pose_B. Output: aligned BEV tensor in B's frame.

### `fusion.py` — Fusion Neck
Concatenates B's own BEV features with the aligned A features along the channel dimension. Passes through 2–3 Conv2D + BatchNorm + ReLU layers to produce a fused feature map. Extension point: conv layers can be replaced with a cross-attention module for an attention-based variant.

### `detector.py` — Detection Head
Lightweight anchor-free 3D detection head (CenterPoint-style) operating on the fused BEV map. Outputs 3D bounding boxes (x, y, z, w, l, h, heading). Evaluated against CARLA ground-truth annotations.

### `eval.py` — Evaluation & Visualization
Runs two passes on the same scene:
1. **Solo:** only `feat_B` fed to detector
2. **Cooperative:** `feat_fused` fed to detector

Computes recall and precision on the occluded pedestrian. Renders a BEV visualization showing ground truth boxes, solo detections, and cooperative detections overlaid with the occlusion zone.

---

## Data Flow

```
CARLA tick
  → carla_env.step()
      → lidar_A, lidar_B, pose_A, pose_B, gt_boxes

  → backbone(lidar_A) → feat_A  (H×W×C)
  → backbone(lidar_B) → feat_B  (H×W×C)

  → v2x_channel.transmit(feat_A, pose_A, pose_B)
      → compressor(feat_A)       → feat_A_compressed
      → [optional: inject latency/dropout]
      → alignment(feat_A_compressed, pose_A, pose_B) → feat_A_aligned

  → fusion(feat_B, feat_A_aligned) → feat_fused

  → detector(feat_fused) → pred_boxes

  → eval(pred_boxes, gt_boxes) → recall, precision, visualization
```

---

## Evaluation Protocol

| Condition | Input to detector | Expected outcome |
|---|---|---|
| Solo (baseline) | `feat_B` only | Misses occluded pedestrian |
| Cooperative | `feat_fused` | Detects occluded pedestrian |

**Primary metric:** Recall on the occluded pedestrian.  
**Secondary metrics:** Precision (no false positives from fusion), optionally mAP over all objects.

---

## File Structure

```
AV_final_test1/
├── carla_env.py        # Scene setup and sensor collection
├── backbone.py         # Shared PointPillars-style BEV encoder
├── compressor.py       # Feature compression for V2X transmission
├── v2x_channel.py      # V2X link simulation (latency, dropout, compression)
├── alignment.py        # Pose-based BEV feature alignment
├── fusion.py           # Concatenation + conv fusion neck
├── detector.py         # 3D detection head
├── eval.py             # Evaluation, metrics, and BEV visualization
├── main.py             # Entry point: runs solo vs. cooperative comparison
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-04-04-v2x-cooperative-perception-design.md
```

---

## Extension Points

- **Attention fusion:** Replace conv layers in `fusion.py` with a cross-attention module (Q from feat_B, K/V from feat_A_aligned)
- **Compression experiment:** Sweep `compression_ratio` in `v2x_channel.py` and plot recall vs. compression
- **Latency experiment:** Inject `latency_ms` and observe detection degradation on moving pedestrians

---

## Dependencies

- CARLA 0.9.x (simulator + Python API)
- PyTorch (backbone, fusion neck, detection head)
- NumPy, OpenCV (point cloud processing, visualization)
- Matplotlib (BEV plots)
