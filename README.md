# V2X Cooperative Perception

A simulation-based implementation of **Cooperative Perception via V2X (Vehicle-to-Everything)** using CARLA. Two vehicles share intermediate BEV feature maps over a simulated wireless link, enabling Vehicle B to detect a pedestrian it cannot see due to occlusion by a truck.

## The Idea

Standard AV perception is limited to what a single vehicle's sensors can see. V2X cooperative perception lets connected vehicles share their perception data so that **Vehicle A can warn Vehicle B about a pedestrian that B can't see because a truck is blocking the view**.

```
Vehicle A ──(BEV features)──► Vehicle B
(sees pedestrian)              (pedestrian occluded by truck)
                                        │
                                  ┌─────▼──────┐
                                  │  Fusion    │
                                  │  Neck      │
                                  └─────┬──────┘
                                        │
                                  Detection Head
                                        │
                                  ✅ Pedestrian detected!
```

## Architecture

| Module | Role |
|---|---|
| `backbone.py` | Shared PointPillars-style BEV encoder (LiDAR → BEV feature map) |
| `compressor.py` | Spatial downsampling + optional int8 quantization for V2X transmission |
| `alignment.py` | Pose-based affine warp of A's features into B's BEV frame |
| `fusion.py` | Concatenation + 3-layer Conv2D neck (solo and cooperative modes) |
| `detector.py` | CenterPoint-style anchor-free 3D detection head |
| `v2x_channel.py` | Simulated V2X link (configurable latency, dropout, compression) |
| `eval.py` | BEV IoU, recall/precision metrics, side-by-side BEV visualization |
| `carla_env.py` | CARLA scene setup: occlusion scenario with truck, pedestrian, two vehicles |
| `main.py` | Entry point: runs solo vs. cooperative comparison over N ticks |

## Intermediate Fusion

This project implements **intermediate (feature-level) fusion** — the most research-relevant V2X fusion strategy:

- **Early fusion** (raw point clouds) — maximum information, huge bandwidth
- **Intermediate fusion** (BEV feature maps) ← **this project** — good trade-off
- **Late fusion** (bounding boxes only) — minimal bandwidth, loses spatial detail

## Occlusion Scenario

```
                    [Pedestrian]   ← occluded from B, visible to A
                        |
                    [Truck]        ← blocker
                        |
[Vehicle A] ────────────────────── [Vehicle B]
(clear LOS)                        (blocked LOS)
```

Vehicle B alone: **misses the pedestrian**
Vehicle B + A's features: **detects the pedestrian**

## Requirements

- Python 3.8+
- PyTorch >= 2.0
- NumPy >= 1.24, < 2.0
- Matplotlib >= 3.7
- CARLA 0.9.x (for the full demo)
- pytest >= 6.2 (for unit tests)

```bash
pip install -r requirements.txt
```

## Unit Tests (no CARLA required)

All perception modules have unit tests that run without a CARLA server:

```bash
pytest tests/ -v
# 39 tests across backbone, compressor, alignment, fusion, detector, v2x_channel, eval
```

## Running the Full Demo

Start a CARLA server first:

```bash
# Linux
./CarlaUE4.sh -RenderOffScreen

# Windows
CarlaUE4.exe
```

Then run the demo:

```bash
python main.py --ticks 50 --output results/ --compression-ratio 2
```

Per-tick output:
```
Tick   1/50 | Solo recall=0.00 prec=0.00 | Coop recall=0.67 prec=0.67
Tick   2/50 | Solo recall=0.00 prec=0.00 | Coop recall=0.67 prec=0.67
...
===== Results =====
Solo        — Recall: 0.000  Precision: 0.000
Cooperative — Recall: 0.667  Precision: 0.667
Recall improvement: +0.667
```

Output files in `results/`:
- `bev_tick_005.png`, `bev_tick_010.png`, … — side-by-side BEV visualizations
- `metrics.txt` — recall/precision summary

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--ticks` | 20 | Number of simulation ticks |
| `--output` | `results/` | Output directory |
| `--compression-ratio` | 2 | V2X feature compression factor |
| `--score-threshold` | 0.3 | Detection confidence threshold |

## Extension Points

**Attention-based fusion** — replace the conv neck in `fusion.py` with cross-attention (Q from Vehicle B, K/V from Vehicle A).

**Compression sweep** — vary `--compression-ratio` and plot recall vs. compression to show the bandwidth vs. accuracy trade-off.

**Latency experiment** — pass `latency_ticks` to `V2XChannel` to model communication delay and observe detection degradation.

## Project Structure

```
AV_final_test1/
├── backbone.py         # BEV encoder + shared constants
├── compressor.py       # Feature compression
├── alignment.py        # Spatial alignment
├── fusion.py           # Fusion neck
├── detector.py         # Detection head
├── v2x_channel.py      # V2X link simulation
├── eval.py             # Metrics + visualization
├── carla_env.py        # CARLA scene
├── main.py             # Entry point
├── requirements.txt
├── tests/              # 39 unit tests
└── docs/
    └── superpowers/
        ├── specs/      # Design spec
        └── plans/      # Implementation plan
```
