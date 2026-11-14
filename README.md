# Multi-Camera Angus Cattle Detection, Segmentation, and Tracking System
## Merck–TAMU Animal Science Hackathon 2026: Agentic Vision for Angus Behavior and BRD Prediction

This is our submission for the AISFS TAMU Hackathon.

> Event data (videos, frames, labels, model outputs) is not in this repository: it must stay within the event.

---

## Executive Summary & Architecture Overview

Phase 1 of the **Agentic Vision for Angus Behavior and BRD Prediction** platform delivers an end-to-end multi-camera computer vision pipeline designed for research feedlots (Pen 6 at Texas A&M Animal Science Feedlot). The system detects, segments, and tracks 10 Angus cattle across three fixed cameras under varying daylight and infrared nighttime illumination, handling real-world occlusions, perspective distortions, and multi-view identity fusion.

```
                           +------------------------------------------------+
                           |  Raw Video / Image Sequences (Cam 1, 2, 3)     |
                           +------------------------------------------------+
                                                   |
                                                   v
                           +------------------------------------------------+
                           |  Adaptive Preprocessing (CLAHE LAB / Denoise)  |
                           +------------------------------------------------+
                                                   |
                                                   v
                           +------------------------------------------------+
                           |  YOLO11 Instance Segmentation & Pen ROI Filter |
                           +------------------------------------------------+
                                                   |
                                                   v
                           +------------------------------------------------+
                           |  Within-Camera ByteTrack Multi-Object Tracking |
                           +------------------------------------------------+
                                                   |
                                                   v
                           +------------------------------------------------+
                           |  Homography Ground-Plane Projection (Meters)   |
                           +------------------------------------------------+
                                                   |
                                                   v
                           +------------------------------------------------+
                           |  Multi-Modal Cross-Camera Global Identity      |
                           |  (Spatial Proximity + Visual ReID + Time Sync) |
                           +------------------------------------------------+
                                                   |
                  +--------------------------------+-------------------------------+
                  |                                                                |
                  v                                                                v
+------------------------------------+                         +---------------------------------------+
|  Synchronized 4-Panel Dashboard    |                         |  Structured Tracking Records (CSV/PQ) |
|  [Cam 1] [Cam 2] / [Cam 3] [2D Map]|                         |  & Global Cow Registry Summary        |
+------------------------------------+                         +---------------------------------------+
```

---

## Model Selection & Empirical Benchmark

We investigated and benchmarked multiple perception candidates on actual Pen 6 daytime and nighttime Angus cattle frames:

| Model Architecture | Weights File | Latency (ms) | Throughput (FPS) | Mean Detected Cows/Frame | Mean Confidence | 50GB Estimated Runtime (1 FPS) | Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLO26-XL-Seg (Default Max-Accuracy)** | `yolo26x-seg.pt` | **237.5 ms** | **4.2 FPS** | **5.00** | **0.622** | **~11.4 hrs** | **Highest Confidence SOTA**: Highest precision and confidence on Angus cattle; best for offline high-fidelity tracking & ground-truth generation. |
| **YOLO26-Medium-Seg (Balanced High-Recall)** | `yolo26m-seg.pt` | **104.9 ms** | **9.5 FPS** | **5.17** | **0.607** | **~5.0 hrs** | **Highest Recall**: Captures the most cattle per frame (5.17) with 2.3x faster inference than XL. |
| **YOLO26-Small-Seg (Real-Time Deployment)** | `yolo26s-seg.pt` | **50.4 ms** | **19.8 FPS** | **3.83** | **0.559** | **~2.4 hrs** | **Real-Time Edge**: Fast, low memory footprint, ideal for live 15+ FPS streaming. |
| **YOLO26-Nano-Seg (Edge)** | `yolo26n-seg.pt` | 39.4 ms | 25.4 FPS | 2.50 | 0.361 | ~1.9 hrs | Lightweight, but drops cattle along the distant back fence. |
| **YOLO11-Small-Seg** | `yolo11s-seg.pt` | 45.4 ms | 22.0 FPS | 3.83 | 0.495 | ~2.2 hrs | Previous generation baseline. |
| **YOLOv8-Small-Seg** | `yolov8s-seg.pt` | 49.0 ms | 20.4 FPS | 1.83 | 0.609 | ~2.4 hrs | Legacy baseline; lower recall in night shadows. |

### Architectural Comparison: XL vs. Medium vs. Small
- **YOLO26-XL (`yolo26x-seg.pt`, 136 MB)**: Features 392 layers and 59M parameters. Achieves the highest individual detection confidence (**0.622**) and tightest segmentation boundary delineation, especially on black Angus cows lying down in dark pen corners.
- **YOLO26-Medium (`yolo26m-seg.pt`, 52 MB)**: Features 280 layers and 22M parameters. Achieved the highest empirical recall (**5.17 cows detected per frame**) while running at over 2x the speed of XL.
- **YOLO26-Small (`yolo26s-seg.pt`, 22 MB)**: Features 260 layers and 10M parameters. Designed for high-throughput live video inference where processing speed is critical.

---

## Repository Structure

```
.
├── configs/
│   ├── cameras.yaml         # Multi-camera positions, orientations, homography calibration points
│   ├── pen_layout.yaml      # Pen 6 metric dimensions, zones, and camera-specific 2D ROI polygons
│   ├── tracking.yaml        # ByteTrack thresholds, buffer sizes, and cross-camera weights
│   └── models.yaml          # Detector parameters, ReID extractor, and preprocessing profiles
├── input/
│   ├── cam1/                # Front Overview camera frames (Day & Night)
│   ├── cam2/                # Top Feed Bunk camera frames (Day & Night)
│   └── cam3/                # Back Overview camera frames (Day & Night)
├── src/
│   ├── video/
│   │   ├── video_metadata.py    # Metadata inspection and 50GB computational resource budgeting
│   │   ├── video_reader.py      # Subsampled frame generator supporting files and folders
│   │   └── synchronization.py   # Multi-camera temporal alignment
│   ├── calibration/
│   │   ├── pen_boundary.py      # Polygonal ROI containment and exclusion filtering
│   │   ├── camera_calibration.py# Planar homography estimation from known landmarks
│   │   ├── coordinate_transform.py # Projective transform between image pixels and ground meters
│   │   └── zone_mapping.py      # Classifies coordinates into Feeding, Drinking, and Resting zones
│   ├── detection/
│   │   ├── preprocessing.py     # Day/night illumination assessment & adaptive CLAHE
│   │   ├── cow_detector.py      # YOLO26 perception wrapper with pen filtering & occlusion scoring
│   │   ├── cow_segmenter.py     # Instance mask extraction, RDP polygon simplification & RLE
│   │   ├── clinical_visibility.py # Optical resolution & BRD facial sign feasibility assessment
│   │   └── clinical_head_detector.py # Anatomical bovine head cropper & Wisconsin BRD clinical scoring
│   ├── tracking/
│   │   ├── local_tracker.py     # 2-stage ByteTrack with Kalman box state prediction
│   │   ├── reidentification.py  # DINOv2 / ResNet50 appearance feature embedding extractor
│   │   ├── appearance_gallery.py# Multi-view visual gallery per cow identity
│   │   ├── global_identity.py   # Cross-camera identity fusion & Global Cow Registry (COW_01-10)
│   │   └── consensus_protocol.py# Lyapunov-stable distributed agreement protocol across camera views
│   ├── dataset/
│   │   ├── frame_extractor.py   # Stratified keyframe extractor for 48h day/night videos
│   │   └── dataset_builder.py   # Auto-labeler generating YOLO polygon annotations & dataset.yaml
│   ├── visualization/
│   │   ├── annotation.py        # Bounding box, segmentation mask, track pills, and ROI overlay
│   │   ├── pen_visualization.py # 2D metric bird's-eye map with real-time cow markers
│   │   └── tracking_dashboard.py# 4-panel synchronized dashboard generator
│   ├── storage/
│   │   ├── database.py          # Structured tracking record schema
│   │   └── export.py            # CSV, Parquet, and JSON exporters
│   ├── evaluation/
│   │   ├── detection_metrics.py # Detection summary and occlusion distribution
│   │   ├── tracking_metrics.py  # Track stability, duration, and fragmentation index
│   │   └── evaluation_report.py # Automated Markdown and JSON evaluation report generator
│   └── pipeline/
│       ├── single_camera.py     # End-to-end single camera pipeline
│       └── multi_camera.py      # Synchronized multi-camera pipeline with Lyapunov consensus
├── tests/                       # Complete automated unit test suite (16 tests)
├── outputs/                     # Generated videos, records, dashboard snapshots, and reports
├── main.py                      # Unified CLI entrypoint
├── train.py                     # YOLO26 fine-tuning and dataset preparation pipeline
├── requirements.txt             # Pinned project dependencies
└── README.md                    # Comprehensive documentation
```

---

## Installation & Setup

```bash
# 1. Clone repository and navigate to workspace
cd /path/to/aisfsHackathon

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Multi-Camera Synchronized Execution (Default)
Processes Front (Cam 1), Top (Cam 2), and Back (Cam 3) camera inputs simultaneously, performs cross-camera identity fusion, and generates a 4-panel dashboard video and structured CSV/Parquet records:
```bash
python main.py --front input/cam1 --top input/cam2 --back input/cam3 --model yolo26s-seg.pt
```

### 2. Single-Camera Execution
Run on a single camera stream independently:
```bash
python main.py --camera front --video input/cam1
python main.py --camera back --video input/cam3
python main.py --camera top --video input/cam2
```

### 3. Video Metadata Inspection & Computational Budgeting
Estimates processing runtime, frame counts, and storage for the 50 GB / 48-hour dataset:
```bash
python main.py --inspect --fps 1.0
```
Output:
```
Total Video: 50.0 GB | 48.0 Hours
Native Frames (15 FPS): 2,592,000 frames
Sampling Target: 1.0 FPS (15.0x downsample)
Processed Frames: 172,800 frames
Estimated GPU Runtime: ~1.37 hours
Estimated Output CSV Size: ~197.8 MB
```

### 4. Candidate Model Benchmarking
Benchmarks candidate models on actual day and night Angus frames:
```bash
python main.py --benchmark
```

### 5. Dataset Extraction & YOLO26 Fine-Tuning
Extract stratified keyframes across the 48-hour day/night video feeds, auto-generate YOLO segmentation polygon annotations, and fine-tune YOLO26:
```bash
# Extract keyframes and build data/training/dataset.yaml
python train.py --prepare-data --stride-seconds 60

# Fine-tune YOLO26-Medium on Pen 6 Angus cattle (freezing backbone)
python train.py --epochs 30 --batch 8 --model yolo26m-seg.pt
```

### 6. Running the Automated Test Suite
```bash
python -m unittest discover tests
```

---

## Cow Identity (tracklet-level, unique, persistent)

`src/tracking/tracklet_identity.py` (default; `--legacy-identity` restores the old greedy matcher).

1. **Tracklet embeddings.** Each ByteTrack track keeps a running embedding: DINOv2 ViT-S/14 plus a coat descriptor (white markings, brown tint, coloured ear tags), because generic embeddings barely separate black Angus.
2. **Unique assignment.** For each camera and step, a Hungarian assignment of tracks to `COW_01…10` with cost = appearance distance to the cow's prototype + distance to where that cow is right now in another camera (or was last seen) + a stickiness bonus for the current ID. Physically impossible jumps (> 3 m/s) are blocked, and each cow is used at most once per camera per frame. Tracks with no acceptable match stay unassigned instead of guessing.
3. **Prototype updates.** A cow's prototype is updated only from assignments that are clearly better than the next-best cow.
4. **Saved gallery.** `--id-gallery` (default `outputs/id_gallery.npz`) is loaded at start and saved at the end, so `COW_xx` stays the same animal across clips and days. Delete it to start fresh.

cam3 positions are not used for identity until its metric calibration is verified against the official pen map.

**Day 1 07:19 clip, 180 s, cam1 + cam3:** 0 duplicate IDs per frame (previously two boxes could share an ID); the distinctive white-faced steer keeps one ID across the clip; 4 of 22 tracks changed ID once.

## Level 1 Behavior Classification (10 official classes)

`classify_behavior.py` turns `tracking_records.csv` (one or many runs) into a per-cow behavior
time series using **only** the official classes: `lying, standing, walking, feeding, drinking,
ruminating (lying), ruminating (standing), grooming, social interaction, other / not visible`.

### How it works (`src/behavior/`)

| Stage | File | What it does |
| :--- | :--- | :--- |
| Cues | `features.py` | Per detection: **posture** from a self-calibrated *standing-height model* (expected standing bbox height vs. image row, fitted per camera from the upper envelope of all tracks; no labels needed; lying ≈ 0.5–0.6× standing height), **motion** in m/s using that standing height (~1.4 m) as the pixel ruler (works the same from any viewing angle and isn't thrown off by far-field homography error), **zones** (feeding apron, waterer distance, cam2 head-over-trough overlap), **grooming** (bbox-width oscillation while the ground contact is still), **social** (nearest-neighbour world distance + same-camera bbox contact, excluding the bunk). |
| Fusion | `fusion.py` | Merges all cameras' observations of a cow in each second, weighting each by detection quality × how much that camera is trusted for that cue (cam2 for feeding, cam1/cam3 for posture), then a factorised model gives P(10 classes). |
| Temporal | `fusion.py` | Sticky HMM with biologically-set mean bout durations (lying 30 min, feeding 7 min, walking 20 s …) and damped impossible jumps (lying→walking). Forward-backward gives posteriors (confidence), Viterbi gives the labels. Short unseen gaps (≤45 s) are bridged; longer ones become `other / not visible`. |
| Rumination | `rumination.py` | **Measured, not guessed**: for still bouts ≥2 min it decodes a 12 s native-fps snippet and looks for a jaw-motion spectral peak at 0.75–1.6 Hz (45–95 chews/min). Opt-in (`--probe-rumination`); also usable as an agent tool. |
| Appearance (optional) | `appearance.py` | Frozen DINOv2 ViT-S/14 (already used for ReID) + logistic-regression probe, fused as a product-of-experts. Trained on event crops you correct + public datasets. |
| Export | `export.py` | `activity.csv`, `behavior_bouts.csv`, `time_budgets.csv`, `behavior_timeline_1s.csv`, `behavior_review_queue.csv`, `behavior_summary.json`. |

Every second is tagged with **evidence**: `observed` (direct geometric measurement),
`observed_chewing`, `estimated_heuristic` (grooming/social until a visual check), `estimated_gap_fill`,
`not_visible`, or `no_video` (minutes with no recording). This separates what the system
observed from what it estimated, as the brief requires. Rumination is **never** assigned
without chewing evidence or the learned expert, so lying/standing time includes unconfirmed rumination.

### Commands

```bash
# 1. tracking (per clip / overlapping clip set) -> tracking_records.csv
python main.py --front "input/Day 1/Cam 01/Event20261107071945006.Avi" \
  --back "input/Day 1/Cam 03/Event20261107071935003.Avi" --top /nonexistent \
  --model runs/segment/runs/angus_train/angus_yolo26m-seg-4/weights/best.pt \
  --fps 1 --no-video --output-dir outputs/run_d1_0719

# 2. behavior for every run, every minute of both days
python classify_behavior.py --records "outputs/run_*/tracking_records.csv" \
  --full-days 2026-11-07 2026-11-08 --out outputs/behavior

# optional: measured rumination + learned appearance expert
python classify_behavior.py --records "outputs/run_*/tracking_records.csv" --video-root input \
  --probe-rumination --appearance-model models/behavior_probe.joblib --out outputs/behavior

# optional: train the appearance probe (crops pre-sorted by the geometric model; fix wrong folders)
python train_behavior_probe.py export-crops --records "outputs/run_*/tracking_records.csv" --video-root input
python train_behavior_probe.py train --event-crops data/behavior_crops \
  --cvb-ava <CVB_in_AVA_format csv> --cvb-frames <raw_frames dir> --cvb-pattern "{video_id}/{frame:05d}.jpg"
```

All thresholds live in `configs/behavior.yaml` and are physical quantities (m/s, m, Hz, s).

### External data and models

| Resource | Use | Licence |
| :--- | :--- | :--- |
| DINOv2 ViT-S/14 (`facebookresearch/dinov2`, torch.hub) | frozen crop embedding for ReID + behavior probe | Apache-2.0 |
| CVB: Cattle Visual Behaviors (CSIRO, arXiv:2305.16555, doi:10.25919/3g3t-p068) | optional probe training: 8 Angus, 11 classes incl. ruminating-lying/standing, grooming | CC BY-NC-SA 4.0 (non-commercial) |
| CBVD-5 (Kaggle `fandaoerji/cbvd-5cow-behavior-video-dataset`) | optional probe training: standing, lying, feeding, drinking, rumination | CC0 |
| Ultralytics YOLO26 | detection / segmentation | AGPL-3.0 |

Class mappings are in `src/behavior/taxonomy.py`. External data is capped per class so it
cannot swamp event crops, and probe accuracy is reported on held-out **event** crops only.

### What this can and cannot support
- Lying / standing / walking / feeding / drinking are measured from geometry and are reliable when a cow is tracked with a stable `global_cow_id`.
- Rumination needs the chew probe (or the learned expert); at 1 fps it is invisible.
- Grooming and social interaction are heuristic cues. They go into `behavior_review_queue.csv` for the agent (VLM) to verify visually.
- Behavior accuracy is capped by identity accuracy. A wrong `global_cow_id` puts the right behavior on the wrong cow.

## Output Data Schema

Structured tracking records are exported to `outputs/tracking_records.csv` and `outputs/tracking_records.parquet` following the exact schema:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `timestamp` | `float` | Elapsed video timestamp in seconds |
| `camera_id` | `string` | Camera identifier (`cam1`, `cam2`, `cam3`) |
| `frame_number` | `int` | Sequential frame index |
| `local_track_id` | `int` | Within-camera ByteTrack ID |
| `global_cow_id` | `string` | Associated global identity (`COW_01` to `COW_10`) |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | `float` | Bounding box coordinates in pixels |
| `center_x`, `center_y` | `float` | 2D bounding box center in pixels |
| `detection_confidence` | `float` | YOLO11 perception confidence score [0.0 - 1.0] |
| `tracking_confidence` | `float` | Filtered tracking stability score [0.0 - 1.0] |
| `segmentation_mask_reference` | `string` | Reference to compact polygon mask |
| `visibility_status` | `string` | `VISIBLE`, `PARTIALLY_OCCLUDED`, `HEAVILY_OCCLUDED` |
| `occlusion_status` | `string` | Fractional overlap with neighboring cattle [0.0 - 1.0] |
| `identity_confidence` | `float` | Cross-camera identity association confidence |
| `world_x`, `world_y` | `float` | Calibrated ground-plane coordinates in Pen 6 meters |
| `discharge_inspection_feasible` | `bool` | True if bovine head has >= 120px optical resolution for clinical inspection |
| `best_clinical_camera` | `string` | Optimal camera viewpoint for BRD inspection (`cam2` when at feed bunk) |
| `clip_10s_reference` | `string` | 10-second video snippet reference window |
| `consensus_energy` | `float` | Inter-camera Lyapunov disagreement energy $V(k)$ |

---

## Future Extensibility: Preparing for Phase 2

Phase 1 outputs were engineered specifically to feed downstream behavior analysis and agentic VLM reasoning:
1. **Feeding & Drinking Detection**: `world_x` and `world_y` automatically map into the metric `Feeding Zone` and `Drinking Zone` polygons, allowing instant computation of time spent at the bunk or waterer.
2. **Resting & Rumination Analysis**: Cow bounding boxes, segmentation masks, and centroid stability indicate standing vs. lying postures.
3. **Agentic VLM Integration**: Downstream LLM/VLM agents can query `tracking_records.csv` and `cow_identity_summary.json` with tools to retrieve video crops for any specific cow and timestamp (e.g. *"Show me all feeding events for COW_04 between 14:00 and 16:00"*).
