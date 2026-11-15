#!/usr/bin/env python3
"""Main CLI entrypoint for Multi-Camera Angus Cattle Detection, Segmentation, and Tracking System.

Merck-TAMU Animal Science Hackathon 2026: Agentic Vision for Angus Behavior and BRD Prediction
"""

import os
import sys
import argparse
import time
from typing import Dict, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.video.video_metadata import VideoMetadataInspector
from src.pipeline.single_camera import SingleCameraPipeline
from src.pipeline.multi_camera import MultiCameraPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Camera Angus Cattle Detection, Segmentation, and Tracking Pipeline"
    )

    # Execution Modes: Single-camera vs Multi-camera
    parser.add_argument(
        "--camera",
        type=str,
        choices=["cam1", "cam2", "cam3", "front", "top", "back"],
        help="Run single camera mode (e.g. --camera front --video path/to/video.mp4)",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to single camera video file or image sequence directory",
    )

    # Multi-camera inputs
    parser.add_argument("--front", type=str, default="input/cam1", help="Path to Front camera video or directory")
    parser.add_argument("--top", type=str, default="input/cam2", help="Path to Top/Bunk camera video or directory")
    parser.add_argument("--back", type=str, default="input/cam3", help="Path to Back camera video or directory")
    parser.add_argument("--pen-map", type=str, help="Path to pen layout map image")

    # Perception & Tracking configuration
    parser.add_argument(
        "--model",
        type=str,
        default="models/angus_det_hprc_m1280.pt",
        help="Cow detector (default: YOLO26m fine-tuned on hand-labelled Pen 6 frames at 1280, HPRC; "
             "held-out test recall 0.88 / precision 0.86)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Detection confidence threshold (default: 0.20)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target frame sampling rate (e.g. 1.0 for 1 FPS exploratory processing)",
    )
    parser.add_argument(
        "--time-range",
        type=str,
        default=None,
        help="Time range to process in seconds, format 'START-END' (e.g. '0-60')",
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Wall-clock start, e.g. '2026-11-07 07:00'. With camera FOLDERS of Event clips the clips "
             "are read as one continuous stream per camera (tracker never restarts at file boundaries)",
    )
    parser.add_argument("--end", type=str, default=None, help="Wall-clock end, e.g. '2026-11-07 09:00'")

    # Operational flags
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Detector inference size (default 1280: 640 loses far-field and edge-cut cows on 4K frames)",
    )
    parser.add_argument(
        "--id-gallery",
        type=str,
        default="outputs/id_gallery.npz",
        help="Saved cow appearance gallery: loaded at start, updated and saved at the end, so COW_xx "
             "stays the same animal across clips and days (delete it to start fresh)",
    )
    parser.add_argument(
        "--legacy-identity",
        action="store_true",
        help="Use the previous greedy GlobalIdentityManager instead of tracklet identity",
    )
    parser.add_argument(
        "--no-reid",
        action="store_true",
        help="Disable appearance embedding extraction (DINOv2)",
    )
    parser.add_argument("--no-assets", action="store_true",
                        help="skip Level 2 risk assets (head-shot gallery, motion energy, bunk signal)")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip annotated video rendering for faster data export",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save videos, records, and reports (default: outputs)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect video metadata and print computational requirements budget without running full inference",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run comparative model benchmark across candidate models on sample frames",
    )

    return parser.parse_args()


def run_benchmark(output_dir: str):
    """Benchmarks candidate models on sample frames from daytime and nighttime."""
    import glob
    import cv2
    import numpy as np
    import pandas as pd
    from ultralytics import YOLO

    print("================================================================================")
    print("RUNNING CANDIDATE MODEL BENCHMARK (DAYTIME & NIGHTTIME ANGUS FRAMES)")
    print("================================================================================")

    candidate_models = [
        ("YOLO26-XL-Seg (Max Accuracy)", "yolo26x-seg.pt"),
        ("YOLO26-Medium-Seg (Balanced)", "yolo26m-seg.pt"),
        ("YOLO26-Small-Seg (Real-Time)", "yolo26s-seg.pt"),
        ("YOLO26-Nano-Seg (Edge)", "yolo26n-seg.pt"),
        ("YOLO11-Small-Seg", "yolo11s-seg.pt"),
        ("YOLOv8-Small-Seg", "yolov8s-seg.pt"),
    ]

    test_images = sorted(glob.glob("input/*/*.png"))
    if not test_images:
        print("No test images found in input/")
        return

    benchmark_rows = []

    for model_desc, model_file in candidate_models:
        if not os.path.exists(model_file):
            continue

        model = YOLO(model_file)
        latencies = []
        cow_counts = []
        confidences = []

        for img_path in test_images:
            img = cv2.imread(img_path)
            t_start = time.time()
            res = model(img, conf=0.15, verbose=False)[0]
            lat = (time.time() - t_start) * 1000.0
            latencies.append(lat)

            cows = [b for b in res.boxes if int(b.cls[0]) == 19]
            cow_counts.append(len(cows))
            for b in cows:
                confidences.append(float(b.conf[0]))

        avg_lat = float(np.mean(latencies))
        avg_cows = float(np.mean(cow_counts))
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        benchmark_rows.append(
            {
                "Model Candidate": model_desc,
                "Model Weights": model_file,
                "Average Latency (ms)": round(avg_lat, 1),
                "Throughput (FPS)": round(1000.0 / max(1.0, avg_lat), 1),
                "Mean Detected Cows/Frame": round(avg_cows, 2),
                "Mean Confidence": round(avg_conf, 3),
            }
        )

    df_bench = pd.DataFrame(benchmark_rows)
    print(df_bench.to_string(index=False))

    bench_path = os.path.join(output_dir, "model_benchmark.csv")
    df_bench.to_csv(bench_path, index=False)
    print(f"\nSaved benchmark comparison table to {bench_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Benchmark mode
    if args.benchmark:
        run_benchmark(args.output_dir)
        return

    # Parse time range
    start_sec = 0.0
    end_sec = None
    if args.time_range:
        parts = args.time_range.split("-")
        start_sec = float(parts[0])
        if len(parts) > 1 and parts[1]:
            end_sec = float(parts[1])

    # 2. Metadata inspection mode
    if args.inspect:
        print("================================================================================")
        print("VIDEO METADATA & COMPUTATIONAL PROCESSING BUDGET")
        print("================================================================================")
        budget = VideoMetadataInspector.estimate_processing_budget(
            total_video_gb=50.0,
            total_hours=48.0,
            native_fps=15.0,
            target_fps=args.fps if args.fps else 1.0,
        )
        for k, v in budget.items():
            print(f"  {k:35s}: {v}")
        return

    # 3. Single-Camera Mode
    if args.camera:
        cam_map = {
            "front": "cam1", "cam1": "cam1",
            "top": "cam2", "cam2": "cam2",
            "back": "cam3", "cam3": "cam3",
        }
        cam_id = cam_map.get(args.camera.lower(), "cam1")
        video_src = args.video
        if not video_src:
            video_src = f"input/{cam_id}"

        print(f"Executing Single-Camera Pipeline: Camera={cam_id}, Source={video_src}")
        pipeline = SingleCameraPipeline(
            camera_id=cam_id,
            video_source=video_src,
            model_path=args.model,
            confidence_threshold=args.conf,
            target_fps=args.fps,
            start_sec=start_sec,
            end_sec=end_sec,
            output_dir=args.output_dir,
            enable_reid=not args.no_reid,
        )
        db = pipeline.run(save_annotated_video=not args.no_video)
        print(f"Single-camera processing complete. {len(db.records)} records created.")
        return

    # 4. Multi-Camera Mode (Default)
    cam_sources = {}
    if os.path.exists(args.front):
        cam_sources["cam1"] = args.front
    if os.path.exists(args.top):
        cam_sources["cam2"] = args.top
    if os.path.exists(args.back):
        cam_sources["cam3"] = args.back

    if not cam_sources:
        print("Error: No valid camera sources found. Please check paths for --front, --top, --back.")
        sys.exit(1)

    print("================================================================================")
    print("MULTI-CAMERA ANGUS CATTLE TRACKING & GLOBAL IDENTITY SYSTEM")
    print("================================================================================")
    print(f"Active Camera Sources: {cam_sources}")
    print(f"Model: {args.model} | Sampling FPS: {args.fps or 'Native'}")

    pipeline = MultiCameraPipeline(
        camera_sources=cam_sources,
        model_path=args.model,
        confidence_threshold=args.conf,
        target_fps=args.fps,
        start_sec=start_sec,
        end_sec=end_sec,
        output_dir=args.output_dir,
        enable_reid=not args.no_reid,
        id_gallery_path=args.id_gallery,
        legacy_identity=args.legacy_identity,
        imgsz=args.imgsz,
        wall_start=args.start,
        wall_end=args.end,
        record_assets=not args.no_assets,
        # resume point every N min of footage (0 = off); see src/pipeline/checkpoint.py
        checkpoint_sec=60.0 * float(os.environ.get("PEN6_CKPT_MIN", "15")),
    )

    results = pipeline.run(save_dashboard_video=not args.no_video)

    print("\n================================================================================")
    print("PIPELINE EXECUTION COMPLETE")
    print("================================================================================")
    for k, v in results.items():
        print(f"  {k:30s}: {v}")


if __name__ == "__main__":
    main()
