"""Automated Activity-Stratified Dataset Extraction and Annotation for Day 1 Videos."""

import os
import sys
import glob
import time
import argparse
from typing import List, Dict, Any
import cv2

sys.path.insert(0, os.path.abspath("."))

from src.dataset.activity_sampler import ActivitySampler
from src.dataset.dataset_builder import YOLODatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="Large-Scale Activity-Stratified Dataset Builder")
    parser.add_argument("--max-clips-per-cam", type=int, default=8, help="Number of 5-minute clips to scan per camera")
    parser.add_argument("--dense-stride", type=float, default=2.0, help="Seconds between frames when crowded/active")
    parser.add_argument("--sparse-stride", type=float, default=45.0, help="Seconds between frames when resting")
    parser.add_argument("--skip-annotation", action="store_true", help="Skip auto-annotation step")
    args = parser.parse_args()

    print("================================================================================")
    print("LARGE-SCALE ACTIVITY-STRATIFIED FRAME EXTRACTION (DAY 1)")
    print(f"Clips per camera: {args.max_clips_per_cam} | Dense Stride: {args.dense_stride}s | Sparse Stride: {args.sparse_stride}s")
    print("================================================================================")
    t0 = time.time()

    sampler = ActivitySampler(output_dir="outputs/activity_sampling")

    cam_folders = {
        "cam1": sorted(glob.glob("input/Day 1/Cam 01/*.Avi"))[:args.max_clips_per_cam],
        "cam2": sorted(glob.glob("input/Day 1/Cam 02/*.Avi"))[:args.max_clips_per_cam],
        "cam3": sorted(glob.glob("input/Day 1/Cam 03/**/*.Avi", recursive=True))[:args.max_clips_per_cam],
    }

    scan_results = []
    for cam_id, files in cam_folders.items():
        for clip_path in files:
            base_name = os.path.splitext(os.path.basename(clip_path))[0]
            print(f"[SCAN] {cam_id.upper()}: {base_name}...")
            res = sampler.scan_clip(clip_path, cam_id, probe_interval_sec=args.sparse_stride)
            status = "CROWDED (Active)" if res["is_crowded_recommendation"] else "QUIESCENT (Resting)"
            print(f"  Bunk: {res['max_bunk_cows']} | Waterer: {res['max_waterer_cows']} -> {status}")
            scan_results.append(res)

    html_dashboard = sampler.generate_html_review_dashboard(scan_results)
    print(f"\n[REVIEW DASHBOARD] Generated at {html_dashboard}")

    print(f"\n[EXTRACTING] Extracting dense ({args.dense_stride}s) and sparse ({args.sparse_stride}s) frames...")
    extract_stats = sampler.extract_activity_dataset(
        scan_results,
        output_train_dir="data/training",
        dense_stride_sec=args.dense_stride,
        sparse_stride_sec=args.sparse_stride,
    )
    print(f"\nExtracted {extract_stats['total_extracted']} total frames (Train: {extract_stats['train_count']}, Val: {extract_stats['val_count']})")

    if not args.skip_annotation:
        print("\n[ANNOTATING] Auto-generating polygon segmentation labels with Pen 6 boundary clipping...")
        builder = YOLODatasetBuilder(base_dir="data/training", model_path="yolo26x-seg.pt")
        n_train = builder.auto_annotate_split("train")
        print(f"Annotated {n_train} training frames.")
        n_val = builder.auto_annotate_split("val")
        print(f"Annotated {n_val} validation frames.")
        yaml_path = builder.write_dataset_yaml()
        print(f"\n[SUCCESS] Dataset YAML created at {yaml_path}")

    elapsed = time.time() - t0
    print(f"Extraction and annotation completed in {elapsed:.1f} seconds!")


if __name__ == "__main__":
    main()
