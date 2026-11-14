#!/usr/bin/env python3
"""Fine-tuning pipeline for YOLO26 / YOLO11 on Angus feedlot video data.

Merck-TAMU Animal Science Hackathon 2026
"""

import os
import sys
import argparse

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ultralytics import YOLO

# Add root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.dataset.frame_extractor import KeyframeExtractor
from src.dataset.dataset_builder import YOLODatasetBuilder


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on Angus Cattle Dataset")
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Extract keyframes and generate pseudo-labeled dataset from input/ before training",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Extract keyframes and generate pseudo-labeled dataset without starting training",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/training",
        help="Path to training dataset directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26m-seg.pt",
        help="Pretrained base model weights to fine-tune (e.g. yolo26m-seg.pt, yolo26x-seg.pt)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu', 'mps', '0')")
    parser.add_argument("--freeze", type=int, default=10, help="Number of backbone layers to freeze")
    parser.add_argument("--stride-seconds", type=float, default=30.0, help="Temporal sampling stride in seconds")
    parser.add_argument("--cache", type=str, default="ram", help="Dataset caching strategy ('ram', 'disk', 'none')")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision (FP16)")
    parser.add_argument("--no-val", action="store_true", help="Skip running the heavy validation loop after every single epoch")

    return parser.parse_args()


def prepare_dataset(data_dir: str, base_model: str, stride_seconds: float = 30.0):
    print("================================================================================")
    print("PREPARING ANGUS CATTLE TRAINING DATASET FROM DAY 1")
    print("================================================================================")
    extractor = KeyframeExtractor(output_dir=data_dir)

    cam_candidates = {
        "cam1": ["input/Day 1/Cam 01", "input/day1/cam1", "input/cam1"],
        "cam2": ["input/Day 1/Cam 02", "input/day1/cam2", "input/cam2"],
        "cam3": ["input/Day 1/Cam 03", "input/day1/cam3", "input/cam3"],
    }

    # Extract keyframes from all available cameras
    for cam_id, paths in cam_candidates.items():
        found_path = None
        for p in paths:
            if os.path.exists(p):
                found_path = p
                break

        if found_path:
            print(f"[{cam_id.upper()}] Reading source: {found_path} (every {stride_seconds}s)")
            stats = extractor.extract_from_source(
                camera_id=cam_id,
                source_path=found_path,
                sample_interval_sec=stride_seconds,
                train_ratio=0.80,
            )
            print(f"[{cam_id.upper()}] Extracted {stats['total_extracted']} frames (Train: {stats['train_count']}, Val: {stats['val_count']})")

    # Generate pseudo-labels and dataset.yaml
    builder = YOLODatasetBuilder(base_dir=data_dir, model_path=base_model)
    print("Generating polygon pseudo-labels for training set...")
    n_train = builder.auto_annotate_split("train", camera_id="cam1")
    print(f"Annotated {n_train} training images.")

    print("Generating polygon pseudo-labels for validation set...")
    n_val = builder.auto_annotate_split("val", camera_id="cam1")
    print(f"Annotated {n_val} validation images.")

    yaml_path = builder.write_dataset_yaml()
    print(f"Created dataset configuration at {yaml_path}")
    return yaml_path


def main():
    args = parse_args()
    yaml_path = os.path.join(args.data_dir, "dataset.yaml")

    if args.prepare_data or args.prepare_only or not os.path.exists(yaml_path):
        yaml_path = prepare_dataset(args.data_dir, args.model, stride_seconds=args.stride_seconds)
        if args.prepare_only:
            print("\n[INFO] Dataset preparation complete (--prepare-only). Training not started.")
            return

    print("\n================================================================================")
    print(f"STARTING FINE-TUNING: {args.model}")
    print("================================================================================")
    print(f"Dataset YAML: {yaml_path}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch} | ImgSz: {args.imgsz} | Device: {args.device} | Cache: {args.cache} | Workers: {args.workers}")

    cache_val = False if args.cache.lower() == "none" else (True if args.cache.lower() == "ram" else args.cache)

    model = YOLO(args.model)
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        freeze=args.freeze,
        cache=cache_val,
        workers=args.workers,
        amp=args.amp,
        val=not args.no_val,
        project="runs/angus_train",
        name=f"angus_{os.path.splitext(os.path.basename(args.model))[0]}",
        save=True,
        verbose=True,
    )
    if args.no_val:
        print("\n[EVALUATION] Running single validation pass on trained model...")
        model.val(data=yaml_path, imgsz=args.imgsz, device=args.device)

    print("\nFine-tuning completed. Trained weights saved to runs/angus_train/")


if __name__ == "__main__":
    main()
