"""Detection performance metrics and daytime vs nighttime evaluation."""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


class DetectionEvaluator:
    """Evaluates detection statistics, confidence distributions, and lighting robustness."""

    @staticmethod
    def evaluate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates detection summary statistics across cameras and lighting."""
        if df.empty:
            return {"total_detections": 0}

        total_dets = len(df)
        avg_conf = float(df["detection_confidence"].mean())
        std_conf = float(df["detection_confidence"].std()) if total_dets > 1 else 0.0

        # Per-camera breakdown
        cam_breakdown = {}
        for cam_id, group in df.groupby("camera_id"):
            cam_breakdown[cam_id] = {
                "detection_count": len(group),
                "mean_confidence": round(float(group["detection_confidence"].mean()), 3),
                "unique_tracks": int(group["local_track_id"].nunique()),
            }

        # Occlusion breakdown
        occlusion_counts = df["visibility_status"].value_counts().to_dict()

        return {
            "total_detections": total_dets,
            "mean_detection_confidence": round(avg_conf, 3),
            "std_detection_confidence": round(std_conf, 3),
            "per_camera_metrics": cam_breakdown,
            "occlusion_distribution": occlusion_counts,
        }
