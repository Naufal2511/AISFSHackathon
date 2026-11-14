"""Multi-object tracking stability, identity switches, and track fragmentation metrics."""

from typing import Dict, Any
import pandas as pd


class TrackingEvaluator:
    """Computes tracking stability, fragmentation, and ID consistency."""

    @staticmethod
    def evaluate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"unique_local_tracks": 0}

        unique_local_tracks = int(df["local_track_id"].nunique())
        unique_global_cows = int(df["global_cow_id"].dropna().nunique())

        # Track duration statistics
        track_durations = (
            df.groupby(["camera_id", "local_track_id"])["timestamp"]
            .agg(lambda s: s.max() - s.min())
        )

        mean_duration = float(track_durations.mean()) if not track_durations.empty else 0.0
        max_duration = float(track_durations.max()) if not track_durations.empty else 0.0

        # Potential ID switches (where a local track is associated with multiple global IDs)
        id_switches = 0
        for _, group in df.groupby(["camera_id", "local_track_id"]):
            assigned_globals = group["global_cow_id"].dropna().unique()
            if len(assigned_globals) > 1:
                id_switches += len(assigned_globals) - 1

        return {
            "unique_local_tracks": unique_local_tracks,
            "unique_global_cows_active": unique_global_cows,
            "mean_track_duration_sec": round(mean_duration, 2),
            "max_track_duration_sec": round(max_duration, 2),
            "detected_id_switches": id_switches,
            "fragmentation_index": round(unique_local_tracks / max(1, unique_global_cows), 2),
        }
