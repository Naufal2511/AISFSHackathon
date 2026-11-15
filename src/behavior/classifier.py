"""End-to-end behavior classifier: tracking_records.csv -> per-cow behavior time series."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from .features import compute_cues, load_records
from .fusion import evidence_label, fuse_cameras, smooth_cow
from .taxonomy import NOT_VISIBLE


class BehaviorClassifier:
    def __init__(self, config_path: str = "configs/behavior.yaml",
                 pen_config: str = "configs/pen_layout.yaml",
                 cameras_config: str = "configs/cameras.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.pen_config = pen_config
        self.cameras_config = cameras_config
        self.rows: Optional[pd.DataFrame] = None
        self.fused: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ stages
    def load(self, records: List[str], clip_start: Optional[str] = None) -> pd.DataFrame:
        rows = load_records(records, self.cameras_config, clip_start)
        n_unassigned = rows["global_cow_id"].isna().sum()
        if n_unassigned:
            print(f"[BEHAVIOR] {n_unassigned}/{len(rows)} rows have no global_cow_id and are excluded")
        self.rows = compute_cues(rows, self.cfg, self.pen_config)
        fits = self.rows.attrs.get("standing_height_fits", {})
        for cam, (a, b) in fits.items():
            print(f"[BEHAVIOR] {cam}: standing height model h = {a:.3f} * y_bottom + {b:.1f} px")
        self.fused = fuse_cameras(self.rows, self.cfg)
        return self.fused

    def attach_appearance(self, app: pd.DataFrame):
        """Merges learned crop probabilities (appearance.predict_rows) into the fused cues."""
        if app is not None and len(app):
            self.fused = self.fused.merge(app, on=["global_cow_id", "sec"], how="left")

    def attach_learned(self, video_index: Dict[str, str], weights: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Scores body-box crops with the fine-tuned classifier (learned.py) and attaches one
        calibrated vector per cow-second; returns per-box predictions in the contract shape."""
        from . import learned
        lc = self.cfg.get("learned_model", {})
        self.model = learned.FineTunedBehaviorModel(weights or lc["weights"], lc.get("device"))
        boxes = learned.score_boxes(self.model, self.rows, video_index, lc)
        if boxes.empty:
            print("[BEHAVIOR] learned model scored no boxes; using the heuristic cue model")
            return None
        self.fused = learned.attach(self.fused, learned.fuse_scores(boxes, self.model, lc),
                                    lc.get("hold_sec", lc.get("every_sec", 2)))
        return learned.contract_predictions(boxes, self.model)

    def decode(self) -> pd.DataFrame:
        parts = []
        for cow, d in self.fused.groupby("global_cow_id"):
            t = smooth_cow(d, self.cfg)
            t.insert(0, "global_cow_id", cow)
            parts.append(t)
        tl = pd.concat(parts, ignore_index=True)
        tl["evidence"] = tl.apply(evidence_label, axis=1)
        return tl

    def probe_rumination(self, tl: pd.DataFrame, video_index: Dict[str, str]) -> int:
        """Agent tool loop: for long still bouts, measure jaw motion and set p_ruminating."""
        from .export import bouts
        from .rumination import chew_score, resolve_video

        rc = self.cfg["rumination"]
        overhead = set(self.cfg["posture"]["overhead_cameras"])
        bt = bouts(tl)
        cand = bt[bt["behavior"].isin(["lying", "standing", "ruminating (lying)", "ruminating (standing)"])
                  & (bt["duration_sec"] >= rc["min_bout_for_probe_sec"])]
        if "p_ruminating" not in self.fused:
            self.fused["p_ruminating"] = np.nan
        n = 0
        for b in cand.itertuples():
            probe_secs = np.linspace(b.start_sec, b.end_sec - rc["probe_duration_sec"], rc["probes_per_bout"] + 2)[1:-1]
            votes = []
            for ps in probe_secs:
                r = self.rows[(self.rows["global_cow_id"] == b.global_cow_id) & (self.rows["sec"].between(ps - 2, ps + 2))
                              & ~self.rows["camera_id"].isin(overhead)]
                if r.empty:
                    continue
                r = r.assign(area=r["bw"] * r["bh"]).sort_values("area").iloc[-1]
                vp = resolve_video(r.get("source_video"), video_index)
                if vp is None:
                    continue
                s = chew_score(vp, float(r["timestamp"]), (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2),
                               rc["probe_duration_sec"], tuple(rc["chew_band_hz"]))
                votes.append(s["peak_ratio"] >= rc["min_peak_ratio"])
                n += 1
            if votes:
                m = (self.fused["global_cow_id"] == b.global_cow_id) & self.fused["sec"].between(b.start_sec, b.end_sec)
                self.fused.loc[m, "p_ruminating"] = 0.85 if any(votes) else 0.0
        return n

    def run(self, records: List[str], clip_start: Optional[str] = None,
            appearance: Optional[pd.DataFrame] = None,
            video_index: Optional[Dict[str, str]] = None, probe_rumination: bool = False) -> pd.DataFrame:
        if self.fused is None:
            self.load(records, clip_start)
        self.attach_appearance(appearance)
        tl = self.decode()
        if probe_rumination and video_index:
            n = self.probe_rumination(tl, video_index)
            print(f"[BEHAVIOR] rumination probes run: {n}")
            tl = self.decode()
        return tl
