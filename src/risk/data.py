"""Level 2 inputs: one herd view over any number of processed runs.

  herd = Herd.load(["outputs/level1_d1_1600_h", "outputs/level1_d1_1950_h"], activity="activity.json")

* cow_state  (per cow per second, from each run's contract)  -> position, zone, nearest cow, speed
* activity   (per cow per minute: the behavior model / validation agent output) -> behavior shares
  Accepted: our activity.csv, or a JSON/CSV with at least cow_id, a minute timestamp and behavior
  (optional confidence and sec_<behavior> columns). If omitted, each run's best activity.csv is used.
* runs       kept for frames, crops and clips (Pen6Run)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..api.flow import EPOCH, Pen6Run

BEHAVIORS = ["lying", "standing", "walking", "feeding", "drinking", "ruminating (lying)",
             "ruminating (standing)", "grooming", "social interaction", "other / not visible"]
_ALIASES = {
    "lie": "lying", "lying down": "lying", "rest": "lying",
    "stand": "standing", "walk": "walking", "moving": "walking",
    "feed": "feeding", "eating": "feeding", "eat": "feeding",
    "drink": "drinking",
    "rum_lying": "ruminating (lying)", "ruminating lying": "ruminating (lying)", "rumination lying": "ruminating (lying)",
    "lying ruminating": "ruminating (lying)", "ruminating_lying": "ruminating (lying)",
    "rum_standing": "ruminating (standing)", "ruminating standing": "ruminating (standing)",
    "rumination standing": "ruminating (standing)", "standing ruminating": "ruminating (standing)",
    "ruminating_standing": "ruminating (standing)",
    "groom": "grooming", "self-grooming": "grooming",
    "social": "social interaction", "social_interaction": "social interaction", "interaction": "social interaction",
    "other": "other / not visible", "not visible": "other / not visible", "other/not visible": "other / not visible",
    "unknown": "other / not visible", "occluded": "other / not visible",
}


def canon_behavior(b) -> str:
    if not isinstance(b, str):
        return "other / not visible"
    s = b.strip().lower().replace("-", " ")
    if s in BEHAVIORS:
        return s
    s2 = s.replace("_", " ")
    return _ALIASES.get(s, _ALIASES.get(s2, s2 if s2 in BEHAVIORS else "other / not visible"))


def _epoch(series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        v = series.astype(float)
        return (v / 1000.0 if v.median() > 1e11 else v).astype(np.int64)  # ms or s
    return ((pd.to_datetime(series) - EPOCH).dt.total_seconds()).astype(np.int64)


def load_activity(path_or_df) -> pd.DataFrame:
    """Per cow per minute: cow_id, minute (epoch s), behavior, confidence, sec_<behavior> columns."""
    if isinstance(path_or_df, pd.DataFrame):
        a = path_or_df.copy()
    elif str(path_or_df).endswith(".json"):
        d = json.load(open(path_or_df))
        if isinstance(d, dict) and "rows" in d:
            d = d["rows"]
        if isinstance(d, dict):  # {cow_id: [ {...}, ...]}
            d = [{**r, "cow_id": k} for k, rows in d.items() for r in rows]
        a = pd.DataFrame(d)
    else:
        a = pd.read_csv(path_or_df)
    tcol = next((c for c in ["minute_start", "minute", "wall_time", "timestamp", "time", "t"] if c in a), None)
    if tcol is None or "cow_id" not in a or "behavior" not in a:
        raise ValueError("activity needs cow_id, behavior and a minute timestamp (minute_start / wall_time / timestamp)")
    a["minute"] = (_epoch(a[tcol]) // 60) * 60
    a["behavior"] = a.behavior.map(canon_behavior)
    if "confidence" not in a:
        a["confidence"] = 1.0
    # seconds per behavior: use sec_* columns when present, else the minute's label gets 60 s
    have = {b: f"sec_{b}" for b in BEHAVIORS if f"sec_{b}" in a}
    for b in BEHAVIORS:
        col = f"sec_{b}"
        if col not in a:
            a[col] = np.where(a.behavior == b, 60.0, 0.0) if not have else 0.0
    if "seconds_observed" not in a:
        a["seconds_observed"] = np.where(a.behavior == "other / not visible", 0, 60)
    keep = ["cow_id", "minute", "behavior", "confidence", "seconds_observed"] + [f"sec_{b}" for b in BEHAVIORS]
    return a[keep].drop_duplicates(["cow_id", "minute"], keep="last").sort_values(["cow_id", "minute"]).reset_index(drop=True)


@dataclass
class Herd:
    runs: List[Pen6Run]
    state: pd.DataFrame            # per cow per second (all runs)
    activity: pd.DataFrame         # per cow per minute
    layout: Dict = field(default_factory=dict)

    @classmethod
    def load(cls, run_dirs: Sequence[str], activity=None, video_root: str = "input",
             layout_path: str = "configs/pen_layout.yaml") -> "Herd":
        import yaml
        runs = [Pen6Run(d, video_root) for d in run_dirs]
        st = pd.concat([r.state.assign(run=r.name) for r in runs], ignore_index=True)
        st = st.sort_values(["cow_id", "sec"]).drop_duplicates(["cow_id", "sec"], keep="first").reset_index(drop=True)
        if activity is None:
            acts = []
            for r in runs:
                try:
                    acts.append(load_activity(r.activity("best")))
                except FileNotFoundError:
                    pass
            act = pd.concat(acts, ignore_index=True) if acts else pd.DataFrame()
        else:
            act = load_activity(activity)
        layout = yaml.safe_load(open(layout_path)) if os.path.exists(layout_path) else {}
        return cls(runs, st, act, layout)

    @property
    def cows(self) -> List[str]:
        return sorted(set(self.state.cow_id.dropna()) | set(self.activity.get("cow_id", pd.Series(dtype=str)).dropna()))

    def run_at(self, sec: int) -> Optional[Pen6Run]:
        for r in self.runs:
            s = r.state.sec
            if s.min() - 5 <= sec <= s.max() + 5:
                return r
        return None

    def span(self):
        return int(self.state.sec.min()), int(self.state.sec.max())

    def zone_poly(self, name: str):
        z = (self.layout.get("zones", {}).get(name) or self.layout.get("landmarks", {}).get(name) or {})
        return np.array(z.get("polygon_metric", []), float)


def iso(sec) -> str:
    return (EPOCH + pd.Timedelta(seconds=int(sec))).strftime("%Y-%m-%dT%H:%M:%S")


def hms(sec) -> str:
    return (EPOCH + pd.Timedelta(seconds=int(sec))).strftime("%m-%d %H:%M:%S")
