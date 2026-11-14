"""Level-1 behavior deliverables: activity.csv, bouts, per-cow time budgets, review queue."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .taxonomy import BEHAVIORS, NOT_VISIBLE

EPOCH = pd.Timestamp("1970-01-01")


def _wall(sec) -> pd.Series:
    return EPOCH + pd.to_timedelta(sec, unit="s")


def bouts(tl: pd.DataFrame) -> pd.DataFrame:
    """Run-length segments of identical behavior per cow (session breaks split bouts)."""
    tl = tl.sort_values(["global_cow_id", "sec"])
    new = (
        (tl["behavior"] != tl["behavior"].shift())
        | (tl["global_cow_id"] != tl["global_cow_id"].shift())
        | (tl["sec"].diff() != 1)
    )
    tl = tl.assign(bout_id=new.cumsum())
    b = tl.groupby("bout_id").agg(
        global_cow_id=("global_cow_id", "first"),
        behavior=("behavior", "first"),
        start_sec=("sec", "min"),
        end_sec=("sec", "max"),
        mean_confidence=("confidence", "mean"),
        evidence=("evidence", lambda s: s.value_counts().index[0]),
        observed_share=("observed", "mean"),
        cameras=("cameras", lambda s: ",".join(sorted({c for v in s.dropna() for c in str(v).split(",")}))),
        source_video=("source_video", lambda s: s.dropna().iloc[0] if s.notna().any() else None)
        if "source_video" in tl else ("sec", lambda s: None),
    )
    b["duration_sec"] = b["end_sec"] - b["start_sec"] + 1
    b["start_time"] = _wall(b["start_sec"])
    b["end_time"] = _wall(b["end_sec"] + 1)
    return b.reset_index(drop=True)


def minute_activity(tl: pd.DataFrame, full_days: Optional[list] = None, cows: Optional[list] = None) -> pd.DataFrame:
    """activity.csv: one row per cow per minute with the dominant behavior and its share.

    full_days=["2026-11-07","2026-11-08"] expands to every minute of those days; minutes
    without any video are labelled 'other / not visible' with evidence 'no_video'.
    """
    tl = tl.assign(minute=(tl["sec"] // 60) * 60)
    counts = tl.pivot_table(index=["global_cow_id", "minute"], columns="behavior", values="sec",
                            aggfunc="count", fill_value=0)
    counts = counts.reindex(columns=BEHAVIORS, fill_value=0)
    tot = counts.sum(axis=1)
    order = np.argsort(-counts.to_numpy(), axis=1)
    arr = counts.to_numpy()
    out = pd.DataFrame(index=counts.index)
    out["behavior"] = [BEHAVIORS[i] for i in order[:, 0]]
    out["share"] = arr[np.arange(len(arr)), order[:, 0]] / tot
    out["secondary_behavior"] = [BEHAVIORS[i] for i in order[:, 1]]
    out["secondary_share"] = arr[np.arange(len(arr)), order[:, 1]] / tot
    g = tl.groupby(["global_cow_id", "minute"])
    out["confidence"] = g["confidence"].mean()
    out["seconds_observed"] = g["observed"].sum()
    out["evidence"] = g["evidence"].agg(lambda s: s.value_counts().index[0])
    out["cameras"] = g["cameras"].agg(lambda s: ",".join(sorted({c for v in s.dropna() for c in str(v).split(",")})))
    for b in BEHAVIORS:
        out[f"sec_{b}"] = counts[b]
    out = out.reset_index()

    if full_days:
        cows = cows or sorted(out["global_cow_id"].unique())
        mins = []
        for d in full_days:
            s0 = int((pd.Timestamp(d) - EPOCH).total_seconds())
            mins.append(np.arange(s0, s0 + 86400, 60))
        grid = pd.MultiIndex.from_product([cows, np.concatenate(mins)], names=["global_cow_id", "minute"])
        out = out.set_index(["global_cow_id", "minute"]).reindex(grid).reset_index()
        nv = out["behavior"].isna()
        out.loc[nv, ["behavior", "evidence"]] = [NOT_VISIBLE, "no_video"]
        out.loc[nv, ["share", "confidence", "seconds_observed"]] = [1.0, 0.0, 0]
        out[[f"sec_{b}" for b in BEHAVIORS]] = out[[f"sec_{b}" for b in BEHAVIORS]].fillna(0).astype(int)

    out["minute_start"] = _wall(out["minute"])
    out["date"] = out["minute_start"].dt.date.astype(str)
    out = out.rename(columns={"global_cow_id": "cow_id"})
    first = ["cow_id", "date", "minute_start", "behavior", "share", "confidence", "evidence",
             "secondary_behavior", "secondary_share", "seconds_observed", "cameras"]
    return out[first + [c for c in out.columns if c not in first and c != "minute"]]


def time_budgets(tl: pd.DataFrame, bt: pd.DataFrame) -> pd.DataFrame:
    """Per cow per day: minutes and % of VISIBLE time per behavior + BRD-relevant bout stats."""
    tl = tl.assign(date=_wall(tl["sec"]).dt.date.astype(str))
    vis = tl[tl["behavior"] != NOT_VISIBLE]
    mins = vis.pivot_table(index=["global_cow_id", "date"], columns="behavior", values="sec",
                           aggfunc="count", fill_value=0).reindex(columns=BEHAVIORS, fill_value=0) / 60.0
    mins = mins.drop(columns=[NOT_VISIBLE])
    visible_min = mins.sum(axis=1)
    out = mins.add_prefix("min_")
    for b in mins.columns:
        out[f"pct_{b}"] = 100 * mins[b] / visible_min.replace(0, np.nan)
    out["visible_minutes"] = visible_min
    out["not_visible_minutes"] = tl[tl["behavior"] == NOT_VISIBLE].groupby(["global_cow_id", "date"]).size() / 60.0
    out["observed_share"] = tl.groupby(["global_cow_id", "date"])["observed"].mean()

    bt = bt.assign(date=bt["start_time"].dt.date.astype(str))
    def bstats(b, name, min_sec):
        x = bt[(bt["behavior"].isin(b)) & (bt["duration_sec"] >= min_sec)].groupby(["global_cow_id", "date"])["duration_sec"]
        out[f"{name}_bouts"] = x.count()
        out[f"{name}_mean_bout_min"] = x.mean() / 60.0
    bstats(["feeding"], "feeding", 30)
    bstats(["lying", "ruminating (lying)"], "lying", 60)
    bstats(["drinking"], "drinking", 5)
    bstats(["social interaction"], "social", 5)
    out = out.fillna({c: 0 for c in out.columns if c.endswith("_bouts")})
    return out.reset_index().rename(columns={"global_cow_id": "cow_id"})


def review_queue(bt: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Segments the agent should verify by looking at video (low confidence or heuristic-only)."""
    rc = cfg["review"]
    q = bt[(bt["duration_sec"] >= rc["min_segment_sec"]) & (bt["behavior"] != NOT_VISIBLE) & (
        (bt["mean_confidence"] < rc["low_confidence"]) | bt["evidence"].str.startswith("estimated_heuristic")
    )].copy()
    q["mid_time"] = q["start_time"] + (q["end_time"] - q["start_time"]) / 2
    q["reason"] = np.where(q["evidence"].str.startswith("estimated"), "heuristic_label_needs_visual_check",
                           "low_posterior_confidence")
    return q.sort_values("mean_confidence")


def write_all(out_dir: str, tl: pd.DataFrame, cfg: Dict, full_days=None, write_1s: bool = True) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    tl = tl.copy()
    tl["wall_time"] = _wall(tl["sec"])
    bt = bouts(tl)
    act = minute_activity(tl, full_days=full_days)
    tb = time_budgets(tl, bt)
    rq = review_queue(bt, cfg)
    paths = {
        "activity_csv": os.path.join(out_dir, "activity.csv"),
        "bouts_csv": os.path.join(out_dir, "behavior_bouts.csv"),
        "time_budgets_csv": os.path.join(out_dir, "time_budgets.csv"),
        "review_queue_csv": os.path.join(out_dir, "behavior_review_queue.csv"),
        "summary_json": os.path.join(out_dir, "behavior_summary.json"),
    }
    act.to_csv(paths["activity_csv"], index=False, float_format="%.3f")
    bt.to_csv(paths["bouts_csv"], index=False, float_format="%.3f")
    tb.to_csv(paths["time_budgets_csv"], index=False, float_format="%.2f")
    rq.to_csv(paths["review_queue_csv"], index=False, float_format="%.3f")
    if write_1s:
        paths["timeline_1s_csv"] = os.path.join(out_dir, "behavior_timeline_1s.csv")
        cols = ["global_cow_id", "wall_time", "behavior", "confidence", "evidence", "observed", "cameras",
                "world_x", "world_y", "speed_mps", "height_ratio", "nn_dist"] + [c for c in tl.columns if c.startswith("p::")]
        tl[[c for c in cols if c in tl.columns]].rename(columns={"global_cow_id": "cow_id"}).to_csv(
            paths["timeline_1s_csv"], index=False, float_format="%.3f")
    summary = {
        "cows": sorted(tl["global_cow_id"].unique().tolist()),
        "span": [str(tl["wall_time"].min()), str(tl["wall_time"].max())],
        "seconds_labelled": int(len(tl)),
        "evidence_share": tl["evidence"].value_counts(normalize=True).round(4).to_dict(),
        "behavior_share": tl["behavior"].value_counts(normalize=True).round(4).to_dict(),
        "review_segments": int(len(rq)),
    }
    with open(paths["summary_json"], "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return paths
