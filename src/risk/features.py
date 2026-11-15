"""Per cow per hour feature table (the team's Level 2 feature list) + comparisons.

features_hourly.csv   cow_id, hour, <feature> (natural units), <feature>__z (vs the pen median that same
                      hour, robust), <feature>__rank (1 = most extreme in the sick direction), hours seen
features_summary.csv  cow_id x {all, day1, day2, change (day2 - day1), day (07-18 h), night}: value, z, rank

Feature numbers follow the team list:
 classifier  1 lying_min 2 lying_bouts 3 longest_lying_bout_min 4 standing_min 5 feeding_min 6 meals
             7 meal_min 8 longest_meal_gap_min 9 drinking_min 10 drinking_visits 11 walking_min
             12 ruminating_min 13 social_min
 tracking    14 distance_m 15 speed_mps 16 bunk_zone_min 17 water_zone_min 18 feed_latency_min
             19 nn_distance_m 20 cows_within_3m 21 edge_or_alone_min (22 stand-up/lie-down duration: needs
             per-second posture; not computed from per-minute labels)
 head signs  23 head_down_share 24 head_height (VLM-sampled, see head_signs.py) 25 ear_droop_share
             26 back_arched_share
 video       27 breathing_bpm 28 breathing_effort
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .indicators import indicators

# feature -> (source indicator, scale to "per hour" units, sick direction)
HOURLY = {
    "lying_min": ("lying_share", 60, 1), "lying_bouts": ("lying_bouts_per_h", 1, -1),
    "longest_lying_bout_min": ("longest_lying_bout_min", 1, 1), "standing_min": ("standing_share", 60, 0),
    "feeding_min": ("feeding_share", 60, -1), "meals": ("meals_per_h", 1, -1), "meal_min": ("meal_min", 1, -1),
    "longest_meal_gap_min": ("longest_meal_gap_min", 1, 1), "drinking_min": ("drinking_share", 60, -1),
    "drinking_visits": ("drinking_visit_rate", 1, -1), "walking_min": ("walking_share", 60, -1),
    "ruminating_min": ("rumination_share", 60, -1), "social_min": ("social_share", 60, -1),
    "distance_m": ("distance_m_per_h", 1, -1), "speed_mps": ("speed_mps", 1, -1),
    "bunk_zone_min": ("bunk_zone_share", 60, -1), "water_zone_min": ("water_zone_share", 60, -1),
    "feed_latency_min": ("feed_call_latency", 1, 1), "nn_distance_m": ("nn_distance_m", 1, 1),
    "cows_within_3m": ("cows_within_3m", 1, -1), "edge_or_alone_min": ("edge_or_alone_share", 60, 1),
}
ASSET = {"head_down_share": 1, "head_height": -1, "ear_droop_share": 1, "back_arched_share": 1,
         "breathing_bpm": 1, "breathing_effort": 1, "discharge_suspect_share": 1}
COVER = {"lying_min": "lying_share", "feeding_min": "feeding_share"}


def _hour_table(m: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    rows = []
    hours = (m.minute // 3600) * 3600
    for h in np.sort(hours.unique()):
        g = m[hours == h]
        g.attrs = m.attrs
        ind = indicators(g, cfg)
        r = pd.DataFrame(index=ind.index)
        for f, (src, scale, _) in HOURLY.items():
            if src in ind:
                r[f] = ind[src] * scale
        r["seen_h"] = ind.get("distance_m_per_h__hours", 0.0)
        r["labelled_h"] = ind.get("feeding_share__hours", 0.0)
        r["hour"] = int(h)
        rows.append(r.reset_index(names="cow_id"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _asset_hourly(assets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    b = assets.get("breathing")
    if b is not None and len(b):
        v = b[b.valid.astype(bool)].assign(hour=lambda d: (d.t0 // 3600 * 3600).astype(int))
        if len(v):
            agg = {"breathing_bpm": ("bpm", "median")}
            if "effort" in v:
                agg["breathing_effort"] = ("effort", "median")
            parts.append(v.groupby(["cow_id", "hour"]).agg(**agg))
    hs = assets.get("head_signs")
    if hs is not None and len(hs):
        h = hs.assign(hour=(hs.wall_t // 3600 * 3600).astype(int))
        st = h[h.standing.astype(str).str.lower().eq("true") & ~h.feeding.astype(str).str.lower().eq("true")]
        f = pd.DataFrame({
            "head_down_share": st.groupby(["cow_id", "hour"]).head_position.apply(lambda s: (s == "below_withers").mean()),
            "head_height": h[h.head_position.isin(["below_withers", "level", "above_withers"])].groupby(["cow_id", "hour"])
                .head_position.apply(lambda s: s.map({"below_withers": -1, "level": 0, "above_withers": 1}).mean()),
            "ear_droop_share": h[h.ears.isin(["droop", "normal"])].groupby(["cow_id", "hour"]).ears.apply(lambda s: (s == "droop").mean()),
            "back_arched_share": h[h.back.isin(["arched", "normal"])].groupby(["cow_id", "hour"]).back.apply(lambda s: (s == "arched").mean()),
            "discharge_suspect_share": h[h.muzzle_box.astype(str).str.startswith("[")].groupby(["cow_id", "hour"])
                .discharge_flag.apply(lambda s: s.astype(str).str.lower().eq("true").mean()),
        })
        parts.append(f)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1).reset_index()


def robust_z(v: pd.Series) -> pd.Series:
    med = v.median()
    mad = 1.4826 * (v - med).abs().median()
    sd = v.std()
    scale = mad if mad > 1e-9 else (sd if sd and np.isfinite(sd) and sd > 1e-9 else np.nan)
    return (v - med) / scale


def add_comparisons(df: pd.DataFrame, by: str, feats) -> pd.DataFrame:
    """<f>__z within each group (vs pen median), <f>__rank (1 = most extreme in the sick direction)."""
    df = df.copy()
    for f in feats:
        if f not in df:
            continue
        d = {**{k: v[2] for k, v in HOURLY.items()}, **ASSET}.get(f, 0)
        df[f + "__z"] = df.groupby(by)[f].transform(robust_z)
        key = df[f + "__z"] * (d if d else 1)
        df[f + "__rank"] = key.groupby(df[by]).rank(ascending=False, method="min")
    return df


def build(m: pd.DataFrame, cfg: Dict, assets: Optional[Dict[str, pd.DataFrame]] = None, min_seen_h: float = 0.25):
    hourly = _hour_table(m, cfg)
    if hourly.empty:
        return hourly, pd.DataFrame()
    ah = _asset_hourly(assets or {})
    if len(ah):
        hourly = hourly.merge(ah, on=["cow_id", "hour"], how="left")
    feats = [f for f in list(HOURLY) + list(ASSET) if f in hourly]
    hourly.loc[hourly.seen_h < min_seen_h, [f for f in HOURLY if f in hourly]] = np.nan
    hourly = add_comparisons(hourly, "hour", feats)
    hourly["time"] = pd.to_datetime(hourly.hour, unit="s")

    # summaries: hour-weighted means per period (each hour counts once, so long-visible cows do not dominate)
    t = pd.to_datetime(hourly.hour, unit="s")
    days = sorted(t.dt.strftime("%Y-%m-%d").unique())
    periods = {"all": np.ones(len(hourly), bool), "day": t.dt.hour.between(7, 17).to_numpy(),
               "night": (~t.dt.hour.between(7, 17)).to_numpy()}
    for i, d in enumerate(days):
        periods[f"day{i + 1}"] = (t.dt.strftime("%Y-%m-%d") == d).to_numpy()
    rows = []
    for name, mask in periods.items():
        g = hourly[mask]
        if g.empty:
            continue
        s = g.groupby("cow_id")[feats].mean()
        s["hours"] = g.groupby("cow_id").seen_h.sum()
        s["period"] = name
        rows.append(s.reset_index())
    summ = pd.concat(rows, ignore_index=True)
    if {"day1", "day2"} <= set(summ.period):
        a = summ[summ.period == "day1"].set_index("cow_id")[feats]
        b = summ[summ.period == "day2"].set_index("cow_id")[feats]
        ch = (b - a).reset_index().assign(period="change")
        summ = pd.concat([summ, ch], ignore_index=True)
    summ = add_comparisons(summ, "period", feats)
    return hourly, summ
