"""Behavioral BRD indicators per cow, from activity (per minute) + fused cow state (per second).

Step 1 builds a per cow per minute table of additive quantities (seconds feeding, bouts started,
metres walked, ...). Step 2 aggregates any subset of minutes into indicators, so the same code gives
whole-period values, per-day values (self-relative trend) and block-bootstrap resamples (rank
stability). Every indicator carries its coverage (observed hours) so thin evidence is shrunk.

Indicator          unit                 sick direction   literature (configs/risk.yaml -> sources)
feeding_share      share of observed    lower            bunk time / head-down time fall before BRD dx
bunk_visit_rate    visits per hour      lower            fewer bunk visits / meals
bunk_visit_min     min per visit        lower            shorter bunk visits
feed_call_latency  min after herd rush  higher           "slow to come to eat" (DART appetite)
lying_share        share of observed    higher           longer lying / inactivity before dx
lying_bout_min     min per bout         higher           longer, fewer lying bouts
distance_m_per_h   metres per hour      lower            reduced activity days before dx
rumination_share   share of observed    lower            reduced rumination (weak: classifier)
drinking_visit_rate visits per hour     lower            water intake falls (visits barely change)
nn_distance_m      m to nearest cow     higher           social withdrawal (sickness behavior)
isolation_share    share > iso_m alone  higher           social withdrawal
desync_share       share off-herd       higher           e.g. lying while the herd feeds
grooming_share     share of observed    lower            self-maintenance drops in sickness behavior
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data import BEHAVIORS, Herd

LYING = ["lying", "ruminating (lying)"]
RUM = ["ruminating (lying)", "ruminating (standing)"]
UPRIGHT_IDLE = ["standing", "ruminating (standing)", "grooming", "social interaction"]


def _in_poly(x, y, poly) -> np.ndarray:
    if poly is None or len(poly) < 3:
        return np.zeros(len(x), bool)
    from matplotlib.path import Path
    return Path(poly).contains_points(np.c_[x, y])


def coarse(b: str) -> str:
    if b in LYING:
        return "lying"
    if b == "feeding":
        return "feeding"
    if b == "other / not visible":
        return "unknown"
    return "upright"


def minute_table(herd: Herd, cfg: Dict) -> pd.DataFrame:
    """One row per cow per minute with additive quantities (seconds, counts, metres)."""
    c = cfg["extraction"]
    st = herd.state.copy()
    st["minute"] = (st.sec // 60) * 60
    real = ~st.predicted.astype(str).str.lower().eq("true")
    x, y = st.x.to_numpy(float), st.y.to_numpy(float)
    st["at_bunk"] = (y <= c["bunk_y_m"]) & real
    st["at_water"] = _in_poly(x, y, herd.zone_poly("drinking_zone")) & real
    st["seen"] = real & np.isfinite(x)
    # walking only: fused positions wobble by ~0.1-0.2 m/s even for a lying cow, which adds up over hours
    sp = st.speed_mps.where(real & st.speed_mps.between(c.get("min_speed_mps", 0.25), c["max_speed_mps"]), 0.0).fillna(0.0)
    st["dist"] = sp  # 1 s rows -> metres
    nn = st.nearest_m.where(real)
    st["nn"] = nn
    st["iso"] = (nn > c["isolation_m"]) & real
    # pen edge: within edge_m of a fence (chute side x=0, Pen 7 side x=12, back fence y=30)
    W, L, e = 12.0, 30.0, c.get("edge_m", 1.0)
    st["edge"] = real & ((x <= e) | (x >= W - e) | (y >= L - e))
    st["within3"] = np.nan
    pos = st[real & np.isfinite(x)]
    if len(pos):
        cnt = np.full(len(pos), np.nan)
        px, py = pos.x.to_numpy(float), pos.y.to_numpy(float)
        for _, idx in pos.groupby("sec").indices.items():
            if len(idx) == 1:
                cnt[idx] = 0
                continue
            d = np.hypot(px[idx, None] - px[None, idx], py[idx, None] - py[None, idx])
            cnt[idx] = (d <= c.get("near_m", 3.0)).sum(1) - 1
        st.loc[pos.index, "within3"] = cnt

    def starts(flag: pd.Series, gap_s: int) -> pd.Series:
        """bout starts: flag turns on after >= gap_s seconds off (per cow, seconds may skip)."""
        out = np.zeros(len(flag), bool)
        for _, idx in st.groupby("cow_id").indices.items():
            f = flag.to_numpy()[idx]
            s = st.sec.to_numpy()[idx]
            last_on = -10 ** 12
            for i in range(len(idx)):
                if f[i]:
                    if s[i] - last_on > gap_s:
                        out[idx[i]] = True
                    last_on = s[i]
        return pd.Series(out, index=flag.index)

    st["bunk_start"] = starts(st.at_bunk, c["visit_gap_s"])
    st["water_start"] = starts(st.at_water, c["visit_gap_s"])
    g = st.groupby(["cow_id", "minute"])
    m = pd.DataFrame({
        "seen_s": g.seen.sum(), "bunk_s": g.at_bunk.sum(), "bunk_starts": g.bunk_start.sum(),
        "water_s": g.at_water.sum(), "water_starts": g.water_start.sum(), "dist_m": g.dist.sum(),
        "nn_sum": g.nn.sum(), "nn_n": g.nn.count(), "iso_s": g.iso.sum(), "edge_s": g.edge.sum(),
        "near_sum": g.within3.sum(), "near_n": g.within3.count(), "edge_or_alone_s": (st.edge | st.iso).groupby([st.cow_id, st.minute]).sum(),
        "x": g.x.median(), "y": g.y.median(),
    }).reset_index()

    a = herd.activity
    if len(a):
        m = m.merge(a, on=["cow_id", "minute"], how="outer")
    else:
        for b in BEHAVIORS:
            m[f"sec_{b}"] = 0.0
        m["behavior"], m["confidence"], m["seconds_observed"] = "other / not visible", 0.0, 0
    num = [col for col in m.columns if col not in ("cow_id", "minute", "behavior")]
    m[num] = m[num].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    m["behavior"] = m.behavior.fillna("other / not visible")
    m["beh_obs_s"] = m[[f"sec_{b}" for b in BEHAVIORS if b != "other / not visible"]].sum(1)

    # lying bouts: a minute that is mostly lying after a minute that was not
    m = m.sort_values(["cow_id", "minute"]).reset_index(drop=True)
    lying_s = m[[f"sec_{b}" for b in LYING]].sum(1)
    m["lying_s"] = lying_s
    is_lying = (lying_s >= 0.5 * m.beh_obs_s.clip(lower=1)) & (m.beh_obs_s > 0)
    prev = is_lying.groupby(m.cow_id).shift(1, fill_value=False)
    prev_min = m.groupby("cow_id").minute.shift(1)
    m["lying_starts"] = (is_lying & (~prev | (m.minute - prev_min > 120))).astype(int)
    # meals: feeding minutes (>= 20 s) merged when the gap is <= meal_gap_min (a meal has short pauses)
    feeding = (m.sec_feeding >= 20).to_numpy()
    starts_meal = np.zeros(len(m), int)
    for _, idx in m.groupby("cow_id").indices.items():
        last = -10 ** 12
        for i in idx:
            if feeding[i]:
                if m.minute.iat[i] - last > 60 * c.get("meal_gap_min", 5):
                    starts_meal[i] = 1
                last = m.minute.iat[i]
    m["meal_starts"] = starts_meal
    m["feeding_min_flag"] = feeding.astype(int)

    # herd synchrony: does this cow do something else than most herdmates this minute?
    m["coarse"] = m.behavior.map(coarse)
    known = m[m.coarse != "unknown"]
    mode = known.groupby("minute").coarse.agg(lambda s: s.value_counts().idxmax())
    share = known.groupby("minute").coarse.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    n_herd = known.groupby("minute").size()
    m["herd_mode"] = m.minute.map(mode)
    ok = (m.minute.map(n_herd) >= c["sync_min_cows"]) & (m.minute.map(share) >= c["sync_min_share"]) & (m.coarse != "unknown")
    m["sync_s"] = np.where(ok, m.beh_obs_s, 0.0)
    m["desync_s"] = np.where(ok & (m.coarse != m.herd_mode), m.beh_obs_s, 0.0)

    # herd feed rushes (feed delivery proxy): bunk occupancy jumps from low to high
    occ = m.assign(b=m.bunk_s >= 30).groupby("minute").agg(n=("seen_s", lambda s: (s > 0).sum()), b=("b", "sum"))
    occ["share"] = occ.b / occ.n.clip(lower=1)
    m.attrs["feed_rushes"] = feed_rushes(occ, c)
    return m


def feed_rushes(occ: pd.DataFrame, c: Dict) -> List[int]:
    """Minutes where the herd moves to the bunk: share >= hi after >= quiet_min of share <= lo."""
    out, last = [], -10 ** 12
    sh = occ.share
    mins = sh.index.to_numpy()
    for i, t in enumerate(mins):
        if sh.iloc[i] < c["rush_hi"] or occ.n.iloc[i] < c["sync_min_cows"]:
            continue
        before = sh[(sh.index >= t - 60 * c["rush_quiet_min"]) & (sh.index < t)]
        if len(before) >= c["rush_quiet_min"] // 2 and (before <= c["rush_lo"]).mean() >= 0.8 and t - last > 3600:
            out.append(int(t))
            last = t
    return out


def indicators(m: pd.DataFrame, cfg: Dict, minutes: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Aggregate a minute table (optionally a subset / resample of minutes) into indicators per cow.
    Returns a wide frame: <indicator> and <indicator>__hours (coverage) per cow."""
    c = cfg["extraction"]
    rushes = list(m.attrs.get("feed_rushes", []))
    if minutes is not None:
        idx = pd.DataFrame({"minute": minutes, "_w": 1}).groupby("minute")._w.sum()
        m = m[m.minute.isin(idx.index)].copy()
        m["_w"] = m.minute.map(idx)
    else:
        m = m.assign(_w=1)
    w = m._w
    S = lambda col: (m[col] * w).groupby(m.cow_id).sum()
    beh_h = S("beh_obs_s") / 3600
    seen_h = S("seen_s") / 3600
    out = pd.DataFrame(index=sorted(m.cow_id.unique()))
    sb = lambda bs: sum(S(f"sec_{b}") for b in bs)
    den = (beh_h * 3600).replace(0, np.nan)
    out["feeding_share"] = sb(["feeding"]) / den
    out["lying_share"] = sb(LYING) / den
    out["rumination_share"] = sb(RUM) / den
    out["grooming_share"] = sb(["grooming"]) / den
    out["desync_share"] = S("desync_s") / S("sync_s").replace(0, np.nan)
    out["bunk_visit_rate"] = S("bunk_starts") / seen_h.replace(0, np.nan)
    out["bunk_visit_min"] = S("bunk_s") / 60 / S("bunk_starts").replace(0, np.nan)
    out["drinking_visit_rate"] = S("water_starts") / seen_h.replace(0, np.nan)
    out["distance_m_per_h"] = S("dist_m") / seen_h.replace(0, np.nan)
    out["nn_distance_m"] = S("nn_sum") / S("nn_n").replace(0, np.nan)
    out["isolation_share"] = S("iso_s") / (seen_h * 3600).replace(0, np.nan)
    out["lying_bout_min"] = S("lying_s") / 60 / S("lying_starts").replace(0, np.nan)
    out["standing_share"] = sb(["standing"]) / den
    out["walking_share"] = sb(["walking"]) / den
    out["drinking_share"] = sb(["drinking"]) / den
    out["social_share"] = sb(["social interaction"]) / den
    out["water_zone_share"] = S("water_s") / (seen_h * 3600).replace(0, np.nan)
    out["bunk_zone_share"] = S("bunk_s") / (seen_h * 3600).replace(0, np.nan)
    out["speed_mps"] = S("dist_m") / (seen_h * 3600).replace(0, np.nan)
    out["cows_within_3m"] = S("near_sum") / S("near_n").replace(0, np.nan)
    out["edge_or_alone_share"] = S("edge_or_alone_s") / (seen_h * 3600).replace(0, np.nan)
    out["lying_bouts_per_h"] = S("lying_starts") / beh_h.replace(0, np.nan)
    out["meals_per_h"] = S("meal_starts") / beh_h.replace(0, np.nan)
    out["meal_min"] = S("feeding_min_flag") / S("meal_starts").replace(0, np.nan)
    seq = sequence_features(m)
    out = out.join(seq, how="left")
    cov = {"feeding_share": beh_h, "lying_share": beh_h, "rumination_share": beh_h, "grooming_share": beh_h,
           "desync_share": S("sync_s") / 3600, "bunk_visit_rate": seen_h, "bunk_visit_min": seen_h,
           "drinking_visit_rate": seen_h, "distance_m_per_h": seen_h, "nn_distance_m": seen_h,
           "isolation_share": seen_h, "lying_bout_min": beh_h,
           "standing_share": beh_h, "walking_share": beh_h, "drinking_share": beh_h, "social_share": beh_h,
           "water_zone_share": seen_h, "bunk_zone_share": seen_h, "speed_mps": seen_h, "cows_within_3m": seen_h,
           "edge_or_alone_share": seen_h, "lying_bouts_per_h": beh_h, "meals_per_h": beh_h, "meal_min": beh_h,
           "longest_lying_bout_min": beh_h, "longest_meal_gap_min": beh_h}
    # feed-call latency: minutes from each herd rush to this cow's first bunk minute
    lat = {}
    for cow, g in m.groupby("cow_id"):
        vals = []
        for t in rushes:
            win = g[(g.minute >= t - 120) & (g.minute <= t + 60 * c["latency_cap_min"])]
            if win.seen_s.sum() < 600:
                continue
            hit = win[win.bunk_s >= 30]
            vals.append(min(max((hit.minute.min() - t) / 60, 0), c["latency_cap_min"]) if len(hit) else c["latency_cap_min"])
        lat[cow] = (np.mean(vals) if vals else np.nan, len(vals))
    out["feed_call_latency"] = pd.Series({k: v[0] for k, v in lat.items()})
    cov["feed_call_latency"] = pd.Series({k: float(v[1]) for k, v in lat.items()})  # number of rushes seen
    for k, v in cov.items():
        out[f"{k}__hours"] = v.reindex(out.index).fillna(0.0)
    return out


def sequence_features(m: pd.DataFrame) -> pd.DataFrame:
    """Order-dependent features on unique minutes: longest lying bout and longest gap between meals
    (minutes). Gaps in the footage (> 2 min without labels) end a bout and are not counted as gaps."""
    rows = {}
    for cow, g in m.drop_duplicates(["cow_id", "minute"]).sort_values("minute").groupby("cow_id"):
        t = g.minute.to_numpy()
        lying = (g.lying_s >= 0.5 * g.beh_obs_s.clip(lower=1)).to_numpy() & (g.beh_obs_s > 0).to_numpy()
        feed = (g.sec_feeding >= 20).to_numpy()
        seen = (g.beh_obs_s > 0).to_numpy()
        best_lie = cur = 0
        for i in range(len(g)):
            cont = i > 0 and t[i] - t[i - 1] <= 120
            cur = (cur + 1 if cont else 1) if lying[i] else 0
            best_lie = max(best_lie, cur)
        gap = best_gap = 0
        last_feed = None
        for i in range(len(g)):
            if i > 0 and t[i] - t[i - 1] > 120:   # footage gap: restart the gap count
                last_feed, gap = None, 0
            if feed[i]:
                if last_feed is not None:
                    best_gap = max(best_gap, (t[i] - last_feed) / 60 - 1)
                last_feed = t[i]
        rows[cow] = {"longest_lying_bout_min": float(best_lie) if seen.any() else np.nan,
                     "longest_meal_gap_min": float(best_gap) if feed.sum() >= 2 else np.nan}
    return pd.DataFrame.from_dict(rows, orient="index")


def per_day(m: pd.DataFrame, cfg: Dict) -> Dict[str, pd.DataFrame]:
    days = pd.to_datetime(m.minute, unit="s").dt.strftime("%Y-%m-%d")
    out = {}
    for d in sorted(days.unique()):
        sub = m[days == d]
        sub.attrs = m.attrs
        out[d] = indicators(sub, cfg)
    return out
