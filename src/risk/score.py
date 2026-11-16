"""Behavior risk score: herd-relative robust z per indicator, literature weights, coverage shrinkage,
optional self-relative trend, then a logistic index. Every number that moves the score is kept as a
contribution so the reasons (and the agent's facts) can quote it."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .indicators import indicators, per_day


def robust_z(v: pd.Series, floor: float) -> pd.Series:
    med = v.median()
    mad = 1.4826 * (v - med).abs().median()
    return (v - med) / max(mad if np.isfinite(mad) else 0.0, floor)


# Indicators that follow the clock (the herd feeds at deliveries, lies at night). Compared with the herd at the
# SAME hours the cow was seen: on the 48 h run COW_01 was missing from a 2 h chunk in which the herd fed 40 % of
# the time, so "4 % vs herd 10 %" overstated its gap (time-matched: 3.8 % vs 8.2 %).
TIME_MATCHED = {"feeding_share", "lying_share", "rumination_share", "grooming_share", "bunk_visit_rate",
                "drinking_visit_rate", "distance_m_per_h", "nn_distance_m", "isolation_share", "desync_share",
                "meals_per_h", "edge_or_alone_share"}


def herd_expectation(m: pd.DataFrame, cfg: Dict, min_cover_h: float = 1 / 6) -> pd.DataFrame:
    """cows x indicators: the herd's value at the hours this cow was seen = the herd MEAN of each hour (cows with
    >= 10 min that hour, coverage-weighted), averaged with this cow's coverage that hour as the weight. Mean, not
    median: in most hours most cows do not feed at all, so the hourly median is 0 and understates the herd.
    NaN where no hour qualifies."""
    hours = (m.minute // 3600) * 3600
    num: Dict = {}
    den: Dict = {}
    for h in np.unique(hours):
        sub = m[hours == h]
        sub.attrs = {k: v for k, v in m.attrs.items() if k != "bunk_occupancy"}
        ind = indicators(sub, cfg)
        for name in TIME_MATCHED:
            if name not in ind:
                continue
            v, cov = ind[name], ind.get(f"{name}__hours", pd.Series(0.0, index=ind.index))
            ok = v.notna() & (cov >= min_cover_h)
            if ok.sum() < 3:
                continue
            med = float(np.average(v[ok], weights=cov[ok]))
            for cow in ind.index[ok]:
                num[(cow, name)] = num.get((cow, name), 0.0) + med * float(cov[cow])
                den[(cow, name)] = den.get((cow, name), 0.0) + float(cov[cow])
    if not num:
        return pd.DataFrame()
    return (pd.Series(num) / pd.Series(den)).unstack()


def contributions(ind: pd.DataFrame, cfg: Dict, trend: Optional[pd.DataFrame] = None,
                  expected: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Long table: cow_id, indicator, value, herd_median, z, signed, hours, shrink, weight, contrib, matched.
    With `expected` (herd_expectation), clock-dependent indicators are scored on value - herd at the same hours,
    and herd_median holds that per-cow reference (matched = True)."""
    sc = cfg["scoring"]
    rows = []
    for name, spec in cfg["indicators"].items():
        if name not in ind:
            continue
        v, h = ind[name], ind.get(f"{name}__hours", pd.Series(0.0, index=ind.index))
        usable = v.notna() & (h >= (sc["min_hours"] if name != "feed_call_latency" else 1))
        matched = expected is not None and name in getattr(expected, "columns", [])
        if matched:
            ref = expected[name].reindex(v.index)
            usable &= ref.notna()
        if usable.sum() < 3:
            continue
        if matched:
            z = robust_z((v - ref)[usable], spec["floor"])
        else:
            ref = pd.Series(float(v[usable].median()), index=v.index)
            z = robust_z(v[usable], spec["floor"])
        for cow in ind.index:
            if not usable.get(cow, False):
                rows.append({"cow_id": cow, "indicator": name, "value": v.get(cow), "hours": h.get(cow, 0.0),
                             "herd_median": v[usable].median(), "z": np.nan, "signed": np.nan, "shrink": 0.0,
                             "weight": spec["weight"], "contrib": 0.0, "kind": "level", "matched": matched})
                continue
            s = float(np.clip(spec["direction"] * z[cow], -sc["z_clip"], sc["z_clip"]))
            hh = float(h[cow]) if name != "feed_call_latency" else float(h[cow]) * 2.0  # each rush ~ 2 h of evidence
            shrink = hh / (hh + sc["shrink_hours"])
            eff = s if s > 0 else sc["protective_factor"] * s
            rows.append({"cow_id": cow, "indicator": name, "value": float(v[cow]), "hours": float(h[cow]),
                         "herd_median": float(ref[cow]), "z": float(z[cow]), "signed": s,
                         "shrink": shrink, "weight": spec["weight"], "contrib": spec["weight"] * shrink * eff,
                         "kind": "level", "matched": matched})
    if trend is not None:
        for name, spec in cfg["indicators"].items():
            if name not in trend:
                continue
            d = trend[name].dropna()
            if len(d) < 3:
                continue
            z = robust_z(d, spec["floor"])
            for cow in d.index:
                s = float(np.clip(spec["direction"] * z[cow], -sc["z_clip"], sc["z_clip"]))
                w = spec["weight"] * sc["trend_weight"]
                rows.append({"cow_id": cow, "indicator": name + "__trend", "value": float(d[cow]), "hours": np.nan,
                             "herd_median": float(d.median()), "z": float(z[cow]), "signed": s, "shrink": 1.0,
                             "weight": w, "contrib": w * (s if s > 0 else sc["protective_factor"] * s), "kind": "trend"})
    return pd.DataFrame(rows)


def score_from(contrib: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    sc = cfg["scoring"]
    if contrib.empty:
        return pd.DataFrame(columns=["behavior_logit", "behavior_score", "evidence_weight"])
    g = contrib.groupby("cow_id")
    w_ind = contrib.groupby("indicator").weight.first()
    total_w = w_ind.sum()
    wsum = g.apply(lambda d: (d.weight * d.shrink).sum(), include_groups=False)
    # weighted Stouffer combination: sum(w * z_eff) / sqrt(sum w^2) over the indicators scored in this herd
    combined = g.contrib.sum() / max(float(np.sqrt((w_ind ** 2).sum())), 1e-9)
    logit = sc["intercept"] + sc["slope"] * combined
    out = pd.DataFrame({"behavior_logit": logit, "behavior_score": 1 / (1 + np.exp(-logit)),
                        "evidence_weight": wsum / max(total_w, 1e-9), "combined_z": combined})
    return out


def trend_table(m: pd.DataFrame, cfg: Dict) -> Optional[pd.DataFrame]:
    days = per_day(m, cfg)
    if len(days) < 2:
        return None
    ks = sorted(days)
    first, last = days[ks[0]], days[ks[-1]]
    cols = [c for c in cfg["indicators"] if c in first and c in last]
    d = last[cols] - first[cols]
    # the herd's own change (feed / weather / handling) is removed by the robust z of the deltas
    return d


def behavior_scores(m: pd.DataFrame, cfg: Dict, with_bootstrap: bool = True, seed: int = 0,
                    extra: Optional[pd.DataFrame] = None):
    """extra: per-cow indicators that are not resampled by time (risk assets: breathing, motion, jerks)."""
    ind = indicators(m, cfg)
    if extra is not None and len(extra.columns):
        ind = ind.join(extra, how="left")
    tr = trend_table(m, cfg)
    expected = herd_expectation(m, cfg)
    con = contributions(ind, cfg, tr, expected)
    sc = score_from(con, cfg)
    sc["rank"] = sc.behavior_score.rank(ascending=False, method="first").astype(int)
    if with_bootstrap and cfg["scoring"]["bootstrap"]:
        sc["p_top3"] = rank_stability(m, cfg, seed, extra, expected)
    return ind, con, sc.sort_values("rank")


def rank_stability(m: pd.DataFrame, cfg: Dict, seed: int = 0, extra: Optional[pd.DataFrame] = None,
                   expected: Optional[pd.DataFrame] = None) -> pd.Series:
    """Block bootstrap over time: how often is each cow in the top 3 when the observed time is resampled?
    (The same-hours herd reference is the full-data one: recomputing it per resample costs 48 indicator passes.)"""
    rng = np.random.default_rng(seed)
    bl = cfg["scoring"]["block_min"] * 60
    mins = np.sort(m.minute.unique())
    blocks = [mins[(mins >= s) & (mins < s + bl)] for s in np.arange(mins.min(), mins.max() + 1, bl)]
    blocks = [b for b in blocks if len(b)]
    hits, n = pd.Series(0.0, index=sorted(m.cow_id.unique())), 0
    for _ in range(cfg["scoring"]["bootstrap"]):
        pick = rng.integers(0, len(blocks), len(blocks))
        minutes = np.concatenate([blocks[i] for i in pick])
        ind = indicators(m, cfg, minutes)
        if extra is not None and len(extra.columns):
            ind = ind.join(extra, how="left")
        s = score_from(contributions(ind, cfg, expected=expected), cfg)
        if s.empty:
            continue
        top = s.behavior_score.sort_values(ascending=False).index[:3]
        hits[hits.index.isin(top)] += 1
        n += 1
    return hits / max(n, 1)


def reasons_text(con: pd.DataFrame, cfg: Dict, cow: str, k: int = 3) -> List[str]:
    d = con[(con.cow_id == cow) & (con.contrib > 0)].sort_values("contrib", ascending=False).head(k)
    out = []
    for _, r in d.iterrows():
        base = r.indicator.replace("__trend", "")
        spec = cfg["indicators"][base]
        f = fmt(r.value, spec["unit"]), fmt(r.herd_median, spec["unit"])
        if r.kind == "trend":
            out.append(f"{spec['label']} changed {f[0]} day-to-day (herd {f[1]})")
        elif base == "feed_call_latency":
            out.append(f"{spec['label']} {f[0]} vs herd median {f[1]} (z {r.z:+.1f}, over {r.hours:.0f} feed calls)")
        elif bool(r.get("matched", False)):
            out.append(f"{spec['label']} {f[0]} vs herd {f[1]} at the same hours (z {r.z:+.1f}, {r.hours:.1f} h observed)")
        else:
            out.append(f"{spec['label']} {f[0]} vs herd median {f[1]} (z {r.z:+.1f}, {r.hours:.1f} h observed)")
    return out


def fmt(v, unit: str) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    if unit == "%":
        return f"{100 * v:.0f}%"
    return f"{v:.1f} {unit}" if abs(v) < 100 else f"{v:.0f} {unit}"
