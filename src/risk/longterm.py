"""Long-term behavior tools (no video needed) + indicators from the risk assets.

Mixed into RiskTools: daily_budget, trend, feed_response, bouts, social, breathing, gallery, jerks.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from .data import hms

BLOCK_LABELS = {0: "00-06", 6: "06-12", 12: "12-18", 18: "18-24"}


def load_assets(run_dirs: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, List[pd.DataFrame]] = {}
    for d in run_dirs:
        a = os.path.join(d, "assets")
        if not os.path.isdir(a):
            continue
        for name in ("gallery", "motion", "breathing", "jerks", "feed_events", "clips", "head_signs"):
            p = os.path.join(a, f"{name}.csv")
            if os.path.exists(p) and os.path.getsize(p) > 5:
                try:
                    df = pd.read_csv(p)
                except pd.errors.EmptyDataError:
                    continue
                for col in ("image", "file"):
                    if col in df and name in ("breathing", "clips"):
                        df[col] = [os.path.join(a, f) for f in df[col]]
                out.setdefault(name, []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def asset_indicators(assets: Dict[str, pd.DataFrame], cows: List[str], hours_seen: pd.Series) -> pd.DataFrame:
    """breathing_bpm (median of valid windows), motion_energy (median while upright and visible),
    jerk_rate (events per hour seen). Coverage columns follow the <name>__hours convention."""
    ind = pd.DataFrame(index=sorted(cows))
    b = assets.get("breathing")
    if b is not None and len(b):
        v = b[b.valid.astype(bool)]
        ind["breathing_bpm"] = v.groupby("cow_id").bpm.median()
        ind["breathing_bpm__hours"] = v.groupby("cow_id").size().reindex(ind.index).fillna(0) * 0.5  # a window ~ 0.5 h of evidence
    m = assets.get("motion")
    if m is not None and len(m):
        # per camera z first (pixel motion depends on distance / lens), then the cow's median
        m = m.dropna(subset=["motion"])
        m = m.assign(mz=m.groupby("camera").motion.transform(lambda s: (s - s.median()) / (1.4826 * (s - s.median()).abs().median() + 1e-6)))
        ind["motion_energy"] = m.groupby("cow_id").mz.median()
        ind["motion_energy__hours"] = m.groupby("cow_id").wall_t.nunique().reindex(ind.index).fillna(0) / 3600
    if b is not None and len(b) and "effort" in b:
        v = b[b.valid.astype(bool)]
        ind["breathing_effort"] = v.groupby("cow_id").effort.median()
        ind["breathing_effort__hours"] = v.groupby("cow_id").size().reindex(ind.index).fillna(0) * 0.5
    hs = assets.get("head_signs")
    if hs is not None and len(hs):
        T = lambda s: s.astype(str).str.lower().eq("true")
        up = hs[T(hs.standing) & ~T(hs.feeding)]
        share = lambda d, col, yes, ok: d[d[col].isin(ok)].groupby("cow_id")[col].apply(lambda s: (s == yes).mean())
        ind["head_down_share"] = share(up, "head_position", "below_withers", ["below_withers", "level", "above_withers"])
        ind["ear_droop_share"] = share(hs, "ears", "droop", ["droop", "normal"])
        ind["back_arched_share"] = share(hs, "back", "arched", ["arched", "normal"])
        mz = hs[hs.muzzle_box.astype(str).str.startswith("[")]
        ind["discharge_suspect_share"] = mz.groupby("cow_id").discharge_flag.apply(lambda s: T(s).mean())
        n = hs.groupby("cow_id").size().reindex(ind.index).fillna(0) * 0.25   # a labelled crop ~ 15 min of evidence
        for k in ("head_down_share", "ear_droop_share", "back_arched_share", "discharge_suspect_share"):
            ind[f"{k}__hours"] = n
    j = assets.get("jerks")
    if j is not None:
        n = j.groupby("cow_id").size() if len(j) else pd.Series(dtype=float)
        ind["jerk_rate"] = (n.reindex(ind.index).fillna(0) / hours_seen.reindex(ind.index).replace(0, np.nan))
        ind["jerk_rate__hours"] = hours_seen.reindex(ind.index).fillna(0)
    return ind


def cusum_onset(dev: pd.Series, k: float = 0.5, h: float = 3.0) -> Optional[int]:
    """One-sided CUSUM on standardised deviations (sick direction positive): first index where S > h."""
    s = 0.0
    for i, x in enumerate(dev.fillna(0).to_numpy()):
        s = max(0.0, s + x - k)
        if s > h:
            return i
    return None


class LongTermTools:
    """Mixin for RiskTools: expects self.m (minute table), self.cfg, self.h (Herd), self.assets, self.store."""

    def _blocks(self, block_h: int = 6) -> pd.DataFrame:
        m = self.m.copy()
        ts = pd.to_datetime(m.minute, unit="s")
        m["block"] = ts.dt.floor(f"{block_h}h")
        return m

    def _ind_by_block(self, block_h: int = 6) -> pd.DataFrame:
        from .indicators import indicators
        m = self._blocks(block_h)
        out = []
        for blk, g in m.groupby("block"):
            g.attrs = self.m.attrs
            ind = indicators(g, self.cfg)
            ind["block"] = blk
            out.append(ind.reset_index(names="cow_id"))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def t_features(self, cow_id: str, period: str = "all"):
        """All extracted features for the cow (period: all, day, night, day1, day2, change) with herd z and rank among the cows."""
        fs = getattr(self, "features_summary", None)
        if fs is None or fs.empty:
            return self._R("no feature table (run_level2 builds it)")
        g = fs[(fs.cow_id == cow_id) & (fs.period == period)]
        if g.empty:
            return self._R(f"no '{period}' row for {cow_id}; periods: {sorted(fs.period.unique())}")
        r = g.iloc[0]
        n = fs[fs.period == period].cow_id.nunique()
        lines = [f"{cow_id} {period} ({n} cows; rank 1 = most extreme in the sick direction)"]
        for c in fs.columns:
            if c.endswith("__z") or c.endswith("__rank") or c in ("cow_id", "period", "hours") or pd.isna(r[c]):
                continue
            z, rk = r.get(c + "__z"), r.get(c + "__rank")
            lines.append(f"{c}: {r[c]:.3g}" + (f"  z {z:+.1f}" if pd.notna(z) else "") + (f"  rank {int(rk)}" if pd.notna(rk) else ""))
        return self._R("\n".join(lines))

    def t_daily_budget(self, cow_id: str):
        """Time budget of the cow per day and per 6-hour block (feeding, lying, ruminating, walking, standing) next to the herd median."""
        m = self._blocks(6)
        cols = {"feeding": ["sec_feeding"], "lying": ["sec_lying", "sec_ruminating (lying)"],
                "ruminating": ["sec_ruminating (lying)", "sec_ruminating (standing)"], "walking": ["sec_walking"],
                "standing": ["sec_standing"]}
        rows = []
        for blk, g in m.groupby("block"):
            den = g.groupby("cow_id").beh_obs_s.sum().replace(0, np.nan)
            share = pd.DataFrame({k: g.groupby("cow_id")[v].sum().sum(1) / den for k, v in cols.items()})
            if cow_id not in share.index:
                continue
            me, herd = share.loc[cow_id], share.median()
            rows.append(f"{blk:%m-%d %H}h ({den.get(cow_id, 0) / 3600:.1f} h seen): " +
                        ", ".join(f"{k} {100 * me[k]:.0f}% (herd {100 * herd[k]:.0f}%)" for k in cols))
        return self._R("\n".join(rows) or f"no labelled minutes for {cow_id}")

    def t_trend(self, cow_id: str, indicator: str = "feeding_share", block_h: int = 6):
        """One indicator per time block for the cow vs the herd band, with a CUSUM onset (when the deviation started)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if indicator not in self.cfg["indicators"] and indicator not in ("breathing_bpm", "motion_energy", "jerk_rate"):
            return self._R(f"unknown indicator; choose from {', '.join(self.cfg['indicators'])}")
        bi = self._ind_by_block(int(block_h))
        if bi.empty or indicator not in bi:
            return self._R("not enough data per block")
        spec = self.cfg["indicators"].get(indicator, {"direction": 1, "floor": 0.05, "label": indicator})
        piv = bi.pivot(index="block", columns="cow_id", values=indicator)
        med = piv.median(1)
        spread = (1.4826 * piv.sub(med, axis=0).abs().median(1)).clip(lower=spec["floor"])
        if cow_id not in piv:
            return self._R(f"{cow_id} has no blocks")
        dev = spec["direction"] * (piv[cow_id] - med) / spread
        onset = cusum_onset(dev)
        fig, ax = plt.subplots(figsize=(9, 3))
        q1, q3 = piv.quantile(0.25, axis=1), piv.quantile(0.75, axis=1)
        ax.fill_between(piv.index, q1, q3, color="#ccc", label="herd IQR")
        ax.plot(piv.index, med, color="#666", label="herd median")
        ax.plot(piv.index, piv[cow_id], "o-", color="#c0392b", label=cow_id)
        if onset is not None:
            ax.axvline(piv.index[onset], color="#c0392b", ls="--", label="CUSUM onset")
        ax.set_title(f"{spec['label']} per {block_h} h"); ax.legend(fontsize=7)
        fig.autofmt_xdate(); fig.tight_layout(); fig.canvas.draw()
        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR)
        plt.close(fig)
        fid = self.store.add_image(img, cow_id, f"{indicator} trend of {cow_id}", {"type": "chart", "indicator": indicator}, "trend")
        lines = [f"{b:%m-%d %H}h: {cow_id} {piv.loc[b, cow_id]:.3g} vs herd {med[b]:.3g} (dev {dev[b]:+.1f})" for b in piv.index]
        msg = f"[{fid}] " + ("no sustained deviation (CUSUM never crossed)" if onset is None else
                             f"sustained deviation starts in block {piv.index[onset]:%m-%d %H}h")
        return self._R(msg + "\n" + "\n".join(lines), [img], {"fact": fid})

    def t_feed_response(self, cow_id: str):
        """For each feed delivery (bunk-trough change) and herd feed rush: minutes until the cow reached the bunk and minutes it stayed in the next 30 min, vs the herd."""
        events = [("herd rush", int(t)) for t in self.m.attrs.get("feed_rushes", [])]
        fe = self.assets.get("feed_events")
        if fe is not None and len(fe):
            events += [(f"trough change {r.camera}", int(r.wall_t)) for r in fe.itertuples()]
        if not events:
            return self._R("no feed deliveries or herd rushes detected in this period")
        lines = []
        for kind, t in sorted(events, key=lambda e: e[1]):
            w = self.m[(self.m.minute >= t - 60) & (self.m.minute <= t + 1800)]
            res = {}
            for cow, g in w.groupby("cow_id"):
                hit = g[g.bunk_s >= 30]
                res[cow] = ((hit.minute.min() - t) / 60 if len(hit) else np.nan, g.bunk_s.sum() / 60)
            if cow_id not in res:
                continue
            lat = pd.Series({c: v[0] for c, v in res.items()})
            stay = pd.Series({c: v[1] for c, v in res.items()})
            me = res[cow_id]
            lines.append(f"{kind} at {hms(t)}: {cow_id} reached bunk after "
                         f"{'never (30 min)' if np.isnan(me[0]) else f'{max(me[0], 0):.0f} min'}, stayed {me[1]:.0f} min; "
                         f"herd median {lat.median():.0f} min / {stay.median():.0f} min")
        return self._R("\n".join(lines) or f"{cow_id} not visible around the feed events")

    def t_bouts(self, cow_id: str, behavior: str = "lying"):
        """Bouts of lying or feeding (start, minutes) and the cow's bout count / mean length vs the herd."""
        col = {"lying": "lying_s", "feeding": "sec_feeding"}.get(behavior)
        if col is None:
            return self._R("behavior must be lying or feeding")
        res = {}
        for cow, g in self.m.sort_values("minute").groupby("cow_id"):
            on = (g[col] >= 30).to_numpy()
            t = g.minute.to_numpy()
            bouts, start = [], None
            for i in range(len(g)):
                cont = on[i] and start is not None and t[i] - t[i - 1] <= 120
                if on[i] and start is None:
                    start = i
                elif start is not None and not cont:
                    bouts.append((t[start], (t[i - 1] - t[start]) / 60 + 1))
                    start = i if on[i] else None
            if start is not None:
                bouts.append((t[start], (t[-1] - t[start]) / 60 + 1))
            res[cow] = bouts
        me = res.get(cow_id, [])
        lens = pd.Series({c: np.mean([b[1] for b in v]) if v else np.nan for c, v in res.items()})
        cnt = pd.Series({c: len(v) for c, v in res.items()})
        head = f"{behavior} bouts: {cow_id} {len(me)} bouts, mean {lens.get(cow_id, np.nan):.0f} min; herd median {cnt.median():.0f} bouts, {lens.median():.0f} min"
        return self._R(head + "\n" + "\n".join(f"{hms(s)} {d:.0f} min" for s, d in me[:40]))

    def t_social(self, cow_id: str):
        """Who the cow spends time nearest to, per day, and whether it drifted away from its usual neighbours."""
        st = self.h.state
        st = st[(st.cow_id == cow_id) & st.nearest_cow.notna()]
        if st.empty:
            return self._R(f"no neighbour data for {cow_id}")
        st = st.assign(day=pd.to_datetime(st.sec, unit="s").dt.strftime("%m-%d"))
        lines = []
        for d, g in st.groupby("day"):
            top = g.nearest_cow.value_counts(normalize=True).head(3)
            lines.append(f"{d}: nearest " + ", ".join(f"{c} {100 * p:.0f}%" for c, p in top.items()) +
                         f"; median distance {g.nearest_m.median():.1f} m; >5 m from all {100 * (g.nearest_m > 5).mean():.0f}%")
        return self._R("\n".join(lines))

    def t_breathing(self, cow_id: str):
        """Breaths per minute measured from still, close windows (flank motion spectrum) vs the herd; shows the clearest window."""
        b = self.assets.get("breathing")
        if b is None or b.empty:
            return self._R("no breathing windows (run scripts/build_risk_assets.py)")
        v = b[b.valid.astype(bool)]
        me = b[b.cow_id == cow_id].sort_values("quality", ascending=False)
        herd = v.groupby("cow_id").bpm.median()
        lines = [f"{cow_id}: {len(me)} windows, {int(me.valid.sum())} valid; median {me[me.valid.astype(bool)].bpm.median():.0f}/min; "
                 f"herd median of cows {herd.median():.0f}/min (n={len(herd)}). Typical resting adult cattle: about 26-50/min."]
        lines += [f"{hms(r.t0)} {r.camera} {r.bpm:.0f}/min quality {r.quality:.1f} {'valid' if r.valid else 'rejected'} box {int(r.box_h)} px"
                  for r in me.head(8).itertuples()]
        imgs, fid = [], None
        if len(me) and os.path.exists(me.iloc[0].image):
            img = cv2.imread(me.iloc[0].image)
            fid = self.store.add_image(img, cow_id, f"breathing window of {cow_id}",
                                       {"time": int(me.iloc[0].t0), "camera": me.iloc[0].camera, "box_px": int(me.iloc[0].box_h),
                                        "daylight": bool(me.iloc[0].daylight), "bpm": float(me.iloc[0].bpm)}, "breathing")
            imgs = [img]
            lines[0] = f"[{fid}] " + lines[0]
        return self._R("\n".join(lines), imgs, {"fact": fid} if fid else {})

    def t_gallery(self, cow_id: str, n: int = 4):
        """The closest daylight crops of this cow saved during the 48 h pass (best material for ears, eyes, nose, discharge)."""
        from .tools import enhance, hcat, label
        g = self.assets.get("gallery")
        if g is None or g.empty or cow_id not in set(g.cow_id):
            return self._R(f"no gallery crops for {cow_id}; use find_moments + get_crop")
        g = g[g.cow_id == cow_id].sort_values("box_h", ascending=False)
        picks, used = [], []
        for r in g.itertuples():
            if all(abs(r.wall_t - u) > 900 for u in used) and os.path.exists(r.path):
                picks.append(r); used.append(r.wall_t)
            if len(picks) >= int(n):
                break
        crops, ids = [], []
        for r in picks:
            img = enhance(cv2.imread(r.path), 768)
            img = label(img, f"{cow_id} {r.camera} {hms(r.wall_t)} {int(r.box_h)}px")
            fid = self.store.add_image(img, cow_id, f"gallery crop of {cow_id}", {"camera": r.camera, "time": int(r.wall_t),
                                                                                  "box_px": int(r.box_h), "daylight": True,
                                                                                  "region": "body", "enhanced": True}, "gallery")
            crops.append(img); ids.append(fid)
        return self._R(f"{len(ids)} gallery crops: " + ", ".join(f"[{i}] {hms(r.wall_t)} {r.camera} {int(r.box_h)} px" for i, r in zip(ids, picks)),
                       [hcat(crops, 420)] if crops else [], {"facts": ids})

    def t_jerks(self, cow_id: str):
        """Sudden movements while not walking (cough / head-shake candidates): rate vs herd and the times to look at."""
        j = self.assets.get("jerks")
        if j is None:
            return self._R("no motion assets (run scripts/build_risk_assets.py)")
        seen = self.m.groupby("cow_id").seen_s.sum() / 3600
        rate = (j.groupby("cow_id").size() if len(j) else pd.Series(dtype=float)).reindex(seen.index).fillna(0) / seen.replace(0, np.nan)
        me = j[j.cow_id == cow_id].sort_values("z", ascending=False) if len(j) else j
        return self._R(f"{cow_id}: {len(me)} events, {rate.get(cow_id, 0):.1f}/h; herd median {rate.median():.1f}/h\n" +
                       "\n".join(f"{hms(r.wall_t)} {r.camera} z {r.z:.0f}" for r in me.head(8).itertuples()) +
                       "\n(check with get_sequence(span_s=6, n=8) around a time: a cough shows neck extension and head drop)")
