"""Level 3 live early warning: one processed video window -> updated risk per cow -> alerts.

A window is one 5-min clip per camera, run through the normal pipeline (stage 1 tracking, identity, behavior,
assets). This module then
  1. maps the window's cows to the 48 h ids (appearance prototypes vs the reference chunk, then the stitched id map),
  2. measures each cow in the window (feeding, bunk, lying, alone, walking) against the herd IN THE SAME WINDOW,
  3. looks for visible clinical signs on close crops (VLM "pose sensor" + discharge image check), and confirms each
     flagged crop with a second, stricter look next to a herdmate crop (the agent checks its own answer),
  4. updates each cow's risk (48 h Level 2 score as the prior, EWMA on window evidence, signs add at once),
  5. decides who gets an alert (confirmed sign, or a risk that crossed the line and is rising), with a cooldown.
Everything it decides is written down with the reason (state.json, risk_history.csv, windows.jsonl).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

NOT_VISIBLE = "other / not visible"


# ----------------------------------------------------------------------------------------------- inputs
def resolve_path(p: str, run: Path) -> Path:
    """Gallery paths are relative to the repo the chunk ran in: try as-is, then against each ancestor of the run."""
    q = Path(p)
    if q.is_absolute() and q.exists():
        return q
    if q.exists():
        return q.resolve()
    for anc in [run, *run.parents]:
        if (anc / q).exists():
            return anc / q
    return q


@dataclass
class Window:
    run: Path                         # pipeline run folder
    start: pd.Timestamp
    end: pd.Timestamp
    minutes: pd.DataFrame             # cow_id, minute_start, behavior (local ids)
    state: pd.DataFrame               # per second: cow_id, wall_time, zone, nearest_m, speed_mps, best_cam_identity
    obs: pd.DataFrame                 # boxes: wall_time, file, offset_s, camera, cow_id, x1..y2
    gallery: pd.DataFrame             # close crops: cow_id, wall_t, camera, path, box_h
    prototypes: Dict[str, np.ndarray] = field(default_factory=dict)


def load_window(run: Path) -> Window:
    run = Path(run)
    l1 = run / "stage1" / "level1"
    fin = run / "final" / "activity_minute.csv"          # stage 2 verified minutes when --verify ran
    if fin.exists():
        m = pd.read_csv(fin, low_memory=False)
        m["minute_start"] = pd.to_datetime(m.minute_start)
    else:
        m = pd.read_csv(l1 / "behavior" / "activity.csv", low_memory=False)
        m["minute_start"] = pd.to_datetime(m.minute_start)
    st = pd.read_csv(l1 / "contract" / "cow_state.csv", low_memory=False)
    st["wall_time"] = pd.to_datetime(st.wall_time)
    obs = pd.read_csv(l1 / "contract" / "observations.csv", low_memory=False,
                      usecols=lambda c: c in {"wall_time", "file", "offset_s", "camera", "cow_id", "x1", "y1", "x2", "y2",
                                              "box_role", "frame_w", "frame_h"})
    obs = obs[obs.cow_id.notna()].copy()
    obs["wall_time"] = pd.to_datetime(obs.wall_time)
    gp = l1 / "assets" / "gallery.csv"
    gal = pd.read_csv(gp) if gp.exists() and gp.stat().st_size > 10 else pd.DataFrame(columns=["cow_id", "wall_t", "camera", "path", "box_h"])
    if len(gal):
        gal["path"] = [str(resolve_path(p, run)) for p in gal.path]
    pp = l1 / "agent" / "memory" / "prototypes.npz"
    protos = {}
    if pp.exists():
        z = np.load(pp)
        protos = {k: z[k] / (np.linalg.norm(z[k]) + 1e-9) for k in z.files}
    t = st.wall_time if len(st) else m.minute_start
    return Window(run, t.min().floor("min"), t.max().ceil("min"), m, st, obs, gal, protos)


# ----------------------------------------------------------------------------------------------- identity
def _gallery_used(run: Path) -> Optional[str]:
    try:
        m = json.loads((run / "pipeline_manifest.json").read_text(encoding="utf-8"))
        return (m.get("settings") or {}).get("id_gallery")
    except (OSError, ValueError):
        return None


def map_ids(win: Window, cfg: Dict, repo: Path) -> pd.DataFrame:
    """local id -> 48 h id, by the most solid route available (each row says which one):
      1. the run IS one of the 48 h chunks: its row in the stitched id map;
      2. the window was tracked from the reference chunk's appearance gallery (live mode): tracking seeds its ids from
         that gallery, so the reference chunk's id map applies;
      3. otherwise appearance only (Hungarian on cosine to the reference prototypes). Black Angus look alike to DINOv2
         (on Day 2 this route got 1 of 8 cows right), so these are flagged as guesses."""
    from scipy.optimize import linear_sum_assignment
    icfg = cfg["identity"]
    locals_ = sorted(set(win.minutes.cow_id.astype(str)) | set(win.state.cow_id.astype(str)))
    idm = pd.read_csv(repo / icfg["id_map"]) if (repo / icfg["id_map"]).exists() else pd.DataFrame()
    if len(idm) and win.run.name in set(idm.chunk):
        r = idm[idm.chunk == win.run.name]
        own = dict(zip(r.local_id.astype(str), r.global_id.astype(str)))
        return pd.DataFrame([{"local_id": l, "ref_id": l, "cow_id": own.get(l, "LIVE_" + l), "similarity": None,
                              "margin": None, "how": "48 h stitched id map" if l in own else "not in the 48 h id map"}
                             for l in locals_])
    chunk_map = {}
    if len(idm):
        r = idm[idm.chunk == icfg["map_chunk"]]
        chunk_map = dict(zip(r.local_id.astype(str), r.global_id.astype(str)))
    used = _gallery_used(win.run)
    if used and Path(used).resolve() == (repo / icfg["gallery"]).resolve() and chunk_map:
        return pd.DataFrame([{"local_id": l, "ref_id": l, "cow_id": chunk_map.get(l, "LIVE_" + l), "similarity": None,
                              "margin": None, "how": f"tracked from the {icfg['map_chunk']} gallery, then the 48 h id map"
                              if l in chunk_map else "new track (not in the gallery)"} for l in locals_])
    ref = {}
    rp = repo / icfg["reference_prototypes"]
    if rp.exists():
        z = np.load(rp)
        ref = {k: z[k] / (np.linalg.norm(z[k]) + 1e-9) for k in z.files}
    rows = []
    usable = [l for l in locals_ if l in win.prototypes]
    refs = [k for k in ref if any(ref[k].shape == win.prototypes[l].shape for l in usable)]
    assign = {}
    if usable and refs:
        S = np.array([[float(win.prototypes[l] @ ref[k]) if ref[k].shape == win.prototypes[l].shape else -1 for k in refs]
                      for l in usable])
        ri, ci = linear_sum_assignment(-S)
        for i, j in zip(ri, ci):
            second = np.sort(S[i])[-2] if S.shape[1] > 1 else -1.0
            assign[usable[i]] = (refs[j], float(S[i, j]), float(S[i, j] - second))
    for l in locals_:
        if l in assign and assign[l][1] >= icfg.get("min_similarity", 0.8):
            ref_id, sim, mar = assign[l]
            g = chunk_map.get(ref_id, ref_id)
            how = "appearance only (guess)" + ("" if mar >= 0.05 else ", low margin")
        else:
            ref_id, sim, mar = l, None, None
            g = "LIVE_" + l
            how = "no match: window id"
        rows.append({"local_id": l, "ref_id": ref_id, "cow_id": g, "similarity": sim, "margin": mar, "how": how})
    df = pd.DataFrame(rows)
    # two locals on one 48 h id: keep the stronger, give the other its own id (flagged)
    if len(df):
        df = df.sort_values("similarity", ascending=False, na_position="last")
        dup = df.duplicated("cow_id")
        df.loc[dup, "how"] = df.loc[dup, "how"] + "; duplicate -> window id"
        df.loc[dup, "cow_id"] = "LIVE_" + df.loc[dup, "local_id"]
    return df.sort_values("cow_id").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- measurement
def window_features(win: Window, ids: pd.DataFrame) -> pd.DataFrame:
    """Per cow in this window: minutes seen and the shares the risk uses."""
    lm = dict(zip(ids.local_id, ids.cow_id))
    m = win.minutes.assign(cow_id=win.minutes.cow_id.astype(str).map(lm).fillna(win.minutes.cow_id.astype(str)))
    st = win.state.assign(cow_id=win.state.cow_id.astype(str).map(lm).fillna(win.state.cow_id.astype(str)))
    out = []
    for cow in sorted(set(m.cow_id) | set(st.cow_id)):
        mc = m[(m.cow_id == cow) & (m.behavior != NOT_VISIBLE)]
        sc = st[st.cow_id == cow]
        seen = len(mc)
        b = mc.behavior.astype(str)
        row = {"cow_id": cow, "seen_min": float(seen),
               "feeding_share": float((b == "feeding").mean()) if seen else np.nan,
               "lying_share": float(b.str.contains("lying").mean()) if seen else np.nan,
               "ruminating_share": float(b.str.startswith("ruminating").mean()) if seen else np.nan,
               "standing_share": float(b.isin(["standing", "ruminating (standing)"]).mean()) if seen else np.nan}
        if len(sc):
            n = len(sc)
            row["bunk_zone_share"] = float((sc.zone.astype(str) == "bunk").mean())
            near = pd.to_numeric(sc.nearest_m, errors="coerce")
            row["isolation_share"] = float((near > 5).mean()) if near.notna().any() else np.nan
            row["nn_distance_m"] = float(near.median()) if near.notna().any() else np.nan
            spd = pd.to_numeric(sc.speed_mps, errors="coerce").fillna(0)
            dt = sc.wall_time.sort_values().diff().dt.total_seconds().clip(upper=5).fillna(1).values
            row["distance_m_per_h"] = float((spd.values * dt).sum() / max(dt.sum(), 1) * 3600)
            row["state_s"] = float(n)
            row["best_cam"] = sc.best_cam_identity.mode().iloc[0] if "best_cam_identity" in sc and sc.best_cam_identity.notna().any() else None
        out.append(row)
    return pd.DataFrame(out)


def robust_z(v: pd.Series, floor: float) -> pd.Series:
    med = v.median()
    mad = (v - med).abs().median() * 1.4826
    return (v - med) / max(mad, floor)


def window_evidence(feat: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Signed, weighted robust z of each indicator against the herd in the same window (+ = more BRD-like)."""
    rc = cfg["risk"]
    f = feat.copy()
    ok = f.seen_min >= rc["min_seen_min"]
    f["evidence"] = 0.0
    f["reasons"] = [[] for _ in range(len(f))]
    for ind, spec in rc["indicators"].items():
        if ind not in f or f.loc[ok, ind].notna().sum() < 3:
            continue
        floor = rc["z_floor"] * (100 if ind == "distance_m_per_h" else 1)
        z = pd.Series(np.nan, index=f.index)
        z[ok] = robust_z(f.loc[ok, ind], floor).clip(-rc["z_cap"], rc["z_cap"])
        f[ind + "__z"] = z
        herd = f.loc[ok, ind].median()
        for i in f.index[ok & z.notna()]:
            s = spec["sign"] * z[i]
            f.at[i, "evidence"] += spec["weight"] * s
            if s >= 1.0:
                val, unit = f.at[i, ind], ("m/h" if ind == "distance_m_per_h" else "%")
                fmt = (lambda x: f"{x:.0f} m/h") if unit == "m/h" else (lambda x: f"{100 * x:.0f}%")
                f.at[i, "reasons"].append({"indicator": ind, "z": round(float(s), 2),
                                           "text": f"{spec['label']} {fmt(val)} vs herd {fmt(herd)} in the same window (z {s:+.1f})"})
    for i in f.index[~ok]:                     # .at per row: .loc would unpack a one-row list of lists
        f.at[i, "reasons"] = [{"indicator": "visibility", "z": 0.0,
                               "text": f"seen only {f.at[i, 'seen_min']:.0f} min in this window: no behavior evidence"}]
    for i in f.index:
        f.at[i, "reasons"] = sorted(f.at[i, "reasons"], key=lambda r: -r["z"])
    return f


# ----------------------------------------------------------------------------------------------- clinical signs
CONFIRM_PROMPT = """You are a feedlot veterinarian checking one camera alert. Image A is the cow under suspicion,
image B is a herdmate seen as close in time as possible (a control). The screening model said image A shows: {sign}.
Look carefully. Black Angus muzzles are dark and wet-shiny; feed crumbs, dust, glare, flies or infrared
reflections are NOT discharge. Ears held back while eating, relaxed ears on a cow that is lying, resting or chewing
cud, and an ear pulled down by the weight of an ear tag are NOT drooping: droop means both ears hanging on a cow
standing up. head_low means a cow STANDING
STILL with her head hanging below the topline, looking dull: a head low while walking, sniffing the ground,
eating, drinking or lying is normal and NOT head_low. Confirm only what image A clearly shows.
Reply JSON only: {{"confirmed": true|false, "confidence": 0-1, "why": "<= 25 words", "visible": true|false}}"""

SIGN_OF = {"ears": ("droop", "ears_droop"), "discharge_flag": (True, "nasal_discharge")}


def screen_signs(win: Window, ids: pd.DataFrame, cfg: Dict, repo: Path, log) -> pd.DataFrame:
    """VLM pose sensor + discharge image check on the window's close crops, then a confirming second look."""
    scfg = cfg["signs"]
    if win.gallery.empty:
        log("signs: no close daylight crops in this window (night, or no cow close enough): not checked")
        return pd.DataFrame()
    try:
        import yaml
        from src.assets import head_signs
        from src.risk.agent import Backends
        acfg = yaml.safe_load(open(repo / "configs" / "agent.yaml", encoding="utf-8"))
        be = Backends(acfg, scfg["backends"])
    except Exception as e:                                  # no key / no network: say so, never pretend
        log(f"signs: VLM not available ({e}); clinical signs NOT checked in this window")
        return pd.DataFrame()
    lm = dict(zip(ids.local_id, ids.cow_id))
    gal = win.gallery.copy()
    gal = gal[gal.box_h >= scfg["min_box_px"]]
    gal = gal.sort_values("box_h", ascending=False).groupby("cow_id").head(scfg["per_window"])
    if gal.empty:
        log("signs: no crop large enough this window")
        return pd.DataFrame()
    hs = head_signs.run(gal, be, per_hour=scfg["per_window"], min_px=scfg["min_box_px"], model_label="live")
    if hs.empty:
        return hs
    hs["local_id"] = hs.cow_id.astype(str)
    hs["cow_id"] = hs.local_id.map(lm).fillna(hs.local_id)
    feeding = hs.feeding.astype(str).str.lower().eq("true")
    standing = hs.standing.astype(str).str.lower().eq("true")
    hs["ears_droop"] = hs.ears.eq("droop") & standing      # pen-riding signs are judged on a standing cow
    for i in hs.index[hs.ears.eq("droop") & ~standing]:
        hs.at[i, "measurement"] = ("ears looked low, but she is not standing (lying / resting): ear carriage is only "
                                   "judged on a standing cow, so not flagged")
    hs["head_low"] = hs.head_position.eq("below_withers") & standing & ~feeding
    hs["nasal_discharge"] = hs.discharge_flag.astype(bool)
    hs["confirmed"] = ""
    # measurement check (no VLM): head_low only counts for a cow standing still. The tracker's speed around the
    # crop's second decides; a head carried low while walking is normal.
    hs["speed_mps"] = np.nan
    if len(win.state):
        st = win.state.assign(t=(win.state.wall_time - pd.Timestamp("1970-01-01")).dt.total_seconds())
        for i, r in hs.iterrows():
            s = st[(st.cow_id.astype(str) == r.local_id) & ((st.t - float(r.wall_t)).abs() <= 3)]
            sp = pd.to_numeric(s.speed_mps, errors="coerce").max() if len(s) else np.nan
            hs.at[i, "speed_mps"] = sp
            if bool(r.head_low) and pd.notna(sp) and sp > scfg.get("still_speed_mps", 0.15):
                hs.at[i, "head_low"] = False
                hs.at[i, "measurement"] = (f"head low flagged, then dropped by measurement: the tracker had her moving "
                                           f"{sp:.2f} m/s (head low while walking is normal)")
                log(f"signs: {r.cow_id} head_low at {pd.to_datetime(r.wall_t, unit='s'):%H:%M:%S} dropped by measurement: "
                    f"moving {sp:.2f} m/s (head low while walking is normal)")
    # what the screening saw, in words, for every crop (the report shows each one)
    hs["screen"] = [f"ears {r.ears}, head {str(r.head_position).replace('_', ' ')}, discharge {r.discharge}"
                    f"{' (pixel check suspects a streak)' if bool(r.discharge_flag) and r.discharge != 'yes' else ''}, "
                    f"standing {r.standing}, feeding {r.feeding}" for r in hs.itertuples()]
    hs["control"] = None
    if "measurement" not in hs:
        hs["measurement"] = None
    if scfg.get("confirm", True):
        from src.agent.evidence import to_jpeg_b64
        import cv2
        for i, r in hs.iterrows():
            flagged = [s for s in ("nasal_discharge", "ears_droop", "head_low") if bool(r[s])]
            if not flagged:
                continue
            ctrl = win.gallery[win.gallery.cow_id.astype(str).map(lm).fillna(win.gallery.cow_id.astype(str)) != r.cow_id]
            ctrl = ctrl.iloc[(ctrl.wall_t - r.wall_t).abs().argsort()] if len(ctrl) else ctrl   # closest in time
            ctrl_path = ctrl.path.iloc[0] if len(ctrl) else None
            imgs = [cv2.imread(r.path)] + ([cv2.imread(ctrl_path)] if ctrl_path else [])
            content = [{"type": "text", "text": CONFIRM_PROMPT.format(sign=", ".join(flagged))}]
            for tag, im in zip("AB", imgs):
                if im is None:
                    continue
                content += [{"type": "text", "text": f"Image {tag}:"},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + to_jpeg_b64(im, be.image_max_side)}}]
            try:
                act, _, model = be.ask([{"role": "user", "content": content}])
            except Exception as e:
                act, model = {"confirmed": False, "why": f"check failed: {e}"}, ""
            act = act or {}
            ok = bool(act.get("confirmed")) and float(act.get("confidence", 0) or 0) >= 0.5
            hs.at[i, "confirmed"] = json.dumps({"signs": flagged, "confirmed": ok, "confidence": act.get("confidence"),
                                                "why": act.get("why", ""), "control": ctrl_path, "model": model})
            hs.at[i, "control"] = ctrl_path
            log(f"signs: {r.cow_id} {', '.join(flagged)} at {pd.to_datetime(r.wall_t, unit='s'):%H:%M:%S} -> "
                f"{'CONFIRMED' if ok else 'not confirmed'} ({act.get('why', '')})")
    return hs


def confirmed_signs(hs: pd.DataFrame) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    if hs is None or hs.empty:
        return out
    for _, r in hs.iterrows():
        if not r.get("confirmed"):
            continue
        c = json.loads(r.confirmed)
        if c.get("confirmed"):
            for s in c["signs"]:
                out.setdefault(r.cow_id, []).append({"sign": s, "time": float(r.wall_t), "camera": r.camera, "path": r.path,
                                                     "why": c.get("why", ""), "confidence": c.get("confidence"),
                                                     "control": c.get("control")})
    return out


# ----------------------------------------------------------------------------------------------- risk state
def load_prior(repo: Path, cfg: Dict) -> pd.DataFrame:
    p = repo / cfg["baseline"] / "brd_risk.csv"
    if not p.exists():
        return pd.DataFrame(columns=["cow_id", "risk_score", "rank", "top_reasons"])
    return pd.read_csv(p)


class RiskState:
    """state.json: per cow the live risk, last window, last alert, and the 48 h prior it started from."""

    def __init__(self, path: Path, prior: pd.DataFrame):
        self.path = path
        self.d = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"cows": {}, "windows": [], "alerts": []}
        for r in prior.itertuples():
            c = self.d["cows"].setdefault(str(r.cow_id), {"prior": float(r.risk_score), "risk": float(r.risk_score),
                                                          "prior_rank": int(r.rank), "history": [], "last_alert": None,
                                                          "alerted_signs": [],
                                                          "prior_reasons": str(r.top_reasons).split(" | ")[:3]})
            c.setdefault("prior", float(r.risk_score))

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, self.path)

    def ranks(self) -> Dict[str, int]:
        order = sorted(self.d["cows"], key=lambda c: -self.d["cows"][c]["risk"])
        return {c: i + 1 for i, c in enumerate(order)}


def update_risk(state: RiskState, ev: pd.DataFrame, signs: Dict[str, List[Dict]], win: Window, cfg: Dict) -> pd.DataFrame:
    """EWMA of (prior + gain * evidence) for the behavior part; confirmed signs add their points at once."""
    rc = cfg["risk"]
    rows = []
    before = state.ranks()
    for _, r in ev.iterrows():
        cow = r.cow_id
        c = state.d["cows"].setdefault(cow, {"prior": 0.1, "risk": 0.1, "prior_rank": None, "history": [], "last_alert": None,
                                             "alerted_signs": [], "prior_reasons": []})
        old = c["risk"]
        seen_ok = r.seen_min >= rc["min_seen_min"]
        target = float(np.clip(c["prior"] + rc["gain"] * r.evidence, 0, 1)) if seen_ok else old
        behav = (1 - rc["smoothing"]) * old + rc["smoothing"] * target if seen_ok else old
        bump = sum(rc["sign_points"].get(s["sign"], 0) for s in {s["sign"]: s for s in signs.get(cow, [])}.values())
        new = float(np.clip(behav + bump, 0, 1))
        c["risk"] = round(new, 4)
        c["history"].append({"window": str(win.start), "risk": c["risk"], "evidence": round(float(r.evidence), 3),
                             "seen_min": r.seen_min, "signs": sorted({s["sign"] for s in signs.get(cow, [])})})
        rows.append({"window_start": win.start, "cow_id": cow, "risk_before": round(old, 4), "risk": c["risk"],
                     "delta": round(new - old, 4), "prior": c["prior"], "evidence": round(float(r.evidence), 3),
                     "seen_min": r.seen_min, "signs": ";".join(sorted({s["sign"] for s in signs.get(cow, [])})),
                     "reasons": " | ".join(x["text"] for x in r.reasons[:3])})
    after = state.ranks()
    for row in rows:
        row["rank_before"], row["rank"] = before.get(row["cow_id"]), after.get(row["cow_id"])
    return pd.DataFrame(rows)


def decide_alerts(state: RiskState, upd: pd.DataFrame, ev: pd.DataFrame, signs: Dict[str, List[Dict]], win: Window,
                  cfg: Dict, decisions: Optional[List[Dict]] = None) -> List[Dict]:
    """Who gets an alert now, and why. A new confirmed sign always alerts; behavior alerts need a high and rising risk
    (or a strong window while already top 3) and respect the cooldown. `decisions` (if given) gets one row per cow
    with the decision and its reason in words, alert or not."""
    ac = cfg["alert"]
    out = []
    evi = ev.set_index("cow_id")

    def note(cow, decision, why):
        if decisions is not None:
            decisions.append({"cow_id": cow, "decision": decision, "why": why})

    for _, u in upd.iterrows():
        cow = u.cow_id
        c = state.d["cows"][cow]
        sig_all = sorted({s["sign"] for s in signs.get(cow, [])})
        new_signs = sorted(set(sig_all) - set(c.get("alerted_signs", [])))
        strong = [x for x in evi.at[cow, "reasons"] if x["z"] >= ac["evidence_z"]] if cow in evi.index else []
        old_tier, new_tier, tier_rule = c.get("tier", "normal"), c.get("tier", "normal"), c.get("tier_rule", "")
        if "tiers" in cfg:
            new_tier, tier_rule = live_tier({**c, "alerted_signs": sorted(set(c.get("alerted_signs", [])) | set(sig_all))},
                                            int(u["rank"] or 99), cfg)
        c["tier"], c["tier_rule"] = new_tier, tier_rule
        tier_up = TIER_ORDER[new_tier] > TIER_ORDER[old_tier] and new_tier in cfg.get("tiers", {}).get("push", [])
        tier_txt = (f"tier {TIER_NAME[old_tier]} -> {TIER_NAME[new_tier]} ({tier_rule}); " if new_tier != old_tier
                    else f"tier stays {TIER_NAME[new_tier]}; ")
        why = None
        if new_signs:
            why = "sign"
        elif u.risk >= ac["score"] and u.delta >= ac["rise"]:
            why = "risk"
        elif len(strong) >= 2 and (u["rank"] or 99) <= 3:
            why = "behavior"
        elif tier_up:
            why = "tier"
        if not why:
            parts = []
            if sig_all:
                parts.append(f"sign(s) {', '.join(sig_all)} already alerted earlier")
            else:
                parts.append("no confirmed visible sign")
            parts.append(f"risk {u.risk:.2f} " + (f"below the alert line {ac['score']:.2f}" if u.risk < ac["score"]
                                                  else f"not rising enough (+{u.delta:.2f} < +{ac['rise']:.2f})"))
            parts.append(f"{len(strong)} indicator(s) at z >= {ac['evidence_z']:.0f}" +
                         ("" if len(strong) >= 2 else " (needs 2)") + (f", rank {u['rank']}" if len(strong) >= 2 else ""))
            note(cow, "no alert", tier_txt + "; ".join(parts))
            continue
        last = pd.Timestamp(c["last_alert"]) if c.get("last_alert") else None
        if why != "sign" and last is not None and (win.end - last).total_seconds() < ac["cooldown_min"] * 60:
            note(cow, "held (cooldown)", tier_txt + f"would alert ({why}), but she was alerted at {last:%H:%M}; one alert per "
                                         f"{ac['cooldown_min']} min unless a new sign appears")
            continue
        note(cow, "ALERT", tier_txt + {"sign": f"new confirmed sign: {', '.join(new_signs)}",
                                       "risk": f"risk {u.risk:.2f} >= {ac['score']:.2f} and up {u.delta:+.2f} this window",
                                       "behavior": f"{len(strong)} indicators at z >= {ac['evidence_z']:.0f} while rank {u['rank']} "
                                                   f"(top 3): " + "; ".join(x['text'] for x in strong[:3]),
                                       "tier": f"moved up to {TIER_NAME[new_tier]}"}[why])
        out.append({"cow_id": cow, "trigger": why, "new_signs": new_signs, "risk": u.risk, "risk_before": u.risk_before,
                    "tier": new_tier, "tier_before": old_tier, "tier_rule": tier_rule,
                    "prior": u.prior, "rank": u["rank"], "rank_before": u.rank_before,
                    "reasons": evi.at[cow, "reasons"] if cow in evi.index else [], "signs": signs.get(cow, []),
                    "window_start": str(win.start), "window_end": str(win.end)})
        c["last_alert"] = str(win.end)
        c["alerted_signs"] = sorted(set(c.get("alerted_signs", [])) | set(new_signs))
    return out


# ----------------------------------------------------------------------------------------------- alert tiers
TIER_ORDER = {"normal": 0, "yellow": 1, "orange": 2, "red": 3}
TIER_NAME = {"red": "Pull and examine", "orange": "Check on the next pen ride", "yellow": "Watch list", "normal": "Normal"}


def baseline_tier(r, cfg: Dict) -> tuple:
    """Tier from one brd_risk.csv row (the 48 h Level 2 result), with the rule that put her there."""
    t = cfg["tiers"]
    sp = str(r.get("signs_present") or "").strip()
    if sp and sp.lower() != "nan":
        return "red", f"Level 2 confirmed a visible sign: {sp}"
    rank, p3 = int(r["rank"]), float(r.get("p_top3") or 0)
    conf = str(r.get("confidence") or "").lower()
    o = t["orange"]
    if rank <= o["rank_max"] and p3 >= o["p_top3_min"] and conf in o["confidence"]:
        return "orange", (f"rank {rank} of 10, in the top 3 in {p3:.0%} of time-block resamples, {conf} confidence "
                          f"(rule: top {o['rank_max']}, >= {o['p_top3_min']:.0%}, confidence {'/'.join(o['confidence'])})")
    y = t["yellow"]
    if rank <= y["rank_max"] or p3 >= y["p_top3_min"]:
        return "yellow", (f"rank {rank}, top 3 in {p3:.0%} of resamples, {conf} confidence "
                          f"(rule: rank <= {y['rank_max']} or >= {y['p_top3_min']:.0%})")
    return "normal", f"rank {rank}, top 3 in {p3:.0%} of resamples: herd-typical"


def live_tier(c: Dict, rank: int, cfg: Dict) -> tuple:
    """Tier after a live window: a confirmed sign -> red; else by live risk and rank."""
    t = cfg["tiers"]
    if c.get("alerted_signs"):
        return "red", f"confirmed visible sign: {', '.join(c['alerted_signs'])}"
    if rank <= t["orange"]["rank_max"] and c["risk"] >= t["orange"]["live_risk_min"]:
        return "orange", f"live risk {c['risk']:.2f} >= {t['orange']['live_risk_min']:.2f} and rank {rank} (top {t['orange']['rank_max']})"
    if rank <= t["yellow"]["rank_max"] or c["risk"] >= t["yellow"]["live_risk_min"]:
        return "yellow", f"live risk {c['risk']:.2f}, rank {rank}"
    return "normal", f"live risk {c['risk']:.2f}, rank {rank}: herd-typical"
