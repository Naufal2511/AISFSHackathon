"""Level 1 deliverables in the form the challenge brief asks for (section 5), from the verified cow-minutes.

  activity.csv         one row per behavior bout, for every cow and every minute of the covered days
  activity_minute.csv  the same minutes one per row (wall time, official label, basis, why)
  time_budgets.csv     per cow per day (and overall): minutes, hours, share of the day, share of the visible time
  traces/              three saved agent reasoning traces: one success, one failure, one at night (+ README.md)

Every row says what it rests on (`basis`), because the brief asks to "clearly distinguish what your system observed
from what it estimated or assumed":
  verified     the BRD validation agent looked at the frames and accepted or corrected the label
  observed     the stage 1 classifier saw the cow and the label passed the agent's rule checks
  unverified   seen, but the agent could not confirm the stage 1 label (kept, marked)
  estimated    not looked at directly: a calm bout inherited its verified label, or stage 1 filled a gap
  not_visible  no camera showed the cow (label "other / not visible", confidence 0)

Used by pipeline/orchestrator.py (export step, one run) and stitch_shares.py (all chunks of the 48 h).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

OFFICIAL = {"lying": "lying", "standing": "standing", "walking": "walking", "feeding": "feeding", "drinking": "drinking",
            "ruminating_lying": "ruminating (lying)", "ruminating_standing": "ruminating (standing)",
            "grooming": "grooming", "social_interaction": "social interaction", "other_not_visible": "other / not visible"}
NOT_VISIBLE = "other / not visible"
BASIS_ORDER = ["verified", "observed", "unverified", "estimated", "not_visible"]


def official(label) -> str:
    if not isinstance(label, str) or not label.strip():
        return NOT_VISIBLE
    s = label.strip()
    if s in OFFICIAL.values():
        return s
    k = s.lower().replace("(", "").replace(")", "").replace("/", " ").replace("-", " ")
    k = "_".join(k.split())
    k = {"other_not_visible": "other_not_visible", "other": "other_not_visible", "not_visible": "other_not_visible",
         "rum_lying": "ruminating_lying", "rum_standing": "ruminating_standing", "social": "social_interaction"}.get(k, k)
    return OFFICIAL.get(k, NOT_VISIBLE)


def basis_of(row) -> str:
    reason = str(row.get("reason") or "")
    source = str(row.get("source") or "")
    flags = str(row.get("flags") or "")
    if reason in ("no_upstream_data", "miss_not_found", "miss_not_recovered", "") or source == "not_visible" \
            or official(row.get("behavior")) == NOT_VISIBLE:
        return "not_visible"
    if reason in ("agent_accept", "agent_revised", "recovered_miss"):
        return "verified"
    if reason == "stable_carry":
        return "estimated"
    if "gap_fill" in flags and reason == "keep_tier1":
        return "estimated"
    if reason == "keep_tier1":
        return "observed"
    return "unverified"


def full_grid(minutes: pd.DataFrame, cows: list[str] | None = None) -> pd.DataFrame:
    """minutes: cow_id, minute_start (wall time) + per-minute columns. Returns every cow x every minute from 00:00 of
    the first day to 23:59 of the last day; minutes nobody saw become 'other / not visible', basis not_visible."""
    m = minutes.copy()
    m["minute_start"] = pd.to_datetime(m.minute_start).dt.floor("min")
    m = m.sort_values(["cow_id", "minute_start"]).drop_duplicates(["cow_id", "minute_start"], keep="first")
    cows = sorted(set(cows or []) | set(m.cow_id.dropna().astype(str)))
    day0 = m.minute_start.min().normalize()
    day1 = m.minute_start.max().normalize() + pd.Timedelta(days=1)
    idx = pd.MultiIndex.from_product([cows, pd.date_range(day0, day1, freq="min", inclusive="left")],
                                     names=["cow_id", "minute_start"])
    g = m.set_index(["cow_id", "minute_start"]).reindex(idx).reset_index()
    seen = g.behavior.notna() & g.reason.notna() & (g.reason != "no_upstream_data")
    g.loc[~seen, ["behavior", "reason", "source"]] = [NOT_VISIBLE, "no_upstream_data", "not_visible"]
    g.loc[~seen, "confidence"] = 0.0
    g["behavior"] = g.behavior.map(official)
    g["basis"] = [basis_of(r) for r in g.to_dict("records")]
    g["date"] = g.minute_start.dt.strftime("%Y-%m-%d")
    g["day"] = (g.minute_start.dt.normalize() - day0).dt.days + 1
    g["minute"] = g.minute_start.dt.hour * 60 + g.minute_start.dt.minute
    return g


def bouts(g: pd.DataFrame) -> pd.DataFrame:
    """Consecutive minutes of one cow with the same label on the same day = one bout."""
    g = g.sort_values(["cow_id", "minute_start"]).reset_index(drop=True)
    new = (g.cow_id != g.cow_id.shift()) | (g.behavior != g.behavior.shift()) | (g.day != g.day.shift()) \
        | (g.minute_start.diff() != pd.Timedelta(minutes=1))
    g = g.assign(bout=new.cumsum(), agent=(pd.to_numeric(g.get("tier"), errors="coerce") == 2).astype(int))
    cams = g.get("cameras_used")

    def cam_set(s):
        out = set()
        for v in s.dropna():
            for c in str(v).strip("[]").replace("'", "").replace('"', "").replace(";", ",").split(","):
                if c.strip():
                    out.add(c.strip())
        return ",".join(sorted(out))

    agg = dict(cow_id=("cow_id", "first"), date=("date", "first"), day=("day", "first"), behavior=("behavior", "first"),
               start=("minute_start", "min"), end=("minute_start", "max"), duration_min=("minute_start", "size"),
               mean_confidence=("confidence", lambda s: round(float(pd.to_numeric(s, errors="coerce").fillna(0).mean()), 4)),
               n_agent_minutes=("agent", "sum"))
    b = g.groupby("bout").agg(**agg)
    shares = g.groupby(["bout", "basis"]).size().unstack(fill_value=0)
    for k in BASIS_ORDER:
        if k not in shares:
            shares[k] = 0
    b["basis"] = shares[BASIS_ORDER[::-1]].idxmax(axis=1)     # majority; a tie goes to the less certain basis
    b["observed_share"] = ((shares["verified"] + shares["observed"] + shares["unverified"]) / b.duration_min).round(3)
    b["cameras"] = g.groupby("bout")["cameras_used"].agg(cam_set) if cams is not None else ""
    b["start_time"] = b.start.dt.strftime("%Y-%m-%d %H:%M")
    b["end_time"] = (b.end + pd.Timedelta(minutes=1)).dt.strftime("%Y-%m-%d %H:%M")   # end is exclusive
    b["start_minute"] = b.start.dt.hour * 60 + b.start.dt.minute
    b["end_minute"] = b.end.dt.hour * 60 + b.end.dt.minute
    return b[["cow_id", "date", "day", "start_time", "end_time", "start_minute", "end_minute", "duration_min", "behavior",
              "mean_confidence", "basis", "observed_share", "n_agent_minutes", "cameras"]].reset_index(drop=True)


def time_budgets(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cow, day), d in list(g.groupby(["cow_id", "date"])) + [((c, "all"), d) for c, d in g.groupby("cow_id")]:
        vis = int((d.behavior != NOT_VISIBLE).sum())
        for beh, n in d.behavior.value_counts().items():
            rows.append({"cow_id": cow, "date": day, "behavior": beh, "minutes": int(n), "hours": round(n / 60, 2),
                         "share_of_time": round(n / len(d), 4),
                         "share_of_visible_time": round(n / vis, 4) if vis and beh != NOT_VISIBLE else None,
                         "visible_hours": round(vis / 60, 2)})
    return pd.DataFrame(rows).sort_values(["cow_id", "date", "minutes"], ascending=[True, True, False]).reset_index(drop=True)


def pick_traces(g: pd.DataFrame, trace_path) -> dict:
    """success: the agent corrected a stage 1 label (else the most confident confirmation); failure: the agent could
    not settle the minute (else the least confident check); night: an agent check on IR night footage.
    trace_path(row) -> Path of the trace .json or None."""
    a = g[(pd.to_numeric(g.get("tier"), errors="coerce") == 2)].copy()
    a["path"] = [trace_path(r) for r in a.to_dict("records")]
    a = a[a.path.notna()]
    if a.empty:
        return {}
    a["conf"] = pd.to_numeric(a.confidence, errors="coerce").fillna(0)
    out = {}
    s = a[a.correction == "behavior_corrected"]
    s = s if len(s) else a[a.basis == "verified"]
    if len(s):
        out["success"] = s.sort_values("conf", ascending=False).iloc[0]
    f = a[a.reason.astype(str).str.contains("unresolved|error_fallback|auto_abstain")]
    if len(f):
        out["failure"] = f.sort_values("conf").iloc[0]
    night = a[pd.to_numeric(a.get("is_night_final"), errors="coerce") == 1]
    other = night[~night.path.isin([r["path"] for r in out.values()])]
    night = other if len(other) else night                 # a different minute from the success/failure picks
    if len(night):
        out["night"] = night.sort_values("conf", ascending=False).iloc[0]
    return out


def save_traces(picks: dict, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Saved agent reasoning traces", "",
             "Picked automatically from this run's BRD validation-agent traces "
             "(pipeline/deliverables.py `pick_traces`).", ""]
    why = {"success": "the agent changed or confirmed the stage 1 label from what it saw in the frames",
           "failure": "the agent could not settle the minute and kept the stage 1 label, marked unverified",
           "night": "an agent check on infrared night footage"}
    written = []
    for kind in ("success", "failure", "night"):
        r = picks.get(kind)
        if r is None:
            lines += [f"## {kind}", "", "No trace of this kind in this run (no night footage, or nothing went wrong).", ""]
            continue
        src = Path(r["path"])
        dst = out_dir / f"{kind}_{src.stem}"
        for ext in (".md", ".json"):
            if src.with_suffix(ext).exists():
                shutil.copy2(src.with_suffix(ext), dst.with_suffix(ext))
        img = src.parent / "img"
        if img.is_dir() and src.with_suffix(".md").exists():          # the images the trace shows
            (out_dir / "img").mkdir(exist_ok=True)
            for name in set(__import__("re").findall(r"img/([^)\s\"']+)", src.with_suffix(".md").read_text(encoding="utf-8"))):
                if (img / name).exists():
                    shutil.copy2(img / name, out_dir / "img" / name)
        lines += [f"## {kind}: {r['cow_id']} at {pd.Timestamp(r['minute_start']):%Y-%m-%d %H:%M}", "",
                  f"Why this one: {why[kind]}.", "",
                  f"Result: **{r['behavior']}** ({r['basis']}, confidence {float(r['conf']):.2f}; {r['reason']}).", "",
                  f"Trace: [{dst.name}.md]({dst.name}.md)", ""]
        written.append(dst.name)
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return written


def write_level1(minutes: pd.DataFrame, out_dir: str | Path, trace_path=None, cows: list[str] | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = full_grid(minutes, cows)
    b = bouts(g)
    tb = time_budgets(g)
    cols = ["cow_id", "date", "day", "minute", "minute_start", "behavior", "confidence", "basis", "reason", "correction",
            "tier1_label", "source", "tier", "is_night_final", "cameras_used", "rationale", "flags"]
    gm = g.assign(minute_start=g.minute_start.dt.strftime("%Y-%m-%d %H:%M"))
    gm[[c for c in cols if c in gm]].to_csv(out / "activity_minute.csv", index=False)
    b.to_csv(out / "activity.csv", index=False)
    tb.to_csv(out / "time_budgets.csv", index=False)
    saved = save_traces(pick_traces(g, trace_path), out / "traces") if trace_path else []
    counts = g.basis.value_counts().to_dict()
    return {"cows": int(g.cow_id.nunique()), "minutes": int(len(g)), "bouts": int(len(b)),
            "days": sorted(g.date.unique().tolist()), "basis": {k: int(v) for k, v in counts.items()}, "traces": saved}


def trace_finder(trace_dir: Path):
    """For one run: trace of a minute is <trace_dir>/<cow>_d<day>_m<minute>.json (brd_agent naming, run clock)."""
    def find(r):
        d, m = r.get("run_day", r.get("day")), r.get("run_minute", r.get("minute"))
        cow = r.get("local_cow", r.get("cow_id"))
        if pd.isna(d) or pd.isna(m):
            return None
        p = Path(r.get("trace_dir") or trace_dir) / f"{cow}_d{int(d)}_m{int(m):04d}.json"
        return str(p) if p.exists() else None
    return find
