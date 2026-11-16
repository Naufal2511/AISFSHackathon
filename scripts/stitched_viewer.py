#!/usr/bin/env python3
"""One self-contained HTML page for a stitched result (stitch_shares.py output): does the end-to-end join hold,
and what do the cows do. Opens from disk; nothing is uploaded or fetched (event data stays on this machine).

  python scripts/stitched_viewer.py                                   # outputs/stitched -> outputs/stitched/viewer.html
  python scripts/stitched_viewer.py --stitched outputs/stitched_test  # the c01 + c02 test stitch

Sections: end-to-end checks, cow-ID joins (id_map.csv), every cow-minute of the stitched window (colour by
behavior or by evidence), each chunk join zoomed in, time budgets, table views.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
# Fixed slot per behavior (colour follows the behavior in every stitch, test or final). Grooming has no slot of its
# own (8 hues max): it is drawn as "other behavior".
BEHAVIORS = ["ruminating (lying)", "ruminating (standing)", "feeding", "standing", "social interaction", "drinking",
             "walking", "lying"]
NOT_VISIBLE = "other / not visible"
BASIS = ["verified", "observed", "unverified", "estimated"]          # strongest evidence first; then not_visible
REASONS = {
    "keep_tier1": "model label kept (confident, nothing to escalate)",
    "auto_abstain": "not visible / no reliable box",
    "agent_accept": "BRD agent checked the frames and agreed",
    "agent_revised": "BRD agent checked the frames and corrected it",
    "unresolved_fallback_tier1": "agent could not settle it: model label kept, unverified",
    "miss_not_recovered": "cow missing in this minute, not recovered",
    "stable_carry": "carried over from the neighbouring stable minutes",
    "no_upstream_data": "outside the tracked window",
}


def _herd_size() -> int:
    try:
        import yaml
        cfg = yaml.safe_load((REPO / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
        return int(((cfg.get("stage1") or {}).get("herd_size")) or 10)
    except Exception:  # noqa: BLE001
        return 10


def build(st: Path) -> dict:
    man = json.loads((st / "stitch_manifest.json").read_text(encoding="utf-8"))
    day1 = pd.Timestamp(man["day1"])
    chunks = sorted(({"name": c["name"], "who": c["who"], "start": pd.Timestamp(c["start"]), "end": pd.Timestamp(c["end"])}
                     for c in man["chunks"]), key=lambda c: c["start"])
    to_min = lambda t: int((t - day1).total_seconds() // 60)  # noqa: E731
    w0, w1 = to_min(chunks[0]["start"]), to_min(max(c["end"] for c in chunks))
    joins = [{"minute": to_min(b["start"]), "left": a["name"], "right": b["name"],
              "gap_min": to_min(b["start"]) - to_min(a["end"])} for a, b in zip(chunks, chunks[1:])]

    m = pd.read_csv(st / "final" / "activity_minute.csv", low_memory=False)
    m["t"] = (m.day.astype(int) - 1) * 1440 + m.minute.astype(int)
    cows = sorted(m.cow_id.unique().tolist())
    days = int(m.day.max())
    w = m[(m.t >= w0) & (m.t < w1)].copy()

    beh_codes = BEHAVIORS + ["other behavior", NOT_VISIBLE]
    w["b"] = w.behavior.map(lambda b: BEHAVIORS.index(b) if b in BEHAVIORS
                            else (len(beh_codes) - 1 if b == NOT_VISIBLE or not isinstance(b, str) else len(BEHAVIORS)))
    basis_codes = BASIS + ["not_visible"]
    w["e"] = w.basis.map(lambda x: BASIS.index(x) if x in BASIS else len(BASIS))
    reason_codes = sorted(w.reason.fillna("").unique().tolist())
    w["r"] = w.reason.fillna("").map(reason_codes.index)
    w["c"] = (w.confidence.fillna(0) * 100).round().astype(int)
    owner = {}
    for c in chunks:
        owner[c["name"]] = (to_min(c["start"]), to_min(c["end"]))
    n = w1 - w0
    grid = {}
    for cow, g in w.groupby("cow_id"):
        arr = {"b": [len(beh_codes) - 1] * n, "e": [len(BASIS)] * n, "r": [0] * n, "c": [0] * n}
        for t, b, e, r, c in zip(g.t, g.b, g.e, g.r, g.c):
            i = t - w0
            arr["b"][i], arr["e"][i], arr["r"][i], arr["c"][i] = int(b), int(e), int(r), int(c)
        grid[cow] = arr

    # ---------------------------------------------------------------- end-to-end checks
    checks = []

    def chk(level, title, detail):
        checks.append({"level": level, "title": title, "detail": detail})

    gaps = [j for j in joins if j["gap_min"] != 0]
    chk("good" if not gaps else "critical", "Chunk windows are back to back",
        " → ".join(f"{c['name']} {c['start']:%H:%M}-{c['end']:%H:%M}" for c in chunks)
        + ("" if not gaps else "; gaps/overlaps: " + ", ".join(f"{j['left']}→{j['right']} {j['gap_min']:+d} min" for j in gaps)))
    pen = _herd_size()
    chk("good" if len(cows) <= pen else "warning", f"{len(cows)} cow ids across all chunks (pen size {pen})",
        "no extra ids from the joins" if len(cows) <= pen else "more ids than cows: check the 'new' rows of id_map.csv")
    per_cow = m.groupby("cow_id").size()
    dup = int(m.duplicated(["cow_id", "t"]).sum())
    ok = dup == 0 and (per_cow == days * 1440).all()
    chk("good" if ok else "critical", "One row per cow per minute (activity_minute.csv)",
        f"{len(m):,} rows = {len(cows)} cows × {days} day(s) × 1,440 min; {dup} duplicates")
    hole = w[w.reason == "no_upstream_data"]
    chk("good" if hole.empty else "warning", "Every minute of the tracked window came from a chunk",
        f"{n} min × {len(cows)} cows = {n * len(cows):,} cow-minutes" + ("" if hole.empty else f"; {len(hole)} have no data"))
    act = pd.read_csv(st / "final" / "activity.csv", low_memory=False)
    act["s"] = (act.day.astype(int) - 1) * 1440 + act.start_minute.astype(int)
    act["e"] = (act.day.astype(int) - 1) * 1440 + act.end_minute.astype(int)
    covered = act.groupby("cow_id").duration_min.sum()
    ok = (covered == days * 1440).all()
    chk("good" if ok else "critical", "Bouts cover every minute exactly once (activity.csv)",
        f"{len(act):,} bouts; per cow {int(covered.min()):,}-{int(covered.max()):,} min (expected {days * 1440:,})")
    for j in joins:
        J = j["minute"]
        cross = act[(act.s < J) & (act.e >= J) & (act.basis != "not_visible")]
        pairs = [(grid[c]["b"][J - 1 - w0], grid[c]["b"][J - w0]) for c in cows if J - 1 >= w0]
        vis = [(a, b) for a, b in pairs if a < len(beh_codes) - 1 and b < len(beh_codes) - 1]
        same = sum(1 for a, b in vis if a == b)
        steps = [(x, y) for c in cows for x, y in zip(grid[c]["b"][:-1], grid[c]["b"][1:])
                 if x < len(beh_codes) - 1 and y < len(beh_codes) - 1]
        base = sum(1 for x, y in steps if x == y) / max(len(steps), 1)
        rate = same / max(len(vis), 1)
        chk("good" if len(cross) else "warning", f"Bouts run through the {j['left']} → {j['right']} join",
            f"{len(cross)} visible bouts span {_hhmm(J)} (they would be cut there if the ids did not line up)")
        chk("good" if rate >= base - 0.15 else "warning", f"Behavior is continuous across {_hhmm(J)}",
            f"{same} of {len(vis)} cows visible on both sides keep their behavior from {_hhmm(J - 1)} to {_hhmm(J)} "
            f"({rate:.0%}); any minute-to-minute step keeps it {base:.0%} of the time. A swapped id usually shows "
            f"as a change at the join, but few cows are visible on both sides and cows rest for long stretches, "
            f"so this is a weak check")
    idm = pd.read_csv(st / "id_map.csv")
    later = idm[idm.chunk != chunks[0]["name"]]
    if len(later):
        stt = later.status.fillna("")
        n_ov = int(stt.str.startswith("overlap (").sum())
        n_sp = int(stt.str.startswith("overlap_split").sum())
        n_ap = int(stt.isin(["matched"]).sum())
        n_pos = int(stt.str.startswith("position").sum())
        n_weak = int(stt.isin(["low_margin", "new"]).sum() + stt.str.startswith("guess").sum())
        lvl = "good" if n_sp + n_weak == 0 else ("warning" if n_weak <= len(later) / 2 else "serious")
        chk(lvl, "Cow ids joined across chunks",
            f"{n_ov} by overlapping boxes (cameras agree), {n_sp} by boxes with the cameras disagreeing, "
            f"{n_ap} by clear appearance, {n_pos} by position across the join, {n_weak} by near-tied appearance or "
            f"no evidence (a guess): review the flagged rows below")
    acts = pd.read_csv(st / "final" / "actions.csv", usecols=["chunk", "actor", "action"], low_memory=False)
    by_chunk = acts.chunk.value_counts().to_dict()
    missing = [c["name"] for c in chunks if not by_chunk.get(c["name"])]
    chk("good" if not missing else "warning", "Agent decisions present for every chunk (actions.csv)",
        ", ".join(f"{k}: {v:,}" for k, v in sorted(by_chunk.items())) + (f"; none for {missing}" if missing else ""))
    vis_share = {c: sum(1 for b in grid[c]["b"] if b < len(beh_codes) - 1) / n for c in cows}
    lo = min(vis_share, key=vis_share.get)
    chk("info", "Visible time per cow",
        f"{min(vis_share.values()):.0%}-{max(vis_share.values()):.0%} of the window (lowest {lo}); the rest is "
        f"'other / not visible' (lying out of view, occluded, night)")

    # ---------------------------------------------------------------- id map, budgets, tiles
    idrows = idm.astype(object).where(idm.notna(), None).to_dict("records")   # NaN -> null, not a JS NaN
    budgets = {}
    for cow in cows:
        b = grid[cow]["b"]
        cnt = [b.count(i) for i in range(len(beh_codes))]
        budgets[cow] = cnt
    basis_tot = [sum(grid[c]["e"].count(i) for c in cows) for i in range(len(basis_codes))]
    agent_calls = int((acts.actor == "vlm").sum())
    return {
        "title": st.name, "day1": str(day1.date()), "w0": w0, "w1": w1, "cows": cows, "chunks": [
            {"name": c["name"], "who": c["who"], "s": to_min(c["start"]), "e": to_min(c["end"])} for c in chunks],
        "joins": joins, "beh": beh_codes, "basis": basis_codes, "reasons": [REASONS.get(r, r or "-") for r in reason_codes],
        "grid": grid, "checks": checks, "idmap": idrows, "budgets": budgets, "basis_tot": basis_tot,
        "tiles": {"cow_minutes": n * len(cows), "visible": sum(sum(v[:-1]) for v in budgets.values()),
                  "verified": basis_tot[0], "agent_steps": agent_calls, "bouts": int((act.basis != "not_visible").sum()),
                  "hours": round(n / 60, 1)},
    }


def _hhmm(minute: int) -> str:
    d, r = divmod(minute, 1440)
    return f"{'' if d == 0 else f'day {d + 1} '}{r // 60:02d}:{r % 60:02d}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stitched", default="outputs/stitched", help="stitch_shares.py --out folder")
    ap.add_argument("--out", help="default: <stitched>/viewer.html")
    a = ap.parse_args()
    st = (REPO / a.stitched).resolve() if not Path(a.stitched).is_absolute() else Path(a.stitched)
    data = build(st)
    html = (REPO / "scripts" / "stitched_viewer.html").read_text(encoding="utf-8")
    out = Path(a.out) if a.out else st / "viewer.html"
    out.write_text(html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    bad = [c for c in data["checks"] if c["level"] in ("critical", "serious", "warning")]
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {len(data['cows'])} cows, {data['tiles']['hours']} h; "
          f"{len(data['checks']) - len(bad)} checks ok, {len(bad)} to look at")
    for c in data["checks"]:
        print(f"  [{c['level']:8s}] {c['title']}: {c['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
