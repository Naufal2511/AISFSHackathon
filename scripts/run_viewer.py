#!/usr/bin/env python3
"""One interactive HTML page for a pipeline run: what every stage produced, and why.

  python scripts/run_viewer.py --run outputs/pipeline/demo_5min          # -> <run>/viewer.html

Self-contained (data + downscaled images embedded), opens from disk; event data stays on this machine.
Sections: run timeline, ethogram (stage 1 per second vs stage 2 verified per minute; click a minute for the agent's
plan, tool calls, images, hypothesis, rule checks and decision), identity (memory cards, identity-agent
investigations, ID-check sheet), BRD risk (Level 2 ranking and the risk agent's findings), assets and files.
"""
from __future__ import annotations

import argparse
import base64
import csv
import glob
import json
import os
import re
from pathlib import Path

import cv2
import pandas as pd

BEH = ["lying", "standing", "walking", "feeding", "drinking", "ruminating_lying", "ruminating_standing", "grooming",
       "social_interaction", "other_not_visible"]


def canon(b) -> str:
    if not isinstance(b, str) or not b.strip():
        return "other_not_visible"
    s = b.strip().lower().replace("(", "").replace(")", "").replace("/", " ").replace("-", " ")
    s = re.sub(r"\s+", "_", s)
    return {"ruminating_lying": "ruminating_lying", "rum_lying": "ruminating_lying", "rum_standing": "ruminating_standing",
            "social": "social_interaction", "other_not_visible": "other_not_visible", "other": "other_not_visible",
            "not_visible": "other_not_visible", "other__not_visible": "other_not_visible"}.get(s, s if s in BEH else "other_not_visible")


class Images:
    def __init__(self, max_side: int, quality: int):
        self.max_side, self.q, self.data, self.ids = max_side, quality, {}, {}

    def add(self, path, max_side=None) -> str | None:
        if not path or not os.path.exists(path):
            return None
        key = (os.path.abspath(path), max_side)
        if key in self.ids:
            return self.ids[key]
        im = cv2.imread(path)
        if im is None:
            return None
        m = max_side or self.max_side
        s = m / max(im.shape[:2])
        if s < 1:
            im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, self.q])
        if not ok:
            return None
        i = f"i{len(self.data)}"
        self.data[i] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        self.ids[key] = i
        return i


def _short(v, n=400):
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= n else s[: n - 1] + "…"


def last_json(path: Path, key: str = "by_reason") -> dict:
    """The last summary dict a step printed in its log that contains `key`."""
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8", errors="replace")
    dec, out, i = json.JSONDecoder(), {}, 0
    while True:
        i = txt.find("{", i)
        if i < 0:
            return out
        try:
            obj, end = dec.raw_decode(txt, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict):
            # the summary may nest its counts one level down (brd_agent: {"by_reason_this_run", "store_status": {...}})
            inner = [v for v in obj.values() if isinstance(v, dict) and key in v]
            if key in obj:
                out = obj
            elif inner:
                out = {**inner[0], **obj}
        i = end


def build(run: Path, out: Path, max_side: int) -> dict:
    L = run / "stage1" / "level1"
    img = Images(max_side, 72)
    state = json.loads((run / "pipeline_state.json").read_text()) if (run / "pipeline_state.json").exists() else {}
    man = json.loads((run / "pipeline_manifest.json").read_text()) if (run / "pipeline_manifest.json").exists() else {}
    steps = [{"name": k, "seconds": v.get("seconds"), "status": v.get("status")} for k, v in state.get("steps", {}).items()]
    settings = man.get("settings", {})

    # ---- stage 2 verified minutes in the window
    am = pd.read_csv(run / "final" / "activity_minute.csv", low_memory=False)
    am = am[am.reason != "no_upstream_data"].copy()
    traces = {Path(p).stem: p for p in glob.glob(str(run / "stage2" / "traces" / "*.json"))}
    acts = pd.read_csv(run / "final" / "actions.csv", low_memory=False) if (run / "final" / "actions.csv").exists() else pd.DataFrame()
    if len(acts):
        acts = acts[acts.action != "no_upstream"]
    minutes = []
    for r in am.itertuples(index=False):
        tid = f"{r.cow_id}_d{int(r.day)}_m{int(r.minute):04d}"
        a = acts[(acts.cow_id == r.cow_id) & (acts.day == r.day) & (acts.minute == r.minute)] if len(acts) else acts
        minutes.append({"cow": r.cow_id, "day": int(r.day), "minute": int(r.minute), "behavior": canon(r.behavior),
                        "conf": None if pd.isna(r.confidence) else round(float(r.confidence), 3), "tier": int(r.tier) if pd.notna(r.tier) else None,
                        "reason": r.reason, "correction": r.correction if isinstance(r.correction, str) else "",
                        "tier1": canon(r.tier1_label) if isinstance(r.tier1_label, str) else None,
                        "rationale": r.rationale if isinstance(r.rationale, str) else "",
                        "flags": r.flags if isinstance(r.flags, str) else "", "night": bool(r.is_night_final) if pd.notna(r.is_night_final) else False,
                        "trace": tid if tid in traces else None,
                        "actions": [{"actor": x.actor, "action": x.action, "reason": _short(x.reason, 300), "outcome": _short(x.outcome, 300)}
                                    for x in a.sort_values("seq").itertuples(index=False)]})
    cows = sorted({m["cow"] for m in minutes})
    m_lo, m_hi = min(m["minute"] for m in minutes), max(m["minute"] for m in minutes)

    # ---- agent traces (stage 2)
    tr_out = {}
    for tid, p in traces.items():
        t = json.loads(Path(p).read_text())
        ev = []
        for e in t.get("events", []):
            ty = e.get("type")
            if ty == "plan":
                ev.append({"type": "plan", "text": e.get("rationale", ""), "steps": [s.get("tool") for s in e.get("steps", [])]})
            elif ty == "tool":
                ims = [img.add(str(Path(p).parent / i), 360) for i in (e.get("images") or [])][:6]
                ev.append({"type": "tool", "tool": e.get("tool"), "reason": e.get("reason", ""), "facts": _short(e.get("facts"), 360),
                           "ok": e.get("ok"), "images": [i for i in ims if i], "captions": (e.get("captions") or [])[:6]})
            elif ty == "hypothesis":
                ev.append({"type": "hypothesis", "stage": e.get("stage"), "label": canon(e.get("label")), "conf": e.get("confidence"),
                           "cues": [str(c) for c in (e.get("cues") or [])][:5], "text": e.get("rationale", "")})
            elif ty == "verifier":
                ev.append({"type": "verifier", "stage": e.get("stage"),
                           "checks": [{"name": c.get("name"), "ok": c.get("passed"), "sev": c.get("severity"), "detail": c.get("detail")}
                                      for c in e.get("checks", [])]})
            elif ty in ("critique", "decision", "night_check", "gate"):
                ev.append({"type": ty, **{k: (_short(v, 500) if not isinstance(v, (int, float, bool)) else v)
                                         for k, v in e.items() if k not in ("type", "t")}})
            elif ty == "vlm":
                ev.append({"type": "vlm", "task": e.get("task"), "latency": e.get("latency_s"),
                           "tokens": (e.get("prompt_tokens") or 0) + (e.get("completion_tokens") or 0)})
        tr_out[tid] = ev

    # ---- stage 1 per-second behavior
    sec = {}
    tl = L / "behavior" / "behavior_timeline_1s.csv"
    t0 = None
    if tl.exists():
        d = pd.read_csv(tl, usecols=["cow_id", "wall_time", "behavior", "confidence", "observed"], low_memory=False)
        d["ts"] = pd.to_datetime(d.wall_time)
        t0 = d.ts.min().floor("min")
        n = int((d.ts.max() - t0).total_seconds()) + 1
        for cow, g in d.groupby("cow_id"):
            arr = [-1] * n
            for r in g.itertuples(index=False):
                k = int((r.ts - t0).total_seconds())
                arr[k] = BEH.index(canon(r.behavior)) if str(r.observed) == "True" else BEH.index(canon(r.behavior)) + 20
            sec[cow] = arr
    sec_start_min = (t0.hour * 60 + t0.minute) if t0 is not None else m_lo

    # ---- identity
    ident = {}
    cards_p = L / "agent" / "memory" / "cards.json"
    if cards_p.exists():
        ident["cards"] = [{"cow": c["cow_id"], "traits": c.get("traits"), "stats": c.get("stats"), "last_seen": c.get("last_seen"),
                           "refs": [i for i in (img.add(r, 260) for r in (c.get("references") or [])[:3]) if i]}
                          for c in json.loads(cards_p.read_text())]
    rep = L / "identity" / "identity_report.json"
    if rep.exists():
        r = json.loads(rep.read_text())
        ident["report"] = {k: r.get(k) for k in ("tracklets", "agreement_merges", "continuity_merges", "appearance_merges",
                                                  "animals_required_by_evidence", "pen_size_cap", "assigned_box_share")}
    inv = []
    for p in sorted(glob.glob(str(L / "agent" / "traces" / "*.json"))):
        t = json.loads(Path(p).read_text())
        f, fin = t.get("flag", {}), t.get("final") or {}
        inv.append({"kind": f.get("kind"), "detail": f.get("detail"), "cows": f.get("cow_ids"), "severity": f.get("severity"),
                    "time": pd.Timestamp(f.get("t0"), unit="s").strftime("%H:%M:%S") if f.get("t0") else "",
                    "decision": fin.get("decision"), "summary": fin.get("summary", ""), "confidence": fin.get("confidence"),
                    "evidence": [str(x) for x in (fin.get("evidence") or [])][:6], "seconds": t.get("seconds"),
                    "steps": [{"thought": s.get("thought", ""), "tool": s.get("tool"), "args": _short(s.get("args"), 200),
                               "result": _short(s.get("result"), 500),
                               "images": [i for i in (img.add(x, 520) for x in (s.get("images") or [])[:2]) if i]}
                              for s in t.get("steps", []) if s.get("tool")]})
    ident["investigations"] = inv
    ident["sheets"] = {k: img.add(str(p), 1400) for k, p in
                       [("id_check", next(iter(sorted((L / "agent" / "id_checks").glob("*.jpg"))), None)),
                        ("identity_sheet", L / "identity" / "identity_sheet.jpg")] if p}
    summ = L / "agent" / "agent_summary.json"
    ident["summary"] = json.loads(summ.read_text()) if summ.exists() else {}

    # ---- level 2
    risk = {}
    rp = run / "level2" / "brd_risk.csv"
    if rp.exists():
        rk = pd.read_csv(rp)
        rows = []
        for r in rk.sort_values("rank").itertuples(index=False):
            tp = run / "level2" / "agent" / "traces" / f"{r.cow_id}.json"
            fin = json.loads(tp.read_text()).get("final", {}) if tp.exists() else {}
            signs = {k: {"status": v.get("status"), "why": _short(v.get("evidence") or v.get("note") or v.get("reason") or "", 240)}
                     for k, v in (fin.get("signs") or {}).items() if isinstance(v, dict)}
            rows.append({"cow": r.cow_id, "rank": int(r.rank), "risk": float(r.risk_score), "behavior": float(r.behavior_score),
                         "visual": float(r.visual_adjustment), "p_top3": None if pd.isna(r.p_top3) else float(r.p_top3),
                         "hours": float(r.hours_observed), "confidence": r.confidence,
                         "reasons": [x.strip() for x in str(r.top_reasons).split("|") if x.strip() and x.strip() != "nan"],
                         "signs": signs, "assessment": _short(fin.get("assessment") or fin.get("summary") or "", 900),
                         "card": f"level2/cards/{r.cow_id}.html" if (run / "level2" / "cards" / f"{r.cow_id}.html").exists() else None})
        risk = {"rows": rows, "index": "level2/index.html" if (run / "level2" / "index.html").exists() else None}

    # ---- assets + stage 2 summary + files
    assets = json.loads((L / "assets" / "summary.json").read_text()) if (L / "assets" / "summary.json").exists() else {}
    gal = {}
    gp = L / "assets" / "gallery.csv"
    if gp.exists():
        g = pd.read_csv(gp)
        for cow, gg in g.groupby("cow_id"):
            gal[cow] = [{"img": i, "cam": x.camera, "h": int(x.box_h)} for x in gg.sort_values("box_h", ascending=False).head(4).itertuples()
                        if (i := img.add(x.path if os.path.isabs(x.path) else str(Path.cwd() / x.path), 300))]
    s2 = last_json(run / "logs" / "4_agent.log")
    files = []
    for p in sorted(run.rglob("*")):
        rel = p.relative_to(run)
        if p.is_file() and rel.parts[0] not in ("input", "cache") and p.suffix in (".csv", ".json", ".html", ".jpg", ".mp4", ".md") \
                and len(rel.parts) <= 4 and "img" not in rel.parts and "traces" not in rel.parts and "crops" not in rel.parts \
                and "gallery" not in rel.parts and "clips" not in rel.parts and "evidence" not in rel.parts and "breathing" not in rel.parts:
            rows = None
            if p.suffix == ".csv":
                with open(p, "rb") as f:
                    rows = max(0, sum(1 for _ in f) - 1)
            files.append({"path": str(rel), "kb": round(p.stat().st_size / 1024), "rows": rows})

    return {"run": run.name, "window": [settings.get("start"), settings.get("end")], "steps": steps,
            "cameras": state.get("steps", {}).get("prepare", {}).get("info", {}).get("cameras", {}),
            "cows": cows, "minutes": minutes, "m_lo": m_lo, "m_hi": m_hi, "sec": sec, "sec_start_min": sec_start_min,
            "traces": tr_out, "identity": ident, "risk": risk, "assets": assets, "gallery": gal, "stage2": s2,
            "files": files, "beh": BEH, "img": img.data}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out")
    ap.add_argument("--max-side", type=int, default=420)
    a = ap.parse_args()
    run = Path(a.run).resolve()
    out = Path(a.out) if a.out else run / "viewer.html"
    data = build(run, out, a.max_side)
    tpl = (Path(__file__).parent / "run_viewer_template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, default=str).replace("</", "<\\/"))
    out.write_text(html, encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 2 ** 20:.1f} MB, {len(data['img'])} images, {len(data['minutes'])} cow-minutes, "
          f"{len(data['traces'])} agent traces, {len(data['identity'].get('investigations', []))} identity investigations)")


if __name__ == "__main__":
    main()
