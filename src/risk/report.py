"""Level 2 outputs: brd_risk.csv, one-page risk cards (HTML) with 2-3 video clips, and an overview page."""
from __future__ import annotations

import html
import json
import os
import subprocess
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from .data import hms, iso
from .score import fmt

KIND_FOR = {"feeding_share": "at_bunk", "bunk_visit_rate": "at_bunk", "bunk_visit_min": "at_bunk",
            "feed_call_latency": "out_of_step", "lying_share": "lying", "lying_bout_min": "lying",
            "distance_m_per_h": "lying", "rumination_share": "lying", "drinking_visit_rate": "walking",
            "nn_distance_m": "isolated", "isolation_share": "isolated", "desync_share": "out_of_step",
            "grooming_share": "lying", "breathing_bpm": "lying", "motion_energy": "lying", "jerk_rate": "close_view"}


def make_clip(tools, cow: str, center: int, out_path: str, camera: Optional[str] = None, span: int = 12,
              fps_out: int = 8, width: int = 960) -> Optional[Dict]:
    """Short H.264 clip of one cow (crop follows the union of its boxes), box drawn, time stamped."""
    o = tools.obs[(tools.obs.cow_id == cow) & (tools.obs.sec.between(center - span // 2, center + span // 2))]
    if camera:
        o = o[o.camera == camera]
    if o.empty:
        return None
    cam = o.groupby("camera").box_h.median().idxmax() if not camera else camera
    o = o[o.camera == cam].sort_values("sec")
    run = tools.h.run_at(int(o.sec.iloc[0]))
    n = _write_clip(run._framestore(), cam, o, out_path, cow, fps_out, width)
    return {"path": out_path, "camera": cam, "t0": int(o.sec.iloc[0]), "t1": int(o.sec.iloc[-1]), "frames": n} if n else None


def _write_clip(fs, cam: str, o: pd.DataFrame, out_path: str, cow: str, fps_out: int = 8, width: int = 960) -> int:
    """H.264 clip from rows with sec, x1..y2 (one camera): crop = union of boxes + margin, box drawn."""
    o = o.sort_values("sec")
    first = fs.frame(cam, float(o.sec.iloc[0]))
    if first is None:
        return 0
    H, W = first.shape[:2]
    x1, y1, x2, y2 = o.x1.min(), o.y1.min(), o.x2.max(), o.y2.max()
    mx, my = 0.35 * (x2 - x1), 0.35 * (y2 - y1)
    X1, Y1, X2, Y2 = int(max(0, x1 - mx)), int(max(0, y1 - my)), int(min(W, x2 + mx)), int(min(H, y2 + my))
    w_out = width
    h_out = max(2, int(round((Y2 - Y1) * w_out / max(X2 - X1, 1) / 2) * 2))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{w_out}x{h_out}", "-r", str(fps_out), "-i", "-", "-c:v", "libx264",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)
    n = 0
    for t in np.arange(o.sec.iloc[0], o.sec.iloc[-1] + 1, 1 / fps_out):
        fr = fs.frame(cam, float(t))
        if fr is None:
            continue
        r = o.iloc[(o.sec - t).abs().argmin()]
        cv2.rectangle(fr, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 215, 255), 6)
        img = cv2.resize(fr[Y1:Y2, X1:X2], (w_out, h_out), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (w_out, 30), (0, 0, 0), -1)
        cv2.putText(img, f"{cow} {cam} {hms(t)}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        proc.stdin.write(img.tobytes())
        n += 1
    proc.stdin.close()
    proc.wait()
    return n


def indicator_chart(con: pd.DataFrame, cfg: Dict, cow: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = con[(con.cow_id == cow) & (con.kind == "level")].copy()
    d["label"] = d.indicator.map(lambda k: cfg["indicators"][k]["label"])
    d = d.sort_values("signed", na_position="first")
    fig, ax = plt.subplots(figsize=(6.2, 0.28 * len(d) + 0.8))
    vals = d.signed.fillna(0)
    ax.barh(d.label, vals, color=["#c0392b" if v > 0.5 else ("#e6a23c" if v > 0 else "#7fb3d5") for v in vals])
    for i, (v, h) in enumerate(zip(d.signed, d.hours)):
        if pd.isna(v):
            ax.text(0.05, i, "not enough data", va="center", fontsize=7, color="#888")
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlim(-3.2, 3.2)
    ax.set_xlabel("deviation from herd, robust z (right = sick direction)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_outputs(out: str, risk: pd.DataFrame, con: pd.DataFrame, cfg: Dict, traces: Dict, store, tools,
                  meta: Dict, n_cards: int = 3, make_clips: bool = True):
    os.makedirs(out, exist_ok=True)
    cols = ["cow_id", "risk_score", "rank", "top_reasons", "behavior_score", "visual_adjustment", "p_top3",
            "hours_observed", "confidence", "signs_present", "artifacts"]
    risk[cols].to_csv(os.path.join(out, "brd_risk.csv"), index=False)
    cards_dir = os.path.join(out, "cards")
    os.makedirs(cards_dir, exist_ok=True)
    clips_all = {}
    for _, r in risk.sort_values("rank").head(n_cards).iterrows():
        cow = r.cow_id
        indicator_chart(con, cfg, cow, os.path.join(cards_dir, f"{cow}_indicators.png"))
        clips = pick_clips(tools, con, traces.get(cow, {}), store, cow, cards_dir) if make_clips else []
        clips_all[cow] = clips
        open(os.path.join(cards_dir, f"{cow}.html"), "w").write(card_html(r, con, cfg, traces.get(cow, {}), store, clips, meta, out))
    open(os.path.join(out, "index.html"), "w").write(index_html(risk, con, cfg, meta, n_cards))
    json.dump({"meta": meta, "clips": clips_all}, open(os.path.join(out, "risk_meta.json"), "w"), indent=1, default=str)


def pick_clips(tools, con, trace, store, cow, cards_dir) -> List[Dict]:
    """2-3 clips: strongest behavior deviation, the image behind a visual finding, the closest view."""
    want = []
    top = con[(con.cow_id == cow) & (con.contrib > 0) & (con.kind == "level")].sort_values("contrib", ascending=False)
    if len(top):
        kind = KIND_FOR.get(top.indicator.iloc[0], "close_view")
        mom = tools.t_find_moments(cow, kind, 1).data.get("moments") or []
        if mom:
            t = mom[0] if isinstance(mom[0], (int, np.integer)) else mom[0]["time"]
            want.append((int(t), None, f"behavior: {top.indicator.iloc[0].replace('_', ' ')} ({kind.replace('_', ' ')})"))
    fin = trace.get("final") or {}
    for s, v in (fin.get("signs") or {}).items():
        if v.get("status") == "present":
            for f in tools_store_images(store, cow, v.get("evidence", [])):
                want.append((int(f["data"]["time"]), f["data"].get("camera"), f"visual: {s.replace('_', ' ')}"))
                break
            if len(want) >= 2:
                break
    cv = tools.t_find_moments(cow, "close_view", 1).data.get("moments") or []
    if cv:
        want.append((int(cv[0]["time"]), cv[0]["camera"], "closest view"))
    clips, used = [], []
    for t, cam, why in want:
        if any(abs(t - u) < 30 for u in used) or len(clips) >= 3:
            continue
        c = make_clip(tools, cow, t, os.path.join(cards_dir, "clips", f"{cow}_{len(clips) + 1}.mp4"), cam)
        if c:
            c["why"] = why
            clips.append(c)
            used.append(t)
    return clips


def tools_store_images(store, cow, ids):
    out = []
    for i in ids or []:
        f = store.get(str(i))
        if not f:
            continue
        if f["kind"] == "visual":
            out += tools_store_images(store, cow, f["data"].get("based_on", []))
        elif f["kind"] == "image" and f["cow_id"] == cow and "time" in f["data"]:
            out.append(f)
    return out


CSS = """body{font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px auto;max-width:980px;color:#1d2433;background:#fff;padding:0 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:22px 0 8px}.muted{color:#667085}.pill{display:inline-block;padding:2px 9px;border-radius:12px;font-weight:600}
.hi{background:#fde2e1;color:#a61b1b}.mid{background:#fdf0d5;color:#8a5a00}.lo{background:#e3f1e8;color:#1e6b3a}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #e4e7ec;padding:5px 6px;text-align:left;vertical-align:top}th{font-size:12px;color:#667085}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.thumbs img{height:150px;margin:0 6px 6px 0;border-radius:4px}video{width:100%;border-radius:6px;background:#000}
.bar{height:10px;background:#c0392b;border-radius:5px}.src{font-size:12px;color:#475467}@media(max-width:700px){.grid{grid-template-columns:1fr}}
@media print{video{display:none}}"""


def level(p):
    return ("hi", "high") if p >= 0.5 else (("mid", "elevated") if p >= 0.25 else ("lo", "low"))


def card_html(r, con, cfg, trace, store, clips, meta, out) -> str:
    cow = r.cow_id
    cls, word = level(r.risk_score)
    fin = trace.get("final") or {}
    e = html.escape
    reasons = "".join(f"<li>{e(x)}</li>" for x in str(r.top_reasons).split(" | ") if x)
    signs_rows = ""
    for s, spec in cfg["visual_signs"].items():
        v = (fin.get("signs") or {}).get(s) or {}
        st = v.get("status", "not reviewed")
        signs_rows += f"<tr><td>{e(spec['label'])}</td><td><b>{e(st)}</b></td><td>{e(', '.join(map(str, v.get('evidence', []))))}</td><td class=muted>{e(str(v.get('note', '')))}</td></tr>"
    imgs = []
    for i in fin.get("key_evidence", []) + [x for v in (fin.get("signs") or {}).values() for x in v.get("evidence", [])]:
        f = store.get(str(i))
        if f and f["kind"] == "image" and f["images"]:
            imgs.append((i, os.path.relpath(os.path.join(store.root, f["images"][0]), os.path.join(out, "cards"))))
    seen, thumbs = set(), ""
    for i, p in imgs:
        if i not in seen:
            seen.add(i)
            thumbs += f'<a href="{e(p)}"><img src="{e(p)}" title="{e(i)}"></a>'
    vids = "".join(f'<div><video controls preload="metadata" src="clips/{e(os.path.basename(c["path"]))}"></video>'
                   f'<div class=muted>{e(c["why"])}: {e(c["camera"])} {hms(c["t0"])} to {hms(c["t1"])[-8:]}</div></div>' for c in clips)
    used = sorted({s for k in con[con.cow_id == cow].indicator for s in cfg["indicators"][k.replace("__trend", "")]["sources"]}
                  | {s for s in ("dart", "white2009")})
    srcs = "".join(f"<li>[{e(k)}] {e(cfg['sources'][k])}</li>" for k in used if k in cfg["sources"])
    arts = "".join(f"<li>{e(a['indicator'])}: {e(a.get('why', ''))} ({e(', '.join(map(str, a.get('evidence', []))))})</li>" for a in fin.get("artifacts", []))
    return f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>{e(cow)} BRD risk card</title><style>{CSS}</style></head><body>
<h1>{e(cow)} <span class="pill {cls}">{word} risk {r.risk_score:.2f}</span> <span class=muted>rank {int(r['rank'])} of {meta['n_cows']}</span></h1>
<div class=muted>Pen 6, {e(meta['span'])} ({meta['hours']:.1f} h of video). Seen for {r.hours_observed:.1f} h. Rank stable in {100 * r.p_top3:.0f}% of time resamples. Confidence: {e(r.confidence)}.</div>
<h2>What to do</h2><p>{e(str(fin.get('summary') or 'Behavior-only flag: the agent review did not run for this cow.'))}</p>
<div class=grid><div><h2>Why</h2><ul>{reasons}</ul>
<p class=muted>Behavior score {r.behavior_score:.2f}; visual signs moved the log-odds by {r.visual_adjustment:+.2f}.</p>
{('<h2>Discounted as artifacts</h2><ul>' + arts + '</ul>') if arts else ''}</div>
<div><h2>Behavior vs herd</h2><img src="{e(cow)}_indicators.png" style="width:100%"></div></div>
<h2>Clips</h2><div class=grid>{vids or '<p class=muted>no clip available</p>'}</div>
<h2>Visible clinical signs (agent review, self-checked)</h2><table><tr><th>sign</th><th>status</th><th>evidence</th><th>note</th></tr>{signs_rows}</table>
<div class=thumbs style="margin-top:8px">{thumbs}</div>
<h2>Agent assessment</h2><p>{e(str(fin.get('behavior_assessment', '')))}</p>
<h2>Sources</h2><ol class=src>{srcs}</ol>
<p class=src>This is a screening index from video behavior, not a diagnosis. Confirm with a hands-on exam (rectal temperature, lung auscultation) before treatment.
Clinical-sign diagnosis of BRD is itself only ~62% sensitive and ~63% specific [white2009].</p></body></html>"""


def index_html(risk, con, cfg, meta, n_cards) -> str:
    e = html.escape
    rows = ""
    for _, r in risk.sort_values("rank").iterrows():
        cls, word = level(r.risk_score)
        link = f'<a href="cards/{e(r.cow_id)}.html">{e(r.cow_id)}</a>' if r["rank"] <= n_cards else e(r.cow_id)
        rows += (f"<tr><td>{int(r['rank'])}</td><td>{link}</td><td><span class='pill {cls}'>{r.risk_score:.2f}</span></td>"
                 f"<td><div class=bar style='width:{max(4, 160 * r.risk_score):.0f}px'></div></td><td>{100 * r.p_top3:.0f}%</td>"
                 f"<td>{r.hours_observed:.1f}</td><td>{e(str(r.top_reasons))}</td></tr>")
    return f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Pen 6 BRD risk</title><style>{CSS}</style></head><body>
<h1>Pen 6 BRD risk ranking</h1><div class=muted>{e(meta['span'])}, {meta['hours']:.1f} h of video, {meta['n_cows']} cows. Behavior: {e(meta['activity_source'])}. Reasoning model: {e(meta['models'])}.</div>
<table style="margin-top:14px"><tr><th>rank</th><th>cow</th><th>risk</th><th></th><th>top-3 stability</th><th>hours seen</th><th>top reasons</th></tr>{rows}</table>
<h2>Method</h2><p>Each cow is compared with its pen-mates on 13 behavior indicators linked to BRD in the literature
(feeding and bunk attendance, lying, activity, rumination, drinking, social isolation, herd synchrony, grooming).
Deviations in the sick direction are weighted by strength of evidence and shrunk when the cow was seen for little time.
A reasoning agent then checks the top cows on video for visible signs (head and ear carriage, nasal discharge, breathing)
and for data artifacts. A self-check accepts only signs backed by a close enough image. "top-3 stability" is how often
the cow stays in the top 3 when the observed time is resampled. Full method: docs/LEVEL2_METHOD.md.</p></body></html>"""
