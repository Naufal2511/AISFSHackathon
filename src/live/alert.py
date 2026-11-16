"""Level 3 alerts: video evidence, a one-page card a pen rider can act on, the live dashboard, and delivery."""
from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

SIGN_TEXT = {"nasal_discharge": "nasal discharge", "ocular_discharge": "eye discharge", "ears_droop": "drooping ears",
             "head_low": "head carried low while standing (not eating)"}
ZONE_TEXT = {"bunk": "at the feed bunk", "rest": "in the resting area", "other": "in the open pen"}


def _ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


VIDEO_ROOTS: List[Path] = []         # extra folders to find a clip in (live_watch.py --video-root)


def _source(win, name: str) -> Optional[Path]:
    """The window's input clip with this file name (hard links under <run>/input/camN), else under --video-root."""
    for root in [win.run / "input", *VIDEO_ROOTS]:
        if root.exists():
            for p in root.rglob(name):
                return p
    return None


def moment(win, ids: pd.DataFrame, alert: Dict) -> Optional[Dict]:
    """Where and when to show the cow: the confirmed sign's crop time, else the second with her largest box."""
    lm_inv = {}
    for r in ids.itertuples():
        lm_inv.setdefault(r.cow_id, []).append(r.local_id)
    locs = set(lm_inv.get(alert["cow_id"], [alert["cow_id"]]))
    o = win.obs[win.obs.cow_id.astype(str).isin(locs)]
    if "box_role" in o:
        o = o[o.box_role.fillna("body") == "body"]
    if o.empty:
        return None
    if alert.get("signs"):
        t = pd.to_datetime(alert["signs"][0]["time"], unit="s")
        cam = alert["signs"][0]["camera"]
        oc = o[o.camera == cam]
        o = oc if len(oc) else o
        r = o.iloc[(o.wall_time - t).abs().argsort().iloc[0]]
    else:
        area = (o.x2 - o.x1) * (o.y2 - o.y1)
        r = o.loc[area.idxmax()]
    return {"time": r.wall_time, "camera": r.camera, "file": r.file, "offset_s": float(r.offset_s),
            "box": [float(r.x1), float(r.y1), float(r.x2), float(r.y2)], "local_id": str(r.cow_id)}


def evidence(win, ids: pd.DataFrame, alert: Dict, out: Path, seconds: int = 12) -> Dict:
    """A clip around the moment (cropped to the cow, 640 px) + a full frame with her box. Paths relative to out."""
    m = moment(win, ids, alert)
    res = {"moment": None, "clip": None, "frame": None, "crops": []}
    if not m:
        return res
    res["moment"] = {**m, "time": str(m["time"])}
    src = _source(win, m["file"])
    stem = f"{alert['cow_id']}_{pd.Timestamp(m['time']):%Y%m%d_%H%M%S}_{m['camera']}"
    ff = _ffmpeg()
    if src and ff:
        # the cameras' .Avi files carry HEVC under a tag ffmpeg does not probe: force the decoder (as the pipeline does)
        inp = (["-vcodec", "hevc"] if src.suffix.lower() in (".avi", ".ghec") else []) + ["-i", str(src)]
        x1, y1, x2, y2 = m["box"]
        pw, ph = (x2 - x1) * 0.6, (y2 - y1) * 0.6
        cx1, cy1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
        cw, ch = int((x2 - x1) + 2 * pw) // 2 * 2, int((y2 - y1) + 2 * ph) // 2 * 2
        start = max(0.0, m["offset_s"] - seconds / 2)
        clip = out / f"{stem}.mp4"
        cmd = [ff, "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", str(seconds), *inp,
               "-vf", f"crop={cw}:{ch}:{cx1}:{cy1},scale=640:-2", "-an", "-c:v", "libx264", "-preset", "veryfast",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(clip)]
        try:
            subprocess.run(cmd, check=True, timeout=180)
            res["clip"] = clip.name
        except Exception as e:                       # crop outside the frame etc.: fall back to the full view
            cmd = [ff, "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", str(seconds), *inp,
                   "-vf", "scale=960:-2", "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(clip)]
            try:
                subprocess.run(cmd, check=True, timeout=180)
                res["clip"] = clip.name
            except Exception as e2:
                res["clip_error"] = f"{e}; {e2}"
        frame = out / f"{stem}_frame.jpg"
        cmd = [ff, "-y", "-loglevel", "error", "-ss", f"{m['offset_s']:.2f}", *inp, "-frames:v", "1", str(frame)]
        try:
            subprocess.run(cmd, check=True, timeout=120)
            img = cv2.imread(str(frame))
            if img is not None:
                x1, y1, x2, y2 = map(int, m["box"])
                t = max(4, img.shape[1] // 400)
                cv2.rectangle(img, (x1, y1), (x2, y2), (40, 140, 255), t)
                cv2.putText(img, alert["cow_id"], (x1, max(30, y1 - 15)), cv2.FONT_HERSHEY_SIMPLEX, t / 2, (40, 140, 255), t)
                s = 1280 / img.shape[1]
                img = cv2.resize(img, (1280, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(frame), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                res["frame"] = frame.name
        except Exception as e:
            res["frame_error"] = str(e)
    for s in alert.get("signs", [])[:3]:
        for key in ("path", "control"):
            p = s.get(key)
            if p and Path(p).exists():
                dst = out / f"{stem}_{s['sign']}_{key}{Path(p).suffix}"
                shutil.copy2(p, dst)
                res["crops"].append({"file": dst.name, "what": SIGN_TEXT.get(s["sign"], s["sign"]) if key == "path"
                                     else "herdmate closest in time (control)"})
    return res


def action_line(alert: Dict, where: Optional[str]) -> str:
    cow = alert["cow_id"]
    signs = [SIGN_TEXT.get(s, s) for s in alert.get("new_signs") or sorted({s["sign"] for s in alert.get("signs", [])})]
    loc = f" She was last seen {where}." if where else ""
    if signs:
        return (f"Walk the pen and look at {cow} now: the camera saw {', '.join(signs)}.{loc} Check her nose and eyes, "
                f"breathing and appetite; take a rectal temperature if she looks dull. Pull her for treatment by your "
                f"BRD protocol if she is febrile (>= 40 C / 104 F).")
    return (f"Look at {cow} on your next pen ride: her behavior moved toward the BRD pattern (see why below).{loc} "
            f"Watch her at the bunk and check for dullness, nasal discharge, drooping ears or fast breathing. "
            f"No visible sign was confirmed by the camera, so this is a watch, not a pull.")


def card_html(alert: Dict, ev: Dict, win_text: str) -> str:
    e = html.escape
    sev = "Check now" if alert["trigger"] == "sign" else "Watch closely"
    reasons = "".join(f"<li>{e(r['text'])}</li>" for r in alert.get("reasons", [])[:4]) or "<li>No behavior deviation in this window.</li>"
    sign_rows = "".join(f"<li><b>{e(SIGN_TEXT.get(s['sign'], s['sign']))}</b> on {e(s['camera'])} at "
                        f"{pd.to_datetime(s['time'], unit='s'):%H:%M:%S}: {e(s.get('why', ''))} "
                        f"<span class='m'>(confirmed by a second look next to a herdmate crop, confidence {e(str(s.get('confidence')))})</span></li>"
                        for s in alert.get("signs", [])) or "<li>None confirmed in this window.</li>"
    media = ""
    if ev.get("clip"):
        media += f"<figure><video src='{e(ev['clip'])}' controls muted loop playsinline></video><figcaption>12 s, cropped to the cow</figcaption></figure>"
    if ev.get("frame"):
        media += f"<figure><img src='{e(ev['frame'])}' alt='full frame'><figcaption>where she is in the pen</figcaption></figure>"
    for c in ev.get("crops", []):
        media += f"<figure><img src='{e(c['file'])}' alt=''><figcaption>{e(c['what'])}</figcaption></figure>"
    mo = ev.get("moment") or {}
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(alert['cow_id'])} Alert</title>
<style>
:root{{--bg:#f9f9f7;--card:#fff;--ink:#0b0b0b;--ink2:#52514e;--m:#898781;--line:#e1e0d9;--warn:#fab219;--crit:#d03b3b;--accent:#0f5a6b}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2c2c2a;--accent:#5fb3c4}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:900px;margin:0 auto;padding:16px}}
.badge{{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:650;font-size:13px;background:{'var(--crit)' if alert['trigger'] == 'sign' else 'var(--warn)'};color:{'#fff' if alert['trigger'] == 'sign' else '#0b0b0b'}}}
h1{{font-size:24px;margin:8px 0 4px}} .m{{color:var(--m)}} section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}}
h2{{font-size:15px;margin:0 0 6px}} .act{{font-size:16px;border-left:4px solid var(--accent);padding-left:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}} figure{{margin:0}} video,img{{width:100%;border-radius:8px;border:1px solid var(--line)}}
figcaption{{font-size:12px;color:var(--m)}} .kv{{display:grid;grid-template-columns:max-content 1fr;gap:2px 12px;font-size:14px}} .kv div:nth-child(odd){{color:var(--m)}}
</style></head><body><main>
<span class="badge">{sev}</span>
<h1>{e(alert['cow_id'])}: {'visible BRD sign' if alert['trigger'] == 'sign' else 'behavior moving toward BRD'}</h1>
<div class="m">{e(win_text)}</div>
<section><h2>What to do</h2><p class="act">{e(alert['action'])}</p></section>
<section class="grid">{media or '<p class="m">No video evidence could be cut for this window.</p>'}</section>
<section><h2>Visible signs (camera)</h2><ul>{sign_rows}</ul></section>
<section><h2>Why (behavior in this window, vs the herd at the same time)</h2><ul>{reasons}</ul></section>
<section><h2>Risk</h2><div class="kv">
<div>Now</div><div><b>{alert['risk']:.2f}</b> (rank {alert['rank']} of 10)</div>
<div>Before this window</div><div>{alert['risk_before']:.2f} (rank {alert['rank_before']})</div>
<div>48 h baseline (Level 2)</div><div>{alert['prior']:.2f}</div>
<div>Seen at</div><div>{e(str(mo.get('time', '')))} on {e(str(mo.get('camera', '')))} ({e(str(mo.get('file', '')))} at {mo.get('offset_s', 0):.0f} s)</div>
<div>Identity</div><div>{e(alert.get('identity', ''))}</div></div>
<p class="m">Risk is a ranking index (0-1), not a probability of disease. Observed: the clip, the crops and the cow's
positions. Estimated: behavior labels from the video model, the risk index. The camera cannot take a temperature:
a pen rider confirms.</p></section>
</main></body></html>"""


def write_email(out: Path, cfg: Dict, subject: str, body_html: str, body_text: str, attachments: List[Path],
                tag: str, log) -> Optional[Dict]:
    """A real RFC 5322 message (.eml) in <out>/outbox: what the pen rider's inbox would receive. With
    delivery.email.mode = smtp (and SMTP_* in .env) the same message is sent; the default 'simulate' never sends."""
    import mimetypes
    import smtplib
    from email.message import EmailMessage
    from email.utils import formatdate, make_msgid
    ec = cfg.get("delivery", {}).get("email") or {}
    if not ec.get("to"):
        return None
    msg = EmailMessage()
    msg["From"], msg["To"] = ec.get("from", "pen6-monitor@ranch.example"), ", ".join(ec["to"])
    msg["Subject"], msg["Date"], msg["Message-ID"] = subject, formatdate(localtime=True), make_msgid(domain="ranch.example")
    msg.set_content(body_text)
    msg.add_alternative(body_html, subtype="html")
    max_mb, att = float(ec.get("max_attachment_mb", 8)), []
    for p in attachments:
        p = Path(p)
        if not p.exists() or p.stat().st_size > max_mb * 2 ** 20:
            continue
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        maj, sub = ctype.split("/", 1)
        msg.add_attachment(p.read_bytes(), maintype=maj, subtype=sub, filename=p.name)
        att.append({"name": p.name, "kb": round(p.stat().st_size / 1024)})
    box = out / "outbox"
    box.mkdir(exist_ok=True)
    f = box / f"{tag}.eml"
    f.write_bytes(bytes(msg))
    rec = {"file": f"outbox/{f.name}", "to": msg["To"], "from": msg["From"], "subject": subject, "date": msg["Date"],
           "body_html": body_html, "attachments": att, "mode": ec.get("mode", "simulate")}
    if ec.get("mode") == "smtp":                    # real sending: only when the operator configured it
        host = os.environ.get("SMTP_HOST")
        if host:
            try:
                with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", 587)), timeout=30) as s:
                    s.starttls()
                    s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
                    s.send_message(msg)
                rec["mode"] = "sent"
            except Exception as e:
                log(f"email send failed ({e}); kept in outbox")
    index = box / "outbox.jsonl"
    with open(index, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    log(f"email ({rec['mode']}): to {rec['to']}: {subject} ({len(att)} attachment(s)) -> {f}")
    return rec


def alert_email(alert: Dict, ev: Dict, out: Path, cfg: Dict, log) -> Optional[Dict]:
    """The email for one live alert: action, why, risk; card, frame, clip and sign crops attached."""
    e = html.escape
    kind = "Check now" if alert["trigger"] == "sign" else "Watch closely"
    subject = f"BRD alert: {alert['cow_id']}, {kind} ({alert.get('window_start', '')[11:16]})"
    reasons = "".join(f"<li>{e(r['text'])}</li>" for r in alert.get("reasons", [])[:3])
    signs = "".join(f"<li><b>{e(SIGN_TEXT.get(s['sign'], s['sign']))}</b>: {e(s.get('why', ''))}</li>" for s in alert.get("signs", []))
    body_html = (f"<h2 style='margin:0'>{e(alert['cow_id'])}: {e(kind)}</h2><p><b>What to do:</b> {e(alert['action'])}</p>"
                 f"{'<p><b>Visible signs (confirmed):</b></p><ul>' + signs + '</ul>' if signs else ''}"
                 f"<p><b>Why:</b></p><ul>{reasons or '<li>see the card</li>'}</ul>"
                 f"<p>Risk {alert['risk']:.2f} (was {alert['risk_before']:.2f}; 48 h baseline {alert['prior']:.2f}), rank {alert['rank']} of 10. "
                 f"Risk is a ranking index, not a diagnosis. The camera cannot take a temperature: a pen rider confirms.</p>"
                 f"<p>Evidence attached: the alert card, the clip and the frame.</p>")
    text = f"{alert['cow_id']}: {kind}\n{alert['action']}\nRisk {alert['risk']:.2f}, rank {alert['rank']}.\nEvidence attached."
    a_dir = out / "alerts"
    files = [out / alert["card"]] + [a_dir / ev[k] for k in ("clip", "frame") if ev.get(k)] + [a_dir / c["file"] for c in ev.get("crops", [])]
    return write_email(out, cfg, subject, body_html, text, files, Path(alert["card"]).stem, log)


def deliver(alert: Dict, card_rel: str, cfg: Dict, log) -> List[str]:
    """Where the alert went. Always: the alerts folder + dashboard + console. Optional webhook (Slack/Teams/ntfy).
    The email is written by alert_email / write_email (simulated by default)."""
    sent = ["alerts folder", "dashboard"]
    if (cfg.get("delivery", {}).get("email") or {}).get("to"):
        sent.append("email (" + cfg["delivery"]["email"].get("mode", "simulate") + ")")
    url = os.environ.get(cfg.get("delivery", {}).get("webhook_env", "LIVE_WEBHOOK_URL"), "")
    if url:
        text = f"BRD early warning: {alert['cow_id']} ({alert['trigger']}). {alert['action']} Card: {card_rel}"
        try:
            req = urllib.request.Request(url, data=json.dumps({"text": text}).encode(), headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=15).read()
            sent.append("webhook")
        except Exception as e:
            log(f"webhook failed: {e}")
    return sent


def dashboard(out: Path, state, windows: List[Dict]) -> Path:
    """live/index.html: risk now vs the 48 h baseline per cow, trend, and the alert feed. Refreshes itself."""
    e = html.escape
    cows = state.d["cows"]
    ranks = state.ranks()
    rows = ""
    for cow in sorted(cows, key=lambda c: ranks[c]):
        c = cows[cow]
        hist = [h["risk"] for h in c.get("history", [])][-24:]
        pts = [c["prior"]] + hist
        w, h = 120, 28
        lo, hi = 0.0, max(0.6, max(pts))
        path = " ".join(f"{i * w / max(len(pts) - 1, 1):.1f},{h - (v - lo) / (hi - lo) * h:.1f}" for i, v in enumerate(pts))
        d = c["risk"] - c["prior"]
        last = c.get("history", [])[-1] if c.get("history") else {}
        sig = ", ".join(SIGN_TEXT.get(s, s) for s in last.get("signs", [])) if last else ""
        from .engine import TIER_NAME
        tier = c.get("tier", "normal")
        bg, fg = TIER_COLOR[tier]
        tcell = (f"<span style='background:{bg};color:{fg};padding:1px 8px;border-radius:999px;font-size:12px;white-space:nowrap' "
                 f"title='{e(str(c.get('tier_rule', '')))}'>{e(TIER_NAME[tier])}</span>")
        rows += (f"<tr><td>{ranks[cow]}</td><td><b>{e(cow)}</b></td><td>{tcell}</td><td class=n>{c['prior']:.2f}</td><td class=n><b>{c['risk']:.2f}</b></td>"
                 f"<td class=n style='color:{'var(--crit)' if d > 0.02 else 'var(--m)'}'>{d:+.2f}</td>"
                 f"<td><svg width={w} height={h} role='img' aria-label='risk trend'><polyline points='{path}' fill='none' stroke='var(--accent)' stroke-width='2'/></svg></td>"
                 f"<td>{e(sig)}</td><td class=m>{e(str(c.get('last_alert') or ''))}</td></tr>")
    feed = ""
    for a in reversed(state.d.get("alerts", [])[-30:]):
        label = {"sign": "visible sign", "risk": "risk rising", "behavior": "behavior", "tier": "moved up a tier"}.get(a["trigger"], a["trigger"])
        feed += (f"<li><a href='{e(a['card'])}'><b>{e(a['cow_id'])}</b> {e(label)}</a> "
                 f"<span class=m>{e(a['window_start'])} · risk {a['risk']:.2f} · sent to {e(', '.join(a.get('sent', [])))}</span><br>{e(a['action'])}</li>")
    wins = "".join(f"<li>{e(w['start'])} to {e(w['end'])}: {w['cows']} cows, {w['alerts']} alert(s), signs {e(w['signs'])} "
                   f"<span class=m>({w['seconds']:.0f} s)</span></li>" for w in reversed(windows[-12:]))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30"><title>Pen 6 Live</title>
<style>
:root{{--bg:#f9f9f7;--card:#fff;--ink:#0b0b0b;--m:#898781;--line:#e1e0d9;--crit:#d03b3b;--accent:#0f5a6b}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--line:#2c2c2a;--accent:#5fb3c4}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}} main{{max-width:1100px;margin:0 auto;padding:16px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0;overflow-x:auto}}
table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}} .n{{text-align:right;font-variant-numeric:tabular-nums}} .m{{color:var(--m)}}
h1{{font-size:20px;margin:4px 0}} h2{{font-size:15px;margin:0 0 8px}} li{{margin:6px 0}} a{{color:var(--accent)}}
</style></head><body><main>
<h1>Pen 6 · live BRD early warning</h1><div class=m>Updates every window (5 min of video from 3 cameras). Risk is a ranking index, not a probability.</div>
<section><h2>Alerts</h2><ul>{feed or '<li class=m>No alerts yet.</li>'}</ul></section>
<section><h2>Risk per cow</h2><table><thead><tr><th>#</th><th>cow</th><th>tier</th><th class=n>48 h baseline</th><th class=n>now</th><th class=n>change</th><th>trend</th><th>signs (last window)</th><th>last alert</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>Windows processed</h2><ul>{wins or '<li class=m>None yet.</li>'}</ul></section>
</main></body></html>"""
    p = out / "index.html"
    p.write_text(page, encoding="utf-8")
    return p


# ----------------------------------------------------------------------------------------------- 48 h baseline alerts
TIER_ACTION = {
    "red": "Pull {cow} and examine her today: a visible BRD sign was confirmed on camera. Take a rectal temperature and "
           "treat by your BRD protocol if she is febrile (>= 40 C / 104 F).",
    "orange": "Check {cow} on your next pen ride. Over the last 48 h her behavior was the most BRD-like in the pen (reasons "
              "below, with clips). Look at her at the bunk after feeding: appetite, dullness, nose and eyes, ears, breathing. "
              "No visible sign was confirmed by the camera, so this is a check, not a pull.",
    "yellow": "Keep {cow} on the watch list: slightly off the herd over 48 h, not enough to walk out for.",
    "normal": "",
}
TIER_COLOR = {"red": ("var(--crit)", "#fff"), "orange": ("#ec835a", "#fff"), "yellow": ("var(--warn)", "#0b0b0b"),
              "normal": ("var(--line)", "var(--ink)")}


def baseline_card_html(cow: str, tier: str, rule: str, row: Dict, clips: List[str], l2_card: Optional[str]) -> str:
    from .engine import TIER_NAME
    e = html.escape
    bg, fg = TIER_COLOR[tier]
    reasons = "".join(f"<li>{e(x.strip())}</li>" for x in str(row.get("top_reasons", "")).split("|") if x.strip())
    vids = "".join(f"<figure><video src='{e(c)}' poster='{e(c[:-4])}.jpg' controls muted loop playsinline preload='metadata'></video>"
                   f"<figcaption>Level 2 evidence clip {i + 1}</figcaption></figure>" for i, c in enumerate(clips))
    link = f"<p><a href='{e(l2_card)}'>Full Level 2 risk card for {e(cow)}</a> (indicator chart, all evidence the risk agent cited)</p>" if l2_card else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(cow)} Risk Alert</title>
<style>
:root{{--bg:#f9f9f7;--card:#fff;--ink:#0b0b0b;--m:#898781;--line:#e1e0d9;--warn:#fab219;--crit:#d03b3b;--accent:#0f5a6b}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--line:#2c2c2a;--accent:#5fb3c4}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}} main{{max-width:900px;margin:0 auto;padding:16px}}
.badge{{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:650;font-size:13px;background:{bg};color:{fg}}}
h1{{font-size:24px;margin:8px 0 4px}} .m{{color:var(--m)}} section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}}
h2{{font-size:15px;margin:0 0 6px}} .act{{font-size:16px;border-left:4px solid var(--accent);padding-left:12px}} a{{color:var(--accent)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}} figure{{margin:0}} video{{width:100%;border-radius:8px;border:1px solid var(--line)}} figcaption{{font-size:12px;color:var(--m)}}
.kv{{display:grid;grid-template-columns:max-content 1fr;gap:2px 12px;font-size:14px}} .kv div:nth-child(odd){{color:var(--m)}}
</style></head><body><main>
<span class="badge">{e(TIER_NAME[tier])}</span>
<h1>{e(cow)}: BRD risk from the last 48 h</h1>
<div class="m">From the Level 2 ranking (brd_risk.csv). Live video keeps updating this tier.</div>
<section><h2>What to do</h2><p class="act">{e(TIER_ACTION[tier].format(cow=cow))}</p></section>
<section><h2>Why (vs the herd at the same hours, over 48 h)</h2><ul>{reasons or '<li>no reason recorded</li>'}</ul>{link}</section>
<section class="grid">{vids or '<p class="m">No Level 2 clip for this cow.</p>'}</section>
<section><h2>Risk</h2><div class="kv">
<div>Risk index</div><div><b>{float(row.get('risk_score', 0)):.2f}</b> (rank {int(row.get('rank', 0))} of 10)</div>
<div>Top 3 in</div><div>{float(row.get('p_top3') or 0):.0%} of time-block resamples</div>
<div>Confidence</div><div>{e(str(row.get('confidence', '')))} ({float(row.get('hours_observed') or 0):.0f} h observed)</div>
<div>Tier rule</div><div>{e(rule)}</div>
<div>Visible signs</div><div>{e(str(row.get('signs_present') or 'none confirmed'))}</div></div>
<p class="m">Risk is a ranking index (0-1), not a probability of disease. The camera cannot take a temperature: a pen rider confirms.</p></section>
</main></body></html>"""


def _poster(clip: Path, t: float = 4.0) -> None:
    """A still next to the clip (<clip>.jpg): players show the scene before play, and screenshots / PDFs are not blank."""
    ff = _ffmpeg()
    if ff and not clip.with_suffix(".jpg").exists():
        try:
            subprocess.run([ff, "-y", "-loglevel", "error", "-ss", str(t), "-i", str(clip), "-frames:v", "1",
                            "-vf", "scale='min(960,iw)':-2", "-q:v", "4", str(clip.with_suffix(".jpg"))], check=True, timeout=60)
        except Exception:
            pass


def baseline_alerts(state, prior: pd.DataFrame, cfg: Dict, repo: Path, out: Path, log) -> List[Dict]:
    """Tier every cow from the 48 h Level 2 result; cards (with Level 2's clips) for the push tiers."""
    from .engine import TIER_NAME, baseline_tier
    l2 = repo / cfg["baseline"]
    res = []
    for _, r in prior.sort_values("rank").iterrows():
        cow = str(r.cow_id)
        tier, rule = baseline_tier(r, cfg)
        c = state.d["cows"].setdefault(cow, {"prior": float(r.risk_score), "risk": float(r.risk_score), "history": []})
        c["tier"], c["tier_rule"] = tier, rule
        item = {"cow_id": cow, "tier": tier, "tier_name": TIER_NAME[tier], "rule": rule, "risk": float(r.risk_score),
                "rank": int(r["rank"]), "p_top3": float(r.get("p_top3") or 0), "confidence": str(r.get("confidence", "")),
                "reasons": [x.strip() for x in str(r.get("top_reasons", "")).split("|") if x.strip()], "card": None, "clips": []}
        if tier in cfg["tiers"].get("push", []):
            clips = []
            for src in sorted((l2 / "cards" / "clips").glob(f"{cow}_*.mp4")):
                dst = out / "alerts" / f"baseline_{src.name}"
                shutil.copy2(src, dst)
                clips.append(dst.name)
                _poster(dst)
            l2c = l2 / "cards" / f"{cow}.html"
            link = os.path.relpath(l2c, out / "alerts").replace("\\", "/") if l2c.exists() else None
            card = out / "alerts" / f"baseline_{cow}.html"
            card.write_text(baseline_card_html(cow, tier, rule, r.to_dict(), clips, link), encoding="utf-8")
            item.update({"card": f"alerts/{card.name}", "clips": [f"alerts/{c}" for c in clips],
                         "action": TIER_ACTION[tier].format(cow=cow)})
            a = {"cow_id": cow, "trigger": f"48 h risk: {TIER_NAME[tier]}", "risk": float(r.risk_score),
                 "window_start": "48 h baseline", "card": item["card"], "action": item["action"]}
            a["sent"] = deliver({**a, "trigger": a["trigger"]}, a["card"], cfg, log)
            e = html.escape
            body = (f"<h2 style='margin:0'>{e(cow)}: {e(TIER_NAME[tier])}</h2><p><b>What to do:</b> {e(item['action'])}</p>"
                    f"<p><b>Why this tier:</b> {e(rule)}</p><p><b>Over the last 48 h, vs the herd at the same hours:</b></p><ul>"
                    + "".join(f"<li>{e(x)}</li>" for x in item["reasons"][:3]) +
                    f"</ul><p>Risk {float(r.risk_score):.2f}, rank {int(r['rank'])} of 10. Two video clips and the alert card are attached. "
                    f"Risk is a ranking index, not a diagnosis.</p>")
            a["email"] = write_email(out, cfg, f"BRD alert: {cow}, {TIER_NAME[tier]} (48 h risk)", body,
                                     f"{cow}: {TIER_NAME[tier]}\n{item['action']}", [card] + [out / "alerts" / c for c in clips],
                                     f"baseline_{cow}", log)
            state.d["alerts"].append(a)
            c["last_alert"] = None                  # the baseline alert does not start a live cooldown
            log(f"48 h risk alert: {cow} -> {TIER_NAME[tier]} ({rule})")
        res.append(item)
    state.d["baseline"] = res
    e = html.escape                                 # the daily BRD report: every cow's tier in one email
    colors = {"red": "#d03b3b", "orange": "#ec835a", "yellow": "#fab219", "normal": "#cccccc"}
    rows = "".join(f"<tr><td><b>{e(b['cow_id'])}</b></td><td><span style='background:{colors[b['tier']]};padding:1px 8px;border-radius:9px'>"
                   f"{e(b['tier_name'])}</span></td><td align=right>{b['risk']:.2f}</td><td align=right>{b['rank']}</td>"
                   f"<td>{e(b['reasons'][0]) if b['reasons'] else ''}</td></tr>" for b in res)
    push = [b for b in res if b["tier"] in cfg["tiers"].get("push", [])]
    body = (f"<h2 style='margin:0'>Pen 6 daily BRD report</h2><p>{len(push)} cow(s) to check today: "
            f"{e(', '.join(b['cow_id'] for b in push)) or 'none'}. Each has its own alert email with video clips.</p>"
            f"<table cellpadding=4 style='border-collapse:collapse'><tr><th align=left>cow</th><th align=left>tier</th><th>risk</th>"
            f"<th>rank</th><th align=left>top reason (48 h, vs the herd at the same hours)</th></tr>{rows}</table>"
            f"<p>Risk is a ranking index (0-1), not a probability of disease. Live video keeps updating these tiers.</p>")
    rp = l2 / "brd_risk.csv"
    write_email(out, cfg, f"Pen 6 daily BRD report: {len(push)} cow(s) to check", body,
                "Pen 6 daily BRD report\n" + "\n".join(f"{b['cow_id']}: {b['tier_name']} ({b['risk']:.2f})" for b in res),
                [rp] if rp.exists() else [], "daily_brd_report", log)
    return res
