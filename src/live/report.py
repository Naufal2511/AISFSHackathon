"""report.html: every decision the live system made, with its evidence, on one page (opens from disk).

For each window: the alerts (action, clip, frame, sign crops), then EVERY cow (photo, how her id was found, her
numbers vs the herd in the same window, risk before -> after, signs, decision and why), then EVERY crop the VLM
looked at (what it saw, the measurement check, the confirming second look next to a herdmate control crop).
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict

import pandas as pd

E = html.escape
IND = {"feeding_share": ("feeding", "%"), "bunk_zone_share": ("at bunk", "%"), "lying_share": ("lying", "%"),
       "isolation_share": ("alone", "%"), "distance_m_per_h": ("walked", "m/h")}
SIGN_TEXT = {"nasal_discharge": "nasal discharge", "ocular_discharge": "eye discharge", "ears_droop": "drooping ears",
             "head_low": "head low while standing"}


def _read(p: Path, kind="csv"):
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame() if kind == "csv" else ({} if kind == "dict" else [])
    if kind == "csv":
        try:
            return pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(v, unit):
    if pd.isna(v):
        return "–"
    return f"{v:.0f} m/h" if unit == "m/h" else f"{100 * v:.0f}%"


def _crop_outcome(r) -> tuple[str, str]:
    """(class, text) for one screened crop."""
    flagged = [s for s in ("nasal_discharge", "ears_droop", "head_low") if str(r.get(s)).lower() == "true"]
    if isinstance(r.get("measurement"), str) and r.get("measurement"):
        return "drop", r["measurement"]
    if isinstance(r.get("confirmed"), str) and r.get("confirmed"):
        c = json.loads(r["confirmed"])
        signs = ", ".join(SIGN_TEXT.get(s, s) for s in c.get("signs", []))
        if c.get("confirmed"):
            return "ok", f"Flagged {signs}; second look CONFIRMED (confidence {c.get('confidence')}): {c.get('why', '')}"
        return "drop", f"Flagged {signs}; second look did NOT confirm: {c.get('why', '')}"
    if flagged:
        return "drop", f"Flagged {', '.join(flagged)}; not confirmed"
    return "none", "Nothing flagged"


TIER_STYLE = {"red": ("#d03b3b", "#fff"), "orange": ("#ec835a", "#fff"), "yellow": ("#fab219", "#0b0b0b"), "normal": ("transparent", "inherit")}
TIER_NAME = {"red": "Pull and examine", "orange": "Check on the next pen ride", "yellow": "Watch list", "normal": "Normal"}


def _baseline_section(state: Dict, cfg: Dict) -> str:
    """The morning list: every cow's tier from the 48 h Level 2 result, the rule, the reasons and Level 2's clips."""
    base = state.get("baseline") or []
    if not base:
        return ""
    t = cfg["tiers"]
    rules = (f"<li><b>Pull and examine</b>: {E(t['red'])}</li>"
             f"<li><b>Check on the next pen ride</b>: rank ≤ {t['orange']['rank_max']}, in the top 3 in ≥ {t['orange']['p_top3_min']:.0%} of "
             f"time-block resamples, confidence {'/'.join(t['orange']['confidence'])} (live: risk ≥ {t['orange']['live_risk_min']:.2f} and top 3)</li>"
             f"<li><b>Watch list</b>: rank ≤ {t['yellow']['rank_max']} or top 3 in ≥ {t['yellow']['p_top3_min']:.0%} of resamples (dashboard only, no push)</li>"
             f"<li><b>Normal</b>: everyone else</li>")
    cards = ""
    for b in base:
        bg, fg = TIER_STYLE[b["tier"]]
        badge = f"<span class='badge' style='background:{bg};color:{fg}'>{E(b['tier_name'])}</span>"
        vids = "".join(f"<figure><video src='{E(c)}' poster='{E(c[:-4])}.jpg' controls muted loop playsinline preload='metadata'></video>"
                       f"<figcaption>Level 2 clip {i + 1}</figcaption></figure>" for i, c in enumerate(b.get("clips", [])))
        reasons = "".join(f"<li>{E(r)}</li>" for r in b.get("reasons", [])[:3])
        if b["tier"] in t.get("push", []):
            cards += (f"<div class='alert' style='border-left-color:{bg}'><div class='ah'>{badge} <b>{E(b['cow_id'])}</b> "
                      f"<span class='m'>risk {b['risk']:.2f} · rank {b['rank']} · top 3 in {b['p_top3']:.0%} · {E(b['confidence'])} confidence · "
                      f"<a href='{E(b['card'])}'>alert card</a></span></div><p><b>Why this tier:</b> {E(b['rule'])}</p>"
                      f"<p class='act'>{E(b.get('action', ''))}</p><ul class='small'>{reasons}</ul><div class='grid'>{vids}</div></div>")
    rows = "".join(f"<tr><td><b>{E(b['cow_id'])}</b></td><td><span class='badge' style='background:{TIER_STYLE[b['tier']][0]};"
                   f"color:{TIER_STYLE[b['tier']][1]}'>{E(b['tier_name'])}</span></td><td class=n>{b['risk']:.2f}</td><td class=n>{b['rank']}</td>"
                   f"<td class=n>{b['p_top3']:.0%}</td><td>{E(b['confidence'])}</td><td class='small'>{E(b['rule'])}</td>"
                   f"<td class='small'>{E(b['reasons'][0]) if b.get('reasons') else ''}</td></tr>" for b in base)
    pushed = sum(b["tier"] in t.get("push", []) for b in base)
    return f"""<section id="baseline"><h2>Alerts from the 48 h BRD risk (the morning list)</h2>
<p class="m">On start, every cow gets a tier from the Level 2 ranking (<code>brd_risk.csv</code>); {pushed} cow(s) got a push alert with Level 2's
video clips. Each live window re-evaluates the tier; moving up is an alert, moving down is logged.</p>
<ul class="small">{rules}</ul>{cards}
<div class="tw"><table><thead><tr><th>cow</th><th>tier</th><th class=n>risk</th><th class=n>rank</th><th class=n>top 3 in</th><th>confidence</th><th>rule</th><th>top reason</th></tr></thead><tbody>{rows}</tbody></table></div></section>"""


def _outbox_section(out: Path, cfg: Dict) -> str:
    """Every email the system wrote, shown as it would arrive (simulated: written to outbox/, not sent)."""
    p = out / "outbox" / "outbox.jsonl"
    if not p.exists():
        return ""
    mails = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    items = ""
    for m in reversed(mails):
        att = " ".join(f"<span class='chip'>📎 {E(a['name'])} ({a['kb']} KB)</span>" for a in m.get("attachments", []))
        items += (f"<details class='mail'><summary><b>{E(m['subject'])}</b> <span class='m small'>to {E(m['to'])} · {E(m['date'])} · "
                  f"{E(m.get('mode', 'simulate'))}</span></summary><div class='mhead small'><div><span class='m'>From</span> {E(m['from'])}</div>"
                  f"<div><span class='m'>To</span> {E(m['to'])}</div><div><span class='m'>Subject</span> {E(m['subject'])}</div></div>"
                  f"<div class='mbody'>{m.get('body_html', '')}</div><div>{att}</div>"
                  f"<p class='small'><a href='{E(m['file'])}'>Open the .eml</a> (a standard email file: opens in Outlook, Mail or Thunderbird)</p></details>")
    mode = (cfg.get("delivery", {}).get("email") or {}).get("mode", "simulate")
    return (f"<section id='outbox'><h2>Email outbox ({'simulated' if mode == 'simulate' else mode})</h2>"
            f"<p class='m'>{len(mails)} email(s) the system wrote for the pen rider and the herd manager: the daily BRD report and one per alert, "
            f"with the evidence attached. In simulate mode they are written as real .eml files to <code>outbox/</code> and never sent; "
            f"<code>delivery.email.mode: smtp</code> plus an SMTP login in <code>.env</code> sends the same messages.</p>{items}</section>")


def build(out: Path, cfg: Dict) -> Path:
    out = Path(out)
    windows = [json.loads(l) for l in open(out / "windows.jsonl", encoding="utf-8")] if (out / "windows.jsonl").exists() else []
    hist = _read(out / "risk_history.csv")
    state = _read(out / "state.json", "dict")
    prior = {c: v.get("prior") for c, v in (state.get("cows") or {}).items()}
    ac = cfg["alert"]
    n_crops = n_flag = n_drop = n_conf = n_alert = 0
    sections = []
    for w in windows:
        wd = out / w.get("dir", f"windows/{pd.Timestamp(w['start']):%Y%m%d_%H%M}")
        ids, feat, dec, signs = (_read(wd / f) for f in ("ids.csv", "features.csv", "decisions.csv", "signs.csv"))
        reasons = {r["cow_id"]: r["reasons"] for r in _read(wd / "reasons.json", "list")}
        thumbs, alerts = _read(wd / "thumbs.json", "dict"), _read(wd / "alerts.json", "list")
        watch = _read(wd / "watch.json", "list")
        h = hist[hist.window_start.astype(str) == str(w["start"])] if len(hist) else pd.DataFrame()
        rel = lambda f: E(f"{wd.relative_to(out).as_posix()}/{f}")          # noqa: E731
        n_alert += len(alerts)
        # ---- alerts
        acards = ""
        for a in alerts:
            ev = a.get("evidence") or {}
            media = ""
            if ev.get("clip"):
                media += f"<figure><video src='alerts/{E(ev['clip'])}' controls muted loop playsinline preload='metadata'></video><figcaption>12 s clip, cropped to the cow</figcaption></figure>"
            if ev.get("frame"):
                media += f"<figure><img src='alerts/{E(ev['frame'])}' loading='lazy' alt=''><figcaption>full frame: where she is</figcaption></figure>"
            for c in ev.get("crops", []):
                media += f"<figure><img src='alerts/{E(c['file'])}' loading='lazy' alt=''><figcaption>{E(c['what'])}</figcaption></figure>"
            if not media:
                media = "<p class='m'>No clip: the raw video of this window was not on the machine that scored it.</p>"
            d = dec[dec.cow_id == a["cow_id"]].why.iloc[0] if len(dec) and (dec.cow_id == a["cow_id"]).any() else ""
            acards += (f"<div class='alert {'sign' if a['trigger'] == 'sign' else 'beh'}'><div class='ah'><span class='badge'>"
                       f"{'Check now' if a['trigger'] == 'sign' else 'Watch closely'}</span> <b>{E(a['cow_id'])}</b> "
                       f"<span class='m'>risk {a['risk_before']:.2f} → {a['risk']:.2f} · <a href='{E(a['card'])}'>alert card</a></span></div>"
                       f"<p><b>Why:</b> {E(str(d))}</p><p class='act'>{E(a['action'])}</p><div class='grid'>{media}</div>"
                       f"<p class='m'>Identity: {E(str(a.get('identity', '')))}</p></div>")
        # ---- check-list cows seen in this window (fresh video every window)
        wcards = ""
        for wv in watch:
            bg, fg = TIER_STYLE.get(wv["tier"], TIER_STYLE["normal"])
            mo = wv.get("moment") or {}
            media = ""
            if wv.get("clip"):
                media += f"<figure><video src='{rel(wv['clip'])}' controls muted loop playsinline preload='metadata'></video><figcaption>12 s from this window, cropped to her</figcaption></figure>"
            if wv.get("frame"):
                media += f"<figure><img src='{rel(wv['frame'])}' loading='lazy' alt=''><figcaption>full frame: where she is ({E(str(mo.get('camera', '')))}, {E(str(mo.get('time', ''))[11:19])})</figcaption></figure>"
            dwhy = dec[dec.cow_id == wv["cow_id"]].why.iloc[0] if len(dec) and (dec.cow_id == wv["cow_id"]).any() else ""
            wcards += (f"<div class='alert' style='border-left-color:{bg}'><div class='ah'><span class='badge' style='background:{bg};color:{fg}'>"
                       f"{E(TIER_NAME.get(wv['tier'], wv['tier']))}</span> <b>{E(wv['cow_id'])}</b> <span class='m'>risk now {wv['risk']:.2f}</span></div>"
                       f"<p class='small'><b>This window:</b> {E(str(dwhy))}</p><div class='grid'>{media}</div></div>")
        # ---- every cow
        rows = ""
        herd = {}
        if len(feat):
            ok = feat.seen_min >= cfg["risk"]["min_seen_min"]
            herd = {k: feat.loc[ok, k].median() for k in IND if k in feat}
        idx_ids = ids.set_index("cow_id") if len(ids) else pd.DataFrame()
        hh = h.set_index("cow_id") if len(h) else pd.DataFrame()
        dd = dec.set_index("cow_id") if len(dec) else pd.DataFrame()
        order = hh.sort_values("rank").index.tolist() if len(hh) else (feat.cow_id.tolist() if len(feat) else [])
        for cow in order:
            f = feat[feat.cow_id == cow].iloc[0] if len(feat) and (feat.cow_id == cow).any() else None
            th = thumbs.get(cow)
            photo = (f"<img class='th' src='{rel(th['file'])}' loading='lazy' alt='{E(cow)}' title='{E(th['camera'])} {E(th['time'])}'>"
                     if th else "<div class='th none'>no close view</div>")
            how = E(str(idx_ids.at[cow, "how"])) if cow in idx_ids.index else ""
            local = E(str(idx_ids.at[cow, "local_id"])) if cow in idx_ids.index else ""
            inds = ""
            if f is not None:
                for k, (lab, unit) in IND.items():
                    if k not in f:
                        continue
                    z = f.get(k + "__z")
                    zs = "" if pd.isna(z) else f"{z:+.1f}"
                    sign = cfg["risk"]["indicators"].get(k, {}).get("sign", 1)
                    hot = (not pd.isna(z)) and sign * z >= 1.0
                    inds += (f"<tr class='{'hot' if hot else ''}'><td>{lab}</td><td class=n>{_fmt(f[k], unit)}</td>"
                             f"<td class='n m'>{_fmt(herd.get(k), unit)}</td><td class=n>{zs}</td></tr>")
                inds = f"<table class='mini'><tr><th></th><th class=n>her</th><th class=n>herd</th><th class=n>z</th></tr>{inds}</table>"
            r = hh.loc[cow] if cow in hh.index else None
            risk = (f"<b>{r.risk_before:.2f} → {r.risk:.2f}</b><br><span class='{'up' if r.delta > 0.005 else 'm'}'>{r.delta:+.2f}</span>"
                    f" <span class='m'>rank {int(r['rank'])}</span><br><span class='m'>48 h: {prior.get(cow, float('nan')):.2f}</span>"
                    if r is not None else "–")
            seen = f"{f.seen_min:.0f} min" if f is not None else "–"
            sg = E(str(r.signs)) if r is not None and isinstance(r.signs, str) and r.signs else "none confirmed"
            decision = dd.at[cow, "decision"] if cow in dd.index else "–"
            why = E(str(dd.at[cow, "why"])) if cow in dd.index else ""
            rows += (f"<tr class='{'alrt' if decision == 'ALERT' else ''}'><td>{photo}</td><td><b>{E(cow)}</b><br><span class='m'>window id {local}</span>"
                     f"<br><span class='m small'>{how}</span></td><td class=n>{seen}</td><td>{inds}</td><td>{risk}</td><td>{sg}</td>"
                     f"<td><b>{E(decision)}</b><br><span class='small'>{why}</span></td></tr>")
        # ---- every crop
        crops = ""
        if len(signs):
            for _, s in signs.iterrows():
                n_crops += 1
                cls, text = _crop_outcome(s)
                n_flag += cls != "none"
                n_drop += cls == "drop"
                n_conf += cls == "ok"
                img = f"<img src='{rel(s['path_file'])}' loading='lazy' alt=''>" if isinstance(s.get("path_file"), str) else ""
                ctrl = (f"<figure><img src='{rel(s['control_file'])}' loading='lazy' alt=''><figcaption>control: herdmate closest in time</figcaption></figure>"
                        if isinstance(s.get("control_file"), str) else "")
                t = pd.to_datetime(s["wall_t"], unit="s")
                sp = f", tracker speed {s['speed_mps']:.2f} m/s" if "speed_mps" in s and pd.notna(s["speed_mps"]) else ""
                crops += (f"<div class='crop {cls}'><div class='pair'><figure>{img}<figcaption>{E(str(s['cow_id']))} · {E(str(s['camera']))} · "
                          f"{t:%H:%M:%S}</figcaption></figure>{ctrl}</div><p class='small'><b>VLM saw:</b> {E(str(s.get('screen', '')))}{E(sp)}</p>"
                          f"<p class='small out'><b>Outcome:</b> {E(text)}</p></div>")
        else:
            crops = "<p class='m'>No close crops were checked in this window (night, nobody close enough, or no VLM key).</p>"
        sections.append(f"""<section id="w{E(pd.Timestamp(w['start']).strftime('%Y%m%d%H%M'))}">
<h2>{pd.Timestamp(w['start']):%b %d %H:%M} – {pd.Timestamp(w['end']):%H:%M}</h2>
<p class='m'>{w.get('cows', 0)} cows · cameras {E(', '.join(w.get('cameras', [])))} · {len(alerts)} alert(s) · signs: {E(str(w.get('signs', '')))} · scored in {w.get('seconds', 0):.0f} s</p>
{('<h3>Alerts</h3>' + acards) if acards else "<p class='m'>No new alert in this window. Each cow's reason is below.</p>"}
{('<h3>Check-list cows in this window (fresh video)</h3>' + wcards) if wcards else ''}
<h3>Every cow: evidence and decision</h3>
<div class='tw'><table class='cows'><thead><tr><th>close view</th><th>cow · how the id was found</th><th class=n>seen</th><th>her numbers vs the herd in this window</th><th>risk</th><th>visible signs</th><th>decision and why</th></tr></thead><tbody>{rows}</tbody></table></div>
<h3>Every crop the VLM checked</h3><div class='crops'>{crops}</div>
</section>""")
    # ---- risk across windows
    trow = ""
    if len(hist):
        cols = [str(w["start"]) for w in windows]
        last = hist.groupby("cow_id").risk.last().sort_values(ascending=False)
        alerted = {(a["cow_id"], str(a["window_start"])) for a in state.get("alerts", [])}
        for cow in last.index:
            cells = ""
            for c in cols:
                r = hist[(hist.cow_id == cow) & (hist.window_start.astype(str) == c)]
                if not len(r):
                    cells += "<td class=n m>–</td>"
                    continue
                r = r.iloc[-1]
                mark = " <span class='badge sm'>alert</span>" if (cow, c) in alerted else ""
                cells += f"<td class=n><b>{r.risk:.2f}</b> <span class='{'up' if r.delta > 0.005 else 'm'}'>{r.delta:+.2f}</span>{mark}</td>"
            trow += f"<tr><td><b>{E(cow)}</b></td><td class='n m'>{prior.get(cow, float('nan')):.2f}</td>{cells}</tr>"
        thead = "".join(f"<th class=n><a href='#w{pd.Timestamp(c):%Y%m%d%H%M}'>{pd.Timestamp(c):%H:%M}</a></th>" for c in cols)
        trow = f"<table class='risk'><thead><tr><th>cow</th><th class=n>48 h baseline</th>{thead}</tr></thead><tbody>{trow}</tbody></table>"
    tiles = [(len(windows), "windows scored"), (n_crops, "close crops checked by the VLM"), (n_flag, "flags raised by screening"),
             (n_drop, "flags rejected by a check"), (n_conf, "signs confirmed"), (n_alert, "alerts sent")]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live Evidence Report</title>
<style>
:root{{--bg:#f9f9f7;--card:#fff;--ink:#0b0b0b;--ink2:#52514e;--m:#898781;--line:#e1e0d9;--warn:#fab219;--crit:#d03b3b;--good:#0ca30c;--accent:#0f5a6b;--hot:#fff4d6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2c2c2a;--accent:#5fb3c4;--hot:#3a2f12}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1240px;margin:0 auto;padding:16px}} h1{{font-size:22px;margin:4px 0}} h2{{font-size:18px;margin:0 0 2px}} h3{{font-size:14px;margin:18px 0 8px;color:var(--ink2)}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:14px 0}}
.m{{color:var(--m)}} .small{{font-size:12px}} .up{{color:var(--crit);font-weight:600}} a{{color:var(--accent)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}} .tile{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}}
.tile .v{{font-size:24px;font-weight:650}} .tile .k{{font-size:12px;color:var(--ink2)}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{font-size:12px;color:var(--ink2);font-weight:600}} .n{{text-align:right;font-variant-numeric:tabular-nums}}
table.mini td,table.mini th{{padding:1px 6px;border:0;font-size:12px}} table.mini tr.hot td{{background:var(--hot);font-weight:600}}
.th{{width:96px;height:96px;object-fit:cover;border-radius:8px;border:1px solid var(--line);display:block}} .th.none{{display:flex;align-items:center;justify-content:center;font-size:11px;color:var(--m);text-align:center}}
tr.alrt td{{background:color-mix(in srgb,var(--warn) 12%,transparent)}}
table.cows{{min-width:1080px}} table.cows td:nth-child(2){{width:170px}} table.cows td:last-child{{width:280px}} table.risk{{min-width:520px}}
.badge{{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;font-weight:650;background:var(--warn);color:#0b0b0b}} .badge.sm{{font-size:10px;padding:0 6px}}
.alert{{border:1px solid var(--line);border-left:5px solid var(--warn);border-radius:8px;padding:10px 12px;margin:8px 0}} .alert.sign{{border-left-color:var(--crit)}} .alert.sign .badge{{background:var(--crit);color:#fff}}
.act{{border-left:3px solid var(--accent);padding-left:10px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px}} figure{{margin:0}} figure img,figure video{{width:100%;border-radius:8px;border:1px solid var(--line)}} figcaption{{font-size:11px;color:var(--m)}}
.crops{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}} .crop{{border:1px solid var(--line);border-radius:8px;padding:8px}}
.crop .pair{{display:grid;grid-template-columns:1fr 1fr;gap:6px}} .crop img{{height:150px;object-fit:cover}}
.crop.ok{{border-color:var(--crit)}} .crop.drop .out{{color:var(--ink2)}} .crop.ok .out{{color:var(--crit);font-weight:600}}
.legend span{{margin-right:14px}} details.mail{{border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:6px 0}} details.mail summary{{cursor:pointer}}
.mhead{{border-bottom:1px solid var(--line);padding:6px 0;margin:6px 0}} .mbody{{padding:6px 0}} .mbody table td,.mbody table th{{border-bottom:1px solid var(--line)}}
.chip{{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 8px;font-size:12px;margin:2px}}
</style></head><body><main>
<h1>Live early warning: every decision and its evidence</h1>
<p class="m">Each window is 5 min (or an hour, in the Day 2 replay) of video from the 3 cameras, scored by <code>live_watch.py</code>.
Per window you see the alerts with their clips, then EVERY cow with her numbers vs the herd in the same window, how her
risk moved and why she was or was not alerted, then EVERY close crop the vision model checked for clinical signs and what
happened to each flag. Risk is a ranking index (0-1, the 48 h Level 2 score as the start), not a probability of disease.</p>
<div class="tiles">{''.join(f"<div class='tile'><div class='v'>{v}</div><div class='k'>{E(k)}</div></div>" for v, k in tiles)}</div>
{_baseline_section(state, cfg)}
{_outbox_section(out, cfg)}
<section><h2>Risk per cow across windows</h2><p class="m small">Alert rules: a new confirmed sign; or risk ≥ {ac['score']:.2f} and up ≥ {ac['rise']:.2f} in a window;
or 2 indicators at z ≥ {ac['evidence_z']:.0f} while in the top 3. One alert per cow per {ac['cooldown_min']} min unless a new sign appears.
Highlighted rows in the cow tables are indicators at z ≥ 1 in the BRD direction (less feeding / bunk / walking, more lying / alone).</p>
<div class="tw">{trow or "<p class='m'>No window scored yet.</p>"}</div></section>
{''.join(reversed(sections))}
</main></body></html>"""
    p = out / "report.html"
    p.write_text(page, encoding="utf-8")
    return p
