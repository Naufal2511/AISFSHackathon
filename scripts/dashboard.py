#!/usr/bin/env python3
"""Local live dashboard: agent memory, reasoning traces, ID checks, corrections and the
per-minute activity grid (before / after the agent). Reads the output folders on every
refresh, so it updates while the pipeline and the agent are running.

  python scripts/dashboard.py --run outputs/run_d1_1600 --agent outputs/agent_d1_1600 --port 8765

Serves only on 127.0.0.1 and only files under the repository's outputs/ folder: event data
stays on this machine.
"""
import argparse
import csv
import glob
import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ALLOWED = os.path.join(ROOT, "outputs")


def read_csv(path, limit=None):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def tail(path, n=40):
    if not os.path.exists(path):
        return []
    with open(path, errors="replace") as f:
        lines = [l.rstrip() for l in f if "warn" not in l.lower() and "xformers" not in l.lower()]
    return lines[-n:]


def state(run_dir, agent_dir):
    s = {"run": run_dir, "agent": agent_dir}
    s["summary"] = json.load(open(os.path.join(agent_dir, "agent_summary.json"))) \
        if os.path.exists(os.path.join(agent_dir, "agent_summary.json")) else None
    s["log"] = tail(os.path.join(agent_dir, "agent_log.txt"))
    cards_p = os.path.join(agent_dir, "memory", "cards.json")
    s["cards"] = json.load(open(cards_p)) if os.path.exists(cards_p) else []
    traces = []
    for p in sorted(glob.glob(os.path.join(agent_dir, "traces", "*.json")), key=os.path.getmtime, reverse=True):
        try:
            traces.append(json.load(open(p)))
        except Exception:
            pass
    s["traces"] = traces
    s["id_checks"] = sorted(glob.glob(os.path.join(agent_dir, "id_checks", "*.jpg")))
    s["corrections"] = read_csv(os.path.join(agent_dir, "corrections.csv"))
    s["activity_before"] = read_csv(os.path.join(run_dir, "behavior", "activity.csv"))
    s["activity_after"] = read_csv(os.path.join(agent_dir, "behavior", "activity.csv"))
    s["budgets"] = read_csv(os.path.join(agent_dir, "behavior", "time_budgets.csv")) or \
        read_csv(os.path.join(run_dir, "behavior", "time_budgets.csv"))
    s["enrollment_sheet"] = os.path.join(agent_dir, "memory", "enrollment_sheet.jpg") \
        if os.path.exists(os.path.join(agent_dir, "memory", "enrollment_sheet.jpg")) else None
    s["dashboard_sample"] = os.path.join(run_dir, "multicam_dashboard_sample.jpg") \
        if os.path.exists(os.path.join(run_dir, "multicam_dashboard_sample.jpg")) else None
    return s


class H(BaseHTTPRequestHandler):
    run_dir = agent_dir = ""

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "max-age=600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            return self._send(200, json.dumps(state(self.run_dir, self.agent_dir), default=str).encode(),
                              "application/json")
        if u.path == "/file":
            p = urllib.parse.parse_qs(u.query).get("p", [""])[0]
            full = os.path.realpath(os.path.join(ROOT, p))
            if not full.startswith(ALLOWED + os.sep) or not os.path.isfile(full):
                return self._send(404, b"not found", "text/plain")
            ctype = "image/jpeg" if full.lower().endswith((".jpg", ".jpeg")) else "text/plain; charset=utf-8"
            return self._send(200, open(full, "rb").read(), ctype, cache=True)
        self._send(404, b"not found", "text/plain")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pen 6 Agent Monitor</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d2330;--muted:#667085;--line:#e3e6ec;--accent:#2f6fde;--ok:#1f9d55;--warn:#c47f00;--bad:#d64545;--chip:#eef2f8}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#11141a;--panel:#191d25;--ink:#e6e9ef;--muted:#98a2b3;--line:#2a303b;--accent:#6c9cff;--ok:#3ccf7c;--warn:#f0b43c;--bad:#ff6b6b;--chip:#232a36}}
:root[data-theme="dark"]{--bg:#11141a;--panel:#191d25;--ink:#e6e9ef;--muted:#98a2b3;--line:#2a303b;--accent:#6c9cff;--ok:#3ccf7c;--warn:#f0b43c;--bad:#ff6b6b;--chip:#232a36}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0}h2{font-size:14px;margin:0 0 10px}
.stat{display:flex;flex-direction:column}.stat b{font-size:18px}.stat span{color:var(--muted);font-size:12px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--ok);display:inline-block;margin-right:6px;animation:p 2s infinite}@keyframes p{50%{opacity:.3}}
nav{display:flex;gap:4px;padding:8px 16px;flex-wrap:wrap}nav button{border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:6px 12px;border-radius:8px;cursor:pointer}
nav button.on{background:var(--accent);border-color:var(--accent);color:#fff}
main{padding:0 16px 32px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:14px;overflow:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.card{border:1px solid var(--line);border-radius:10px;padding:10px}.card h3{margin:0 0 6px;font-size:14px;display:flex;justify-content:space-between}
.thumbs{display:flex;gap:4px;overflow-x:auto;margin:6px 0}.thumbs img{height:84px;border-radius:6px;cursor:zoom-in}
.chip{display:inline-block;background:var(--chip);border-radius:999px;padding:1px 8px;font-size:12px;margin:2px 4px 2px 0}
.muted{color:var(--muted)}.small{font-size:12px}
table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;font-size:12px;white-space:nowrap}
.heat td.c{width:30px;height:24px;padding:0;border:1px solid var(--bg)}.heat th{font-weight:500;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;margin-right:10px;font-size:12px}.legend i{width:12px;height:12px;border-radius:3px;margin-right:4px;display:inline-block}
.trace{border:1px solid var(--line);border-radius:10px;margin-bottom:8px}.trace summary{padding:8px 10px;cursor:pointer;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.trace .body{padding:0 12px 12px}.step{border-left:3px solid var(--line);padding:4px 10px;margin:8px 0}.step.final{border-color:var(--accent)}
pre{background:var(--chip);padding:8px;border-radius:6px;white-space:pre-wrap;font-size:12px;margin:4px 0;max-height:260px;overflow:auto}
.step img{max-width:100%;max-height:340px;border-radius:6px;cursor:zoom-in}
.b-relabel{background:#2f6fde22;color:var(--accent)}.b-uncertain{background:#c47f0022;color:var(--warn)}.b-confirm{background:#1f9d5522;color:var(--ok)}
.b-set_behavior{background:#7a4cd922;color:#8b5cf6}.b-escalate{background:#d6454522;color:var(--bad)}.b-note{background:var(--chip)}
.pill{border-radius:6px;padding:1px 6px;font-size:12px;font-weight:600}
.sev5,.sev4{color:var(--bad)}.sev3{color:var(--warn)}
#zoom{position:fixed;inset:0;background:#000c;display:none;align-items:center;justify-content:center;z-index:9;cursor:zoom-out}#zoom img{max-width:96vw;max-height:94vh}
.log{font:12px ui-monospace,Menlo,monospace;white-space:pre-wrap}
@media (max-width:600px){header{gap:10px}.grid{grid-template-columns:1fr}}
</style></head><body>
<header><h1><span class="dot" id="live"></span>Pen 6 agent monitor</h1>
<div class="stat"><b id="s-inv">-</b><span>investigations</span></div>
<div class="stat"><b id="s-dec">-</b><span>decisions</span></div>
<div class="stat"><b id="s-lat">-</b><span>avg s / investigation</span></div>
<div class="stat"><b id="s-be">-</b><span>reasoning backend</span></div>
<div class="stat"><b id="s-upd">-</b><span>last refresh</span></div></header>
<nav id="tabs"></nav><main id="main"></main><div id="zoom" onclick="this.style.display='none'"><img></div>
<script>
const BEH={"lying":"#4c78a8","standing":"#9ecae9","walking":"#f58518","feeding":"#54a24b","drinking":"#2bb3c9","ruminating (lying)":"#7b4fbd","ruminating (standing)":"#b279a2","grooming":"#e45756","social interaction":"#eeca3b","other / not visible":"#bab0ac","no_video":"#00000000"};
const TABS=["Activity","Agent traces","Memory cards","ID checks","Corrections","Log"];let tab=localStorageGet("tab")||"Activity",S=null,actView="after",open=new Set();
function localStorageGet(k){try{return localStorage.getItem(k)}catch(e){return null}}function localStorageSet(k,v){try{localStorage.setItem(k,v)}catch(e){}}
const f=p=>"/file?p="+encodeURIComponent(p);const esc=s=>String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function zoom(src){const z=document.getElementById("zoom");z.querySelector("img").src=src;z.style.display="flex"}
function nav(){document.getElementById("tabs").innerHTML=TABS.map(t=>`<button class="${t==tab?"on":""}" onclick="tab='${t}';localStorageSet('tab',tab);render()">${t}${t=="Agent traces"&&S?` (${S.traces.length})`:""}</button>`).join("")}
function header(){const s=S.summary,tr=S.traces;const dec={};tr.forEach(t=>{const d=(t.final||{}).decision;dec[d]=(dec[d]||0)+1});
document.getElementById("s-inv").textContent=tr.length;document.getElementById("s-dec").textContent=Object.entries(dec).map(([k,v])=>`${v} ${k}`).join(" · ")||"-";
document.getElementById("s-lat").textContent=tr.length?(tr.reduce((a,t)=>a+(t.seconds||0),0)/tr.length).toFixed(1):"-";
document.getElementById("s-be").textContent=(tr[0]||{}).backend||(s||{}).backend||"-";document.getElementById("s-upd").textContent=new Date().toLocaleTimeString()}
function activity(){const rows=actView=="after"&&S.activity_after.length?S.activity_after:S.activity_before;
if(!rows.length)return `<div class="panel muted">No activity.csv yet.</div>`;
const mins=[...new Set(rows.map(r=>r.minute_start))].sort(),cows=[...new Set(rows.map(r=>r.cow_id))].sort();const m={};rows.forEach(r=>m[r.cow_id+"|"+r.minute_start]=r);
let h=`<div class="panel"><h2>Behavior per cow per minute</h2><div style="margin-bottom:8px">
<button onclick="actView='before';render()" ${actView=="before"?'style="font-weight:700"':""}>pipeline output</button>
<button onclick="actView='after';render()" ${actView=="after"?'style="font-weight:700"':""} ${S.activity_after.length?"":"disabled"}>after agent corrections</button>
<span class="muted small"> showing: ${actView=="after"&&S.activity_after.length?"after agent":"pipeline"} · hover a cell for share / confidence / evidence</span></div>
<table class="heat"><tr><th></th>${mins.map(x=>`<th>${x.slice(11,16)}</th>`).join("")}</tr>`;
cows.forEach(c=>{h+=`<tr><th>${c}</th>`+mins.map(x=>{const r=m[c+"|"+x];if(!r)return `<td class="c"></td>`;
const col=BEH[r.behavior]||"#999",op=Math.max(.35,Math.min(1,+r.confidence||0));const est=(r.evidence||"").startsWith("estimated")||r.evidence=="not_visible";
return `<td class="c" title="${c} ${x.slice(11,16)}\n${r.behavior} ${(+r.share*100).toFixed(0)}% of minute\nconfidence ${(+r.confidence).toFixed(2)}\nevidence ${r.evidence}\nseen ${r.seconds_observed}s on ${r.cameras}" style="background:${col};opacity:${op};${est?"background-image:repeating-linear-gradient(45deg,#fff5 0 3px,transparent 3px 6px)":""}"></td>`}).join("")+"</tr>"});
h+=`</table><div class="legend" style="margin-top:8px">${Object.entries(BEH).filter(([k])=>k!="no_video").map(([k,v])=>`<span><i style="background:${v}"></i>${k}</span>`).join("")}<span class="muted">striped = estimated, faded = low confidence</span></div></div>`;
if(S.budgets.length){const keys=Object.keys(S.budgets[0]).slice(0,10);h+=`<div class="panel"><h2>Time budgets</h2><table><tr>${keys.map(k=>`<th>${esc(k)}</th>`).join("")}</tr>${S.budgets.map(r=>`<tr>${keys.map(k=>`<td>${esc(r[k])}</td>`).join("")}</tr>`).join("")}</table></div>`}
if(S.dashboard_sample)h+=`<div class="panel"><h2>Three cameras + pen map (first step)</h2><img src="${f(S.dashboard_sample)}" style="max-width:100%;border-radius:8px;cursor:zoom-in" onclick="zoom(this.src)"></div>`;return h}
function traces(){if(!S.traces.length)return `<div class="panel muted">No investigations yet - the agent writes one trace per flag.</div>`;
return `<div class="panel">`+S.traces.map((t,i)=>{const fl=t.flag,fi=t.final||{},id=fl.kind+fl.t0+(fl.track_id??"")+fl.camera;
const tm=new Date(fl.t0*1000).toISOString().slice(11,19);
const steps=(t.steps||[]).map(s=>s.tool?`<div class="step"><b>Thought:</b> ${esc(s.thought)}<br><b>Tool:</b> <code>${esc(s.tool)}</code> <span class="muted small">${esc(JSON.stringify(s.args))} · evidence: ${esc(s.evidence_kind)}</span><pre>${esc(s.result)}</pre>${(s.images||[]).map(p=>`<img src="${f(p)}" onclick="zoom(this.src)">`).join("")}</div>`
:`<div class="step final"><b>Decision:</b> ${esc(s.thought)}<pre>${esc(JSON.stringify(s.final,null,1))}</pre><b>Self-check:</b> ${s.verified?'<span style="color:var(--ok)">passed</span>':'<span style="color:var(--bad)">failed: '+esc((s.check||[]).join("; "))+"</span>"}</div>`).join("");
return `<details class="trace" ${open.has(id)?"open":""} ontoggle="this.open?open.add('${id}'):open.delete('${id}')"><summary><span class="sev${fl.severity}"><b>${esc(fl.kind)}</b></span><span>${tm}</span><span class="chip">${esc((fl.cow_ids||[]).slice(0,3).join(", "))}</span>${fl.camera?`<span class="chip">${fl.camera}${fl.track_id!=null?" t"+fl.track_id:""}</span>`:""}<span class="pill b-${fi.decision}">${esc(fi.decision)}</span><span class="muted small">${t.seconds}s · ${(t.steps||[]).length} steps</span><span class="small">${esc(fi.summary||"")}</span></summary>
<div class="body"><div class="muted small">Trigger: ${esc(fl.detail)} · backend ${esc(t.backend)}</div>${steps}</div></details>`}).join("")+`</div>`}
function memory(){if(!S.cards.length)return `<div class="panel muted">No memory cards yet (enrollment runs first).</div>`;
let h=`<div class="grid">`+S.cards.filter(c=>!c.same_as).map(c=>`<div class="card"><h3><span>${c.cow_id}</span><span class="small ${c.confirmed?"":"muted"}">${c.confirmed?"✓ human-confirmed":"not confirmed"}</span></h3>
<div class="thumbs">${(c.references||[]).map(p=>`<img src="${f(p)}" title="${esc(p.split("/").pop())}" onclick="zoom(this.src)">`).join("")}</div>
${c.ear_tag?`<span class="chip">ear tag: ${esc(c.ear_tag)}</span>`:""}${(c.stats.cameras||[]).map(x=>`<span class="chip">${x}</span>`).join("")}
<div class="small" style="margin:4px 0">${esc(c.traits)||'<span class="muted">no traits yet</span>'}</div>
<div class="small muted">last seen ${esc((c.last_seen||{}).time||"-")} on ${esc((c.last_seen||{}).camera||"-")}</div>
${(c.notes||[]).slice(-4).map(n=>`<div class="small">• <span class="muted">${esc(n.time)}</span> ${esc(n.text)}</div>`).join("")}</div>`).join("")+`</div>`;
if(S.enrollment_sheet)h+=`<div class="panel" style="margin-top:12px"><h2>Enrollment sheet (confirm once)</h2><img src="${f(S.enrollment_sheet)}" style="max-width:100%;cursor:zoom-in" onclick="zoom(this.src)"></div>`;return h}
function idchecks(){if(!S.id_checks.length)return `<div class="panel muted">No ID-check sheets yet.</div>`;
return S.id_checks.slice().reverse().map(p=>`<div class="panel"><h2>${esc(p.split("/").pop())}</h2><div class="muted small">row = cow · card reference | cam1 | cam2 | cam3 box carrying that ID · red label = looks closer to another card</div><img src="${f(p)}" style="max-width:100%;cursor:zoom-in" onclick="zoom(this.src)"></div>`).join("")}
function corrections(){if(!S.corrections.length)return `<div class="panel muted">No corrections yet.</div>`;const keys=["flag","flag_time","decision","camera","track_id","cow_id","new_cow_id","behavior","t0","t1","confidence","summary"];
return `<div class="panel"><table><tr>${keys.map(k=>`<th>${k}</th>`).join("")}</tr>${S.corrections.map(r=>`<tr>${keys.map(k=>`<td>${k=="decision"?`<span class="pill b-${r[k]}">${esc(r[k])}</span>`:esc(r[k])}</td>`).join("")}</tr>`).join("")}</table></div>`}
function log(){return `<div class="panel"><h2>Agent log (tail)</h2><div class="log">${esc(S.log.join("\n"))||"(empty)"}</div></div>`}
function render(){if(!S)return;nav();header();const y=window.scrollY;document.getElementById("main").innerHTML=
{"Activity":activity,"Agent traces":traces,"Memory cards":memory,"ID checks":idchecks,"Corrections":corrections,"Log":log}[tab]();window.scrollTo(0,y)}
let last="";async function tick(){try{const txt=await (await fetch("/api/state")).text();document.getElementById("live").style.background="var(--ok)";document.getElementById("s-upd").textContent=new Date().toLocaleTimeString();if(txt!==last){last=txt;S=JSON.parse(txt);render()}}catch(e){document.getElementById("live").style.background="var(--bad)"}}
tick();setInterval(tick,4000);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/run_d1_1600")
    ap.add_argument("--agent", default="outputs/agent_d1_1600")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    os.chdir(ROOT)
    H.run_dir, H.agent_dir = a.run, a.agent
    print(f"dashboard: http://127.0.0.1:{a.port}  (run {a.run}, agent {a.agent})")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
