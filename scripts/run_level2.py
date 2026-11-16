#!/usr/bin/env python3
"""Level 2: BRD risk ranking with explanations, evidence and clips.

  python scripts/run_level2.py --runs outputs/level1_d1_1600_h outputs/level1_d1_1950_h \
      --activity path/to/activity.json --out outputs/level2

  --activity   per cow per minute behavior (JSON or CSV; see src/risk/data.py). Default: each run's best activity.csv
  --skip-agent behavior score only (no TAMU calls)
  --deep N     full visual review for the top N cows (others get a short check); default configs/risk.yaml

Writes <out>/brd_risk.csv, cards/<COW>.html (+ clips), index.html, indicators.csv, contributions.csv,
evidence/facts.jsonl + evidence/img/, agent/traces/<COW>.json.
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import yaml

from src.risk.agent import Backends, RiskAgent, visual_logit
from src.risk.data import Herd, hms
from src.risk.indicators import confirmed_deliveries, minute_table
from src.risk.longterm import asset_indicators, load_assets
from src.risk.features import build as build_features
from src.risk.report import write_outputs
from src.risk.score import behavior_scores, reasons_text, score_from
from src.risk.store import EvidenceStore
from src.risk.tools import RiskTools, is_daylight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--activity", default=None)
    ap.add_argument("--out", default="outputs/level2")
    ap.add_argument("--video-root", default="input")
    ap.add_argument("--skip-agent", action="store_true")
    ap.add_argument("--deep", type=int, default=None)
    ap.add_argument("--short-budget", type=int, default=3)
    ap.add_argument("--cows", nargs="*", default=None, help="only review these cows with the agent")
    ap.add_argument("--cards", type=int, default=3)
    ap.add_argument("--no-clips", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument("--min-hours", type=float, default=None, help="override scoring.min_hours (short test slots)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/risk.yaml"))
    if a.min_hours is not None:
        cfg["scoring"]["min_hours"] = a.min_hours
    if a.bootstrap is not None:
        cfg["scoring"]["bootstrap"] = a.bootstrap
    t_start = time.time()
    if os.path.exists(os.path.join(a.out, "evidence")):
        shutil.rmtree(os.path.join(a.out, "evidence"))
    os.makedirs(a.out, exist_ok=True)

    herd = Herd.load(a.runs, a.activity, a.video_root)
    m = minute_table(herd, cfg)
    assets = load_assets(a.runs)
    fe = assets.get("feed_events")
    if fe is not None and len(fe):   # trough changes the herd answers join the herd-rush proxy (not every flicker)
        kept = confirmed_deliveries(fe.wall_t, m.attrs["bunk_occupancy"], cfg["extraction"])
        print(f"feed deliveries: {len(kept)} of {len(fe)} trough changes confirmed by the herd going to the bunk")
        ev = sorted(set(m.attrs["feed_rushes"]) | set(kept))
        m.attrs["feed_rushes"] = [t for i, t in enumerate(ev) if i == 0 or t - ev[i - 1] > 1800]
    hourly, summary = build_features(m, cfg, assets)
    if len(hourly):
        hourly.to_csv(os.path.join(a.out, "features_hourly.csv"), index=False)
        summary.to_csv(os.path.join(a.out, "features_summary.csv"), index=False)
        print(f"features: {len(hourly)} cow-hours x {sum(1 for c in hourly.columns if '__' not in c) - 4} features")
    extra = asset_indicators(assets, herd.cows, m.groupby("cow_id").seen_s.sum() / 3600)
    print("risk assets:", {k: len(v) for k, v in assets.items()} or "none (scripts/build_risk_assets.py)")
    ind, con, sc = behavior_scores(m, cfg, extra=extra)
    if sc.empty or sc.behavior_score.isna().all():
        raise SystemExit(f"no indicator has >= {cfg['scoring']['min_hours']} h of coverage for 3+ cows: "
                         "process more footage or pass --min-hours for a short test")
    ind.to_csv(os.path.join(a.out, "indicators.csv"))
    con.to_csv(os.path.join(a.out, "contributions.csv"), index=False)
    lo, hi = herd.span()
    hours = m.minute.nunique() / 60
    act_src = os.path.basename(a.activity) if a.activity else "baseline classifier (each run's activity.csv)"
    print(f"{len(herd.cows)} cows, {hours:.1f} h of minutes, feed rushes: {[hms(t) for t in m.attrs['feed_rushes']]}")
    print(sc.round(3).to_string())

    store = EvidenceStore(os.path.join(a.out, "evidence"))
    for _, r in con.iterrows():   # indicator facts the agent and the cards cite
        store.add("indicator", r.cow_id, f"{r.indicator} = {r.value}", {k: r[k] for k in ["value", "herd_median", "z", "signed", "hours", "contrib"]},
                  source="src/risk/indicators.py", fid=f"F-{r.cow_id}-{r.indicator}")
    for t in m.attrs["feed_rushes"]:
        store.add("context", None, f"herd feed rush at {hms(t)}", {"time": t}, source="feed_rushes")
    tools = RiskTools(herd, m, ind, con, sc, cfg, store)
    tools.features_hourly, tools.features_summary = hourly, summary

    traces, models = {}, "none (behavior only)"
    if not a.skip_agent:
        acfg = yaml.safe_load(open("configs/agent.yaml"))
        backends = Backends(acfg, [cfg["backend"]] + cfg.get("fallback", []))
        agent = RiskAgent(tools, store, backends, cfg, os.path.join(a.out, "agent"))
        deep = cfg["deep_review_top"] if a.deep is None else a.deep
        smp = tools.obs.groupby("run", group_keys=False).apply(lambda g: g.sample(min(4, len(g)), random_state=0))
        night = 1 - np.mean([is_daylight(tools._frame(cam, int(t))) for cam, t in smp[["camera", "sec"]].itertuples(index=False)])
        notes = (f"behavior labels from {act_src}; {hours:.1f} h of video between {hms(lo)} and {hms(hi)}; "
                 f"about {100 * night:.0f}% of sampled frames are IR night. Feed rushes detected: {len(m.attrs['feed_rushes'])}. "
                 f"Short coverage makes every indicator noisy: weigh the 'hours' column.")
        for cow in sc.index:
            if a.cows and cow not in a.cows:
                continue
            budget = cfg["max_steps"] if int(sc.loc[cow, "rank"]) <= deep else a.short_budget
            print(f"[agent] {cow} (rank {int(sc.loc[cow, 'rank'])}, budget {budget}) ...", flush=True)
            tr = agent.investigate(cow, sc, notes, budget)
            traces[cow] = tr
            fin = tr["final"]
            print(f"   {tr['seconds']} s, {tr['tool_calls']} tools, signs present: "
                  f"{[s for s, v in (fin.get('signs') or {}).items() if v.get('status') == 'present']}, "
                  f"artifacts: {[x['indicator'] for x in fin.get('artifacts', [])]}, self-check: {len(tr['self_check_first'])} -> {len(tr['self_check_after_revision'])}")
        models = ", ".join(sorted(set(backends.used))) or "none answered"

    # ---- final score: behavior (minus agreed artifacts) + visible signs
    con_adj = con.copy()
    for cow, tr in traces.items():
        for art in tr["final"].get("artifacts", []):
            con_adj.loc[(con_adj.cow_id == cow) & (con_adj.indicator.str.startswith(art["indicator"])), "contrib"] = 0.0
    beh = score_from(con_adj, cfg)
    rows = []
    for cow in beh.index:
        fin = traces.get(cow, {}).get("final", {})
        vl, present = visual_logit(fin, cfg) if fin else (0.0, [])
        logit = beh.loc[cow, "behavior_logit"] + vl
        hrs = float(m[m.cow_id == cow].seen_s.sum() / 3600)
        p3 = float(sc.loc[cow, "p_top3"]) if "p_top3" in sc else np.nan
        reasons = reasons_text(con_adj, cfg, cow) + [f"visible: {p}" for p in present]
        arts = [x["indicator"] for x in fin.get("artifacts", [])]
        conf = "low" if hrs < 2 or (not np.isnan(p3) and p3 < 0.4) else ("high" if hrs >= 12 and p3 >= 0.7 else "medium")
        rows.append({"cow_id": cow, "risk_score": round(float(1 / (1 + np.exp(-logit))), 3),
                     "behavior_score": round(float(beh.loc[cow, "behavior_score"]), 3), "visual_adjustment": round(vl, 2),
                     "p_top3": round(p3, 2), "hours_observed": round(hrs, 2), "confidence": conf,
                     "top_reasons": " | ".join(reasons) or "no deviation from the herd",
                     "signs_present": ";".join(present), "artifacts": ";".join(arts)})
    risk = pd.DataFrame(rows).sort_values("risk_score", ascending=False)
    risk["rank"] = np.arange(1, len(risk) + 1)
    meta = {"span": f"{hms(lo)} to {hms(hi)}", "hours": hours, "n_cows": len(risk), "activity_source": act_src,
            "models": models, "runs": a.runs, "seconds": round(time.time() - t_start, 1)}
    write_outputs(a.out, risk, con_adj, cfg, traces, store, tools, meta, a.cards, not a.no_clips)
    print(risk[["rank", "cow_id", "risk_score", "behavior_score", "visual_adjustment", "p_top3", "confidence", "top_reasons"]].to_string(index=False))
    print(f"-> {a.out}/brd_risk.csv, {a.out}/index.html ({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
