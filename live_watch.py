#!/usr/bin/env python3
"""Level 3: live BRD early warning. Watches new video, updates each cow's risk, sends an alert with video evidence.

  # the judges' clip (any one camera's file; the other cameras' files for the same 5 min are picked up next to it)
  python live_watch.py --clip "<videos>/Day 2/Cam 01/Event20261108142130006.Avi"

  # ranch mode: watch the folder the DVR writes to, process every new 5-min window as it completes
  python live_watch.py --watch "<videos>" [--poll 30]

  # analyze a window that already went through run_pipeline.py (no tracking)
  python live_watch.py --run outputs/pipeline/<run>

Per window: run_pipeline.py (stage 1 tracking + identity + behavior + close crops; --verify adds the stage 2 BRD
agent) -> cows mapped to the 48 h ids -> behavior vs the herd in the same window -> visible clinical signs on close
crops (VLM) confirmed by a second look next to a herdmate -> risk update (48 h Level 2 score as the prior) -> alerts.
Writes live/: index.html (dashboard), alerts/<cow>_<time>.html (+ clip, frame, crops), state.json,
risk_history.csv, windows.jsonl, live.log. Settings: configs/live.yaml.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from pipeline import cameras as camlib          # noqa: E402
from src.live import alert as al                # noqa: E402
from src.live import engine as en               # noqa: E402


class Live:
    def __init__(self, a):
        self.a = a
        self.cfg = yaml.safe_load(open(REPO / a.config, encoding="utf-8"))
        if a.alert_score is not None:
            self.cfg["alert"]["score"] = a.alert_score
        self.out = Path(a.out)
        (self.out / "alerts").mkdir(parents=True, exist_ok=True)
        self.state = en.RiskState(self.out / "state.json", en.load_prior(REPO, self.cfg))
        self.logf = open(self.out / "live.log", "a", encoding="utf-8")
        self.windows = [json.loads(l) for l in open(self.out / "windows.jsonl", encoding="utf-8")] \
            if (self.out / "windows.jsonl").exists() else []
        if "baseline" not in self.state.d or a.baseline:       # the morning list: alerts from the 48 h BRD risk
            al.baseline_alerts(self.state, en.load_prior(REPO, self.cfg), self.cfg, REPO, self.out, self.log)
            self.state.save()
        al.dashboard(self.out, self.state, self.windows)
        from src.live import report
        report.build(self.out, self.cfg)

    def log(self, msg: str):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    # ------------------------------------------------------------------------------------------- one window
    def run_pipeline(self, videos: Path, start: pd.Timestamp, end: pd.Timestamp) -> Path:
        name = f"live_{start:%Y%m%d_%H%M}"
        root = self.out / "runs"
        gal = REPO / self.cfg["identity"]["gallery"]
        cmd = [sys.executable, "run_pipeline.py", "--videos", str(videos), "--name", name, "--out-root", str(root),
               "--start", f"{start:%Y-%m-%d %H:%M}", "--end", f"{end:%Y-%m-%d %H:%M}"]
        cmd += ([] if self.a.verify else list(self.cfg["window"]["pipeline_args"]))
        if gal.exists():
            cmd += ["--id-gallery", str(gal)]
        if not self.has_key():
            cmd += ["--no-id-agent"] + (["--vlm", "mock"] if self.a.verify else [])
        if self.a.fps:
            cmd += ["--fps", str(self.a.fps)]
        self.log(f"window {start:%Y-%m-%d %H:%M}-{end:%H:%M}: running the pipeline ({' '.join(cmd[2:])})")
        t = time.time()
        rc = subprocess.run(cmd, cwd=REPO, env={**os.environ, "PYTHONUTF8": "1"}).returncode
        if rc != 0:
            raise RuntimeError(f"run_pipeline.py exited {rc}; see {root / name / 'logs'}")
        self.log(f"pipeline done in {time.time() - t:.0f} s")
        return root / name

    @staticmethod
    def has_key() -> bool:
        if os.environ.get("TAMU_API_KEY"):
            return True
        p = REPO / ".env"
        return p.exists() and any(l.strip().startswith("TAMU_API_KEY=") and len(l.strip()) > 13
                                  for l in p.read_text(encoding="utf-8").splitlines())

    def analyze(self, run: Path) -> dict:
        t0 = time.time()
        win = en.load_window(run)
        ids = en.map_ids(win, self.cfg, REPO)
        self.log(f"window {win.start} .. {win.end}: {len(ids)} cows; ids: " +
                 ", ".join(f"{r.local_id}->{r.cow_id}" for r in ids.itertuples()))
        feat = en.window_features(win, ids)
        ev = en.window_evidence(feat, self.cfg)
        hs = en.screen_signs(win, ids, self.cfg, REPO, self.log) if not self.a.no_signs else pd.DataFrame()
        if self.a.no_signs:
            self.log("signs: skipped (--no-signs)")
        signs = en.confirmed_signs(hs)
        upd = en.update_risk(self.state, ev, signs, win, self.cfg)
        decisions = []
        alerts = en.decide_alerts(self.state, upd, ev, signs, win, self.cfg, decisions)
        wdir = self.out / "windows" / f"{win.start:%Y%m%d_%H%M}"
        (wdir / "crops").mkdir(parents=True, exist_ok=True)
        ids.to_csv(wdir / "ids.csv", index=False)
        ev.assign(reasons=ev.reasons.map(lambda r: " | ".join(x["text"] for x in r))).to_csv(wdir / "features.csv", index=False)
        ev[["cow_id", "reasons"]].to_json(wdir / "reasons.json", orient="records", indent=1)
        pd.DataFrame(decisions).to_csv(wdir / "decisions.csv", index=False)
        if len(hs):                                  # keep every crop the VLM saw (and its control) with the window
            for col in ("path", "control"):
                rel = []
                for p in hs[col]:
                    if isinstance(p, str) and Path(p).exists():
                        dst = wdir / "crops" / f"{Path(p).parent.name}_{Path(p).name}"
                        shutil.copy2(p, dst)
                        rel.append(f"crops/{dst.name}")
                    else:
                        rel.append(None)
                hs[col + "_file"] = rel
            hs.to_csv(wdir / "signs.csv", index=False)
        thumbs = {}                                  # one close crop per cow, for the report (every cow, every window)
        if len(win.gallery):
            lm = dict(zip(ids.local_id, ids.cow_id))
            g = win.gallery.assign(gid=win.gallery.cow_id.astype(str).map(lm).fillna(win.gallery.cow_id.astype(str)))
            for cow, gg in g.sort_values("box_h", ascending=False).groupby("gid"):
                p = Path(gg.path.iloc[0])
                if p.exists():
                    dst = wdir / "crops" / f"thumb_{cow}{p.suffix}"
                    shutil.copy2(p, dst)
                    thumbs[cow] = {"file": f"crops/{dst.name}", "time": str(pd.to_datetime(gg.wall_t.iloc[0], unit="s")),
                                   "camera": gg.camera.iloc[0]}
        (wdir / "thumbs.json").write_text(json.dumps(thumbs, indent=1), encoding="utf-8")
        hist = self.out / "risk_history.csv"
        upd.to_csv(hist, mode="a", header=not hist.exists(), index=False)
        for r in upd.sort_values("rank").itertuples():
            self.log(f"  #{r.rank} {r.cow_id}: risk {r.risk_before:.2f} -> {r.risk:.2f} (seen {r.seen_min:.0f} min, "
                     f"evidence {r.evidence:+.2f}{', signs ' + r.signs if r.signs else ''})")
        idx = ids.set_index("cow_id")
        for a in alerts:
            a["identity"] = (f"{idx.at[a['cow_id'], 'how']}" + (f", similarity {idx.at[a['cow_id'], 'similarity']:.2f}"
                             if a["cow_id"] in idx.index and pd.notna(idx.at[a['cow_id'], 'similarity']) else "")) \
                if a["cow_id"] in idx.index else ""
            st = win.state[win.state.cow_id.astype(str).isin(ids[ids.cow_id == a["cow_id"]].local_id)]
            where = al.ZONE_TEXT.get(str(st.zone.iloc[-1])) if len(st) else None
            a["action"] = al.action_line(a, where)
            stamp = f"{a['cow_id']}_{win.start:%Y%m%d_%H%M}"
            evd = al.evidence(win, ids, a, self.out / "alerts", self.cfg["alert"]["clip_seconds"])
            card = self.out / "alerts" / f"{stamp}.html"
            card.write_text(al.card_html(a, evd, f"Window {win.start:%Y-%m-%d %H:%M}-{win.end:%H:%M}, cameras "
                                         f"{', '.join(sorted(win.obs.camera.unique()))}"), encoding="utf-8")
            a["card"] = f"alerts/{card.name}"
            a["evidence"] = evd
            a["sent"] = al.deliver(a, a["card"], self.cfg, self.log)
            a["email"] = al.alert_email(a, evd, self.out, self.cfg, self.log)
            (self.out / "alerts" / f"{stamp}.json").write_text(json.dumps(a, indent=1, default=str), encoding="utf-8")
            self.state.d["alerts"].append({k: a[k] for k in ("cow_id", "trigger", "risk", "window_start", "card", "action", "sent")})
            self.log(f"ALERT {a['cow_id']} ({a['trigger']}): {a['action']}  -> {card}")
        # fresh video of every cow already on the check list (push tiers) that this window saw, alert or not:
        # the pen rider gets a new look at her every window
        watch = []
        alerted = {a["cow_id"] for a in alerts}
        push = set(self.cfg.get("tiers", {}).get("push", []))
        (wdir / "watch").mkdir(exist_ok=True)
        for cow, c in self.state.d["cows"].items():
            if c.get("tier") not in push or cow in alerted or cow not in set(ids.cow_id):
                continue
            ev_c = al.evidence(win, ids, {"cow_id": cow, "signs": []}, wdir / "watch", self.cfg["alert"]["clip_seconds"])
            if ev_c.get("clip") or ev_c.get("frame"):
                watch.append({"cow_id": cow, "tier": c["tier"], "tier_rule": c.get("tier_rule", ""), "risk": c["risk"],
                              "clip": f"watch/{ev_c['clip']}" if ev_c.get("clip") else None,
                              "frame": f"watch/{ev_c['frame']}" if ev_c.get("frame") else None, "moment": ev_c.get("moment")})
                self.log(f"check-list cow {cow} ({c['tier']}): fresh clip {ev_c.get('clip')}")
        (wdir / "watch.json").write_text(json.dumps(watch, indent=1, default=str), encoding="utf-8")
        (wdir / "alerts.json").write_text(json.dumps([{k: a[k] for k in ("cow_id", "trigger", "card", "action", "evidence",
                                                                            "identity", "risk", "risk_before")}
                                                      for a in alerts], indent=1, default=str), encoding="utf-8")
        w = {"start": str(win.start), "end": str(win.end), "run": str(run), "cows": int(len(ids)), "alerts": len(alerts),
             "dir": f"windows/{wdir.name}", "cameras": sorted(win.obs.camera.dropna().unique().tolist()),
             "signs": ", ".join(f"{c}: {'/'.join(sorted({s['sign'] for s in v}))}" for c, v in signs.items()) or "none confirmed"
             if len(hs) else ("not checked" if self.a.no_signs or win.gallery.empty or not self.has_key() else "none"),
             "seconds": time.time() - t0}
        self.windows.append(w)
        with open(self.out / "windows.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(w) + "\n")
        self.state.d["windows"].append(w["start"])
        self.state.save()
        al.dashboard(self.out, self.state, self.windows)
        from src.live import report
        report.build(self.out, self.cfg)
        self.log(f"window done: {len(alerts)} alert(s); dashboard {self.out / 'index.html'}, evidence {self.out / 'report.html'}")
        return w

    # ------------------------------------------------------------------------------------------- modes
    def window_of(self, clips, t: pd.Timestamp):
        mins = self.cfg["window"]["minutes"]
        start = t.floor(f"{mins}min") if t.second == 0 and t.minute % mins == 0 else t.floor("min")
        return start, start + pd.Timedelta(minutes=mins)

    def clip_mode(self, files):
        files = [Path(f).resolve() for f in files]
        ccfg = yaml.safe_load(open(REPO / "configs" / "pipeline.yaml", encoding="utf-8"))["cameras"]
        root = files[0].parent
        for _ in range(4):                       # walk up until the other cameras' folders are in view
            try:
                cams = {c.camera for c in camlib.assign_cameras(root, ccfg)}
            except camlib.CameraError:
                cams = set()
            if len(cams) >= 3 or root.parent == root:
                break
            root = root.parent
        clips = camlib.assign_cameras(root, ccfg)
        mine = [c for c in clips if Path(c.path).resolve() in files] or clips
        t = min(pd.Timestamp(c.start) for c in mine)
        start, end = self.window_of(clips, t)
        self.log(f"clip mode: {', '.join(f.name for f in files)} -> window {start:%H:%M}-{end:%H:%M} "
                 f"(clips from {root}, cameras {sorted({c.camera for c in camlib.clips_in_window(clips, str(start), str(end))})})")
        return self.analyze(self.run_pipeline(root, start, end))

    def watch_mode(self, folder: Path):
        ccfg = yaml.safe_load(open(REPO / "configs" / "pipeline.yaml", encoding="utf-8"))["cameras"]
        done = set(self.state.d.get("windows", []))
        first_seen = {}
        self.log(f"watching {folder} every {self.a.poll} s (Ctrl+C to stop)")
        while True:
            try:
                clips = camlib.assign_cameras(folder, ccfg)
            except camlib.CameraError as e:
                self.log(f"waiting for clips: {e}")
                clips = []
            mins = self.cfg["window"]["minutes"]
            by = {}
            for c in clips:
                s = pd.Timestamp(c.start).floor(f"{mins}min")
                by.setdefault(s, set()).add(c.camera)
            for s in sorted(by):
                if str(s) in done or any(str(s) == w[:19] for w in done):
                    continue
                first_seen.setdefault(s, time.time())
                # the next window's clips exist -> this one is complete on disk; else wait for all cameras / timeout
                complete = any(o > s for o in by) and (len(by[s]) >= 3 or time.time() - first_seen[s] > self.cfg["window"]["wait_all_cameras_s"])
                if not complete:
                    continue
                try:
                    self.analyze(self.run_pipeline(folder, s, s + pd.Timedelta(minutes=mins)))
                except Exception as e:
                    self.log(f"window {s} FAILED: {e}")
                done.add(str(s))
            if self.a.once:
                break
            time.sleep(self.a.poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--clip", nargs="+", help="one or more Event*.Avi files of one 5-min window (the judges' clip)")
    g.add_argument("--watch", help="folder the cameras write to; processes each new 5-min window")
    g.add_argument("--run", nargs="+", help="analyze already-processed run_pipeline.py run folder(s), in time order")
    ap.add_argument("--out", default="outputs/live", help="live state, dashboard and alerts")
    ap.add_argument("--config", default="configs/live.yaml")
    ap.add_argument("--verify", action="store_true", help="also run the stage 2 BRD agent on the window (slower, more VLM)")
    ap.add_argument("--no-signs", action="store_true", help="skip the clinical-sign VLM check")
    ap.add_argument("--fps", type=float, help="tracking frames per second (default configs/pipeline.yaml: 1.0)")
    ap.add_argument("--video-root", action="append", default=[],
                    help="where to find the raw clips for evidence when a run has no input/ folder (repeatable)")
    ap.add_argument("--alert-score", type=float, help="override alert.score")
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--once", action="store_true", help="watch mode: one pass over the folder, then exit")
    ap.add_argument("--baseline", action="store_true",
                    help="(re)issue the alerts from the 48 h BRD risk (done automatically on a fresh start)")
    ap.add_argument("--reset", action="store_true", help="start from the 48 h baseline again (clears state, keeps old alerts)")
    a = ap.parse_args()
    os.chdir(REPO)
    al.VIDEO_ROOTS.extend(Path(v).resolve() for v in a.video_root)
    if a.reset:
        for f in ("state.json", "windows.jsonl", "risk_history.csv"):
            p = Path(a.out) / f
            if p.exists():
                p.unlink()
    lv = Live(a)
    if a.clip:
        lv.clip_mode(a.clip)
    elif a.watch:
        lv.watch_mode(Path(a.watch).resolve())
    elif a.run:
        for r in a.run:
            lv.analyze(Path(r))
    elif not a.baseline:
        ap.error("give --clip, --watch, --run, or --baseline (only the alerts from the 48 h BRD risk)")


if __name__ == "__main__":
    main()
