#!/usr/bin/env python3
"""Local live dashboard for the whole Level 1 chain.

  python scripts/dashboard.py --run outputs/level1_d1_1600 --agent outputs/level1_d1_1600/agent

Replay (second by second): the three camera frames with every tracked box, its cow ID and
behavior, a live pen map, the cow table, and the agent's investigations around that moment
with their full reasoning; a per-cow behavior strip over the whole slot with flag markers.
Other tabs: activity grid (before / after the agent), traces, memory cards, identity, ID
checks, corrections, log. Everything is re-read from disk, so it updates while things run.

Serves only on 127.0.0.1 and only files under outputs/ (plus frames decoded from input/):
event data stays on this machine.
"""
import argparse
import csv
import glob
import json
import os
import sys
import threading
import urllib.parse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
ALLOWED = os.path.join(ROOT, "outputs")
PAGE_PATH = os.path.join(os.path.dirname(__file__), "dashboard_page.html")


def read_csv(path, limit=None):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def tail(path, n=60):
    if not os.path.exists(path):
        return []
    with open(path, errors="replace") as f:
        lines = [l.rstrip() for l in f if "warn" not in l.lower() and "xformers" not in l.lower()]
    return lines[-n:]


def load_json(p):
    try:
        return json.load(open(p)) if os.path.exists(p) else None
    except Exception:
        return None


class Data:
    """Replay data, reloaded when the underlying files change."""

    def __init__(self, run_dir, agent_dir, records, video_root):
        self.run_dir, self.agent_dir, self.video_root = run_dir, agent_dir, video_root
        self.records_arg = records
        self.lock = threading.Lock()
        self.sig = None
        self.frames = None
        self.jpeg = OrderedDict()

    def records_path(self):
        for p in (self.records_arg, os.path.join(self.run_dir, "identity", "tracking_records_resolved.csv"),
                  os.path.join(self.run_dir, "tracking_records.csv")):
            if p and os.path.exists(p):
                return p
        return None

    def timeline_path(self):
        for p in (os.path.join(self.agent_dir, "behavior", "behavior_timeline_1s.csv"),
                  os.path.join(self.run_dir, "behavior", "behavior_timeline_1s.csv")):
            if os.path.exists(p):
                return p
        return None

    def refresh(self):
        import numpy as np
        import pandas as pd
        rp, tp = self.records_path(), self.timeline_path()
        sig = tuple((p, os.path.getmtime(p)) for p in (rp, tp) if p)
        if sig == self.sig:
            return
        with self.lock:
            from src.behavior.features import load_records
            from src.agent.evidence import FrameStore
            self.rec = load_records([rp]) if rp else None
            if self.rec is not None:
                self.rec["sec"] = np.round(self.rec.t).astype(np.int64)
                self.by_sec = {s: g for s, g in self.rec.groupby("sec")}
                if self.frames is not None:
                    self.frames.close()
                self.frames = FrameStore(self.rec, self.video_root)
                self.jpeg.clear()
            self.tl = None
            if tp:
                tl = pd.read_csv(tp)
                tl["sec"] = ((pd.to_datetime(tl.wall_time) - pd.Timestamp("1970-01-01")).dt.total_seconds()).astype(np.int64)
                self.tl = tl
                self.tl_by_sec = {s: g for s, g in tl.groupby("sec")}
            self.sig = sig

    def meta(self):
        import yaml
        self.refresh()
        if self.rec is None:
            return {"ready": False}
        cams = yaml.safe_load(open(os.path.join(ROOT, "configs/cameras.yaml")))["cameras"]
        pen = yaml.safe_load(open(os.path.join(ROOT, "configs/pen_layout.yaml")))
        cows = sorted(c for c in self.rec.global_cow_id.dropna().unique())
        flags = []
        for p in glob.glob(os.path.join(self.agent_dir, "traces", "*.json")):
            t = load_json(p)
            if t:
                f, fi = t["flag"], t.get("final") or {}
                flags.append({"file": os.path.relpath(p, ROOT), "kind": f["kind"], "t0": f["t0"], "t1": f["t1"],
                              "severity": f.get("severity"), "cows": f.get("cow_ids", [])[:3], "camera": f.get("camera"),
                              "decision": fi.get("decision"), "summary": fi.get("summary", "")})
        zones = {k: v.get("polygon_metric") for k, v in (pen.get("zones") or {}).items() if v.get("polygon_metric")}
        return {"ready": True, "t_min": int(self.rec.sec.min()), "t_max": int(self.rec.sec.max()),
                "cameras": sorted(self.rec.camera_id.unique()), "cows": cows, "flags": sorted(flags, key=lambda x: x["t0"]),
                "cam_pos": {k: v.get("mount_position_estimated") for k, v in cams.items()},
                "cam_names": {k: v.get("name") for k, v in cams.items()}, "zones": zones,
                "records": os.path.relpath(self.records_path(), ROOT)}

    def at(self, sec):
        import numpy as np
        self.refresh()
        g = self.by_sec.get(sec)
        boxes, pos = [], {}
        if g is not None:
            for r in g.itertuples():
                cid = r.global_cow_id if isinstance(r.global_cow_id, str) else None
                boxes.append({"cam": r.camera_id, "cow": cid, "track": int(r.local_track_id),
                              "x1": r.bbox_x1 / r.frame_width, "y1": r.bbox_y1 / r.frame_height,
                              "x2": r.bbox_x2 / r.frame_width, "y2": r.bbox_y2 / r.frame_height,
                              "det": round(float(r.detection_confidence), 2),
                              "idc": None if not np.isfinite(r.identity_confidence) else round(float(r.identity_confidence), 2)})
                if cid and np.isfinite(r.world_x) and np.isfinite(r.world_y):
                    pos.setdefault(cid, []).append((float(r.world_x), float(r.world_y), r.camera_id))
        beh = {}
        if self.tl is not None:
            h = self.tl_by_sec.get(sec)
            if h is not None:
                for r in h.itertuples():
                    beh[r.cow_id] = {"behavior": r.behavior, "conf": round(float(r.confidence), 2),
                                     "evidence": r.evidence, "cameras": r.cameras if isinstance(r.cameras, str) else ""}
        cows = {}
        for cid, ps in pos.items():
            a = np.array([(x, y) for x, y, _ in ps])
            cows[cid] = {"x": round(float(a[:, 0].mean()), 2), "y": round(float(a[:, 1].mean()), 2),
                         "cams": sorted({c for *_, c in ps}),
                         "spread": round(float(np.max(np.linalg.norm(a[:, None] - a[None], axis=2))), 2) if len(a) > 1 else 0.0}
        return {"sec": sec, "boxes": boxes, "cows": cows, "behavior": beh}

    def strip(self):
        self.refresh()
        if self.tl is None:
            return {"rows": {}}
        rows = {}
        for cid, g in self.tl.sort_values("sec").groupby("cow_id"):
            seg, cur = [], None
            for s, b in zip(g.sec, g.behavior):
                if cur and cur[2] == b and s - cur[1] <= 1:
                    cur[1] = s
                else:
                    if cur:
                        seg.append(cur)
                    cur = [int(s), int(s), b]
            if cur:
                seg.append(cur)
            rows[cid] = seg
        return {"rows": rows}

    def frame_jpeg(self, cam, sec, width=960):
        import cv2
        self.refresh()
        key = (cam, sec, width)
        with self.lock:
            if key in self.jpeg:
                self.jpeg.move_to_end(key)
                return self.jpeg[key]
            fr = self.frames.frame(cam, float(sec)) if self.frames else None
            if fr is None:
                return None
            s = width / fr.shape[1]
            fr = cv2.resize(fr, (width, int(fr.shape[0] * s)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 78])
            self.jpeg[key] = buf.tobytes()
            while len(self.jpeg) > 400:
                self.jpeg.popitem(last=False)
            return self.jpeg[key]


def state(run_dir, agent_dir):
    s = {"run": run_dir, "agent": agent_dir}
    s["summary"] = load_json(os.path.join(agent_dir, "agent_summary.json"))
    s["log"] = tail(os.path.join(agent_dir, "agent_log.txt"))
    s["chain_log"] = tail(os.path.join(run_dir, "level1_log.txt"), 30)
    s["cards"] = load_json(os.path.join(agent_dir, "memory", "cards.json")) or []
    traces = []
    for p in sorted(glob.glob(os.path.join(agent_dir, "traces", "*.json")), key=os.path.getmtime, reverse=True):
        t = load_json(p)
        if t:
            t["_file"] = os.path.relpath(p, ROOT)
            traces.append(t)
    s["traces"] = traces
    s["id_checks"] = sorted(glob.glob(os.path.join(agent_dir, "id_checks", "*.jpg")))
    s["corrections"] = read_csv(os.path.join(agent_dir, "corrections.csv"))
    s["activity_before"] = read_csv(os.path.join(run_dir, "behavior", "activity.csv"))
    s["activity_after"] = read_csv(os.path.join(agent_dir, "behavior", "activity.csv"))
    s["budgets"] = read_csv(os.path.join(agent_dir, "behavior", "time_budgets.csv")) or \
        read_csv(os.path.join(run_dir, "behavior", "time_budgets.csv"))
    s["identity"] = load_json(os.path.join(run_dir, "identity", "identity_report.json"))
    sheet = os.path.join(run_dir, "identity", "identity_sheet.jpg")
    s["identity_sheet"] = sheet if os.path.exists(sheet) else None
    en = os.path.join(agent_dir, "memory", "enrollment_sheet.jpg")
    s["enrollment_sheet"] = en if os.path.exists(en) else None
    return s


class H(BaseHTTPRequestHandler):
    data: Data = None
    run_dir = agent_dir = ""

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "max-age=600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj, default=str).encode(), "application/json")

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        try:
            if u.path == "/":
                return self._send(200, open(PAGE_PATH, "rb").read(), "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self._json(state(self.run_dir, self.agent_dir))
            if u.path == "/api/meta":
                return self._json(self.data.meta())
            if u.path == "/api/at":
                return self._json(self.data.at(int(float(q["t"][0]))))
            if u.path == "/api/strip":
                return self._json(self.data.strip())
            if u.path == "/api/frame":
                b = self.data.frame_jpeg(q["cam"][0], int(float(q["t"][0])), int(q.get("w", ["960"])[0]))
                return self._send(200, b, "image/jpeg", cache=True) if b else self._send(404, b"no frame", "text/plain")
            if u.path == "/file":
                p = q.get("p", [""])[0]
                full = os.path.realpath(os.path.join(ROOT, p))
                if not full.startswith(ALLOWED + os.sep) or not os.path.isfile(full):
                    return self._send(404, b"not found", "text/plain")
                ctype = "image/jpeg" if full.lower().endswith((".jpg", ".jpeg")) else "text/plain; charset=utf-8"
                return self._send(200, open(full, "rb").read(), ctype, cache=True)
        except Exception as e:  # keep the server alive
            return self._send(500, f"{type(e).__name__}: {e}".encode(), "text/plain")
        self._send(404, b"not found", "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/level1_d1_1600")
    ap.add_argument("--agent", default=None)
    ap.add_argument("--records", default=None, help="records to replay (default: <run>/identity/tracking_records_resolved.csv)")
    ap.add_argument("--video-root", default="input")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    os.chdir(ROOT)
    agent = a.agent or os.path.join(a.run, "agent")
    H.run_dir, H.agent_dir = a.run, agent
    H.data = Data(a.run, agent, a.records, a.video_root)
    print(f"dashboard: http://127.0.0.1:{a.port}  (run {a.run}, agent {agent})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
