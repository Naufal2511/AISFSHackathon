"""HTTP API for the downstream behavior model and validation agent (stdlib only, no installs).

  python -m src.api.server --port 8800                         # local only (default 127.0.0.1)
  python -m src.api.server --host 0.0.0.0 --token SECRET       # share on the team network

GET  /health
GET  /runs                                         processed slots
GET  /runs/{run}/manifest                          span, cows, which outputs exist
GET  /runs/{run}/cows
GET  /runs/{run}/observations?t0&t1&cow_id&camera&limit&offset      per box per second per camera
GET  /runs/{run}/cow_state?t0&t1&cow_id&limit&offset                per cow per second (fused)
GET  /runs/{run}/frame.jpg?camera&t&boxes=1&width=1280
GET  /runs/{run}/crop.jpg?camera&t&cow_id|track_id&pad=0.15&size=448
GET  /runs/{run}/tools                             agent tools and their arguments
GET  /runs/{run}/tools/{name}?arg=value...         run one tool -> {text, images:[base64 jpeg], data}
POST /runs/{run}/behavior?model=NAME               body: CSV or JSON list of per-box predictions
POST /runs/{run}/verdicts                          body: CSV or JSON list of verdicts
GET  /runs/{run}/activity?stage=best|final|model|baseline
GET  /runs/{run}/time_budgets?stage=...

Times: ISO wall clock ("2026-11-07T16:01:00") or epoch seconds. Contract: docs/DATA_CONTRACT.md.
Event data stays on this machine unless you deliberately bind to the network with a token.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

from .flow import Pen6Run, list_runs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _jpeg(img, width=None, quality=85) -> bytes:
    import cv2
    if width and img.shape[1] > width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _records(df: pd.DataFrame, limit: int, offset: int):
    total = len(df)
    df = df.iloc[offset: offset + limit]
    return {"total": total, "offset": offset, "count": len(df),
            "rows": json.loads(df.to_json(orient="records", date_format="iso"))}


def _arg(v: str):
    try:
        return json.loads(v)
    except Exception:
        return v


class API(BaseHTTPRequestHandler):
    runs_root = "outputs"
    token = None
    video_root = "input"
    _runs: dict = {}
    _lock = threading.Lock()

    def log_message(self, fmt, *a):
        print("[api]", self.address_string(), fmt % a, flush=True)

    # -------------------------------------------------------------- helpers
    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=str).encode())

    def _err(self, code, msg):
        self._json({"error": msg}, code)

    def _auth(self) -> bool:
        if not self.token:
            return True
        h = self.headers.get("Authorization", "")
        if h == f"Bearer {self.token}" or self.headers.get("X-Token") == self.token:
            return True
        self._err(401, "missing or wrong token (Authorization: Bearer <token>)")
        return False

    def _run(self, name) -> Pen6Run:
        if name not in list_runs(self.runs_root):
            raise KeyError(f"unknown run '{name}' (GET /runs)")
        with self._lock:
            if name not in self._runs:
                self._runs[name] = Pen6Run(os.path.join(self.runs_root, name), self.video_root)
            return self._runs[name]

    def _body_table(self) -> pd.DataFrame:
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n).decode("utf-8")
        ctype = self.headers.get("Content-Type", "")
        if "json" in ctype or raw.lstrip().startswith(("[", "{")):
            data = json.loads(raw)
            return pd.DataFrame(data["rows"] if isinstance(data, dict) and "rows" in data else data)
        return pd.read_csv(io.StringIO(raw))

    # -------------------------------------------------------------- routes
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Token")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if not self._auth():
            return
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        parts = [p for p in u.path.split("/") if p]
        try:
            if parts == ["health"]:
                return self._json({"ok": True})
            if parts == ["runs"]:
                return self._json({"runs": list_runs(self.runs_root)})
            if len(parts) < 3 or parts[0] != "runs":
                return self._err(404, "see module docstring for routes")
            run, what = self._run(parts[1]), parts[2]
            limit, offset = int(q.get("limit", 5000)), int(q.get("offset", 0))
            if what == "manifest":
                return self._json(run.manifest())
            if what == "cows":
                return self._json({"cows": run.cows()})
            if what == "observations":
                return self._json(_records(run.observations(q.get("t0"), q.get("t1"), q.get("cow_id"), q.get("camera")), limit, offset))
            if what == "cow_state":
                return self._json(_records(run.cow_state(q.get("t0"), q.get("t1"), q.get("cow_id")), limit, offset))
            if what == "frame.jpg":
                img = run.frame(q["camera"], _arg(q["t"]), boxes=q.get("boxes") in ("1", "true"))
                return self._err(404, "no frame") if img is None else self._send(200, _jpeg(img, int(q.get("width", 1280))), "image/jpeg")
            if what == "crop.jpg":
                img = run.crop(q["camera"], _arg(q["t"]), q.get("cow_id"),
                               int(q["track_id"]) if "track_id" in q else None,
                               float(q.get("pad", 0.15)), int(q.get("size", 448)))
                return self._err(404, "no such box at that second") if img is None else self._send(200, _jpeg(img), "image/jpeg")
            if what == "tools" and len(parts) == 3:
                return self._json({"tools": run.tools()})
            if what == "tools" and len(parts) == 4:
                res = run.tool(parts[3], **{k: _arg(v) for k, v in q.items()})
                return self._json({"text": res.text, "data": res.data,
                                   "images": [base64.b64encode(_jpeg(im, 1280)).decode() for im in res.images]})
            if what == "activity":
                return self._json(_records(run.activity(q.get("stage", "best")), limit, offset))
            if what == "time_budgets":
                return self._json(_records(run.time_budgets(q.get("stage", "best")), limit, offset))
            return self._err(404, f"unknown endpoint {what}")
        except KeyError as e:
            return self._err(404 if "unknown run" in str(e) else 400, f"missing/unknown parameter: {e}")
        except (ValueError, FileNotFoundError) as e:
            return self._err(400, str(e))
        except Exception as e:
            traceback.print_exc()
            return self._err(500, f"{type(e).__name__}: {e}")

    def do_POST(self):
        if not self._auth():
            return
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        parts = [p for p in u.path.split("/") if p]
        try:
            if len(parts) != 3 or parts[0] != "runs":
                return self._err(404, "POST /runs/{run}/behavior or /runs/{run}/verdicts")
            run = self._run(parts[1])
            table = self._body_table()
            with self._lock:
                if parts[2] == "behavior":
                    return self._json(run.submit_behavior(table, model=q.get("model")))
                if parts[2] == "verdicts":
                    return self._json(run.submit_verdicts(table, base=q.get("base", "auto")))
            return self._err(404, f"unknown endpoint {parts[2]}")
        except (ValueError, KeyError) as e:
            return self._err(400, str(e))
        except Exception as e:
            traceback.print_exc()
            return self._err(500, f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--runs-root", default="outputs")
    ap.add_argument("--video-root", default="input")
    ap.add_argument("--token", default=os.environ.get("PEN6_API_TOKEN"))
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.host not in ("127.0.0.1", "localhost") and not a.token:
        raise SystemExit("binding to the network needs --token (event data must stay within the team)")
    API.runs_root, API.token, API.video_root = a.runs_root, a.token, a.video_root
    print(f"Pen 6 API on http://{a.host}:{a.port}  runs: {list_runs(a.runs_root)}", flush=True)
    ThreadingHTTPServer((a.host, a.port), API).serve_forever()


if __name__ == "__main__":
    main()
