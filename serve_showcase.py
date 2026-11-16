#!/usr/bin/env python3
"""See every result in a browser: a small local web server for the showcase (no extra packages).

  python serve_showcase.py                     # then open http://localhost:8000
  docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/outputs:/app/outputs" pen6     # the image's default command

  http://localhost:8000/        the showcase (submission/index.html): Levels 1, 2 and 3, with links to every page
  http://localhost:8000/live/   your own Level 3 run (live_watch.py --clip ... writes outputs/live/)

Serves only submission/, docs/, README.md and outputs/live/: never the code folders or .env. Local only by default
(127.0.0.1). Inside Docker it listens on 0.0.0.0 so the port can be published; publish it as 127.0.0.1:8000:8000 so
only this computer can open it. The pages contain event video frames: keep them within the event.
"""
from __future__ import annotations

import argparse
import mimetypes
import os
import posixpath
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent
mimetypes.add_type("text/markdown; charset=utf-8", ".md")
mimetypes.add_type("text/csv; charset=utf-8", ".csv")
mimetypes.add_type("message/rfc822", ".eml")
mimetypes.add_type("application/x-ndjson; charset=utf-8", ".jsonl")

NO_LIVE_RUN = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>No Live Run</title><style>body{{font:16px/1.55 system-ui,"Segoe UI",sans-serif;max-width:720px;margin:48px auto;padding:0 16px;color:#18212b;background:#f6f5f1}}
code,pre{{background:#e9e7e0;border-radius:6px;padding:2px 6px}} pre{{padding:12px;overflow-x:auto}} a{{color:#0f5a6b}}</style></head><body>
<h1>No live run yet</h1>
<p>Run Level 3 on a clip first; its dashboard, evidence report, alert cards and emails then appear here.</p>
<pre>{cmd}</pre>
<p>Meanwhile, the saved example runs are in the <a href="/submission/index.html#l3">showcase</a>.</p></body></html>"""


class Showcase(SimpleHTTPRequestHandler):
    live_dir: Path = REPO / "outputs" / "live"
    roots = {"submission": REPO / "submission", "docs": REPO / "docs"}

    def log_message(self, fmt, *args):
        pass

    def _target(self) -> Path | None:
        """URL path -> file, only inside the allowed folders (no '..' escapes)."""
        path = posixpath.normpath(urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)).lstrip("/")
        if path == "README.md":
            return REPO / "README.md"
        head, _, rest = path.partition("/")
        base = self.live_dir if head == "live" else self.roots.get(head)
        if base is None:
            return None
        p = (base / rest).resolve()
        if p != base.resolve() and base.resolve() not in p.parents:
            return None
        if p.is_dir():
            for name in ("report.html", "index.html") if head == "live" else ("index.html",):
                if (p / name).is_file():
                    return p / name
        return p

    def do_GET(self):
        url_path = urllib.parse.urlsplit(self.path).path
        if url_path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/submission/index.html")
            self.end_headers()
            return
        if url_path.rstrip("/") == "/live" and not url_path.endswith("/"):
            self.send_response(301)
            self.send_header("Location", "/live/")
            self.end_headers()
            return
        p = self._target()
        if url_path.startswith("/live") and (p is None or not p.exists()) and url_path in ("/live/", "/live/report.html"):
            body = NO_LIVE_RUN.format(cmd='python live_watch.py --clip "/videos/Day 2/Cam 01/Event20261108142130006.Avi"').encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if p is None or not p.is_file():
            self.send_error(404, "Not found (the showcase serves submission/, docs/, README.md and /live/ only)")
            return
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") and "charset" not in ctype:
            ctype += "; charset=utf-8"
        size = p.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store" if url_path.startswith("/live") else "max-age=600")
        self.end_headers()
        with open(p, "rb") as f:
            while chunk := f.read(1 << 16):
                self.wfile.write(chunk)

    def do_HEAD(self):      # GET only (the base class would serve HEAD from the current folder)
        self.send_error(405, "GET only")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("SHOWCASE_HOST", "127.0.0.1"),
                    help="127.0.0.1 = this computer only (default); the Docker image uses 0.0.0.0 inside the container")
    ap.add_argument("--port", type=int, default=int(os.environ.get("SHOWCASE_PORT", 8000)))
    ap.add_argument("--live", default=str(REPO / "outputs" / "live"), help="live_watch.py --out folder shown at /live/")
    a = ap.parse_args()
    Showcase.live_dir = Path(a.live).resolve()
    srv = ThreadingHTTPServer((a.host, a.port), Showcase)
    print(f"Pen 6 showcase:  http://localhost:{a.port}/        (Levels 1, 2 and 3)")
    print(f"Your live run:   http://localhost:{a.port}/live/   (after live_watch.py --clip ...)")
    print("Ctrl+C to stop.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
