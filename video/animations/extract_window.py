#!/usr/bin/env python3
"""Cut one time window out of a stitched run for the video animations.

  python3 video/animations/extract_window.py --run outputs/stitched/runs/hprc_day2_c18 \
      --start "2026-11-08 16:28:00" --seconds 120 --out <dir> [--videos input] [--speed 4]

Writes <dir>/world.json (per-second cow state + per-camera visibility), <dir>/boxes.json
(per-second boxes, normalised) and <dir>/frames/cam1..cam3/NNNN.jpg (640x480, one per output
frame at 30 fps for the given playback speed).
"""
import argparse
import csv
import collections
import datetime as dt
import glob
import json
import os
import subprocess


def secs(w):
    return int(dt.datetime.fromisoformat(w[:19]).timestamp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", default="input")
    ap.add_argument("--speed", type=float, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = int(dt.datetime.fromisoformat(a.start).timestamp())
    n = a.seconds

    state = [dict() for _ in range(n)]
    for r in csv.DictReader(open(os.path.join(a.run, "contract", "cow_state.csv"))):
        s = secs(r["wall_time"]) - t0
        if 0 <= s < n:
            f = lambda k: float(r[k]) if r[k] not in ("", None) else None
            state[s][r["cow_id"]] = dict(x=f("x"), y=f("y"), h=f("heading_deg"), p=r["predicted"] == "True",
                                         sd=f("pos_sd_m"), cv=r["cams_visible"], z=r["zone"], b=r["behavior_fused"])

    vis = [collections.defaultdict(dict) for _ in range(n)]
    boxes = collections.defaultdict(list)
    first = {}
    for r in csv.DictReader(open(os.path.join(a.run, "contract", "observations.csv"))):
        s = secs(r["wall_time"]) - t0
        if not (0 <= s < n) or r["box_role"] != "body":
            continue
        fw, fh = float(r["frame_w"]), float(r["frame_h"])
        x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
        if r["cow_id"]:
            vis[s][r["cow_id"]][r["camera"]] = dict(v=r["visibility"], o=r["occluded_by"], f=r["facing"],
                                                    a=round((x2 - x1) * (y2 - y1) / (fw * fh) * 100, 2))
        boxes[s].append([r["camera"], r["cow_id"], int(r["track_id"]), round(x1 / fw, 4), round(y1 / fh, 4),
                         round(x2 / fw, 4), round(y2 / fh, 4), r["visibility"]])
        cand = (s, float(r["offset_s"]) - s, r["file"])
        if r["camera"] not in first or cand[0] < first[r["camera"]][0]:
            first[r["camera"]] = cand

    start = dt.datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S")
    json.dump({"start": start, "n": n, "state": state, "vis": vis}, open(os.path.join(a.out, "world.json"), "w"))
    json.dump({str(k): v for k, v in boxes.items()}, open(os.path.join(a.out, "boxes.json"), "w"))

    fps_src = 30 / a.speed
    for cam, (_, off0, fname) in sorted(first.items()):
        path = glob.glob(os.path.join(a.videos, "**", fname), recursive=True)
        if not path:
            print("missing clip", fname)
            continue
        d = os.path.join(a.out, "frames", cam)
        os.makedirs(d, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-vcodec", "hevc", "-ss", f"{off0:.2f}", "-i", path[0],
                        "-t", str(n), "-vf", f"fps={fps_src},scale=640:480", "-q:v", "3", os.path.join(d, "%04d.jpg")])
        print(cam, fname, f"offset {off0:.1f}s", len(os.listdir(d)), "frames")
    print("window", start, n, "s")


if __name__ == "__main__":
    main()
