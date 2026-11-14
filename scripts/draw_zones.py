#!/usr/bin/env python3
"""Click-to-draw pen ROI, feed bunk and waterer polygons on a real frame.

  python scripts/draw_zones.py --camera cam1 --video "input/Day 1/Cam 01/Event20261107071945006.Avi" --t 30
  python scripts/draw_zones.py --camera cam2 --video "input/Day 1/Cam 02/<clip>.Avi" --t 30
  python scripts/draw_zones.py --camera cam3 --video "input/Day 1/Cam 03/Cam 03/Event20261107071935003.Avi" --t 30

Controls (window must have focus):
  1 / 2 / 3     select layer: 1 = pen ROI, 2 = feed bunk, 3 = waterer
  left click    add a point to the selected layer
  u             undo last point          c   clear selected layer
  s             save to configs/camera_zones.yaml
  q / Esc       quit (unsaved changes are lost)

What to draw (all layers are polygons, 3+ points):
  pen ROI    Pen 6 ground where a hoof can stand. Follow the fence bases; run to the frame
             edge where cows are cut off (e.g. along the bunk). Exclude Pen 7 and the alley.
  feed bunk  The band a head occupies while eating: from the neck rail to the far trough lip.
  waterer    The waterer plus ~0.5 m of head reach around it.

Non-interactive (scripting/tests):
  --set feed_bunk="0.1,0.8;0.9,0.8;0.9,1.0;0.1,1.0" --save
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from src.calibration.camera_zones import DEFAULT_PATH, LAYERS, load_zones, save_zones

COLORS = {"roi": (0, 255, 255), "feed_bunk": (0, 0, 255), "waterer": (255, 128, 0)}
NAMES = {"roi": "1: PEN ROI", "feed_bunk": "2: FEED BUNK", "waterer": "3: WATERER"}


def read_frame(video: str, t: float) -> np.ndarray:
    if video.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(video)
    else:
        from src.video.video_reader import FrameReader
        cap = cv2.VideoCapture(FrameReader(video).source)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        cap.release()
        img = img if ok else None
    if img is None:
        sys.exit(f"Could not read a frame from {video}")
    return img


def render(small, layers, active, cursor=None, saved_msg=""):
    """Draw on the DISPLAY-sized image so markers stay big whatever the source resolution."""
    img = small.copy()
    h, w = img.shape[:2]
    overlay = img.copy()
    for name, pts in layers.items():
        if not pts:
            continue
        P = (np.array(pts) * [w, h]).round().astype(np.int32)
        is_active = name == active
        if len(P) >= 3:
            cv2.fillPoly(overlay, [P], COLORS[name])
        cv2.polylines(img, [P], len(P) >= 3, COLORS[name], 4 if is_active else 2, cv2.LINE_AA)
        for i, p in enumerate(P):
            cv2.circle(img, tuple(int(v) for v in p), 9 if is_active else 6, (0, 0, 0), -1)
            cv2.circle(img, tuple(int(v) for v in p), 7 if is_active else 4, COLORS[name], -1)
            if is_active:
                cv2.putText(img, str(i + 1), (int(p[0]) + 10, int(p[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(img, str(i + 1), (int(p[0]) + 10, int(p[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if is_active and cursor is not None:  # rubber band: last point -> cursor -> first point
            cv2.line(img, tuple(int(v) for v in P[-1]), cursor, COLORS[name], 2, cv2.LINE_AA)
            if len(P) >= 2:
                cv2.line(img, cursor, tuple(int(v) for v in P[0]), COLORS[name], 1, cv2.LINE_AA)
    img = cv2.addWeighted(overlay, 0.22, img, 0.78, 0)
    lines = [f"LAYER {NAMES[active]}   points: {len(layers[active])}   (polygon closes automatically)",
             "keys: 1 roi | 2 feed bunk | 3 waterer | click = add point | u undo | c clear | s SAVE | q quit"]
    if saved_msg:
        lines.append(saved_msg)
    for i, line in enumerate(lines):
        y = 28 + 28 * i
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--video", help="clip (.Avi/.mp4) or a still image")
    ap.add_argument("--t", type=float, default=30.0, help="seconds into the clip")
    ap.add_argument("--zones", default=DEFAULT_PATH)
    ap.add_argument("--max-width", type=int, default=1600)
    ap.add_argument("--set", action="append", default=[], help='layer="x,y;x,y;..." normalized')
    ap.add_argument("--save", action="store_true", help="with --set: save without opening a window")
    a = ap.parse_args()

    zones = load_zones(a.zones)
    layers = {k: [list(p) for p in zones.get(a.camera, {}).get(k, [])] for k in LAYERS}

    for spec in a.set:
        name, pts = spec.split("=", 1)
        assert name in LAYERS, f"layer must be one of {LAYERS}"
        layers[name] = [[float(v) for v in p.split(",")] for p in pts.strip('"').split(";")]
    if a.save:
        zones[a.camera] = {k: v for k, v in layers.items() if len(v) >= 3}
        save_zones(zones, a.zones)
        print(f"saved {a.camera}: {', '.join(k for k, v in zones[a.camera].items())} -> {a.zones}")
        return

    if not a.video:
        sys.exit("--video is required for interactive drawing")
    base = read_frame(a.video, a.t)
    h, w = base.shape[:2]
    scale = min(1.0, a.max_width / w)
    small = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    state = {"active": "roi", "dirty": False, "cursor": None, "msg": ""}
    win = f"draw zones - {a.camera}"

    def on_mouse(event, x, y, flags, _):
        state["cursor"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            layers[state["active"]].append([round(min(max(x / sw, 0.0), 1.0), 4), round(min(max(y / sh, 0.0), 1.0), 4)])
            state["dirty"] = True

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        active = state["active"]
        cv2.imshow(win, render(small, layers, active, state["cursor"], state["msg"]))
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("1"), ord("2"), ord("3")):
            state["active"] = LAYERS[k - ord("1")]
        elif k == ord("u") and layers[active]:
            layers[active].pop(); state["dirty"] = True
        elif k == ord("c"):
            layers[active] = []; state["dirty"] = True
        elif k == ord("s"):
            zones[a.camera] = {kk: v for kk, v in layers.items() if len(v) >= 3}
            save_zones(zones, a.zones)
            state["msg"] = "SAVED " + ", ".join(f"{kk} ({len(v)} pts)" for kk, v in zones[a.camera].items())
            print(f"saved {a.camera} -> {a.zones}: {state['msg']}")
            state["dirty"] = False
        elif k in (ord("q"), 27):
            if state["dirty"]:
                print("quit WITHOUT saving the last changes (press s before q to keep them)")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
