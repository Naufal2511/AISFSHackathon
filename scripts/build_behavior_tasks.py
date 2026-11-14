#!/usr/bin/env python3
"""Per-cow 5-second behavior clips for a LOCAL Label Studio (one task = one cow, one window).

  # 1. build clips + tasks (append more any time with a new --seed)
  python scripts/build_behavior_tasks.py build --windows 60 --out data/behavior_annotation

  # 2. after labelling: Export -> JSON, then build the YOLO classification dataset
  python scripts/build_behavior_tasks.py to-cls --export ~/Downloads/project-2-at-....json

What it does per window (5 s of one clip, one camera)
  * decodes the window at --fps (default 10) and --decode-width (default 2560 px)
  * runs the fine-tuned cow detector + ByteTrack, keeps tracks seen in >= --min-cover of the
    frames whose hoof point is inside the drawn Pen 6 ROI (configs/camera_zones.yaml)
  * per track writes
      clips/<task>.mp4      zoomed, smoothed crop following the cow (H.264, plays in the browser),
                            target cow outlined so neighbours are not confused with it
      context/<task>.jpg    full frame, target cow in a thick box, other cows thin
      crops/<task>/fNN.jpg  CLEAN crops (no drawing) at --crops-per-track times -> training images
  * a cautious pre-selection (feeding / drinking / walking only, from bunk/waterer overlap and
    hoof-point speed); everything else is left for you to choose

Train/val split is by clip (md5 hash, same rule as the detector sampler), so no clip leaks.
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd

from src.behavior.taxonomy import BEHAVIORS
from src.calibration.camera_zones import bbox_fraction_in, load_zones

EVENT_RE = re.compile(r"Event(\d{14})")
CAM_RE = re.compile(r"Cam\s*0?(\d)", re.I)
HOTKEYS = "1234567890"
QUALITY = ["bad track (ID switch / two cows / not a cow)", "mostly occluded", "unsure"]


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def label_config():
    behav = "\n".join(f'      <Choice value="{b}" hotkey="{HOTKEYS[i]}"/>' for i, b in enumerate(BEHAVIORS))
    qual = "\n".join(f'      <Choice value="{q}"/>' for q in QUALITY)
    return f"""<View>
  <Style>
    .row {{ display: flex; gap: 12px; align-items: flex-start; flex-wrap: wrap; }}
    .col {{ flex: 1 1 420px; min-width: 320px; }}
  </Style>
  <Header value="Cow $track_label  ·  $camera  ·  $clip_start  (+$t_start s)" size="5"/>
  <View className="row">
    <View className="col">
      <Header value="Zoomed 5 s clip (target cow outlined)" size="6"/>
      <Video name="video" value="$video" framerate="$fps" muted="true" height="420"/>
    </View>
    <View className="col">
      <Header value="Where it is in the pen" size="6"/>
      <Image name="context" value="$context" zoom="true" zoomControl="true"/>
    </View>
  </View>
  <Header value="Main behavior during these 5 s" size="5"/>
  <Choices name="behavior" toName="video" choice="single-radio" showInline="true" required="true">
{behav}
  </Choices>
  <Header value="Flags (optional)" size="6"/>
  <Choices name="quality" toName="video" choice="multiple" showInline="true">
{qual}
  </Choices>
</View>
"""


# ---------------------------------------------------------------- sampling

def list_clips(root):
    rows = []
    for p in glob.glob(os.path.join(root, "**", "*.[Aa]vi"), recursive=True):
        m, c = EVENT_RE.search(os.path.basename(p)), CAM_RE.search(p)
        if not (m and c):
            continue
        ts = pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S")
        hour = ts.hour + ts.minute / 60
        rows.append(dict(path=p, camera=f"cam{c.group(1)}", start=ts, clip=m.group(0),
                         night=not (7.0 <= hour < 17.5)))
    return pd.DataFrame(rows).sort_values(["camera", "start"]).reset_index(drop=True)


def plan_windows(clips, n, share, seed):
    rng = random.Random(seed)
    share = {k: v for k, v in share.items() if k in set(clips.camera)}
    tot = sum(share.values())
    plan = []
    for cam, s in share.items():
        n_cam = int(round(n * s / tot))
        cc = clips[clips.camera == cam]
        strata = [g for _, g in cc.groupby("night")]
        per = [n_cam // len(strata) + (1 if i < n_cam % len(strata) else 0) for i in range(len(strata))]
        for g, k in zip(strata, per):
            if k == 0:
                continue
            # spread over the day, with a random offset so a new seed gives new clips
            off = rng.random() * len(g) / max(k, 1)
            idx = sorted({int((off + i * len(g) / k)) % len(g) for i in range(k)})
            plan.extend(g.iloc[idx].to_dict("records"))
    return plan


def is_val(clip, frac):
    return int(hashlib.md5(clip.encode()).hexdigest(), 16) % 1000 < frac * 1000


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def decode_window(path, t0, secs, fps, width):
    """Frames [t0, t0+secs) at fps, resized to `width` (keeps aspect). HEVC is forced first."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", path],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip().split(",")
    W, H = int(probe[0]), int(probe[1])
    w = min(width, W)
    w -= w % 2
    h = int(round(H * w / W))
    h -= h % 2
    for forced in (["-vcodec", "hevc"], []):
        cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{t0:.2f}", *forced, "-i", path, "-t", f"{secs:.2f}",
               "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        n = len(r.stdout) // (w * h * 3)
        if n >= fps * secs * 0.8:
            return [np.frombuffer(r.stdout, np.uint8, w * h * 3, i * w * h * 3).reshape(h, w, 3).copy()
                    for i in range(n)]
    return []


# ---------------------------------------------------------------- tracking + rendering

def track_window(model, frames, imgsz, conf):
    tracks = defaultdict(dict)  # tid -> {frame_idx: (x1,y1,x2,y2,conf)}
    for i, f in enumerate(frames):
        r = model.track(f, persist=i > 0, tracker="bytetrack.yaml", imgsz=imgsz, conf=conf, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            continue
        for b, tid, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy().astype(int),
                             r.boxes.conf.cpu().numpy()):
            tracks[int(tid)][i] = (*map(float, b), float(c))
    return tracks


def in_roi(roi, box, w, h):
    if not roi:
        return True
    poly = (np.asarray(roi, np.float32) * np.array([w, h], np.float32)).reshape(-1, 1, 2)
    x1, y1, x2, y2 = box[:4]
    return cv2.pointPolygonTest(poly, ((x1 + x2) / 2, y2 - 0.03 * (y2 - y1)), False) >= 0


def interp_boxes(obs, n):
    """Per-frame boxes for all n frames: linear interpolation inside, hold at the ends, then smooth."""
    idx = np.array(sorted(obs))
    arr = np.array([obs[i][:4] for i in idx], np.float32)
    out = np.stack([np.interp(np.arange(n), idx, arr[:, k]) for k in range(4)], 1)
    k = 5
    pad = np.pad(out, ((k // 2, k // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], np.ones(k) / k, mode="valid") for j in range(4)], 1)


def crop_square(frame, cx, cy, side, out):
    h, w = frame.shape[:2]
    half = side / 2
    M = np.array([[out / side, 0, -(cx - half) * out / side], [0, out / side, -(cy - half) * out / side]], np.float32)
    return cv2.warpAffine(frame, M, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(40, 40, 40)), M


def write_mp4(frames, path, fps):
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate(b"".join(f.tobytes() for f in frames))
    return p.returncode == 0


def guess_behavior(cam, boxes, zones, w, h, fps):
    """Only cues we trust: head in the bunk, at the waterer, or clearly moving. Else no guess."""
    z = zones.get(cam, {})
    mid = boxes[len(boxes) // 2]
    if z.get("feed_bunk"):
        fr = np.median([bbox_fraction_in(z["feed_bunk"], b, w, h) for b in boxes[::3]])
        if fr >= 0.12:
            return "feeding"
    if z.get("waterer"):
        fr = np.median([bbox_fraction_in(z["waterer"], b, w, h) for b in boxes[::3]])
        if fr >= 0.05:
            return "drinking"
    foot = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], 1)
    body = np.median(np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]))
    speed = np.linalg.norm(foot[-1] - foot[0]) / max(body, 1) / (len(boxes) / fps)  # body lengths / s
    if speed > 0.25:
        return "walking"
    return None


def build(a):
    from ultralytics import YOLO
    random.seed(a.seed)
    np.random.seed(a.seed)
    out = os.path.abspath(a.out)
    root = os.path.abspath(a.doc_root)
    for d in ("clips", "context", "crops"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    zones = load_zones(a.zones)
    clips = list_clips(a.input)
    if clips.empty:
        sys.exit(f"no Event*.Avi clips under {a.input}")
    share = {k: float(v) for k, v in (kv.split("=") for kv in a.camera_share.split(","))}
    plan = plan_windows(clips, a.windows, share, a.seed)
    model = YOLO(a.model)

    mpath = os.path.join(out, "manifest.csv")
    old = pd.read_csv(mpath) if os.path.exists(mpath) else pd.DataFrame()
    done_ids = set(old["task_id"]) if len(old) else set()
    rows, tasks = [], []
    rng = random.Random(a.seed)
    for wi, c in enumerate(plan):
        dur = duration(c["path"])
        if dur < a.secs + 10:
            continue
        t0 = round(rng.uniform(3, dur - a.secs - 3), 1)
        frames = decode_window(c["path"], t0, a.secs, a.fps, a.decode_width)
        if not frames:
            print(f"[skip] decode failed {c['path']}")
            continue
        n = len(frames)
        H, W = frames[0].shape[:2]
        tracks = track_window(model, frames, a.imgsz, a.conf)
        roi = zones.get(c["camera"], {}).get("roi")
        keep = {}
        for tid, obs in tracks.items():
            inside = sum(in_roi(roi, b, W, H) for b in obs.values())
            if len(obs) >= a.min_cover * n and inside >= 0.5 * len(obs):
                keep[tid] = interp_boxes(obs, n)
        split = "val" if is_val(c["clip"], a.val_frac) else "train"
        mid = n // 2
        ctx_base = frames[mid].copy()
        for tid, bx in keep.items():
            x1, y1, x2, y2 = map(int, bx[mid])
            cv2.rectangle(ctx_base, (x1, y1), (x2, y2), (200, 200, 200), max(2, W // 900))
        made = 0
        for k, (tid, bx) in enumerate(sorted(keep.items()), 1):
            task_id = f"{c['camera']}_{c['clip']}_t{int(t0):03d}_id{tid:02d}"
            if task_id in done_ids:
                continue
            bw, bh = bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]
            side = float(np.percentile(np.maximum(bw, bh), 90) * a.pad)
            side = max(side, 96.0)
            cxs = (bx[:, 0] + bx[:, 2]) / 2
            cys = (bx[:, 1] + bx[:, 3]) / 2
            vid, crop_idx = [], set(np.linspace(0, n - 1, a.crops_per_track).round().astype(int))
            cdir = os.path.join(out, "crops", task_id)
            os.makedirs(cdir, exist_ok=True)
            for i in range(n):
                img, M = crop_square(frames[i], cxs[i], cys[i], side, a.crop_size)
                if i in crop_idx:
                    cv2.imwrite(os.path.join(cdir, f"f{i:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                p1 = M @ np.array([bx[i, 0], bx[i, 1], 1.0])
                p2 = M @ np.array([bx[i, 2], bx[i, 3], 1.0])
                img = img.copy()
                cv2.rectangle(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 220, 255), 2)
                vid.append(img)
            vpath = os.path.join(out, "clips", task_id + ".mp4")
            if not write_mp4(vid, vpath, a.fps):
                print(f"[skip] encode failed {task_id}")
                continue
            ctx = ctx_base.copy()
            x1, y1, x2, y2 = map(int, bx[mid])
            th = max(4, W // 400)
            cv2.rectangle(ctx, (x1, y1), (x2, y2), (0, 220, 255), th)
            cv2.putText(ctx, f"#{tid}", (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, W / 1600, (0, 220, 255), th)
            s = a.context_width / W
            ctx = cv2.resize(ctx, (a.context_width, int(H * s)), interpolation=cv2.INTER_AREA)
            cpath = os.path.join(out, "context", task_id + ".jpg")
            cv2.imwrite(cpath, ctx, [cv2.IMWRITE_JPEG_QUALITY, 88])

            guess = guess_behavior(c["camera"], bx, zones, W, H, a.fps)
            data = {"video": f"/data/local-files/?d={os.path.relpath(vpath, root)}",
                    "context": f"/data/local-files/?d={os.path.relpath(cpath, root)}",
                    "task_id": task_id, "camera": c["camera"], "clip": c["clip"], "split": split,
                    "clip_start": str(c["start"]), "t_start": t0, "track_label": f"#{tid}", "fps": a.fps}
            task = {"data": data}
            if guess:
                task["predictions"] = [{"model_version": "geometry_cues_v1", "score": 0.3, "result": [{
                    "from_name": "behavior", "to_name": "video", "type": "choices",
                    "value": {"choices": [guess]}}]}]
            tasks.append(task)
            rows.append(dict(task_id=task_id, split=split, camera=c["camera"], clip=c["clip"],
                             clip_path=c["path"], clip_start=str(c["start"]), t_start=t0, secs=a.secs,
                             fps=a.fps, track_id=tid, night=c["night"], n_seen=len(tracks[tid]), n_frames=n,
                             crop_side_px=round(side), decode_w=W,
                             guess=guess or "", seed=a.seed))
            made += 1
        print(f"[{wi + 1}/{len(plan)}] {split:5s} {c['camera']} {c['clip']} t={t0:5.1f}s  "
              f"tracks={len(tracks)} kept={len(keep)} new_tasks={made}", flush=True)

    new = pd.DataFrame(rows)
    allm = pd.concat([old, new], ignore_index=True) if len(old) else new
    allm.to_csv(mpath, index=False)
    tpath = os.path.join(out, f"labelstudio_tasks_seed{a.seed}.json")
    with open(tpath, "w") as f:
        json.dump(tasks, f)
    with open(os.path.join(out, "labelstudio_config.xml"), "w") as f:
        f.write(label_config())
    print(f"\n{len(tasks)} new tasks -> {tpath}  (import this file)")
    print(f"label config -> {os.path.join(out, 'labelstudio_config.xml')}")
    if len(new):
        print(new.groupby(["split", "camera", "night"]).size().to_string())
        print("pre-selected:", new.guess.replace("", "(none)").value_counts().to_dict())


# ---------------------------------------------------------------- export -> YOLO-cls

def _local(url, doc_root):
    q = parse_qs(urlparse(url).query)
    return os.path.join(doc_root, unquote(q["d"][0]) if "d" in q else unquote(url))


def to_cls(a):
    tasks = json.load(open(a.export))
    out = os.path.abspath(a.out)
    root = os.path.abspath(a.doc_root)
    if os.path.isdir(out) and a.clean:
        shutil.rmtree(out)
    rows, skipped = [], defaultdict(int)
    for t in tasks:
        anns = [x for x in t.get("annotations", []) if not x.get("was_cancelled")]
        if not anns:
            skipped["not annotated"] += 1
            continue
        ann = max(anns, key=lambda x: x.get("updated_at") or x.get("created_at") or "")
        behav, flags = None, []
        for r in ann.get("result", []):
            if r.get("from_name") == "behavior":
                behav = (r["value"].get("choices") or [None])[0]
            elif r.get("from_name") == "quality":
                flags = r["value"].get("choices") or []
        d = t["data"]
        rows.append(dict(task_id=d["task_id"], split=d["split"], camera=d["camera"], clip=d["clip"],
                         clip_start=d["clip_start"], t_start=d["t_start"], behavior=behav,
                         bad_track=QUALITY[0] in flags, occluded=QUALITY[1] in flags, unsure=QUALITY[2] in flags))
        if behav is None:
            skipped["no behavior chosen"] += 1
            continue
        if QUALITY[0] in flags:
            skipped["bad track"] += 1
            continue
        if QUALITY[2] in flags and not a.keep_unsure:
            skipped["unsure"] += 1
            continue
        cdir = os.path.join(os.path.dirname(os.path.dirname(_local(d["video"], root))), "crops", d["task_id"])
        dst = os.path.join(out, d["split"], slug(behav))
        os.makedirs(dst, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(cdir, "*.jpg"))):
            shutil.copy2(f, os.path.join(dst, f"{d['task_id']}_{os.path.basename(f)}"))
    df = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    csv = os.path.join(os.path.dirname(out), "behavior_labels.csv")
    df.to_csv(csv, index=False)
    for sp in ("train", "val"):  # ultralytics needs every class folder in both splits
        for b in BEHAVIORS:
            os.makedirs(os.path.join(out, sp, slug(b)), exist_ok=True)
    print(f"clip-level labels -> {csv}  ({len(df)} annotated tasks)")
    if skipped:
        print("skipped:", dict(skipped))
    print(f"YOLO-cls dataset -> {out}")
    for sp in ("train", "val"):
        counts = {b: len(os.listdir(os.path.join(out, sp, slug(b)))) for b in BEHAVIORS}
        print(f"  {sp:5s} images per class: " + ", ".join(f"{slug(k)}={v}" for k, v in counts.items()))
    print(f"\ntrain:  yolo classify train model=yolo11s-cls.pt data={out} imgsz=224 epochs=60 batch=64")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--input", default="input")
    b.add_argument("--out", default="data/behavior_annotation")
    b.add_argument("--doc-root", default=".", help="LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT (project root)")
    b.add_argument("--zones", default="configs/camera_zones.yaml")
    b.add_argument("--model", default="models/angus_det_hand_e32.pt")
    b.add_argument("--windows", type=int, default=60, help="5 s windows to sample (each gives several cows)")
    b.add_argument("--camera-share", default="cam1=0.35,cam2=0.3,cam3=0.35")
    b.add_argument("--secs", type=float, default=5.0)
    b.add_argument("--fps", type=int, default=10)
    b.add_argument("--decode-width", type=int, default=2560)
    b.add_argument("--imgsz", type=int, default=1280)
    b.add_argument("--conf", type=float, default=0.3)
    b.add_argument("--min-cover", type=float, default=0.7, help="track must be seen in this share of frames")
    b.add_argument("--pad", type=float, default=1.6, help="crop side = 90th pct box side x pad")
    b.add_argument("--crop-size", type=int, default=384)
    b.add_argument("--crops-per-track", type=int, default=5)
    b.add_argument("--context-width", type=int, default=1280)
    b.add_argument("--val-frac", type=float, default=0.25)
    b.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("to-cls")
    c.add_argument("--export", required=True, help="Label Studio JSON export of the behavior project")
    c.add_argument("--out", default="data/behavior_annotation/cls")
    c.add_argument("--doc-root", default=".")
    c.add_argument("--keep-unsure", action="store_true")
    c.add_argument("--clean", action="store_true", default=True)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else to_cls(a)


if __name__ == "__main__":
    main()
