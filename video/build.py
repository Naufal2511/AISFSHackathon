#!/usr/bin/env python3
"""Build the demo video from our real outputs.

  python3 video/build.py                 # everything (prep data, voice-over, render, assemble)
  python3 video/build.py --only 5 6      # re-render some scenes, then re-assemble
  python3 video/build.py --assemble      # only re-assemble from rendered scenes

Voice-over: video/vo/vo_NN.(wav|m4a|mp3) when present (scene 14 may be vo_14a/b/c), otherwise a scratch
computer voice (macOS `say`) so the timing can be checked before recording. Each scene lasts as long as its
voice-over plus a short pause. Lines come from docs/VIDEO_SCRIPT.md.

Writes outputs/video/demo.mp4 and outputs/video/demo.srt. Work files go to outputs/video/cache/
(git-ignored: event data must stay within the event).
"""
import argparse
import csv
import collections
import datetime as dt
import glob
import json
import math
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VID = os.path.join(ROOT, "video")
CACHE = os.path.join(ROOT, "outputs", "video", "cache")
OUT = os.path.join(ROOT, "outputs", "video")
ST = os.path.join(ROOT, "outputs", "stitched")
SCRATCH_VOICE = {"A": "Daniel", "B": "Samantha", "C": "Karen"}
LEAD, TAIL = 0.45, 0.55

FEED = dict(run="hprc_day2_c18", start="2026-11-08 16:28:00", seconds=120)   # the feed rush
WORLD = dict(run="afeef_training_c04", start="2026-11-07 15:58:00", seconds=240)


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def probe_dur(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


def ts(w):
    return int(dt.datetime.fromisoformat(w[:19]).timestamp())


# ---------------------------------------------------------------- script
def read_script():
    t = open(os.path.join(ROOT, "docs", "VIDEO_SCRIPT.md")).read()
    out = {}
    for m in re.finditer(r"^\*\*(\d+) · ([A-C, ]+)\*\*\n> (.+)$", t, re.M):
        out[int(m.group(1))] = (m.group(2).strip(), m.group(3).strip())
    return out


# ---------------------------------------------------------------- data prep
def page(scene_dir, template, data=None, extra=None):
    os.makedirs(scene_dir, exist_ok=True)
    common = open(os.path.join(VID, "scenes", "common.html")).read()
    s = open(template).read().replace("<!--COMMON-->", common)
    if data is not None:
        s = s.replace("__DATA__", json.dumps(data))
    for k, v in (extra or {}).items():
        s = s.replace(k, v)
    open(os.path.join(scene_dir, "index.html"), "w").write(s)


def window(name, cfg, frames=True):
    d = os.path.join(CACHE, name)
    if not os.path.exists(os.path.join(d, "world.json")) or (frames and not os.path.isdir(os.path.join(d, "frames", "cam1"))):
        cmd = ["python3", os.path.join(VID, "animations", "extract_window.py"), "--run", os.path.join(ST, "runs", cfg["run"]),
               "--start", cfg["start"], "--seconds", str(cfg["seconds"]), "--out", d, "--videos", os.path.join(ROOT, "input")]
        sh(cmd + ([] if frames else ["--no-frames"]), stderr=subprocess.DEVNULL)
    return d


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def find_clip(name):
    p = glob.glob(os.path.join(ROOT, "input", "**", name), recursive=True)
    return p[0] if p else None


def extract(clip, off, secs, out_dir, vf, fps=15):
    os.makedirs(out_dir, exist_ok=True)
    if len(os.listdir(out_dir)) > 10:
        return len(os.listdir(out_dir))
    sh(["ffmpeg", "-v", "error", "-y", "-vcodec", "hevc", "-ss", str(off), "-i", clip, "-t", str(secs),
        "-vf", f"fps={fps},{vf}", "-q:v", "3", os.path.join(out_dir, "%04d.jpg")], stderr=subprocess.DEVNULL)
    return len(os.listdir(out_dir))


def prep_s01(d):
    fill = "scale=1920:1440,crop=1920:1080"
    n_night = extract(find_clip("Event20261108020058006.Avi"), 60, 10, os.path.join(d, "frames", "s01", "night"), fill)
    n_day = extract(find_clip("Event20261108162542003.Avi"), 138, 10, os.path.join(d, "frames", "s01", "day"), fill)
    page(d, os.path.join(VID, "scenes", "s01_cold_open.html"),
         dict(night=n_night, day=n_day, fps=15, nightTag="FRONT CAMERA · 8 NOV 02:01 · INFRARED", dayTag="BACK CAMERA · 8 NOV 16:28"))


def prep_sync(d, focus=None):
    w = window("win_feed", FEED)
    os.makedirs(d, exist_ok=True)
    link(os.path.join(w, "frames"), os.path.join(d, "frames"))
    tpl = open(os.path.join(VID, "animations", "sync_view.html")).read()
    s = (tpl.replace("__WORLD__", open(os.path.join(w, "world.json")).read())
            .replace("__BOXES__", open(os.path.join(w, "boxes.json")).read())
            .replace("__FOCUS__", json.dumps(focus)))
    open(os.path.join(d, "index.html"), "w").write(s)


def prep_s04(d):
    w = window("win_world", WORLD, frames=False)
    os.makedirs(d, exist_ok=True)
    s = open(os.path.join(VID, "animations", "world_state.html")).read().replace("__DATA__", open(os.path.join(w, "world.json")).read())
    open(os.path.join(d, "index.html"), "w").write(s)


def prep_s05(d):
    w = window("win_feed", FEED)
    world = json.load(open(os.path.join(w, "world.json")))
    boxes = json.load(open(os.path.join(w, "boxes.json")))
    W_HEAD = {"toward": 2.0, "side-on": 0.8}
    W_ID = {"side-on": 1.5}
    best = None
    for sec, v in enumerate(world["vis"]):
        for cow, cams in v.items():
            if len(cams) < 3:
                continue
            fac = {c["f"] for c in cams.values()}
            if "unknown" in fac:
                continue
            sc = lambda W: {k: c["a"] * (1.0 if c["v"] == "visible" else 0.4) * W.get(c["f"], 0.2 if W is W_HEAD else 1.0) for k, c in cams.items()}
            h, i = sc(W_HEAD), sc(W_ID)
            diff = max(h, key=h.get) != max(i, key=i.get)
            stat = len({c["v"] for c in cams.values()}) > 1
            q = (diff, stat, len(fac), min(c["a"] for c in cams.values()))
            if best is None or q > best[0]:
                best = (q, sec, cow, cams)
    _, sec, cow, cams = best
    # full-resolution crops of that second
    run = os.path.join(ST, "runs", FEED["run"], "contract", "observations.csv")
    t0 = ts(FEED["start"])
    rows = {}
    for r in csv.DictReader(open(run)):
        if ts(r["wall_time"]) - t0 == sec and r["cow_id"] == cow and r["box_role"] == "body":
            rows[r["camera"]] = r
    crops = []
    os.makedirs(os.path.join(d, "crops"), exist_ok=True)
    for cam in ("cam1", "cam2", "cam3"):
        r = rows[cam]
        x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
        fw, fh = float(r["frame_w"]), float(r["frame_h"])
        cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)
        side_w = max(bw * 1.6, bh * 1.6 * 4 / 3, 640)
        side_h = side_w * 3 / 4
        X = int(min(max(0, cx - side_w / 2), fw - side_w)); Y = int(min(max(0, cy - side_h / 2), fh - side_h))
        out = os.path.join(d, "crops", f"{cam}.jpg")
        bx, by = int((x1 - X) / side_w * 960), int((y1 - Y) / side_h * 720)
        bw2, bh2 = int(bw / side_w * 960), int(bh / side_h * 720)
        sh(["ffmpeg", "-v", "error", "-y", "-vcodec", "hevc", "-ss", f"{float(r['offset_s']):.2f}", "-i", find_clip(r["file"]),
            "-frames:v", "1", "-vf", f"crop={int(side_w)}:{int(side_h)}:{X}:{Y},scale=960:720,drawbox=x={bx}:y={by}:w={bw2}:h={bh2}:color=0xa85a24@0.95:t=5",
            "-q:v", "2", out], stderr=subprocess.DEVNULL)
        c = cams[cam]
        crops.append(dict(cam=cam, img=f"crops/{cam}.jpg", status=c["v"], facing=c["f"], area=c["a"], occl=c["o"]))
    hhmmss = (dt.datetime.fromisoformat(FEED["start"]) + dt.timedelta(seconds=sec)).strftime("%H:%M:%S")
    # the minute funnel (48 h, 10 cows)
    reasons = collections.Counter()
    for r in csv.DictReader(open(os.path.join(ST, "final", "activity_minute.csv"))):
        if r["date"] != "2026-11-09":
            reasons[r["reason"].split(":")[0]] += 1
    total = sum(reasons.values())
    agent = reasons["agent_accept"] + reasons["agent_revised"] + reasons["unresolved_fallback_tier1"]
    funnel = dict(total=total, rows=[
        ["All cow-minutes", "10 cows × 48 h", total, "#c9d0d7"],
        ["Passed cheap rule checks", "speed, zone, chewing screen · no model call", reasons["keep_tier1"], "#2e8b6e"],
        ["Not visible to any camera", "reported as not visible, never guessed", reasons["no_upstream_data"] + reasons["miss_not_recovered"], "#9fb0c0"],
        ["Abstained up front", "too little to go on", reasons["auto_abstain"], "#c9d0d7"],
        ["Calm bout carried", "verified once, then carried", reasons["stable_carry"], "#7fb3c0"],
        ["Agent looked", "flagged or audited minutes only", agent, "#a85a24"]])
    page(d, os.path.join(VID, "scenes", "s05_where_to_look.html"), dict(cow=cow, time=f"8 Nov {hhmmss}", crops=crops, funnel=funnel))


def prep_s06(d):
    w = window("win_feed", FEED)
    # a real anchor second: the most cows "clear" at once (observed and >= 2.5 m from every other cow)
    run = os.path.join(ST, "runs", FEED["run"], "contract", "cow_state.csv")
    per = collections.defaultdict(dict)
    for r in csv.DictReader(open(run)):
        if r["predicted"] == "False" and r["x"]:
            per[r["wall_time"][:19]][r["cow_id"]] = (float(r["x"]), float(r["y"]))
    best = None
    for k, cows in per.items():
        if len(cows) < 9:
            continue
        clear = [c for c, p in cows.items() if all(math.dist(p, q) >= 2.5 for o, q in cows.items() if o != c)]
        if best is None or len(clear) > len(best[2]):
            best = (k, cows, clear)
    k, cows, clear = best
    anchor = dict(time=f"8 Nov {k[11:19]}", n_clear=len(clear), n=len(cows),
                  cows=[dict(id=c, x=x, y=y, clear=c in clear) for c, (x, y) in sorted(cows.items())])
    # example beliefs over the 10 cows for one track in each camera (the consensus update itself is the real one)
    beliefs = [[.04, .03, .05, .42, .04, .26, .04, .04, .04, .04],
               [.05, .04, .04, .28, .05, .40, .04, .03, .04, .03],
               [.08, .08, .09, .20, .09, .12, .09, .08, .09, .08]]
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(ST, "id_agent", "hprc_day2_c18", "id_checks", "id_check_20261108_163000.jpg"), os.path.join(d, "id_check.jpg"))
    page(d, os.path.join(VID, "scenes", "s06_identity.html"), dict(anchor=anchor, beliefs=beliefs))


def prep_s07(d):
    import numpy as np
    mem = os.path.join(ROOT, "outputs", "agent_d1_1600", "memory")
    card = next(c for c in json.load(open(os.path.join(mem, "cards.json"))) if c["cow_id"] == "COW_01")
    os.makedirs(os.path.join(d, "crops"), exist_ok=True)
    crops = []
    for i, p in enumerate(card["references"][:6]):
        shutil.copy(os.path.join(ROOT, p), os.path.join(d, "crops", f"{i}.jpg")); crops.append(f"crops/{i}.jpg")
    proto = np.load(os.path.join(mem, "prototypes.npz"))
    v = proto["COW_01"] if "COW_01" in proto.files else proto[proto.files[0]]
    v = np.asarray(v, dtype=float).ravel()[:40]
    v = ((v - v.min()) / (np.ptp(v) + 1e-9)).round(3).tolist()
    facts = [dict(time="16:01:54", source="enrolment", text="reference crops from the front, bunk and back cameras; confirmed once by a person"),
             dict(time="16:02", source="VLM · traits", text=f"ear tag {card['ear_tag']}; {card['traits']}")]
    facts += [dict(time=n["time"], source=n["source"], text=n["text"].replace("[uncertain] ", "")) for n in card["notes"][-1:]]
    trace = os.path.join(ST, "id_agent", "hprc_day2_c18", "traces")
    shutil.copy(os.path.join(trace, "img", "weak_id_1794154820_cam1_12_s2_0.jpg"), os.path.join(d, "query.jpg"))
    query = dict(title="Two boxes on the front camera both say COW_05", img="query.jpg", steps=[
        "<code>compare_to_cards</code>: track 12 vs COW_05's card · similarity <b>0.204</b>, too weak to decide",
        "<code>cow_state</code> / <code>all_views</code>: track 9 already holds COW_05 at the <b>same spot</b>",
        "self-check: a relabel would need two kinds of evidence, and only one agrees"],
        outcome="<b>uncertain</b> · a duplicate detection of COW_05; the ID is <b>not forced</b>")
    page(d, os.path.join(VID, "scenes", "s07_memory.html"), dict(card=dict(
        id=card["cow_id"], ear_tag=card["ear_tag"], traits=card["traits"],
        last_seen=f"{card['last_seen']['camera']} · {card['last_seen']['time'][11:16]}",
        cams=f"front · bunk · back · {card['stats']['n_observations']:,} observations", crops=crops, proto=v, facts=facts), query=query))


def prep_s08(d):
    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "behavior_annotation", "manifest.csv"))))
    pick, seen = [], collections.Counter()
    for r in man:
        key = (r["camera"], r["night"])
        f = os.path.join(ROOT, "data", "behavior_annotation", "crops", r["task_id"], "f24.jpg")
        if os.path.exists(f) and seen[key] < 3:
            seen[key] += 1; pick.append((r, f))
    os.makedirs(os.path.join(d, "tiles"), exist_ok=True)
    tiles = []
    names = {"cam1": "FRONT", "cam2": "BUNK", "cam3": "BACK"}
    for i, (r, f) in enumerate(pick[:18]):
        shutil.copy(f, os.path.join(d, "tiles", f"{i}.jpg"))
        tiles.append(dict(img=f"tiles/{i}.jpg", tag=f"{names.get(r['camera'], r['camera'])} · {'IR' if r['night'] == 'True' else 'DAY'}"))
    page(d, os.path.join(VID, "scenes", "s08_behavior.html"), dict(tiles=tiles, n_tasks=len(glob.glob(os.path.join(ROOT, "data", "behavior_annotation", "crops", "*", "f24.jpg"))), n_cams=3))


def prep_s09(d):
    tdir = os.path.join(ST, "final", "traces")
    md = open(os.path.join(tdir, "success_COW_02_d1_m0862.md")).read()
    readme = open(os.path.join(tdir, "README.md")).read()
    m = re.search(r"## success: (COW_\d+) at ([0-9-]+ [0-9:]+)", readme)
    gid, when = m.group(1), m.group(2)
    steps = [dict(label="gate", kind="", html="sent to the agent: <b>cameras disagree</b>, rare class at modest confidence")]
    want = ["best_camera", "zone_query", "track_history", "temporal_window", "zoom", "get_frames"]
    for line in md.splitlines():
        mt = re.match(r"\d+\. \*\*Tool\*\* `(\w+)` .*?_\(why: (.+?);", line)
        if mt and mt.group(1) in want:
            want.remove(mt.group(1))
            why = mt.group(2).rstrip(".")
            why = why[:110] + ("…" if len(why) > 110 else "")
            steps.append(dict(label="tool", kind="tool", html=f"<code>{mt.group(1)}</code> <span class='why'>{why}</span>"))
    steps.append(dict(label="verify", kind="", html="hypothesis checked by rules: <b>0 hard, 0 soft</b> violations"))
    strip = []
    os.makedirs(os.path.join(d, "img"), exist_ok=True)
    for i, p in enumerate(re.findall(r"\]\((img/[^)]+)\)", md)[:6]):
        shutil.copy(os.path.join(tdir, p), os.path.join(d, "img", f"{i}.png")); strip.append(f"img/{i}.png")
    data = {"title": f"{gid} · {dt.datetime.fromisoformat(when).strftime('%-d %b %H:%M')} · drinking or walking?",
            "steps": steps[:9], "strip": strip, "from": "drinking (0.69)", "to": "walking · 0.96",
            "evidence": "upright and changing position across frames; 9.81 m covered in the minute; 0 % of the minute in the waterer zone."}
    page(d, os.path.join(VID, "scenes", "s09_agent.html"), data)


def prep_s10(d):
    basis = collections.Counter()
    for r in csv.DictReader(open(os.path.join(ST, "final", "activity_minute.csv"))):
        if r["date"] != "2026-11-09":
            basis[r["basis"]] += 1
    total = sum(basis.values())
    B = [["observed · label passed rule checks", basis["observed"], "#0f5a6b"],
         ["verified · the agent looked at the frames", basis["verified"], "#a85a24"],
         ["unverified · seen, agent could not confirm", basis["unverified"], "#d69a2d"],
         ["estimated · calm bout carried / gap filled", basis["estimated"], "#7fb3c0"],
         ["not visible · no camera saw the cow", basis["not_visible"], "#c9d0d7"]]
    corr = collections.Counter(r["correction"] for r in csv.DictReader(open(os.path.join(ST, "final", "activity_minute.csv"))))
    bouts = sum(1 for _ in open(os.path.join(ST, "final", "activity.csv"))) - 1
    tiles = [[f"{bouts:,}", "behavior bouts in activity.csv"], [f"{corr['behavior_corrected']:,}", "minutes corrected by the agent", "w"],
             [f"{corr['confirmed']:,}", "minutes confirmed on video"], [f"{corr['abstained']:,}", "minutes it refused to guess", "w"]]
    rows = list(csv.DictReader(open(os.path.join(ST, "final", "activity.csv"))))
    sel = [r for r in rows if r["cow_id"] == "COW_02" and r["start_time"] >= "2026-11-08 15:00"][:26]
    csvl = [f"{'cow':7} {'start':5} {'end':5} {'min':>4}  {'behavior':24} {'basis':10}"]
    for r in sel:
        csvl.append(f"{r['cow_id']:7} {r['start_time'][-5:]:5} {r['end_time'][-5:]:5} {r['duration_min']:>4}  {r['behavior'][:24]:24} {r['basis'][:10]:10}")
    page(d, os.path.join(VID, "scenes", "s10_results.html"), dict(basis=B, total=total, tiles=tiles, csv=csvl))


def prep_s11(d):
    m = json.load(open(os.path.join(ST, "stitch_manifest.json")))
    ch = [dict(who=re.sub(r"_c\d+$", "", c["name"]), s=ts(c["start"]), e=ts(c["end"])) for c in m["chunks"]]
    st = collections.Counter(r["status"].split(" ")[0] for r in csv.DictReader(open(os.path.join(ST, "id_map.csv"))))
    joins = sum(v for k, v in st.items() if k != "first_chunk")
    box = st["overlap"] + st["overlap_split"]
    stats = [[str(len(ch)), "chunks: Day 1 on three laptops, Day 2 as an HPRC GPU array job", "#18212b"],
             [f"{box} / {joins}", "cow joins from the same boxes in the same frames (2-min overlaps)", "#0f5a6b"],
             [str(st["matched"]), "joins from a clear appearance margin", "#2e8b6e"],
             [str(st["low_margin"] + st["guess"]), "weak joins, listed in id_map.csv for review", "#a85a24"]]
    t0 = ts("2026-11-07 00:00:00")
    page(d, os.path.join(VID, "scenes", "s11_stitch.html"), dict(chunks=ch, t0=t0, t1=t0 + 48 * 3600, stats=stats))


def risk_rows():
    rows = list(csv.DictReader(open(os.path.join(ST, "level2", "brd_risk.csv"))))
    rows.sort(key=lambda r: int(r["rank"]))
    why = {"COW_02": "27 vs 19 min to reach the bunk after 14 feed rushes; walks less (88 vs 101 m/h); feeds 7 vs 10 %",
           "COW_01": "feeds 4 vs 8 % of the time; bunk visits 5.8 vs 9.5 min; lies 57 vs 51 %",
           "COW_10": "short bunk visits; longest lying bout 139 vs 100 min; bunk visits falling faster than the herd's"}
    return [dict(id=r["cow_id"], score=float(r["risk_score"]), p_top3=float(r["p_top3"] or 0), why=why.get(r["cow_id"], "")) for r in rows]


def prep_s12(d):
    page(d, os.path.join(VID, "scenes", "s12_ranking.html"), dict(rows=risk_rows()))


def prep_s13(d):
    top = [[r["id"], f"{r['score']:.2f}", round(100 * r["p_top3"])] for r in risk_rows()[:3]]
    page(d, os.path.join(VID, "scenes", "s13_honest.html"), dict(top=top))


def prep_s14(d):
    page(d, os.path.join(VID, "scenes", "s14_close.html"), {})


# scene -> list of (part name, prep function, share of the scene's time or None for the rest)
FOCUS = {"id": "COW_02", "from": 0.6, "score": "0.37", "why": "27 vs 19 min to reach the bunk after a feed rush", "box": [300, 690]}
SCENES = {
    1: [("s01", prep_s01, None)],
    2: [("s02", prep_sync, None)],
    3: [("s03", lambda d: page(d, os.path.join(VID, "scenes", "s03_pipeline.html")), None)],
    4: [("s04", prep_s04, None)],
    5: [("s05", prep_s05, None)],
    6: [("s06", prep_s06, None)],
    7: [("s07", prep_s07, None)],
    8: [("s08", prep_s08, None)],
    9: [("s09", prep_s09, None)],
    10: [("s10", prep_s10, None)],
    11: [("s11", prep_s11, None)],
    12: [("s12a", lambda d: prep_sync(d, FOCUS), 0.36), ("s12b", prep_s12, None)],
    13: [("s13", prep_s13, None)],
    14: [("s14", prep_s14, None)],
}


# ---------------------------------------------------------------- voice-over
def voice(n, who, text):
    vo = os.path.join(CACHE, "vo")
    os.makedirs(vo, exist_ok=True)
    out = os.path.join(vo, f"vo_{n:02d}.wav")
    mine = sorted(glob.glob(os.path.join(VID, "vo", f"vo_{n:02d}.*")))
    parts = sorted(glob.glob(os.path.join(VID, "vo", f"vo_{n:02d}[abc].*")))
    if mine or parts:
        srcs = mine[:1] or parts
        inputs = sum((["-i", s] for s in srcs), [])
        sh(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex",
            "".join(f"[{i}:a]" for i in range(len(srcs))) + f"concat=n={len(srcs)}:v=0:a=1,aresample=48000,pan=mono|c0=c0[a]", "-map", "[a]", out])
        return out, "recorded"
    if n == 14:
        segs = re.findall(r"([ABC]): ([^:]+?)(?= [ABC]:|$)", text)
    else:
        segs = [(who.split(",")[0].strip(), text)]
    pieces = []
    for i, (w, t) in enumerate(segs):
        aiff = os.path.join(vo, f"tmp_{n}_{i}.aiff")
        sh(["say", "-v", SCRATCH_VOICE.get(w, "Daniel"), "-r", "158", "-o", aiff, t.replace("COW_", "cow ").replace("4K", "four K")])
        pieces.append(aiff)
    inputs = sum((["-i", p] for p in pieces), [])
    gap = "".join(f"[{i}:a]apad=pad_dur=0.35[p{i}];" for i in range(len(pieces)))
    sh(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex",
        gap + "".join(f"[p{i}]" for i in range(len(pieces))) + f"concat=n={len(pieces)}:v=0:a=1,aresample=48000[a]", "-map", "[a]", out])
    return out, "scratch"


# ---------------------------------------------------------------- render + assemble
def render(scene_dir, secs, out):
    sh(["node", os.path.join(VID, "capture.js"), os.path.join(scene_dir, "index.html"), out, f"{secs:.3f}", "30"],
       stdout=subprocess.DEVNULL)


def srt_time(s):
    h, r = divmod(s, 3600); m, r = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(r):02d},{int(round((r % 1) * 1000)) % 1000:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    if not os.path.isdir(os.path.join(VID, "node_modules", "puppeteer-core")):
        sh(["npm", "install", "--silent"], cwd=VID)
    script = read_script()
    plan = []
    for n in sorted(SCENES):
        who, text = script[n]
        wav, kind = voice(n, who, text)
        dur = max(4.0, probe_dur(wav) + LEAD + TAIL)
        plan.append(dict(n=n, wav=wav, kind=kind, dur=dur, text=text, who=who))
    total = sum(p["dur"] for p in plan)
    print(f"timeline {total:.1f} s ({total / 60:.2f} min); voice: {collections.Counter(p['kind'] for p in plan)}")

    jobs = []
    for p in plan:
        rest = p["dur"]
        p["parts"] = []
        for name, prep, share in SCENES[p["n"]]:
            secs = round(p["dur"] * share, 2) if share else round(rest, 2)
            rest -= secs if share else 0
            sdir = os.path.join(CACHE, "scenes", name)
            mp4 = os.path.join(CACHE, "scenes", f"{name}.mp4")
            p["parts"].append(mp4)
            if a.assemble:
                continue
            if a.only and p["n"] not in a.only and os.path.exists(mp4) and abs(probe_dur(mp4) - secs) < 0.1:
                continue
            prep(sdir)
            jobs.append((sdir, secs, mp4))
    with ThreadPoolExecutor(a.jobs) as ex:
        for f in [ex.submit(render, *j) for j in jobs]:
            f.result()
    print(f"rendered {len(jobs)} parts")

    # audio: each scene's voice placed after LEAD, padded to the scene length
    lst_v, lst_a, subs, t = [], [], [], 0.0
    for p in plan:
        sa = os.path.join(CACHE, f"a_{p['n']:02d}.wav")
        sh(["ffmpeg", "-v", "error", "-y", "-i", p["wav"], "-af", f"adelay={int(LEAD * 1000)},apad", "-t", f"{p['dur']:.3f}", "-ar", "48000", "-ac", "1", sa])
        lst_a.append(sa)
        lst_v += p["parts"]
        # captions: sentences spread over the voice
        sents = [s for s in re.split(r"(?<=[.?!:])\s+", re.sub(r"\b[ABC]: ", "", p["text"])) if s]
        vo_len = p["dur"] - LEAD - TAIL
        L = sum(len(s) for s in sents)
        c = t + LEAD
        for s in sents:
            d = vo_len * len(s) / L
            subs.append((c, c + d, s)); c += d
        t += p["dur"]
    with open(os.path.join(CACHE, "v.txt"), "w") as f:
        f.writelines(f"file '{v}'\n" for v in lst_v)
    with open(os.path.join(CACHE, "a.txt"), "w") as f:
        f.writelines(f"file '{x}'\n" for x in lst_a)
    sh(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", os.path.join(CACHE, "v.txt"), "-c", "copy", os.path.join(CACHE, "video.mp4")])
    sh(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", os.path.join(CACHE, "a.txt"),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", os.path.join(CACHE, "audio.wav")])
    out = os.path.join(OUT, "demo.mp4")
    sh(["ffmpeg", "-v", "error", "-y", "-i", os.path.join(CACHE, "video.mp4"), "-i", os.path.join(CACHE, "audio.wav"),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out])
    with open(os.path.join(OUT, "demo.srt"), "w") as f:
        for i, (a0, a1, s) in enumerate(subs, 1):
            f.write(f"{i}\n{srt_time(a0)} --> {srt_time(a1)}\n{s}\n\n")
    print(f"wrote {out} ({probe_dur(out):.1f} s) and demo.srt")


if __name__ == "__main__":
    main()
