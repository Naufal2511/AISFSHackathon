import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from brd_agent.config import VideoCfg
from brd_agent.store import Store as BrdDB
from brd_agent.video import Clock, VideoIndex, parse_event_filename, parse_timestamp
from brd_agent.video_backend import VideoBackend
from brd_agent.yolo import aggregate_minutes, import_yolo

ROOT = Path(__file__).resolve().parents[1]
NAME = "Event20261107033930006.Avi"
W, H, FPS, SECS = 320, 240, 10, 120


def make_clip(path: Path, night=False):
    """C01 (box 40..120 x 80..160) walks right; C02 (box 200..280 x 60..140) is still but its head flickers at 1 Hz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(FPS * SECS):
        t = i / FPS
        base = 18 if night else 120
        f = np.full((H, W, 3), base, np.uint8)
        if night:
            f[..., 1:] = f[..., :1]                              # grayscale / infrared look
        x = int(40 + t * 1.0)
        cv2.rectangle(f, (x, 80), (x + 80, 160), (60, 60, 60) if night else (30, 30, 200), -1)
        flick = 80 if int(t * 2) % 2 == 0 else 0                 # 1 cycle per second
        cv2.rectangle(f, (200, 60), (280, 90), (base + flick,) * 3, -1)
        cv2.rectangle(f, (200, 90), (280, 140), (50, 50, 50), -1)
        vw.write(f)
    vw.release()


@pytest.fixture()
def world(tmp_path):
    root = tmp_path / "videos"
    make_clip(root / "cam_a" / NAME)
    rows = []
    for s in range(0, SECS, 1):
        x = int(40 + s * 1.0)
        rows.append({"file": NAME, "timestamp": s, "cow_id": "C01", "behavior": "walking", "confidence": 0.8,
                     "x1": x, "y1": 80, "x2": x + 80, "y2": 160})
        rows.append({"file": NAME, "timestamp": s, "cow_id": "C02", "behavior": "ruminating_standing", "confidence": 0.7,
                     "x1": 200, "y1": 60, "x2": 280, "y2": 140})
    yp = tmp_path / "yolo.csv"
    with open(yp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    db = BrdDB(str(tmp_path / "d_store"))
    cfg = VideoCfg(root=str(root))
    idx = VideoIndex.scan(root)
    return db, idx, cfg, yp, tmp_path


def test_event_filename_and_timestamp_parsing():
    assert parse_event_filename(NAME) == datetime(2026, 11, 7, 3, 39, 30, 6000)
    assert parse_event_filename("nothing.avi") is None
    start = parse_event_filename(NAME)
    assert parse_timestamp(12.5, start)[0] == 12.5
    assert parse_timestamp("00:01:05.5", start)[0] == 65.5
    off, absdt = parse_timestamp("2026-11-07 03:40:00", start)
    assert round(off, 1) == 29.99 or round(off, 1) == 30.0 and absdt.minute == 40
    c = Clock.from_first(start, "midnight")
    assert c.to_dm(datetime(2026, 11, 7, 3, 40, 0)) == (1, 220)
    assert c.to_dm(datetime(2026, 11, 8, 0, 0, 5)) == (2, 0)


def test_index_scan_probes_clip_and_locates(world):
    db, idx, cfg, yp, _ = world
    assert len(idx.clips) == 1 and idx.clips[0].camera == "back"           # first unmapped folder -> 'back' (assumption)
    clip = idx.clips[0]
    assert abs(clip.duration_s - SECS) < 1 and (clip.width, clip.height) == (W, H)
    assert idx.locate("back", datetime(2026, 11, 7, 3, 40, 0)) is not None
    assert idx.locate("back", datetime(2026, 11, 7, 4, 0, 0)) is None


def test_import_and_aggregate_minutes_and_find_missing(world):
    db, idx, cfg, yp, _ = world
    st = import_yolo(db, yp, idx, cfg)
    assert st["rows"] == 2 * SECS and not st["skipped"]
    res = aggregate_minutes(db, cfg, idx, expected_cows=["C01", "C02", "C03"])
    assert res["rows"] >= 2 and res["find_missing"] > 0
    r = db.get_row("C01", 1, 220)
    assert r["tier1_label"] == "walking" and r["task"] == "verify" and r["src_file"] == NAME and r["src_camera"] == "back"
    assert r["speed_px_s"] and 0.5 < r["speed_px_s"] < 2.0 and r["vis_back"] > 0.9
    m = db.get_row("C03", 1, 220)
    assert m["task"] == "find_missing" and m["tier1_top_json"] is None and "no_yolo_record" in m["yolo_flags"]


def test_backend_measures_pixels_not_labels(world):
    db, idx, cfg, yp, tmp = world
    import_yolo(db, yp, idx, cfg); aggregate_minutes(db, cfg, idx, ["C01", "C02"])
    be = VideoBackend(str(tmp / "d_store"), idx, cfg, workdir=str(tmp / "w"))
    p = be.crop("C01", 1, 220, "back", 0, "body")
    assert p and Path(p).exists()
    head = be.crop("C02", 1, 220, "back", 0, "head", 2.0)
    assert head and cv2.imread(head).shape[0] >= 224
    assert be.crop("C01", 1, 220, "front") is None                          # no such camera
    th = be.track_history("C01", 1, 220, 60)
    assert 0.5 < th["speed_mean"] < 2.0 and th["posture_mode"] == "unknown"
    assert be.track_history("C02", 1, 220, 60)["speed_mean"] < 0.1
    # rumination: C02's head flickers at 1 Hz, C01's does not
    jaw2 = be.burst_motion("C02", 1, 220, "back", 20)
    jaw1 = be.burst_motion("C01", 1, 220, "back", 20)
    assert jaw2["available"] and jaw2["jaw_motion_score"] > 0.3 and 0.7 <= jaw2["periodicity_hz"] <= 1.3
    assert jaw1["jaw_motion_score"] < jaw2["jaw_motion_score"] / 2
    assert be.zone_info("C01", 1, 220) == {"available": False, "note": "no pen map configured"}
    nb = be.neighbors("C01", 1, 220)
    assert nb["nn_cow_id"] == "C02" and nb["nn_dist_px"] > 0
    assert be.exemplar_similarity("C01", p) is None                        # no gallery -> honest "unavailable"
    scene = be.crop("C01", 1, 220, "back", 0, "scene")
    assert scene and Path(scene).exists()


# ---------------------------------------------------------------- end to end on synthetic video
def _boxrow(cow, s, box, beh="standing", conf=0.95, file=NAME):
    return {"file": file, "timestamp": s, "cow_id": cow, "behavior": beh, "confidence": conf,
            "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}


def _write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def test_end_to_end_video_backend_checks_yolo_and_recovers_a_miss(tmp_path):
    from brd_agent.backend import DataBackend
    from brd_agent.config import AgentConfig
    from brd_agent.pipeline import run_db
    from brd_agent.vlm_client import MockVLM
    root = tmp_path / "videos"
    make_clip(root / "cam_a" / NAME)
    rows = []
    for s in range(SECS):
        x = int(40 + s)
        rows.append(_boxrow("C01", s, (x, 80, x + 80, 160), "standing"))               # YOLO says standing, she is walking (1 px/s)
        rows.append(_boxrow("C02", s, (200, 60, 280, 140), "standing"))                # YOLO says standing, head flickers at 1 Hz
        if s < 30:                                                                      # C03 seen in minute 219, then missed
            rows.append(_boxrow("C03", s, (200, 160, 280, 235), "standing"))
    yp = tmp_path / "y.csv"; _write_rows(yp, rows)
    dbp = str(tmp_path / "e_store"); db = BrdDB(dbp)
    cfgv = VideoCfg(root=str(root)); idx = VideoIndex.scan(root)
    import_yolo(db, yp, idx, cfgv); aggregate_minutes(db, cfgv, idx, ["C01", "C02", "C03"]); db.close()

    def detector(img):                                    # a fake "your own YOLO weights"
        return [(40 + 60, 80, 120 + 60, 160, 0.9), (200, 60, 280, 140, 0.9), (203, 163, 283, 238, 0.8)]

    cfg = AgentConfig(behaviors_path=str(ROOT / "configs" / "behaviors.yaml")); cfg.verify.audit_rate = 0.0
    cfg.video = cfgv
    run_db(dbp, cfg, lambda: VideoBackend(dbp, idx, cfgv, workdir=str(tmp_path / "w"), detector=detector), MockVLM(),
           trace_dir=str(tmp_path / "tr"), cows=["C01", "C02", "C03"], days=[1], progress_every=0)
    d = BrdDB(dbp)
    r2 = d.get_row("C02", 1, 220)
    # YOLO said 'standing' with 95% confidence, but the head ROI cycles at 1 Hz: the screen catches it, the agent corrects it
    assert r2["agent_status"] == "done" and r2["tier1_label"] == "standing" and r2["behavior"] == "ruminating_standing"
    assert r2["correction"] == "behavior_corrected" and r2["src_file"] == NAME and "YOLO said 'standing'" in r2["rationale"]
    r1 = d.get_row("C01", 1, 220)
    assert r1["behavior"] == "standing" and r1["agent_status"] == "passthrough"          # no jaw motion: trusted, with the check recorded
    assert any(a["action"] == "rumination_screen" for a in d.actions_for("C01", 1, 220))
    r3 = d.get_row("C03", 1, 220)
    assert r3["task"] == "find_missing" and r3["correction"] in ("recovered_miss", "miss_not_found")
    acts = d.actions_for("C03", 1, 220)
    assert acts[0]["action"] == "tool:recover_box" and acts[0]["reason"] and '"ok": true' in acts[0]["outcome"].lower().replace('"ok":true', '"ok": true')
    assert all(a["reason"].strip() for a in acts)


def test_night_clip_end_to_end(tmp_path):
    from brd_agent.config import AgentConfig
    from brd_agent.pipeline import run_db
    from brd_agent.vlm_client import MockVLM
    name = "Event20261107230000000.Avi"
    root = tmp_path / "videos"; make_clip(root / "cam_a" / name, night=True)
    rows = [_boxrow("C01", s, (int(40 + s), 80, int(40 + s) + 80, 160), "walking", 0.9, name) for s in range(SECS)]
    yp = tmp_path / "n.csv"; _write_rows(yp, rows)
    dbp = str(tmp_path / "n_store"); db = BrdDB(dbp); cfgv = VideoCfg(root=str(root)); idx = VideoIndex.scan(root)
    import_yolo(db, yp, idx, cfgv); aggregate_minutes(db, cfgv, idx, ["C01"]); db.close()
    cfg = AgentConfig(behaviors_path=str(ROOT / "configs" / "behaviors.yaml")); cfg.verify.audit_rate = 0.0; cfg.video = cfgv
    run_db(dbp, cfg, lambda: VideoBackend(dbp, idx, cfgv, workdir=str(tmp_path / "w")), MockVLM(), cows=["C01"], days=[1], progress_every=0)
    r = BrdDB(dbp).get_row("C01", 1, 1381)
    assert r["is_night_final"] == 1 and "night" in (r["flags"] or "") and r["agent_status"] == "done"
    assert (r["confidence"] or 0) <= 0.85
    acts = [a["action"] for a in BrdDB(dbp).actions_for("C01", 1, 1381)]
    assert "night_check" in acts and "tool:enhance_night" in acts


def test_file_camera_mapping_wins_over_folder_layout(tmp_path):
    """Tracking observations say which camera each file belongs to, so files can sit in any folder (even all in one)."""
    root = tmp_path / "v"
    make_clip(root / "dump" / NAME)
    idx = VideoIndex.scan(root, None, file_camera={NAME.lower(): "back"})
    assert idx.cameras == ["back"] and idx.find_by_name(NAME).camera == "back"
