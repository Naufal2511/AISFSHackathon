"""Stage 1 signals reused by stage 2 instead of decoding video again: day/night per minute, rumination already scored."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from brd_agent.config import AgentConfig
from brd_agent.ingest import import_tracking
from brd_agent.store import Store

ROOT = Path(__file__).resolve().parents[1]
P = ["p_lying", "p_standing", "p_walking", "p_feeding", "p_drinking", "p_rum_lying", "p_rum_standing", "p_grooming", "p_social", "p_other"]


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _inputs(tmp: Path, rum: float):
    obs, st, fr = [], [], []
    for minute, day in ((0, 0), (1, 1)):                     # 03:00 is IR night, 03:01 is a colour frame
        for s in range(3):
            wt = f"2026-11-07T03:0{minute}:{10 + s:02d}"
            obs.append({"wall_time": wt, "file": "Event20261107030000006.Avi", "offset_s": minute * 60 + 10 + s, "camera": "cam1",
                        "camera_name": "Front Camera 1", "track_id": 1, "cow_id": "COW_01", "id_conf": 0.9, "id_evidence": "",
                        "x1": 10, "y1": 10, "x2": 300, "y2": 300, "world_x": 1, "world_y": 1, "visibility": "visible", "occluded_by": ""})
            st.append({"wall_time": wt, "cow_id": "COW_01", "speed_mps": 0.0, "zone": "rest", "n_cams": 1, "behavior_fused": "lying",
                       "beh_conf_fused": 0.95, "beh_evidence": "observed", "anomaly": "",
                       **{p: (0.95 - rum if p == "p_lying" else (rum if p == "p_rum_lying" else 0.0)) for p in P}})
            t = (datetime.fromisoformat(wt) - datetime(1970, 1, 1)).total_seconds()
            fr.append({"wall_t": t, "camera": "cam1", "source_video": "Event20261107030000006.Avi", "offset_s": 0,
                       "daylight": day, "bunk_mean": 0, "bunk_std": 0, "frame_motion": 0})
    _write(tmp / "obs.csv", obs)
    _write(tmp / "st.csv", st)
    _write(tmp / "frames.csv", fr)


def test_night_and_rumination_come_from_stage1(tmp_path):
    _inputs(tmp_path, rum=0.2)
    db = Store(str(tmp_path / "s"))
    res = import_tracking(db, tmp_path / "obs.csv", tmp_path / "st.csv", AgentConfig(behaviors_path=str(ROOT / "configs" / "behaviors.yaml")),
                          frames=tmp_path / "frames.csv")
    assert res["night_minutes_from_stage1"] == 1 and res["stage1_rumination"] is True
    assert db.get_meta("night_source") == "stage1_frames"
    assert db.get_row("COW_01", 1, 180)["is_night"] == 1 and db.get_row("COW_01", 1, 181)["is_night"] == 0
    db.close()


def test_without_frames_night_is_left_to_the_frame_check(tmp_path):
    _inputs(tmp_path, rum=0.0)
    db = Store(str(tmp_path / "s"))
    res = import_tracking(db, tmp_path / "obs.csv", tmp_path / "st.csv", AgentConfig(behaviors_path=str(ROOT / "configs" / "behaviors.yaml")))
    assert res["night_minutes_from_stage1"] is None and res["stage1_rumination"] is False
    assert db.get_meta("night_source") == "frame_check" and db.get_row("COW_01", 1, 180)["is_night"] is None
    db.close()


def test_open_identity_holds_only_that_cow(tmp_path):
    import json
    _inputs(tmp_path, rum=0.0)
    tr = tmp_path / "id_traces"
    tr.mkdir()
    t0 = (datetime(2026, 11, 7, 3, 1, 5) - datetime(1970, 1, 1)).total_seconds()
    (tr / "a.json").write_text(json.dumps({"flag": {"kind": "weak_id", "t0": t0, "t1": t0 + 10, "cow_ids": ["COW_01"]},
                                           "final": {"decision": "uncertain"}}))
    (tr / "b.json").write_text(json.dumps({"flag": {"kind": "weak_id", "t0": t0 - 60, "t1": t0 - 50, "cow_ids": ["COW_01"]},
                                           "final": {"decision": "confirm"}}))
    db = Store(str(tmp_path / "s"))
    res = import_tracking(db, tmp_path / "obs.csv", tmp_path / "st.csv", AgentConfig(behaviors_path=str(ROOT / "configs" / "behaviors.yaml")),
                          id_traces=tr)
    assert res["id_hold_minutes"] == 1
    assert "id_unresolved" in (db.get_row("COW_01", 1, 181)["yolo_flags"] or "")
    assert "id_unresolved" not in (db.get_row("COW_01", 1, 180)["yolo_flags"] or "")   # confirmed case: no hold
    db.close()
