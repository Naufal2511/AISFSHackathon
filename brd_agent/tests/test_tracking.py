"""Tracking input (observations.csv + cow_state.csv) and the calm-cow shortcut."""
from __future__ import annotations

from pathlib import Path

from brd_agent.backend import MockBackend
from brd_agent.calm import CalmChain, is_calm
from brd_agent.config import AgentConfig
from brd_agent.ingest import import_tracking
from brd_agent.pipeline import run_db
from brd_agent.store import Store, row_to_brief
from brd_agent.video import VideoIndex
from brd_agent.vlm_client import MockVLM

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "tracking_sample"

# The sample is real tracking output from the event footage, which must stay off GitHub (git-ignored), so these tests
# only run where the file exists locally.
pytestmark = pytest.mark.skipif(not (SAMPLE / "observations.csv").exists(),
                                reason="examples/tracking_sample/ (event data, not in git) is not present")


def _cfg() -> AgentConfig:
    c = AgentConfig(behaviors_path=str(ROOT / "configs" / "behaviors.yaml"))
    c.verify.audit_rate = 0.0
    return c


def _imported(tmp_path):
    db = Store(str(tmp_path / "s"))
    res = import_tracking(db, SAMPLE / "observations.csv", SAMPLE / "cow_state.csv", _cfg())
    db.close()
    return str(tmp_path / "s"), res


def _backend(dbp):
    base = {}
    for r in Store(dbp).all_rows():
        lab = r.get("tier1_label")
        if lab:
            sc = {"posture_mode": "lying" if lab == "lying" else "standing", "speed_mean": r["speed_px_s"] or 0.0}
            if lab == "feeding":
                sc.update(head_in_bunk_frac=0.8, zone_fractions={"feed_bunk": 0.8, "waterer": 0, "resting": 0, "open": 0.2})
            base[(r["cow_id"], r["day"], r["minute"])] = sc
    return lambda: MockBackend(base)


def test_import_maps_cameras_files_and_minutes(tmp_path):
    dbp, res = _imported(tmp_path)
    assert res["cameras"] == {"back": "Back Camera", "front": "Front Camera 1", "top": "Front Camera 2 (bunk)"}
    assert res["unidentified_boxes"] > 0 and res["minutes_with_state"] > 0
    db = Store(dbp)
    fc = db.get_meta("file_camera")
    assert fc["event20261107160019006.avi"] == "front" and fc["event20261107160029017.avi"] == "top" and fc["event20261107160008003.avi"] == "back"
    assert db.get_meta("speed_unit") == "m_s" and db.get_meta("clock_origin").startswith("2026-11-07T00:00")
    r = db.get_row("COW_01", 1, 960)                          # 16:00 -> minute 960 of day 1
    assert r["tier1_label"] == "lying" and r["stability"] == 1.0 and r["src_file"].startswith("Event2026110716")
    # a second import adds nothing (idempotent)
    assert import_tracking(db, SAMPLE / "observations.csv", SAMPLE / "cow_state.csv", _cfg())["boxes_added"] == 0


def test_source_pointer_matches_wall_time(tmp_path):
    """file start (from its name) + offset must equal the minute's time stamp: that is what lets us seek the right camera file."""
    from datetime import datetime, timedelta
    from brd_agent.video import parse_event_filename
    dbp, _ = _imported(tmp_path)
    for r in Store(dbp).all_rows():
        if r.get("src_file"):
            t = parse_event_filename(r["src_file"]) + timedelta(seconds=r["src_offset_s"])
            assert abs((t - datetime.fromisoformat(r["src_abs_time"])).total_seconds()) < 1.0


def test_calm_minutes_inherit_and_do_not_touch_video_or_vlm(tmp_path):
    dbp, _ = _imported(tmp_path)
    res = run_db(dbp, _cfg(), _backend(dbp), MockVLM(), progress_every=0, days=[1])
    rows = [r for r in Store(dbp).all_rows() if r["agent_status"] != "pending" and r.get("tier1_label")]
    carried = [r for r in rows if r["reason"] == "stable_carry"]
    assert carried and res["by_reason_this_run"]["stable_carry"] == len(carried)
    for r in carried:
        assert r["agent_status"] == "carried" and r["n_vlm_calls"] == 0 and r["n_tool_calls"] == 0 and r["behavior"] == r["tier1_label"]
        acts = Store(dbp).actions_for(r["cow_id"], r["day"], r["minute"])
        assert [a["action"] for a in acts] == ["stable_carry"] and "not doing anything" in acts[0]["reason"]
    # an unstable / camera-conflict minute is never carried
    r = Store(dbp).get_row("COW_02", 1, 961)
    assert "camera_conflict" in (r["yolo_flags"] or "") or "label_unstable" in (r["yolo_flags"] or "") or r["stability"] < 0.98
    assert r["reason"] != "stable_carry"


def test_verify_all_and_disabled_turn_the_shortcut_off(tmp_path):
    for name, tweak in (("all", lambda c: setattr(c.verify, "verify_all", True)), ("off", lambda c: setattr(c.calm, "enabled", False))):
        dbp, _ = _imported(tmp_path / name)
        cfg = _cfg(); tweak(cfg)
        res = run_db(dbp, cfg, _backend(dbp), MockVLM(), progress_every=0, days=[1])
        assert "stable_carry" not in res["by_reason_this_run"], name


def test_recheck_interval_and_label_change_break_the_chain():
    cfg = _cfg(); cfg.calm.recheck_every_min = 3
    from brd_agent.schemas import Brief, Decision

    def brief(label, m):
        return Brief(cow_id="C", day=1, minute=m, tier1_top=[(label, 0.99)], tier1_entropy=0.05, visible_frac={"back": 1.0}, id_conf=0.95)

    def row(label, m):
        return {"cow_id": "C", "day": 1, "minute": m, "t_abs": m, "stability": 1.0, "min_conf": 1.0, "obs_frac": 1.0, "speed_p90": 0.05}
    chain = CalmChain()
    d = Decision(cow_id="C", day=1, minute=0, behavior="lying", confidence=0.99, reason="keep_tier1", tier=1, tier1_label="lying", correction="confirmed")
    chain.observe(brief("lying", 0), row("lying", 0), d, cfg)
    assert chain.try_carry(brief("lying", 1), row("lying", 1), cfg).reason == "stable_carry"
    assert chain.try_carry(brief("lying", 2), row("lying", 2), cfg) is not None
    assert chain.try_carry(brief("lying", 3), row("lying", 3), cfg) is None         # recheck_every_min=3 -> re-verify
    chain.observe(brief("lying", 0), row("lying", 0), d, cfg)
    assert chain.try_carry(brief("standing", 1), row("standing", 1), cfg) is None   # label changed
    assert chain.try_carry(brief("lying", 5), row("lying", 5), cfg) is None         # gap in time
    ok, why = is_calm({**row("lying", 1), "speed_p90": 0.9}, brief("lying", 1), cfg.calm)
    assert not ok and "moving" in why


def test_corrected_anchor_is_inherited(tmp_path):
    from brd_agent.schemas import Brief, Decision
    cfg = _cfg()
    b = lambda m: Brief(cow_id="C", day=1, minute=m, tier1_top=[("standing", 0.99)], tier1_entropy=0.05, visible_frac={"back": 1.0}, id_conf=0.95)
    r = lambda m: {"cow_id": "C", "day": 1, "minute": m, "t_abs": m, "stability": 1.0, "min_conf": 1.0, "obs_frac": 1.0, "speed_p90": 0.05}
    d = Decision(cow_id="C", day=1, minute=0, behavior="ruminating_standing", confidence=0.8, reason="agent_revised", tier=2,
                 tier1_label="standing", correction="behavior_corrected")
    chain = CalmChain(); chain.observe(b(0), r(0), d, cfg)
    c = chain.try_carry(b(1), r(1), cfg)
    assert c.behavior == "ruminating_standing" and c.correction == "behavior_corrected"
