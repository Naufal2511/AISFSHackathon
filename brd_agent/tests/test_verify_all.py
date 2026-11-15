import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_db_and_providers import _write_minutes, cfg, run  # noqa: E402

from brd_agent.backend import MockBackend
from brd_agent.store import Store as BrdDB
from brd_agent.pipeline import run_db
from brd_agent.vlm_client import MockVLM


def setup(tmp_path, overrides=None, n=30):
    dbp = str(tmp_path / "v_store"); db = BrdDB(dbp); db.init_rows(["C01"], [1], n_minutes=n)
    csvp = tmp_path / "v.csv"; _write_minutes(csvp, n=n, overrides=overrides); db.import_minutes_csv(csvp); db.close()
    return dbp


def row(dbp, minute):
    return BrdDB(dbp).get_row("C01", 1, minute)


def test_confident_but_wrong_yolo_label_is_caught_by_rule_verification_and_corrected(tmp_path):
    # YOLO is 97% sure the cow is walking at minute 15, but the track says she barely moves and the images agree she stands
    dbp = setup(tmp_path, overrides={15: "walking:0.97;standing:0.02"})
    run(dbp, scen={("C01", 1, 15): {"speed_mean": 0.5, "posture_mode": "standing"}})
    r = row(dbp, 15)
    assert r["agent_status"] == "done" and r["tier1_label"] == "walking"
    assert r["behavior"] == "standing" and r["correction"] == "behavior_corrected"
    acts = BrdDB(dbp).actions_for("C01", 1, 15)
    rv = next(a for a in acts if a["action"] == "rule_verify")
    assert "rule_contradiction:posture_walking" in rv["outcome"] and rv["reason"]


def test_verified_confident_minute_records_why_it_was_trusted(tmp_path):
    dbp = setup(tmp_path)
    run(dbp)
    acts = BrdDB(dbp).actions_for("C01", 1, 3)
    assert [a["action"] for a in acts][:2] == ["night_check", "gate:keep_tier1"] or "rule_verify" in [a["action"] for a in acts]
    assert any(a["action"] == "rule_verify" and a["outcome"] == "no contradiction found" for a in acts)
    assert row(dbp, 3)["correction"] == "confirmed" and row(dbp, 3)["rationale"]


def test_verify_all_sends_every_minute_to_the_agent(tmp_path):
    dbp = setup(tmp_path)
    c = cfg(); c.verify.verify_all = True
    chk = BrdDB(dbp)
    run_db(dbp, c, lambda: MockBackend({}), MockVLM(), progress_every=0)
    assert BrdDB(dbp).status()["by_status"].get("done", 0) == 30


def test_audit_rate_samples_confident_minutes_deterministically(tmp_path):
    dbp = setup(tmp_path, n=200) if False else None
    dbp = str(tmp_path / "a_store"); db = BrdDB(dbp); db.init_rows(["C01"], [1], n_minutes=200)
    csvp = tmp_path / "a.csv"; _write_minutes(csvp, n=200, overrides={m: "standing:0.97;lying:0.02" for m in range(200)})
    db.import_minutes_csv(csvp); db.close()
    c = cfg(); c.verify.audit_rate = 0.10
    run_db(dbp, c, lambda: MockBackend({}), MockVLM(), progress_every=0)
    n_done = BrdDB(dbp).status()["by_status"].get("done", 0)
    assert 8 <= n_done <= 35                                   # ~10% of 200, deterministic by hash
    dbp2 = str(tmp_path / "b_store"); import shutil
    # same inputs -> same audit sample
    db = BrdDB(dbp2); db.init_rows(["C01"], [1], n_minutes=200); db.import_minutes_csv(csvp); db.close()
    run_db(dbp2, c, lambda: MockBackend({}), MockVLM(), progress_every=0)
    pick = lambda p: {r["minute"] for r in BrdDB(p).all_rows() if r["agent_status"] == "done"}
    assert pick(dbp) == pick(dbp2)


def test_night_frame_regates_a_confident_minute(tmp_path):
    dbp = setup(tmp_path, overrides={5: "standing:0.70;lying:0.20"})
    run(dbp, scen={("C01", 1, 5): {"brightness": 20.0}})
    r = row(dbp, 5)
    assert r["agent_status"] == "done" and r["is_night_final"] == 1 and "night" in (r["flags"] or "")
    assert (r["confidence"] or 0) <= 0.85


def test_budget_exhaustion_leaves_flagged_unverified_not_silent(tmp_path):
    dbp = setup(tmp_path, overrides={15: "walking:0.97;standing:0.02", 16: "walking:0.97;standing:0.02"})
    res = run(dbp, scen={("C01", 1, 15): {"speed_mean": 0.5}, ("C01", 1, 16): {"speed_mean": 0.5}}, max_agent_runs=1)
    flagged = [r for r in (row(dbp, 15), row(dbp, 16)) if "over_budget_unverified" in (r["flags"] or "")]
    assert len(flagged) >= 1 and all(r["confidence"] <= 0.6 and r["correction"] == "unresolved" for r in flagged)


def test_find_missing_without_detector_is_honest_not_guessed(tmp_path):
    dbp = str(tmp_path / "m_store"); db = BrdDB(dbp); db.init_rows(["C01"], [1], n_minutes=3)
    db.set_task = None
    r = db._load("C01", 1)[1]; r.update(task="find_missing", yolo_flags="no_yolo_record"); db._dirty.add(("C01", 1)); db.flush()
    db.close()
    run(dbp)
    r = row(dbp, 1)
    assert r["behavior"] == "other_not_visible" and r["correction"] == "miss_not_found" and "yolo_miss" in r["flags"]
    assert "cannot" in (r["rationale"] or "") or "could not" in (r["rationale"] or "")
