"""Level 3 live early warning (src/live): no videos, no models, no API calls."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.live import alert as al
from src.live import engine as en

REPO = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(REPO / "configs" / "live.yaml", encoding="utf-8"))


def _feat(rows):
    return pd.DataFrame(rows)


def _herd(sick=None):
    rows = []
    for i in range(1, 9):
        r = {"cow_id": f"COW_{i:02d}", "seen_min": 5.0, "feeding_share": 0.4 + 0.02 * i, "lying_share": 0.3 + 0.01 * i,
             "bunk_zone_share": 0.45, "isolation_share": 0.1 + 0.01 * i, "distance_m_per_h": 100.0 + i}
        rows.append(r)
    if sick:
        rows.append({"cow_id": sick, "seen_min": 5.0, "feeding_share": 0.0, "lying_share": 1.0, "bunk_zone_share": 0.0,
                     "isolation_share": 1.0, "distance_m_per_h": 5.0})
    return _feat(rows)


def _win(start="2026-11-08 14:00"):
    t = pd.Timestamp(start)
    return en.Window(Path("."), t, t + pd.Timedelta(minutes=5), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def _state(tmp_path, prior=None):
    prior = prior if prior is not None else pd.DataFrame(
        [{"cow_id": f"COW_{i:02d}", "risk_score": 0.1 + (0.3 if i == 9 else 0), "rank": 1 if i == 9 else i + 1,
          "top_reasons": "a | b"} for i in range(1, 10)])
    return en.RiskState(tmp_path / "state.json", prior)


def test_evidence_points_at_the_brd_pattern():
    ev = en.window_evidence(_herd("COW_09"), CFG)
    e = ev.set_index("cow_id").evidence
    assert e["COW_09"] == e.max() and e["COW_09"] > 3          # not eating, lying, alone, not walking
    top = ev.set_index("cow_id").at["COW_09", "reasons"]
    assert {r["indicator"] for r in top} >= {"feeding_share", "lying_share", "isolation_share"}


def test_unseen_cow_gets_no_behavior_evidence(tmp_path):
    f = _herd()
    f.loc[0, "seen_min"] = 0.5
    ev = en.window_evidence(f, CFG)
    assert ev.loc[0, "evidence"] == 0 and "seen only" in ev.loc[0, "reasons"][0]["text"]
    st = _state(tmp_path)
    upd = en.update_risk(st, ev, {}, _win(), CFG)
    assert upd.set_index("cow_id").at["COW_01", "delta"] == 0                # risk held, not pulled toward the prior


def test_confirmed_sign_alerts_at_once_and_only_once(tmp_path):
    st = _state(tmp_path)
    ev = en.window_evidence(_herd(), CFG)
    signs = {"COW_03": [{"sign": "nasal_discharge", "time": 0.0, "camera": "cam2", "path": "x.jpg", "why": "streak"}]}
    upd = en.update_risk(st, ev, signs, _win(), CFG)
    a = en.decide_alerts(st, upd, ev, signs, _win(), CFG)
    assert [x["cow_id"] for x in a] == ["COW_03"] and a[0]["trigger"] == "sign"
    assert upd.set_index("cow_id").at["COW_03", "delta"] >= CFG["risk"]["sign_points"]["nasal_discharge"] - 0.05
    upd2 = en.update_risk(st, ev, signs, _win("2026-11-08 14:05"), CFG)      # same sign next window: no repeat
    assert en.decide_alerts(st, upd2, ev, signs, _win("2026-11-08 14:05"), CFG) == []


def test_behavior_alert_needs_high_rising_risk_and_respects_cooldown(tmp_path):
    cfg = json.loads(json.dumps(CFG))
    cfg["alert"]["score"] = 0.42
    st = _state(tmp_path)
    ev = en.window_evidence(_herd("COW_09"), cfg)
    t = pd.Timestamp("2026-11-08 14:00")
    alerts = []
    for k in range(6):
        w = _win(str(t + pd.Timedelta(minutes=5 * k)))
        upd = en.update_risk(st, ev, {}, w, cfg)
        alerts += en.decide_alerts(st, upd, ev, {}, w, cfg)
    assert [a["cow_id"] for a in alerts] == ["COW_09"]                      # once, then the 2 h cooldown holds
    assert st.d["cows"]["COW_09"]["risk"] > 0.42


def test_ids_from_the_stitched_map_and_from_the_gallery(tmp_path):
    idm = tmp_path / "id_map.csv"
    pd.DataFrame([{"chunk": "c25", "local_id": "COW_01", "global_id": "COW_07"},
                  {"chunk": "c25", "local_id": "COW_02", "global_id": "COW_05"},
                  {"chunk": "runA", "local_id": "COW_01", "global_id": "COW_03"}]).to_csv(idm, index=False)
    gal = tmp_path / "gal.npz"
    np.savez(gal, x=np.zeros(3))
    cfg = json.loads(json.dumps(CFG))
    cfg["identity"].update({"id_map": str(idm), "gallery": str(gal), "map_chunk": "c25",
                            "reference_prototypes": str(tmp_path / "none.npz")})
    m = pd.DataFrame({"cow_id": ["COW_01", "COW_02"], "behavior": ["lying", "feeding"]})
    st = pd.DataFrame({"cow_id": ["COW_01"]})
    run = tmp_path / "runA"
    run.mkdir()
    w = en.Window(run, None, None, m, st, pd.DataFrame(), pd.DataFrame())
    ids = en.map_ids(w, cfg, Path("/"))                                      # run is a 48 h chunk: its own map row
    assert dict(zip(ids.local_id, ids.cow_id)) == {"COW_01": "COW_03", "COW_02": "LIVE_COW_02"}
    run2 = tmp_path / "live_x"
    run2.mkdir()
    (run2 / "pipeline_manifest.json").write_text(json.dumps({"settings": {"id_gallery": str(gal)}}))
    ids = en.map_ids(en.Window(run2, None, None, m, st, pd.DataFrame(), pd.DataFrame()), cfg, Path("/"))
    assert dict(zip(ids.local_id, ids.cow_id)) == {"COW_01": "COW_07", "COW_02": "COW_05"}
    assert ids.how.str.contains("gallery").all()


def test_action_line_and_card_say_what_to_do():
    a = {"cow_id": "COW_02", "trigger": "sign", "new_signs": ["nasal_discharge"], "signs": [], "reasons": [],
         "risk": 0.6, "risk_before": 0.35, "prior": 0.37, "rank": 1, "rank_before": 1}
    a["action"] = al.action_line(a, "at the feed bunk")
    assert "nasal discharge" in a["action"] and "temperature" in a["action"] and "feed bunk" in a["action"]
    page = al.card_html(a, {"clip": "c.mp4", "frame": None, "crops": []}, "Window 14:00-14:05")
    assert "Check now" in page and "c.mp4" in page and "not a probability" in page
    b = {**a, "trigger": "risk", "new_signs": []}
    assert "watch, not a pull" in al.action_line(b, None)


def test_every_cow_gets_a_decision_with_a_reason(tmp_path):
    st = _state(tmp_path)
    ev = en.window_evidence(_herd(), CFG)
    signs = {"COW_02": [{"sign": "ears_droop", "time": 0.0, "camera": "cam1", "path": "x.jpg", "why": "ears hang"}]}
    upd = en.update_risk(st, ev, signs, _win(), CFG)
    dec = []
    en.decide_alerts(st, upd, ev, signs, _win(), CFG, dec)
    d = {r["cow_id"]: r for r in dec}
    assert set(d) == set(upd.cow_id)
    assert d["COW_02"]["decision"] == "ALERT" and "ears_droop" in d["COW_02"]["why"]
    assert d["COW_01"]["decision"] == "no alert" and "no confirmed visible sign" in d["COW_01"]["why"]


def test_tiers_from_the_48h_risk_and_live_escalation(tmp_path):
    l2 = pd.read_csv(REPO / CFG["baseline"] / "brd_risk.csv") if (REPO / CFG["baseline"] / "brd_risk.csv").exists() else None
    rows = [{"cow_id": "COW_A", "risk_score": 0.37, "rank": 1, "p_top3": 0.92, "confidence": "high", "signs_present": None},
            {"cow_id": "COW_B", "risk_score": 0.33, "rank": 2, "p_top3": 0.40, "confidence": "medium", "signs_present": None},
            {"cow_id": "COW_C", "risk_score": 0.20, "rank": 4, "p_top3": 0.05, "confidence": "low", "signs_present": None},
            {"cow_id": "COW_D", "risk_score": 0.12, "rank": 9, "p_top3": 0.00, "confidence": "low", "signs_present": None},
            {"cow_id": "COW_E", "risk_score": 0.15, "rank": 7, "p_top3": 0.1, "confidence": "low", "signs_present": "ears_droop"}]
    tiers = {r["cow_id"]: en.baseline_tier(pd.Series(r), CFG)[0] for r in rows}
    assert tiers == {"COW_A": "orange", "COW_B": "yellow", "COW_C": "yellow", "COW_D": "normal", "COW_E": "red"}
    if l2 is not None:                     # the real 48 h result: the top 3 are the ones a pen rider checks
        real = {r.cow_id: en.baseline_tier(r, CFG)[0] for _, r in l2.iterrows()}
        assert sorted(c for c, t in real.items() if t == "orange") == sorted(l2.sort_values("rank").cow_id[:3])
    # live: a cow that gets a confirmed sign jumps to red, and that is an alert even without a new behavior trigger
    st = _state(tmp_path)
    for c in st.d["cows"].values():
        c["tier"] = "normal"
    ev = en.window_evidence(_herd(), CFG)
    signs = {"COW_04": [{"sign": "nasal_discharge", "time": 0.0, "camera": "cam2", "path": "x.jpg"}]}
    upd = en.update_risk(st, ev, signs, _win(), CFG)
    dec = []
    a = en.decide_alerts(st, upd, ev, signs, _win(), CFG, dec)
    assert st.d["cows"]["COW_04"]["tier"] == "red" and any(x["cow_id"] == "COW_04" and x["tier"] == "red" for x in a)
    assert "Pull and examine" in {r["cow_id"]: r for r in dec}["COW_04"]["why"]


def test_simulated_email_is_a_real_message_and_never_sent(tmp_path, monkeypatch):
    import email
    import smtplib
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", lambda *a, **k: sent.append(a))          # would record any real send
    (tmp_path / "card.html").write_text("<p>card</p>")
    cfg = json.loads(json.dumps(CFG))
    rec = al.write_email(tmp_path, cfg, "BRD alert: COW_02", "<p>check her</p>", "check her", [tmp_path / "card.html"],
                         "t1", lambda m: None)
    assert rec["mode"] == "simulate" and not sent
    msg = email.message_from_bytes((tmp_path / rec["file"]).read_bytes())
    assert msg["Subject"] == "BRD alert: COW_02" and "ranch.example" in msg["To"]
    assert [p.get_filename() for p in msg.walk() if p.get_filename()] == ["card.html"]
    assert json.loads((tmp_path / "outbox" / "outbox.jsonl").read_text().splitlines()[0])["subject"] == "BRD alert: COW_02"
    cfg["delivery"]["email"]["to"] = []
    assert al.write_email(tmp_path, cfg, "x", "", "", [], "t2", lambda m: None) is None   # no recipients: nothing written
