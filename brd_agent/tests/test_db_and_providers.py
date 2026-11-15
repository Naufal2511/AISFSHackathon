import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from brd_agent import dbcli
from brd_agent.backend import MockBackend
from brd_agent.config import AgentConfig, HistoryCfg
from brd_agent.store import Store as BrdDB, HistoryStore
from brd_agent.env import ConfigError, load_settings
from brd_agent.orchestrator import BehaviorAgent
from brd_agent.pipeline import run_db
from brd_agent.schemas import Brief, Decision
from brd_agent.vlm_client import MockVLM, VLMHTTPError, make_vlm

ROOT = Path(__file__).resolve().parents[1]


def cfg(**kw) -> AgentConfig:
    c = AgentConfig(behaviors_path=str(ROOT / "configs" / "behaviors.yaml"))
    c.vlm.cache_dir = None
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ================================================================== env / provider switch
def test_env_defaults_openai_and_inline_comments(tmp_path):
    envf = tmp_path / ".env"
    envf.write_text("OPENAI_API_KEY=sk-abcdef123456\nOPENAI_MODEL=my-model   # pick one\nOPENAI_TEMPERATURE=\n", encoding="utf-8")
    s = load_settings(envf, environ={})
    assert s.provider == "openai" and s.use_local_vlm is False          # default is OpenAI
    assert s.openai_model == "my-model" and s.openai_temperature is None
    assert "sk-abcdef123456" not in s.describe() and s.describe().count("3456") == 1   # key masked


def test_boolean_switch_and_real_env_overrides_file(tmp_path):
    envf = tmp_path / ".env"
    envf.write_text("USE_LOCAL_VLM=false\nLOCAL_VLM_MODEL=from-file\n", encoding="utf-8")
    s = load_settings(envf, environ={"USE_LOCAL_VLM": "TRUE"})
    assert s.provider == "local" and s.local_model == "from-file"


def test_make_vlm_config_errors():
    c = cfg()
    base = dict(environ={})
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        make_vlm(load_settings("nope.env", **base), c)
    with pytest.raises(ConfigError, match="OPENAI_MODEL"):
        make_vlm(load_settings("nope.env", environ={"OPENAI_API_KEY": "sk-x"}), c)
    with pytest.raises(ConfigError, match="LOCAL_VLM_MODEL"):
        make_vlm(load_settings("nope.env", environ={"USE_LOCAL_VLM": "1"}), c)


# ================================================================== fake OpenAI-compatible server
REPLIES = {
    "plan": {"rationale": "check bunk", "steps": [{"tool": "zone_query", "args": {}}, {"tool": "zoom", "args": {"region": "head"}}]},
    "hypothesis": {"label": "feeding", "confidence": 0.9, "cues": ["head in bunk zone"], "alternatives": [["standing", 0.05]],
                   "tags": ["head_butting", "tongue_rolling", "bogus"], "needs": None},
    "critique": {"verdict": "accept", "best_alternative": None, "reason": "holds", "requested_camera": None},
}


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        srv = self.server
        srv.log.append({"headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
        if srv.fail_first > 0:
            srv.fail_first -= 1
            self.send_response(srv.fail_status); self.send_header("Retry-After", "0"); self.end_headers()
            self.wfile.write(b'{"error":"nope"}'); return
        text = body["messages"][1]["content"][0]["text"]
        task = text.split("\n", 1)[0].replace("TASK:", "").strip()
        out = json.dumps({"choices": [{"message": {"content": json.dumps(REPLIES[task])}}],
                          "usage": {"prompt_tokens": 100, "completion_tokens": 20}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    srv.log, srv.fail_first, srv.fail_status = [], 0, 429
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    srv.base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    yield srv
    srv.shutdown()


def _img(tmp_path):
    return MockBackend(workdir=str(tmp_path / "img")).crop("C01", 1, 1, "front")


def _openai(server, tmp_path, **env):
    e = {"OPENAI_API_KEY": "sk-test-1234567890", "OPENAI_MODEL": "my-model", "OPENAI_BASE_URL": server.base, **env}
    c = cfg(); c.vlm.cache_dir = str(tmp_path / "cache")
    v = make_vlm(load_settings("nope.env", environ=e), c); v.backoff_s = 0
    return v


def test_openai_request_shape_cache_and_tokens(server, tmp_path):
    from brd_agent.schemas import Hypothesis
    v = _openai(server, tmp_path)
    obj, info = v.ask("hypothesis", "SYS (JSON)", "TASK: hypothesis\nx", [_img(tmp_path)], Hypothesis)
    r = server.log[-1]
    assert r["headers"]["authorization"] == "Bearer sk-test-1234567890"
    b = r["body"]
    assert b["model"] == "my-model" and b["max_completion_tokens"] == 800 and "max_tokens" not in b
    assert b["response_format"] == {"type": "json_object"} and "temperature" not in b
    assert b["messages"][1]["content"][1]["image_url"]["detail"] == "low"
    assert obj.label == "feeding" and obj.tags == ["head_butting", "tongue_rolling"]      # 'bogus' tag dropped
    assert (info["prompt_tokens"], info["completion_tokens"]) == (100, 20)
    n = len(server.log)
    _, info2 = v.ask("hypothesis", "SYS (JSON)", "TASK: hypothesis\nx", [_img(tmp_path)], Hypothesis)
    assert info2["cached"] is True and len(server.log) == n                              # cache hit: no request


def test_openai_model_and_temperature_come_from_env(server, tmp_path):
    from brd_agent.schemas import Hypothesis
    v = _openai(server, tmp_path, OPENAI_MODEL="another-model", OPENAI_TEMPERATURE="0.2", OPENAI_IMAGE_DETAIL="high")
    v.ask("hypothesis", "SYS (JSON)", "TASK: hypothesis\ny", [_img(tmp_path)], Hypothesis)
    b = server.log[-1]["body"]
    assert b["model"] == "another-model" and b["temperature"] == 0.2
    assert b["messages"][1]["content"][1]["image_url"]["detail"] == "high"


def test_local_request_shape(server, tmp_path):
    from brd_agent.schemas import Hypothesis
    e = {"USE_LOCAL_VLM": "true", "LOCAL_VLM_BASE_URL": server.base, "LOCAL_VLM_MODEL": "local-m"}
    v = make_vlm(load_settings("nope.env", environ=e), cfg())
    v.ask("hypothesis", "SYS (JSON)", "TASK: hypothesis\nz", [_img(tmp_path)], Hypothesis)
    r = server.log[-1]
    assert "authorization" not in r["headers"]
    b = r["body"]
    assert b["model"] == "local-m" and b["max_tokens"] == 600 and "max_completion_tokens" not in b
    assert b["temperature"] == 0.0 and "response_format" not in b
    assert "detail" not in b["messages"][1]["content"][1]["image_url"]


def test_retry_on_429_then_success_and_no_retry_on_400(server, tmp_path):
    from brd_agent.schemas import Hypothesis
    v = _openai(server, tmp_path)
    server.fail_first = 2
    obj, _ = v.ask("hypothesis", "SYS (JSON)", "TASK: hypothesis\nr", [], Hypothesis)
    assert obj.label == "feeding" and len(server.log) == 3
    server.log.clear(); server.fail_first, server.fail_status = 1, 400
    with pytest.raises(VLMHTTPError, match="400"):
        v.ask("hypothesis", "SYS (JSON)", "TASK: hypothesis\nr2", [], Hypothesis)
    assert len(server.log) == 1


def test_agent_end_to_end_over_http_records_provider_tokens_tags(server, tmp_path):
    v = _openai(server, tmp_path)
    scen = {("C01", 1, 100): {"head_in_bunk_frac": 0.8}}
    a = BehaviorAgent(cfg(), MockBackend(scen), v)
    b = Brief(cow_id="C01", day=1, minute=100, tier1_top=[["standing", 0.41], ["feeding", 0.33]], tier1_entropy=1.3,
              visible_frac={"back": 0.3, "front": 0.8, "top": 0.6}, id_conf=0.9)
    d = a.run(b)
    assert d.behavior == "feeding" and d.provider == "openai" and d.model == "my-model"
    assert d.tokens_in >= 200 and d.tokens_out >= 40
    assert d.behavior_tags == ["tongue_rolling"]          # head_butting is not valid for 'feeding' -> filtered
    assert d.prompt_version == "v3"


# ================================================================== database
def _write_minutes(path, cow="C01", n=30, overrides=None):
    overrides = overrides or {}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cow_id", "day", "minute", "tier1_top", "speed_px_s", "posture", "vis_back", "vis_front", "vis_top", "id_conf"])
        for m in range(n):
            top = overrides.get(m) or ("lying:0.95;standing:0.03" if m < 10 else "feeding:0.95;standing:0.03")
            w.writerow([cow, 1, m, top, 2.0, "lying" if top.startswith(("lying", "ruminating_lying")) else "standing", 0.6, 0.8, 0.7, 0.9])


@pytest.fixture
def db_env(tmp_path):
    dbp = str(tmp_path / "t_store")
    db = BrdDB(dbp); db.init_rows(["C01"], [1], n_minutes=30)
    csvp = tmp_path / "m.csv"; _write_minutes(csvp)
    db.import_minutes_csv(csvp)
    db.close()
    return dbp, csvp


def tmp_dir(dbp):
    from pathlib import Path
    return Path(dbp)


def run(dbp, scen=None, trace_dir=None, **kw):
    """Run the pipeline with a MockBackend whose measurements are CONSISTENT with each minute's upstream label, so confident
    minutes pass the rule pre-verification (explicit `scen` entries override per key). Audits are off for determinism."""
    chk = BrdDB(dbp)
    rows = [r for r in chk.all_rows() if r["tier1_label"] is not None]
    chk.close()
    base = {}
    for r in rows:
        lab = r["tier1_label"]
        sc = {"posture_mode": "lying" if lab.endswith("lying") or lab == "lying" else "standing", "speed_mean": 1.0}
        if lab == "walking":
            sc.update(speed_mean=30.0)
        if lab == "feeding":
            sc.update(head_in_bunk_frac=0.8, zone_fractions={"feed_bunk": 0.8, "waterer": 0, "resting": 0, "open": 0.2})
        base[(r["cow_id"], r["day"], r["minute"])] = sc
    for k, v in (scen or {}).items():
        base[k] = {**base.get(k, {}), **v}
    c = cfg(); c.verify.audit_rate = 0.0; c.verify.night_audit_rate = 0.0
    return run_db(dbp, c, lambda: MockBackend(base), MockVLM(), trace_dir=trace_dir, progress_every=0, **kw)


def test_run_writes_result_to_every_row_and_is_resumable(db_env):
    dbp, _ = db_env
    res = run(dbp)
    assert res["processed"] == 30
    db = BrdDB(dbp)
    rows = list(db.all_rows())
    assert all(r["behavior"] for r in rows) and {r["agent_status"] for r in rows} == {"passthrough", "done"}   # minute 10: lying -> feeding is implausible, so the rules escalate it
    assert sum(len(open(p, encoding="utf-8").readlines()) for p in (tmp_dir(dbp) / "runs").glob("*.jsonl")) == 30
    assert len(db.rebuild_bouts()) == 2
    assert run(dbp)["processed"] == 0                                           # idempotent: nothing pending


def test_parallel_cows_with_sqlite(tmp_path):
    dbp = str(tmp_path / "p_store"); db = BrdDB(dbp); db.init_rows(["C01", "C02", "C03"], [1], n_minutes=30)
    for c in ("C01", "C02", "C03"):
        _write_minutes(tmp_path / f"{c}.csv", cow=c); db.import_minutes_csv(tmp_path / f"{c}.csv")
    db.close()
    assert run(dbp, workers=3)["processed"] == 90
    assert BrdDB(dbp).status()["by_status"] == {"passthrough": 87, "done": 3}


def test_history_summary_facts(db_env):
    dbp, _ = db_env
    run(dbp)
    h = HistoryStore(BrdDB(dbp), HistoryCfg())
    f = h.summary("C01", 1, 15)
    assert f["current_bout"] == {"label": "feeding", "minutes": 5, "ends_minutes_ago": 1, "final": True}
    assert f["minutes_since"]["lying"] == 6 and f["minutes_since"]["feeding"] == 1
    assert f["recent_runs"] == "lying x10 > feeding x5" and f["next_tier1"] == ["feeding", "feeding"]
    assert "next_tier1" not in HistoryStore(BrdDB(dbp), HistoryCfg(lookahead_min=0)).summary("C01", 1, 15)   # live mode
    assert HistoryStore(BrdDB(dbp), HistoryCfg()).summary("C01", 1, 0)["empty"] is True


def test_history_marks_unprocessed_minutes_provisional(db_env):
    dbp, _ = db_env
    f = HistoryStore(BrdDB(dbp), HistoryCfg()).summary("C01", 1, 12)       # nothing processed yet -> tier-1 labels
    assert "?" in f["recent_runs"] and f["n_provisional_in_window"] > 0 and f["current_bout"]["final"] is False


def test_agent_uses_db_history_for_rumination_and_does_not_flag_same_posture_refinement(tmp_path):
    dbp = str(tmp_path / "r_store"); db = BrdDB(dbp); db.init_rows(["C01"], [1], n_minutes=30)
    csvp = tmp_path / "r.csv"
    _write_minutes(csvp, overrides={**{m: "lying:0.95;standing:0.03" for m in range(30)}, 20: "lying:0.50;ruminating_lying:0.40"})
    db.import_minutes_csv(csvp); db.close()
    scen = {("C01", 1, 20): {"jaw_motion_score": 0.7, "periodicity_hz": 0.9, "posture_mode": "lying", "speed_mean": 1.0}}
    run(dbp, scen=scen, trace_dir=str(tmp_path / "tr"))
    row = BrdDB(dbp).get_row("C01", 1, 20)
    assert row["behavior"] == "ruminating_lying" and row["tier"] == 2 and row["agent_status"] == "done"
    assert row["used_history"] == 1 and row["provider"] == "mock" and row["prompt_version"] == "v3"
    assert "breaks_long_bout" not in (row["flags"] or "")                      # lying -> ruminating_lying is a refinement
    tr = json.loads((tmp_path / "tr" / "C01_d1_m0020.json").read_text(encoding="utf-8"))
    assert any(e["type"] == "tool" and e["tool"] == "cow_history" and e["ok"] for e in tr["events"])


def test_stale_when_upstream_changes_and_only_that_row_reprocessed(db_env, tmp_path):
    dbp, _ = db_env
    run(dbp)
    csv2 = tmp_path / "m2.csv"; _write_minutes(csv2, overrides={12: "walking:0.9;standing:0.05"})
    db = BrdDB(dbp); db.import_minutes_csv(csv2)
    assert db.status()["by_status"] == {"passthrough": 28, "done": 1, "stale": 1}
    db.close()
    assert run(dbp)["processed"] == 1
    assert BrdDB(dbp).get_row("C01", 1, 12)["behavior"] == "walking"


def test_rows_without_upstream_are_abstained_not_guessed(tmp_path):
    dbp = str(tmp_path / "g_store"); db = BrdDB(dbp); db.init_rows(["C01"], [1], n_minutes=30)
    csvp = tmp_path / "g.csv"; _write_minutes(csvp, n=6); db.import_minutes_csv(csvp); db.close()
    run(dbp)
    r = BrdDB(dbp).get_row("C01", 1, 20)
    assert tuple(r[k] for k in ("behavior", "source", "reason", "agent_status")) == ("other_not_visible", "not_visible", "no_upstream_data", "abstained")


def test_write_decision_requires_existing_row(tmp_path):
    db = BrdDB(tmp_path / "x_store")
    with pytest.raises(KeyError):
        db.write_decision(Decision(cow_id="C99", day=1, minute=0, behavior="lying", confidence=0.5))


def test_export_bouts_cover_all_minutes_and_eval_ablation(db_env, tmp_path):
    dbp, _ = db_env
    run(dbp)
    db = BrdDB(dbp)
    hl = tmp_path / "h.csv"
    hl.write_text("cow_id,day,minute,behavior,labeler\n" + "\n".join(f"C01,1,{m},{'lying' if m < 10 else 'feeding'},a" for m in range(30)), encoding="utf-8")
    db.import_human_labels_csv(hl)
    out = db.export_activity(tmp_path / "out")
    assert out == {"minute_rows": 30, "bouts": 2, "rows_without_label": 0}
    with open(tmp_path / "out" / "activity.csv", encoding="utf-8") as f:
        bouts = list(csv.DictReader(f))
    assert sum(int(b["duration_min"]) for b in bouts) == 30 and bouts[0]["behavior"] == "lying"
    ev = db.evaluate_vs_human()
    assert ev["n"] == 30 and ev["accuracy_final"] == 1.0 and ev["accuracy_tier1_only"] == 1.0 and "lying" in ev["per_class"]


def test_cli_end_to_end(tmp_path, capsys):
    envf = tmp_path / ".env"; envf.write_text(f"BRD_STORE_DIR={tmp_path / 'cli_store'}\n", encoding="utf-8")
    csvp = tmp_path / "m.csv"; _write_minutes(csvp)
    g = ["--env", str(envf), "--config", str(ROOT / "configs" / "agent.yaml")]
    dbcli.main(g + ["init", "--cows", "C01", "--days", "1", "--n-minutes", "30"])
    dbcli.main(g + ["import-minutes", "--csv", str(csvp)])
    dbcli.main(g + ["run", "--vlm", "mock", "--trace-dir", str(tmp_path / "tr")])
    dbcli.main(g + ["export", "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert '"processed": 30' in out and (tmp_path / "o" / "activity.csv").exists()


def test_store_is_plain_files_and_survives_reopen(tmp_path):
    dbp = str(tmp_path / "s"); db = BrdDB(dbp); db.init_rows(["C01"], [1], n_minutes=5); db.close()
    f = tmp_path / "s" / "minutes" / "C01_d1.csv"
    assert f.exists() and f.read_text(encoding="utf-8").splitlines()[0].startswith("cow_id,day,minute,t_abs")
    assert len(f.read_text(encoding="utf-8").splitlines()) == 6                      # header + 5 minutes
    assert json.loads((tmp_path / "s" / "meta.json").read_text(encoding="utf-8"))["schema_version"] == 3
    assert BrdDB(dbp).status()["by_status"] == {"pending": 5}


def test_actions_with_reasons_persist(tmp_path):
    dbp = str(tmp_path / "a_store"); db = BrdDB(dbp); db.init_rows(["C01"], [1], n_minutes=30)
    csvp = tmp_path / "a.csv"
    _write_minutes(csvp, overrides={m: "standing:0.95;feeding:0.03" for m in range(30)} | {10: "standing:0.45;feeding:0.40"})
    db.import_minutes_csv(csvp); db.close()
    run(dbp, scen={("C01", 1, 10): {"head_in_bunk_frac": 0.8, "zone_fractions": {"feed_bunk": 0.8, "waterer": 0, "resting": 0, "open": 0.2}}})
    d = BrdDB(dbp)
    row = d.get_row("C01", 1, 10)
    assert row["rationale"] and row["correction"] in ("confirmed", "behavior_corrected")
    acts = d.actions_for("C01", 1, 10)
    assert len(acts) >= 5 and all(a["reason"].strip() and a["reason"] != "(no reason supplied)" for a in acts)
    easy = len(d.actions_for("C01", 1, 3))
    assert easy >= 2                                  # even passthrough minutes record why nothing was escalated
