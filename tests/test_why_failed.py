"""why_failed.py: finds the failed step, shows only the latest attempt's traceback, names the cause."""
import json

import why_failed as wf

OLD = """===== track 2026-11-14 22:00
Traceback (most recent call last):
  File "main.py", line 311, in <module>
ValueError: operands could not be broadcast together with shapes (384,) (420,)
"""


def _run(tmp_path, log, status="failed", error="step 'track' failed (exit 1)"):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "1_track.log").write_text(log)
    (tmp_path / "pipeline_state.json").write_text(json.dumps(
        {"steps": {"prepare": {"status": "ok"}, "track": {"status": status, "error": error}}}))
    return tmp_path


def test_known_crash_is_named(tmp_path, capsys):
    assert wf.report_run(_run(tmp_path, OLD)) is False
    out = capsys.readouterr().out
    assert "FAILED" in out and "4124414" in out and "could not be broadcast" in out


def test_only_latest_attempt_counts(tmp_path, capsys):
    new = OLD + """===== track 2026-11-15 06:48
Traceback (most recent call last):
  File "x.py", line 3, in <module>
OSError: [Errno 28] No space left on device
"""
    wf.report_run(_run(tmp_path, new))
    out = capsys.readouterr().out
    assert "disk" in out and "4124414" not in out


def test_unknown_error_says_so(tmp_path, capsys):
    wf.report_run(_run(tmp_path, "===== track\nTraceback (most recent call last):\nZeroDivisionError: x\n"))
    assert "not a known one" in capsys.readouterr().out


def test_patterns():
    for text, word in [("openai.AuthenticationError: 401 Unauthorized from chat-api.tamu.ai", "TAMU"),
                       ("RuntimeError: MPS backend out of memory", "memory"),
                       ("requests.exceptions.ConnectionError: Max retries exceeded", "network"),
                       ("ModuleNotFoundError: No module named 'ultralytics'", "package"),
                       ("_pickle.UnpicklingError: invalid load key, 'v'", "LFS")]:
        assert word.lower() in wf.diagnose(text)[0].lower(), text


# ---- a quiet step is not a dead step: check the process (level1's behavior model prints nothing for ~40 min)
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402


def _running(d, quiet_min):
    (d / "logs").mkdir(parents=True)
    log = d / "logs" / "2_level1.log"
    log.write_text("===== level1 2026-11-15T10:39\n== behavior: classify_behavior.py --records ...\n")
    t = time.time() - quiet_min * 60
    os.utime(log, (t, t))
    (d / "pipeline_state.json").write_text(json.dumps(
        {"steps": {"track": {"status": "ok"}, "level1": {"status": "running"}}}))
    return d


def test_quiet_step_with_live_process_is_running(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(wf, "alive", lambda run: [4242])
    wf.report_run(_running(tmp_path, 40))
    out = capsys.readouterr().out
    assert "INTERRUPTED" not in out and "process 4242 alive" in out


def test_no_process_is_interrupted_with_a_cause(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(wf, "alive", lambda run: [])
    assert wf.report_run(_running(tmp_path, 2)) is False        # even a fresh log: nothing is running it
    out = capsys.readouterr().out
    assert "INTERRUPTED (no process" in out and "process is gone" in out and "not a known one" not in out


def test_unchecked_falls_back_to_log_age(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(wf, "alive", lambda run: None)          # HPRC login node / no psutil
    wf.report_run(_running(tmp_path / "a", 40))
    assert "INTERRUPTED" not in capsys.readouterr().out          # 40 min quiet is normal for behavior
    wf.report_run(_running(tmp_path / "b", 90))
    out = capsys.readouterr().out
    assert "INTERRUPTED?" in out and "squeue" in out


@pytest.mark.skipif(shutil.which("squeue") is not None, reason="cluster login node: process check is off")
def test_alive_finds_real_processes(tmp_path):
    pytest.importorskip("psutil")
    share = tmp_path.resolve() / "share"
    run, run2 = share / "x_c01", share / "x_c02"
    run.mkdir(parents=True)
    sleep = [sys.executable, "-c", "import time; time.sleep(30)"]
    p1 = subprocess.Popen(sleep + ["--records", str(run / "stage1" / "r.csv")])        # a step: has the folder
    p2 = subprocess.Popen(sleep + ["--name", "x_c02", "--out-root", str(share)])        # run_pipeline.py
    try:
        deadline = time.time() + 10
        while time.time() < deadline and not (p1.pid in wf.alive(run) and p2.pid in wf.alive(run2)):
            time.sleep(0.2)
        assert p1.pid in wf.alive(run) and p2.pid not in wf.alive(run)
        assert p2.pid in wf.alive(run2) and p1.pid not in wf.alive(run2)
        assert wf.alive(share / "x_c0") == [] or not {p1.pid, p2.pid} & set(wf.alive(share / "x_c0"))
    finally:
        for p in (p1, p2):
            p.kill()
            p.wait()
    assert p1.pid not in wf.alive(run)
