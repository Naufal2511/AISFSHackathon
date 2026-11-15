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
