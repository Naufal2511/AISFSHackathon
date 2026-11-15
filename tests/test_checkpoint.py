"""Tracking checkpoints: atomic save/load, stale ones ignored, asset CSVs cut back to the checkpoint on resume."""
import os
import pickle

import numpy as np

from src.assets.recorder import AssetRecorder
from src.pipeline.checkpoint import NAME, Checkpointer, log_error


def test_save_load_and_clear(tmp_path):
    c = Checkpointer(str(tmp_path), {"a": 1}, 900)
    assert c.load() is None
    c.save({"step": 7, "records": list(range(1000))})
    d = Checkpointer(str(tmp_path), {"a": 1}, 900).load()
    assert d["step"] == 7 and len(d["records"]) == 1000
    assert not (tmp_path / (NAME + ".tmp")).exists()
    c.clear()
    assert not (tmp_path / NAME).exists()


def test_other_settings_or_torn_file_start_over(tmp_path):
    Checkpointer(str(tmp_path), {"fps": 1.0}, 900).save({"step": 1})
    assert Checkpointer(str(tmp_path), {"fps": 0.5}, 900).load() is None
    assert (tmp_path / (NAME + ".stale")).exists() and not (tmp_path / NAME).exists()
    (tmp_path / NAME).write_bytes(b"\x80\x05garbage")
    assert Checkpointer(str(tmp_path), {"fps": 1.0}, 900).load() is None


def test_asset_csvs_cut_back_to_checkpoint(tmp_path):
    rec = AssetRecorder(str(tmp_path))
    frame = np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    rec.frame("cam1", frame, "Event20261107070000001.Avi", 1.0)
    rec.track("cam1", 3, (10, 10, 200, 200))
    sizes = rec.snapshot()
    state = pickle.loads(pickle.dumps(rec))          # what the checkpoint holds
    rec.frame("cam1", frame, "Event20261107070000001.Avi", 2.0)  # rows written after the checkpoint...
    rec.track("cam1", 3, (10, 10, 200, 200))
    rec.close()                                      # ...then the process dies
    state.reopen(sizes)
    state.frame("cam1", frame, "Event20261107070000001.Avi", 2.0)
    state.close()
    rows = open(os.path.join(rec.dir, "frames.csv")).read().strip().splitlines()
    assert len(rows) == 3 and rows[1].split(",")[3] == "1.00" and rows[2].split(",")[3] == "2.00"
    assert len(open(os.path.join(rec.dir, "motion.csv")).read().strip().splitlines()) == 2


def test_error_log(tmp_path):
    try:
        raise ValueError("boom")
    except ValueError as e:
        log_error(str(tmp_path), "step 5", e)
    t = (tmp_path / "step_errors.log").read_text()
    assert "step 5" in t and "ValueError: boom" in t
