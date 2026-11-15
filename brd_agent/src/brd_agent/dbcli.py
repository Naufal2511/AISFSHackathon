"""Database-driven workflow.

  python -m brd_agent.dbcli provider                                   # show which VLM provider .env selects (no secrets)
  python -m brd_agent.dbcli init --cows C01,C02,...,C10 --days 1,2     # pre-create all cow-minute rows
  python -m brd_agent.dbcli import-minutes --csv upstream_minutes.csv  # Student 1's tier-1 output -> upstream columns
  python -m brd_agent.dbcli import-obs --csv track_obs.csv             # per-second tracker/identity output (optional)
  python -m brd_agent.dbcli import-human --csv human_labels.csv        # hand labels for evaluation
  python -m brd_agent.dbcli import-yolo --yolo yolo_out.csv --video-root videos/   # standalone YOLO output (file + timestamp + cow + behaviour)
  python -m brd_agent.dbcli aggregate --video-root videos/ --cows C01,...,C10      # YOLO records -> per-cow-minute rows (+ find_missing rows)
  python -m brd_agent.dbcli run --backend video --video-root videos/ [--pen-map pen.yaml] [--detector-weights best.pt]
                                [--verify-all] [--audit-rate 0.05] [--workers 8] [--max-agent-runs 4000] [--vlm mock] [--live] [--rerun]
  python -m brd_agent.dbcli status | export --out outputs/level1 | eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import load_config
from .store import Store
from .env import ConfigError, load_settings
from .pipeline import run_db


def _list(s, cast=str):
    return [cast(x) for x in s.split(",") if x.strip()] if s else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=".env", help="path to the .env file")
    ap.add_argument("--store", default=None, help="store folder (CSV+JSONL files; default: BRD_STORE_DIR from .env)")
    ap.add_argument("--config", default="configs/agent.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("provider")
    p = sub.add_parser("init"); p.add_argument("--cows", required=True); p.add_argument("--days", default="1,2"); p.add_argument("--n-minutes", type=int, default=1440)
    for n in ("import-minutes", "import-obs", "import-human"):
        p = sub.add_parser(n); p.add_argument("--csv", required=True); p.add_argument("--source", default="student1")
    p = sub.add_parser("run")
    p.add_argument("--cows"); p.add_argument("--days"); p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-agent-runs", type=int); p.add_argument("--limit-per-cow", type=int)
    p.add_argument("--vlm", default="auto", choices=["auto", "mock"], help="auto = provider from .env (USE_LOCAL_VLM)")
    p.add_argument("--backend", default="mock", help="'mock' or 'package.module:Class' (frames/zones/tracks provider)")
    p.add_argument("--trace-dir", default="outputs/traces")
    p.add_argument("--live", action="store_true", help="no look-ahead (streaming mode): history.lookahead_min=0")
    p.add_argument("--rerun", action="store_true", help="reset agent columns first and reprocess")
    p.add_argument("--video-root"); p.add_argument("--pen-map"); p.add_argument("--exemplars", help="folder with <cow_id>/*.jpg reference crops (identity check)")
    p.add_argument("--detector-weights", help="Ultralytics weights used only to look for cows YOLO missed")
    p.add_argument("--verify-all", action="store_true", help="send EVERY minute through the full agent")
    p.add_argument("--audit-rate", type=float, help="override verify.audit_rate")
    p.add_argument("--no-find-missing", action="store_true")
    for n in ("import-yolo", "aggregate"):
        p = sub.add_parser(n); p.add_argument("--video-root"); p.add_argument("--pen-map"); p.add_argument("--cows")
        if n == "import-yolo":
            p.add_argument("--yolo", required=True)
    p = sub.add_parser("import-tracking", help="tracking pipeline output: observations.csv + cow_state.csv")
    p.add_argument("--observations", required=True); p.add_argument("--cow-state", required=True); p.add_argument("--video-root")
    sub.add_parser("status")
    p = sub.add_parser("export"); p.add_argument("--out", default="outputs/level1")
    sub.add_parser("eval")
    a = ap.parse_args(argv)

    settings = load_settings(a.env)
    dbp = a.store or settings.store_dir
    cfg = load_config(a.config if Path(a.config).exists() else None)
    for attr, key in (("video_root", "root"), ("pen_map", "pen_map")):
        if getattr(a, attr, None):
            setattr(cfg.video, key, getattr(a, attr))
    if getattr(a, "verify_all", False):
        cfg.verify.verify_all = True
    if getattr(a, "audit_rate", None) is not None:
        cfg.verify.audit_rate = a.audit_rate
    if getattr(a, "no_find_missing", False):
        cfg.verify.find_missing = False

    if a.cmd == "provider":
        print(settings.describe()); return
    if a.cmd == "run":
        from .backend import MockBackend
        from .cli import load_backend
        from .vlm_client import MockVLM, make_vlm
        if a.live:
            cfg.history.lookahead_min = 0
        try:
            vlm = MockVLM() if a.vlm == "mock" else make_vlm(settings, cfg)
        except ConfigError as e:
            sys.exit(f"config error: {e}")
        print(f"VLM: {'mock' if a.vlm == 'mock' else settings.describe()} | store={dbp} | prompt={cfg.prompt_version} | live={a.live}")
        _db = Store(dbp)
        _has_rows = bool(_db.cows())
        _db.close()
        if not _has_rows:
            sys.exit(f"store '{dbp}' has no cow-minute rows, so there is nothing to run. Import tracking output first:\n"
                     f"  python -m brd_agent.dbcli --store {dbp} import-tracking --observations data/tracking/observations.csv "
                     f"--cow-state data/tracking/cow_state.csv")
        if a.backend == "video":
            from .video import VideoIndex
            from .video_backend import VideoBackend, ultralytics_detector
            if not cfg.video.root:
                sys.exit("config error: --backend video needs --video-root (or video.root in the config)")
            index = VideoIndex.scan(cfg.video.root, cfg.video.camera_map, file_camera=Store(dbp).get_meta("file_camera"))
            print(f"video index: {len(index.clips)} clips, cameras={index.cameras}")
            detector = ultralytics_detector(a.detector_weights) if a.detector_weights else None
            wd = Path(a.trace_dir).parent / "work"
            factory = lambda: VideoBackend(dbp, index, cfg.video, workdir=str(wd / "frames"), detector=detector, exemplar_dir=a.exemplars)
        else:
            factory = lambda: load_backend(a.backend)
        res = run_db(dbp, cfg, factory, vlm, trace_dir=a.trace_dir, workers=a.workers,
                     cows=_list(a.cows), days=_list(a.days, int), max_agent_runs=a.max_agent_runs,
                     rerun=a.rerun, limit_per_cow=a.limit_per_cow)
        print(json.dumps(res, indent=1)); return

    db = Store(dbp)
    if a.cmd == "import-tracking":
        from .ingest import import_tracking
        print(json.dumps(import_tracking(db, a.observations, a.cow_state, cfg), indent=1, default=str))
        db.close(); return
    if a.cmd in ("import-yolo", "aggregate"):
        from .video import VideoIndex
        from .yolo import aggregate_minutes, import_yolo
        index = VideoIndex.scan(cfg.video.root, cfg.video.camera_map) if cfg.video.root else None
        if a.cmd == "import-yolo":
            print(json.dumps(import_yolo(db, a.yolo, index, cfg.video), indent=1))
        else:
            cows = _list(a.cows) or db.cows()
            print(json.dumps(aggregate_minutes(db, cfg.video, index, cows), indent=1))
        db.close(); return
    if a.cmd == "init":
        print(f"created/ensured {db.init_rows(_list(a.cows), _list(a.days, int), a.n_minutes)} cow-minute rows in {dbp}")
    elif a.cmd == "import-minutes":
        print(db.import_minutes_csv(a.csv, a.source))
    elif a.cmd == "import-obs":
        print(db.import_track_obs_csv(a.csv, a.source))
    elif a.cmd == "import-human":
        print(db.import_human_labels_csv(a.csv))
    elif a.cmd == "status":
        print(json.dumps(db.status(), indent=1))
    elif a.cmd == "export":
        print(json.dumps(db.export_activity(a.out), indent=1))
    elif a.cmd == "eval":
        print(json.dumps(db.evaluate_vs_human(), indent=1))


if __name__ == "__main__":
    main()
