"""One command from raw videos to the agent's CSVs.

  python run_pipeline.py --videos <folder of Event*.Avi> [--name d1_1600] [--vlm auto|mock]

Stages (each step is the existing, unchanged entry point, run as a subprocess):
  prepare   sort clips into cameras, hard-link them into <run>/input/cam1..cam3
  track     stage 1  main.py                      -> stage1/tracking/tracking_records.csv
  level1    stage 1  scripts/run_level1.py --skip-agent
                     (identity, world state, anchors, behavior, contract export)
                                                  -> stage1/level1/contract/{observations,cow_state}.csv
  import    stage 2  brd_agent dbcli import-tracking   -> stage2/store/
  agent     stage 2  brd_agent dbcli run --backend video -> verified cow-minutes + traces
  export    stage 2  brd_agent dbcli export           -> final/{activity,activity_minute,actions}.csv

Resumable: finished steps are recorded in <run>/pipeline_state.json and skipped next time (same --name).
--from STEP reruns STEP and everything after it. Stage 1's own supervisor agent is not run: stage 2 replaces it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from . import cameras as camlib

REPO = Path(__file__).resolve().parent.parent
STEPS = ["prepare", "track", "level1", "import", "agent", "export"]
FINAL_FILES = ["activity.csv", "activity_minute.csv", "actions.csv"]


# ----------------------------------------------------------------------------------------------- helpers
def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Run:
    def __init__(self, root: Path):
        self.root = root
        self.input = root / "input"
        self.remux = root / "cache" / "remux"
        self.logs = root / "logs"
        self.s1 = root / "stage1"
        self.tracking = self.s1 / "tracking"
        self.level1 = self.s1 / "level1"
        self.contract = self.level1 / "contract"
        self.s2 = root / "stage2"
        self.store = self.s2 / "store"
        self.traces = self.s2 / "traces"
        self.final = root / "final"
        self.state_file = root / "pipeline_state.json"
        self.manifest_file = root / "pipeline_manifest.json"

    def load_state(self) -> dict:
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"steps": {}}

    def save_state(self, st: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
        os.replace(tmp, self.state_file)


def _env(run: Run, a) -> dict:
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
           # one remux (lossless HEVC->MP4 stream copy) per clip, shared by tracking, identity crops, rumination
           # probes and the agent's frame reads, instead of each step redoing it
           "PEN6_REMUX_DIR": str(run.remux), "PEN6_KEEP_REMUX": "1"}
    if a.ffmpeg_dir:
        env["PATH"] = str(Path(a.ffmpeg_dir)) + os.pathsep + env.get("PATH", "")
    return env


def _run_cmd(name: str, cmd: list[str], cwd: Path, env: dict, log: Path, quiet: bool) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"   $ {' '.join(str(c) for c in cmd)}", flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== {name} {_now()}\n$ {' '.join(map(str, cmd))}\n(cwd {cwd})\n")
        f.flush()
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        tail: list[str] = []
        for line in p.stdout:
            f.write(line)
            tail = (tail + [line.rstrip()])[-25:]
            if not quiet:
                print("     " + line.rstrip(), flush=True)
        rc = p.wait()
    if rc != 0:
        msg = "\n".join("     " + t for t in tail) if quiet else ""
        raise RuntimeError(f"step '{name}' failed (exit {rc}); full log: {log}" + (f"\n{msg}" if msg else ""))


def _need_file(p: Path, what: str) -> None:
    if not p.exists() or p.stat().st_size == 0:
        raise RuntimeError(f"expected {what} at {p}, but it is missing or empty")


# ----------------------------------------------------------------------------------------------- preflight
def preflight(a, cfg: dict, run: Run, env: dict) -> None:
    errs = []
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool, path=env.get("PATH")) is None:
            errs.append(f"'{tool}' is not on PATH (the HEVC clips need it). Pass --ffmpeg-dir <folder with ffmpeg.exe> "
                        "or add it to PATH. Note: the conda ffmpeg on this machine is broken; use the standalone build.")
    det = Path(a.detector)
    if not det.is_absolute():
        det = REPO / det
    if a.first_step <= STEPS.index("track"):
        if not det.exists():
            errs.append(f"detector weights not found: {det} (the fine-tuned weights are in Git LFS: `git lfs pull`)")
        elif det.stat().st_size < 1_000_000:
            errs.append(f"{det} is only {det.stat().st_size} bytes: an LFS pointer, not the weights. Run `git lfs pull`.")
    s2 = REPO / cfg["stage2"]["dir"]
    if not (s2 / "src" / "brd_agent" / "dbcli.py").exists():
        errs.append(f"stage 2 package not found at {s2}")
    if a.vlm != "mock" and not (s2 / ".env").exists():
        errs.append(f"{s2 / '.env'} is missing (copy .env.example and set the VLM provider), or use --vlm mock")
    mods = ["numpy", "cv2", "yaml", "pandas", "pydantic", "PIL"]
    if a.first_step <= STEPS.index("level1"):
        mods += ["torch", "ultralytics", "scipy", "sklearn", "tabulate"]
    r = subprocess.run([a.python, "-c", "import importlib.util as u,sys; m=[x for x in sys.argv[1:] if not u.find_spec(x)];"
                        " print(','.join(m))", *mods], capture_output=True, text=True, env=env)
    missing = [m for m in r.stdout.strip().split(",") if m]
    if r.returncode or missing:
        errs.append(f"python {a.python} is missing: {', '.join(missing) or r.stderr.strip()[-200:]} "
                    f"(pip install -r requirements.txt -r {cfg['stage2']['dir']}/requirements.txt)")
    if errs:
        raise SystemExit("preflight failed:\n  - " + "\n  - ".join(errs))
    a.detector = str(det)


# ----------------------------------------------------------------------------------------------- steps
def step_prepare(a, cfg, run: Run, env) -> dict:
    clips = camlib.assign_cameras(a.videos, cfg["cameras"])
    res = {} if a.no_camera_check else camlib.check_resolutions(clips, cfg["cameras"].get("expected_resolution", {}))
    stats = camlib.link_inputs(clips, run.input)
    (run.root / "clips.json").write_text(json.dumps(camlib.as_dicts(clips), indent=1), encoding="utf-8")
    summ = camlib.summary(clips)
    for cam, s in summ.items():
        print(f"   {cam}: {s['clips']} clips {s['first']} .. {s['last']}  (by {', '.join(s['rules'])}"
              f"{', ' + 'x'.join(map(str, res[cam])) if cam in res else ''})")
    print(f"   input folders: {run.input}  {stats}")
    if len(summ) < 3:
        print(f"   NOTE: only {', '.join(summ)} present; stage 1 runs with the cameras it has.")
    return {"cameras": summ, "resolution": res, "links": stats}


def step_track(a, cfg, run: Run, env) -> dict:
    gallery = Path(a.id_gallery) if a.id_gallery else run.tracking / "id_gallery.npz"
    if not a.id_gallery and gallery.exists():
        gallery.unlink()          # rerunning tracking: start identities fresh, as a first run would
    cam = {c: run.input / c for c in camlib.CAMERAS}
    cmd = [a.python, "main.py",
           "--front", cam["cam1"] if cam["cam1"].exists() else "",
           "--top", cam["cam2"] if cam["cam2"].exists() else "",
           "--back", cam["cam3"] if cam["cam3"].exists() else "",
           "--model", a.detector, "--fps", a.fps, "--imgsz", a.imgsz,
           "--output-dir", run.tracking, "--id-gallery", gallery]
    if not a.render_video:
        cmd.append("--no-video")
    if a.start:
        cmd += ["--start", a.start]
    if a.end:
        cmd += ["--end", a.end]
    _run_cmd("track", cmd, REPO, env, run.logs / "1_track.log", a.quiet)
    _need_file(run.tracking / "tracking_records.csv", "stage 1 tracking records")
    return {"records": str(run.tracking / "tracking_records.csv")}


def step_level1(a, cfg, run: Run, env) -> dict:
    cmd = [a.python, "scripts/run_level1.py", "--records", run.tracking / "tracking_records.csv",
           "--out", run.level1, "--video-root", run.input, "--pen-size", a.herd_size, "--skip-agent"]
    _run_cmd("level1", cmd, REPO, env, run.logs / "2_level1.log", a.quiet)
    for f in ("observations.csv", "cow_state.csv"):
        _need_file(run.contract / f, f"stage 1 contract {f}")
    return {"observations": str(run.contract / "observations.csv"), "cow_state": str(run.contract / "cow_state.csv")}


def _dbcli(a, cfg, run: Run) -> tuple[list, Path]:
    s2 = REPO / cfg["stage2"]["dir"]
    return [a.python, "-m", "brd_agent.dbcli", "--env", s2 / ".env", "--store", run.store], s2


def _s2_env(env: dict, s2: Path) -> dict:
    return {**env, "PYTHONPATH": str(s2 / "src") + os.pathsep + env.get("PYTHONPATH", "")}


def step_import(a, cfg, run: Run, env) -> dict:
    base, s2 = _dbcli(a, cfg, run)
    if run.store.exists():
        shutil.rmtree(run.store)   # import is not idempotent: (re)build the store from the current contract
    cmd = base + ["import-tracking", "--observations", run.contract / "observations.csv",
                  "--cow-state", run.contract / "cow_state.csv", "--video-root", run.input]
    _run_cmd("import", cmd, s2, _s2_env(env, s2), run.logs / "3_import.log", a.quiet)
    return {"store": str(run.store)}


def step_agent(a, cfg, run: Run, env) -> dict:
    base, s2 = _dbcli(a, cfg, run)
    cmd = base + ["run", "--backend", "video", "--video-root", run.input, "--vlm", a.vlm,
                  "--trace-dir", run.traces, "--workers", a.workers] + list(a.agent_args)
    _run_cmd("agent", cmd, s2, _s2_env(env, s2), run.logs / "4_agent.log", a.quiet)
    return {"traces": str(run.traces)}


def step_export(a, cfg, run: Run, env) -> dict:
    base, s2 = _dbcli(a, cfg, run)
    _run_cmd("export", base + ["export", "--out", run.final], s2, _s2_env(env, s2), run.logs / "5_export.log", a.quiet)
    for f in FINAL_FILES:
        _need_file(run.final / f, f"final {f}")
    return {f: str(run.final / f) for f in FINAL_FILES}


STEP_FN = {"prepare": step_prepare, "track": step_track, "level1": step_level1,
           "import": step_import, "agent": step_agent, "export": step_export}


# ----------------------------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", help="folder with the Event*.Avi clips (flat, or Cam 0N subfolders)")
    ap.add_argument("--name", help="run name (folder under --out-root); reuse it to resume. Default: from the clips")
    ap.add_argument("--out-root", default="outputs/pipeline", help="where runs are kept (relative to the repo)")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--from", dest="from_step", choices=STEPS, help="rerun from this step (and all later ones)")
    ap.add_argument("--until", choices=STEPS, help="stop after this step")
    ap.add_argument("--start", help="wall-clock start for stage 1, e.g. '2026-11-07 16:00'")
    ap.add_argument("--end", help="wall-clock end for stage 1")
    g = ap.add_argument_group("stage 1")
    g.add_argument("--detector", help="detector weights (default: configs/pipeline.yaml stage1.detector)")
    g.add_argument("--fps", type=float)
    g.add_argument("--imgsz", type=int)
    g.add_argument("--herd-size", type=int)
    g.add_argument("--id-gallery", help="shared appearance gallery (keeps COW_xx the same across runs/days); "
                                       "default: a fresh one inside the run")
    g.add_argument("--render-video", action="store_true", help="also render stage 1's dashboard video (slow)")
    g.add_argument("--no-camera-check", action="store_true", help="skip the per-camera resolution check")
    g = ap.add_argument_group("stage 2")
    g.add_argument("--vlm", choices=["auto", "mock"])
    g.add_argument("--workers", type=int)
    g.add_argument("--agent-args", nargs=argparse.REMAINDER, default=[],
                   help="passed to `dbcli run` unchanged, e.g. --agent-args --max-agent-runs 500 --audit-rate 0.1 (keep last)")
    g = ap.add_argument_group("runtime")
    g.add_argument("--python", default=sys.executable, help="interpreter for both stages (default: this one)")
    g.add_argument("--ffmpeg-dir", help="folder with a working ffmpeg/ffprobe, prepended to PATH")
    g.add_argument("--keep-cache", action="store_true", help="keep the remuxed clips after a successful run")
    g.add_argument("--quiet", action="store_true", help="step output only to the logs")
    g.add_argument("--dry-run", action="store_true", help="assign cameras and print the plan; run nothing")
    return ap.parse_args(argv)


def _default_name(videos: str, cfg: dict) -> str:
    clips = camlib.assign_cameras(videos, cfg["cameras"])
    first = min(c.start for c in clips)
    return "run_" + first.replace("-", "").replace(":", "").replace("T", "_")[:13]


def main(argv=None) -> int:
    a = parse_args(argv)
    os.chdir(REPO)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    s1, s2c = cfg.get("stage1", {}), cfg.get("stage2", {})
    a.detector = a.detector or s1.get("detector", "models/angus_det_hprc_m1280.pt")
    a.fps = a.fps or s1.get("fps", 1.0)
    a.imgsz = a.imgsz or s1.get("imgsz", 1280)
    a.herd_size = a.herd_size or s1.get("herd_size", 8)
    a.vlm = a.vlm or s2c.get("vlm", "auto")
    a.workers = a.workers or s2c.get("workers", 1)
    cfg.setdefault("stage2", {}).setdefault("dir", "brd_agent")

    root = Path(a.out_root)
    if a.name is None:
        if not a.videos:
            raise SystemExit("give --videos (and optionally --name), or --name of an existing run to resume")
        a.name = _default_name(a.videos, cfg)
    run = Run((root / a.name).resolve())
    state = run.load_state()
    if a.videos is None:
        a.videos = state.get("videos")
        if not a.videos:
            raise SystemExit(f"run {run.root} has no recorded --videos; pass --videos")
    a.videos = str(Path(a.videos).resolve())
    if state.get("videos") and state["videos"] != a.videos and not a.from_step:
        raise SystemExit(f"run '{a.name}' was made from {state['videos']}, not {a.videos}. "
                         f"Use another --name, or --from prepare to rebuild it from the new folder.")

    done = state.get("steps", {})
    a.first_step = STEPS.index(a.from_step) if a.from_step else next(
        (i for i, s in enumerate(STEPS) if done.get(s, {}).get("status") != "ok"), len(STEPS))
    last = STEPS.index(a.until) if a.until else len(STEPS) - 1
    todo = STEPS[a.first_step:last + 1]

    env = _env(run, a)
    print(f"pipeline run: {run.root}")
    print(f"  videos: {a.videos}")
    print(f"  stage 1: detector={a.detector} fps={a.fps} imgsz={a.imgsz} herd={a.herd_size}"
          f"{' window=' + str(a.start) + '..' + str(a.end) if a.start or a.end else ''}")
    print(f"  stage 2: vlm={a.vlm} workers={a.workers} {' '.join(a.agent_args)}")
    print(f"  steps:  {' -> '.join(todo) if todo else '(all done; use --from STEP to redo)'}")
    if a.dry_run:
        clips = camlib.assign_cameras(a.videos, cfg["cameras"])
        print(json.dumps(camlib.summary(clips), indent=1))
        return 0
    if not todo:
        print(f"final outputs: {run.final}")
        return 0
    preflight(a, cfg, run, env)

    state["videos"] = a.videos
    state["started"] = state.get("started") or _now()
    for s in STEPS[a.first_step:]:
        done.pop(s, None)             # everything after a rerun step is stale
    state["steps"] = done
    run.save_state(state)
    run.remux.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    for name in todo:
        print(f"\n== [{STEPS.index(name) + 1}/{len(STEPS)}] {name}", flush=True)
        t = time.time()
        done[name] = {"status": "running", "started": _now()}
        run.save_state(state)
        try:
            info = STEP_FN[name](a, cfg, run, env)
        except (RuntimeError, camlib.CameraError) as e:
            done[name].update(status="failed", error=str(e)[:2000], seconds=round(time.time() - t, 1))
            run.save_state(state)
            print(f"\nFAILED at '{name}': {e}\nFix it and rerun the same command: finished steps are skipped.")
            return 1
        done[name].update(status="ok", ended=_now(), seconds=round(time.time() - t, 1), info=info)
        run.save_state(state)
        print(f"   ok in {time.time() - t:.0f} s", flush=True)

    finished = all(done.get(s, {}).get("status") == "ok" for s in STEPS)
    if finished and not a.keep_cache and run.remux.exists():
        shutil.rmtree(run.remux, ignore_errors=True)
    manifest = {"run": str(run.root), "videos": a.videos, "finished": finished, "updated": _now(),
                "settings": {k: getattr(a, k) for k in ("detector", "fps", "imgsz", "herd_size", "vlm", "workers",
                                                        "start", "end", "id_gallery", "agent_args")},
                "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                             text=True, cwd=REPO).stdout.strip() or None,
                "steps": done,
                "outputs": {f: str(run.final / f) for f in FINAL_FILES} if finished else {}}
    run.manifest_file.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\n{'done' if finished else 'stopped'} in {time.time() - t_all:.0f} s -> "
          f"{run.final if finished else run.root}")
    if finished:
        for f in FINAL_FILES:
            print(f"   {run.final / f}")
    return 0
