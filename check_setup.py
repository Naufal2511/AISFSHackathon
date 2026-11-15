#!/usr/bin/env python3
"""Is this laptop ready for its share of the 48 h run? Prints READY or NOT READY, and the exact fix for each problem.

  python check_setup.py                                   # asks who you are and where the clips are
  python check_setup.py --who afeef_training --videos "D:\\Pen6\\all clips" [--ffmpeg-dir "C:\\ffmpeg\\bin"]

Nothing here changes your files. Two tiny TAMU requests are sent (one per agent client) to prove the key works.
run_share.py runs these same checks before it starts and refuses to start on a FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

PEOPLE = ["naufal_training", "afeef_training", "nooh_training"]
OK, WARN, FAIL = "PASS", "WARN", "FAIL"
USE_COLOR = sys.stdout.isatty() and os.name != "nt" or os.environ.get("WT_SESSION")
C = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", "end": "\033[0m", "b": "\033[1m"} if USE_COLOR else \
    {OK: "", WARN: "", FAIL: "", "end": "", "b": ""}


class Report:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, status: str, name: str, detail: str = "", fix: str = ""):
        self.rows.append({"status": status, "check": name, "detail": detail, "fix": fix})
        mark = {OK: "✔", WARN: "!", FAIL: "✖"}[status]
        print(f"  {C[status]}{mark} {status:4s}{C['end']}  {name}" + (f"  —  {detail}" if detail else ""), flush=True)
        if fix and status != OK:
            for line in fix.splitlines():
                print(f"           {C[status]}FIX:{C['end']} {line}")

    @property
    def failed(self):
        return [r for r in self.rows if r["status"] == FAIL]


def _env_file() -> dict:
    out = {}
    p = REPO / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def section(title: str):
    print(f"\n{C['b']}{title}{C['end']}")


# ----------------------------------------------------------------------------------------------- checks
def check_python(r: Report):
    v = sys.version_info
    good = (3, 10) <= (v.major, v.minor) <= (3, 12)
    r.add(OK if good else FAIL, "Python version", f"{platform.python_version()} at {sys.executable}",
          "use Python 3.10-3.12 (3.11 recommended): create the venv with that interpreter")


def check_repo(r: Report):
    if not (REPO / ".git").exists():
        r.add(WARN, "git checkout", "not a git checkout (zip?)", "clone the repo so you get fixes: git clone ... && git lfs pull")
        return
    try:
        subprocess.run(["git", "fetch", "-q", "origin"], cwd=REPO, timeout=60, capture_output=True)
        br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        behind = subprocess.run(["git", "rev-list", "--count", "HEAD..origin/main"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        r.add(WARN, "git checkout", f"could not ask git: {e}", "install git, or check your network")
        return
    if behind and behind != "0":
        r.add(FAIL, "code is up to date", f"{behind} commit(s) behind origin/main (on {br} @ {head})",
              "git checkout main && git pull && git lfs pull")
    else:
        r.add(OK, "code is up to date", f"{br} @ {head} (all three laptops should show the same commit)")
    if dirty:
        r.add(WARN, "no local code edits", f"{len(dirty.splitlines())} modified file(s)",
              "git stash   (results must come from the shared code)")


def check_packages(r: Report):
    need = {"numpy": "numpy", "cv2": "opencv-python", "torch": "torch", "torchvision": "torchvision", "timm": "timm",
            "ultralytics": "ultralytics", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
            "yaml": "pyyaml", "pydantic": "pydantic", "PIL": "pillow", "requests": "requests", "tabulate": "tabulate"}
    missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        r.add(FAIL, "Python packages", "missing: " + ", ".join(missing),
              f'"{sys.executable}" -m pip install -r requirements.txt -r brd_agent/requirements.txt')
    else:
        r.add(OK, "Python packages", "all present")


def check_device(r: Report) -> str:
    try:
        import torch
        if torch.cuda.is_available():
            dev = f"cuda ({torch.cuda.get_device_name(0)})"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps (Apple GPU)"
        else:
            dev = "cpu"
    except Exception as e:
        r.add(FAIL, "PyTorch works", str(e)[:200], "reinstall torch for your platform (pytorch.org)")
        return "none"
    r.add(OK if dev != "cpu" else WARN, "GPU for tracking", dev,
          "no GPU: tracking runs on the CPU and is several times slower. Use fewer chunks in parallel, or HPRC.")
    return dev


def check_weights(r: Report):
    for p in [REPO / "models" / "angus_det_hprc_m1280.pt", REPO / "models" / "behavior" / "beh_finetuned.pt"]:
        if not p.exists():
            r.add(FAIL, f"model {p.relative_to(REPO)}", "missing", "git lfs install && git lfs pull")
        elif p.stat().st_size < 1_000_000:
            r.add(FAIL, f"model {p.relative_to(REPO)}", f"only {p.stat().st_size} bytes: a Git LFS pointer, not the weights",
                  "git lfs install && git lfs pull")
        else:
            r.add(OK, f"model {p.relative_to(REPO)}", f"{p.stat().st_size / 2 ** 20:.0f} MB")
    hub = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub" / "facebookresearch_dinov2_main"
    r.add(OK if hub.exists() else WARN, "DINOv2 (re-ID) cached", str(hub) if hub.exists() else "not downloaded yet",
          "it downloads itself on the first run (~90 MB, needs internet); the smoke test does that")


def check_ffmpeg(r: Report, ffmpeg_dir: str | None) -> dict:
    env = dict(os.environ)
    if ffmpeg_dir:
        env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool, path=env.get("PATH"))
        if not path:
            r.add(FAIL, f"{tool} on PATH", "not found",
                  "Mac: brew install ffmpeg   Windows: download the 'full_build' zip from gyan.dev, unzip, then pass\n"
                  "--ffmpeg-dir \"C:\\path\\to\\ffmpeg\\bin\" (the conda ffmpeg is known to be broken on Windows)")
            continue
        try:
            out = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=20, env=env)
            ok = out.returncode == 0
            r.add(OK if ok else FAIL, f"{tool} runs", out.stdout.splitlines()[0][:70] if ok else f"exit {out.returncode}",
                  "this ffmpeg is broken: install the standalone build and pass --ffmpeg-dir")
        except (OSError, subprocess.TimeoutExpired) as e:
            r.add(FAIL, f"{tool} runs", str(e)[:120], "install the standalone ffmpeg build and pass --ffmpeg-dir")
    return env


def check_env(r: Report) -> str | None:
    env = _env_file()
    if not (REPO / ".env").exists():
        r.add(FAIL, ".env file", f"missing at {REPO / '.env'}",
              "copy .env.example to .env and put the TAMU key in it:  TAMU_API_KEY=<key>")
        return None
    key = os.environ.get("TAMU_API_KEY") or env.get("TAMU_API_KEY")
    if not key:
        r.add(FAIL, "TAMU_API_KEY", "empty", "edit .env:  TAMU_API_KEY=<key>   (no quotes, no spaces)")
        return None
    r.add(OK, "TAMU_API_KEY", f"set ({len(key)} characters, ends ...{key[-4:]})")
    b = REPO / "brd_agent" / ".env"
    if b.exists():
        r.add(WARN, "brd_agent/.env", "exists, so stage 2 uses ITS provider instead of TAMU",
              "delete brd_agent/.env unless you mean to use another VLM for stage 2")
    for k in ("AGENT_BACKEND", "AGENT_VLM_BASE_URL"):
        if env.get(k) or os.environ.get(k):
            r.add(WARN, f"{k} override", env.get(k) or os.environ.get(k),
                  f"remove {k} from .env / your shell unless intended (all laptops must use the same models)")
    return key


def check_tamu(r: Report):
    """One tiny request through each agent's real client, exactly as the pipeline sends them."""
    import yaml
    try:
        from pipeline.orchestrator import _tamu_for_stage2
        e = _tamu_for_stage2(yaml.safe_load((REPO / "configs" / "pipeline.yaml").read_text(encoding="utf-8")))
        sys.path.insert(0, str(REPO / "brd_agent" / "src"))
        from brd_agent.config import AgentConfig
        from brd_agent.env import load_settings
        from brd_agent.schemas import Hypothesis
        from brd_agent.vlm_client import make_vlm
        old = {k: os.environ.get(k) for k in e}
        os.environ.update(e)
        try:
            cfg = AgentConfig(behaviors_path=str(REPO / "brd_agent" / "configs" / "behaviors.yaml"))
            cfg.vlm.cache_dir = None
            v = make_vlm(load_settings(str(REPO / "brd_agent" / ".env")), cfg)
            t = time.time()
            h = v.ask("hypothesis", "Connection test. Answer in JSON matching the schema.",
                      "Return label 'other_not_visible', confidence 0.1, cues [].", [], Hypothesis)
            h = h[0] if isinstance(h, tuple) else h
            r.add(OK, "TAMU from the BRD agent (stage 2)", f"{v.model}, answered in {time.time() - t:.1f} s")
        finally:
            for k, val in old.items():
                if val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = val
    except Exception as ex:
        r.add(FAIL, "TAMU from the BRD agent (stage 2)", f"{type(ex).__name__}: {str(ex)[:160]}",
              "check the key in .env, your internet / VPN, and that tamu.ai is up")
    try:
        from src.agent.vlm import make_backend
        cfg = yaml.safe_load((REPO / "configs" / "agent.yaml").read_text(encoding="utf-8"))
        prof = yaml.safe_load((REPO / "configs" / "pipeline.yaml").read_text(encoding="utf-8")).get("stage1", {}).get(
            "id_agent_backend", "tamu_luna")
        be = make_backend(cfg, prof, with_fallback=False)          # TAMU only: a pass must not be the local fallback
        t = time.time()
        txt = be.complete([{"role": "user", "content": [{"type": "text", "text": "Reply with the word ok."}]}])
        label = getattr(be, "label", getattr(be, "name", "backend"))
        if txt:
            r.add(OK, "TAMU from the identity agent (stage 1)", f"{str(label).split(' @')[0]}, answered in {time.time() - t:.1f} s")
        else:
            r.add(FAIL, "TAMU from the identity agent (stage 1)", f"no answer from {label}",
                  "check TAMU_API_KEY in .env; remove AGENT_BACKEND overrides")
    except Exception as ex:
        r.add(FAIL, "TAMU from the identity agent (stage 1)", f"{type(ex).__name__}: {str(ex)[:160]}",
              "check TAMU_API_KEY in .env and your internet / VPN")


def check_videos(r: Report, videos: str | None, who: str | None, env: dict) -> dict:
    info = {}
    if not videos:
        r.add(FAIL, "video folder", "not given", "pass --videos <folder with the Event*.Avi clips>")
        return info
    import yaml
    from pipeline import cameras as camlib
    root = Path(videos).expanduser()
    if not root.is_dir():
        r.add(FAIL, "video folder", f"{root} does not exist", "check the path (quote it if it has spaces)")
        return info
    cfg = yaml.safe_load((REPO / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
    try:
        clips = camlib.assign_cameras(str(root), cfg["cameras"])
    except Exception as e:
        r.add(FAIL, "clips sorted into cameras", str(e)[:200], "only Event*.Avi clips from the 3 pen cameras in that folder")
        return info
    summ = camlib.summary(clips)
    r.add(OK if len(summ) == 3 else WARN, "clips sorted into cameras",
          ", ".join(f"{c}: {s['clips']} ({s['first'][:16]} .. {s['last'][:16]})" for c, s in summ.items()),
          "a camera is missing from this folder" if len(summ) < 3 else "")
    plan_p = REPO / "outputs" / "shares" / "plan.json"
    if not plan_p.exists():
        r.add(WARN, "shared plan", "outputs/shares/plan.json not here yet",
              "ONE person runs:  python run_share.py --plan-only --videos <ALL clips>  and sends plan.json to the\n"
              "others, who put it in outputs/shares/. All three must use the same file.")
    else:
        digest = hashlib.sha256(plan_p.read_bytes()).hexdigest()[:10]
        plan = json.loads(plan_p.read_text(encoding="utf-8"))
        r.add(OK, "shared plan", f"plan.json fingerprint {digest} (compare: all three laptops must show the same)")
        if who:
            share = plan["shares"][who]
            have = {c.name.lower() for c in clips}
            miss = [n for n in share["clips"] if n.lower() not in have]
            r.add(FAIL if len(miss) > 0.02 * len(share["clips"]) else (WARN if miss else OK),
                  f"clips for {who}", f"{len(share['clips']) - len(miss)} of {len(share['clips'])} present "
                  f"({share['start']} -> {share['end']})" + (f"; missing e.g. {miss[:2]}" if miss else ""),
                  "copy the missing clips into the video folder (keep the original names)")
            info["share_bytes"] = sum(os.path.getsize(c.path) for c in clips if c.name in set(share["clips"]))
    # one real decode, the way the pipeline does it: the AVIs carry HEVC under a proprietary 'GHEC' tag, so ffmpeg
    # stream-copies them into an MP4 (forcing the hevc decoder) and OpenCV reads that
    c0 = next(iter(sorted(clips, key=lambda c: c.start)), None)
    ff = shutil.which("ffmpeg", path=env.get("PATH"))
    if c0 is not None and ff:
        import tempfile
        import cv2
        with tempfile.TemporaryDirectory() as td:
            mp4 = os.path.join(td, "t.mp4")
            p = subprocess.run([ff, "-v", "error", "-y", "-vcodec", "hevc", "-i", c0.path, "-c:v", "copy", "-tag:v", "hvc1",
                                "-t", "3", mp4], capture_output=True, text=True, timeout=120, env=env)
            frame = None
            if p.returncode == 0 and os.path.exists(mp4):
                cap = cv2.VideoCapture(mp4)
                ok, frame = cap.read()
                cap.release()
            good = frame is not None
            r.add(OK if good else FAIL, "decode a clip (remux + read)",
                  f"{c0.name}: {frame.shape[1]}x{frame.shape[0]}" if good else (p.stderr.strip()[-160:] or "OpenCV could not read the remuxed MP4"),
                  "install the standalone ffmpeg full build (Windows: gyan.dev) and pass --ffmpeg-dir; "
                  "make sure opencv-python is installed")
    return info


def check_machine(r: Report, info: dict):
    out = REPO / "outputs"
    out.mkdir(exist_ok=True)
    free = shutil.disk_usage(out).free / 2 ** 30
    need = max(30.0, 1.3 * info.get("share_bytes", 0) / 2 ** 30 / 3)   # remux cache of the chunks running at once
    r.add(OK if free >= need else (WARN if free >= 15 else FAIL), "free disk for outputs + remux cache",
          f"{free:.0f} GB free (want >= {need:.0f} GB)", "free up disk space, or put the repo on a bigger drive")
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        ram = None
    if ram:
        r.add(OK if ram >= 16 else WARN, "memory", f"{ram:.0f} GB", "under 16 GB: run 1 chunk at a time (--parallel 1)")


def run_checks(who: str | None, videos: str | None, ffmpeg_dir: str | None, tamu: bool = True) -> Report:
    r = Report()
    print(f"{C['b']}Setup check for {who or '(no share chosen)'} on {platform.node()} ({platform.system()}){C['end']}")
    section("Code and Python")
    check_python(r)
    check_repo(r)
    check_packages(r)
    if r.failed:          # nothing below works without the packages
        return _verdict(r, who)
    check_device(r)
    check_weights(r)
    section("Video tools")
    env = check_ffmpeg(r, ffmpeg_dir)
    section("TAMU access (.env)")
    if check_env(r) and tamu:
        check_tamu(r)
    section("Footage")
    info = check_videos(r, videos, who, env)
    section("This machine")
    check_machine(r, info)
    return _verdict(r, who)


def _verdict(r: Report, who: str | None) -> Report:
    n_w = sum(1 for x in r.rows if x["status"] == WARN)
    print()
    if r.failed:
        print(f"{C[FAIL]}{C['b']}NOT READY{C['end']}: {len(r.failed)} problem(s) to fix (see FIX lines above), then run this again.")
    else:
        print(f"{C[OK]}{C['b']}READY{C['end']}" + (f" ({n_w} warning(s) above: read them)" if n_w else ""))
    if who:
        p = REPO / "outputs" / "shares" / who / "setup_check.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(),
                                 "ready": not r.failed, "rows": r.rows}, indent=1), encoding="utf-8")
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--who", choices=PEOPLE)
    ap.add_argument("--videos")
    ap.add_argument("--ffmpeg-dir")
    ap.add_argument("--no-tamu", action="store_true", help="skip the two live TAMU requests")
    a = ap.parse_args(argv)
    os.chdir(REPO)
    if sys.stdin.isatty():
        if not a.who:
            print("Who is running this laptop?")
            for i, p in enumerate(PEOPLE, 1):
                print(f"  {i}) {p}")
            v = input("choose 1-3: ").strip()
            a.who = PEOPLE[int(v) - 1] if v in ("1", "2", "3") else None
        if not a.videos:
            a.videos = input("video folder (Event*.Avi clips): ").strip().strip('"') or None
    r = run_checks(a.who, a.videos, a.ffmpeg_dir, tamu=not a.no_tamu)
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
