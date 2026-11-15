#!/usr/bin/env python3
"""Download from the organizers' Google Drive exactly the clips a share needs that are not on this machine yet.

  python fetch_clips.py --who afeef_training --dest "<your video folder>"     # only what is missing for your share
  python fetch_clips.py --check                                             # can rclone see the Drive folder?
  python fetch_clips.py --list                                              # every clip name -> outputs/shares/all_clips.txt
  python fetch_clips.py --all --dest "<folder>"                             # everything (stitching + Level 2)

The Drive layout is kept (Day 1/Cam 01/Event...Avi), which is what the pipeline expects. Copies are resumable:
run it again and only missing files are downloaded. Where the footage lives is in configs/pipeline.yaml (drive:).

One-time rclone setup (your own Google account, the one the folder is shared with):
  install   Mac: brew install rclone     Windows: winget install Rclone.Rclone     HPRC: scripts/hprc/pipeline_setup.sh
  connect   rclone config create gdrive drive scope=drive.readonly        (a browser opens: sign in, allow)
            on HPRC (no browser) see docs/HPRC_PIPELINE.md step 3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from pipeline import cameras as camlib  # noqa: E402

SHARES = REPO / "outputs" / "shares"


def rclone_bin(explicit: str | None = None) -> str:
    for c in [explicit, shutil.which("rclone"), str(REPO / "tools" / "rclone"), str(REPO / "tools" / "rclone.exe")]:
        if c and Path(c).exists():
            return c
    raise SystemExit("rclone is not installed.\n  Mac: brew install rclone\n  Windows: winget install Rclone.Rclone "
                     "(then open a NEW terminal)\n  HPRC: bash scripts/hprc/pipeline_setup.sh")


def drive_cfg() -> dict:
    d = yaml.safe_load((REPO / "configs" / "pipeline.yaml").read_text(encoding="utf-8")).get("drive") or {}
    return {"remote": d.get("remote", "gdrive"), "folder": d.get("folder", ""), "shared": bool(d.get("shared_with_me", True))}


def base_args(rc: str, d: dict) -> tuple[list[str], str]:
    remotes = subprocess.run([rc, "listremotes"], capture_output=True, text=True).stdout.split()
    if f"{d['remote']}:" not in remotes:
        raise SystemExit(f"rclone has no remote '{d['remote']}' yet. Connect your Google account once:\n"
                         f"  {rc} config create {d['remote']} drive scope=drive.readonly\n"
                         "(a browser opens; sign in with the account the hackathon folder is shared with).\n"
                         "On HPRC there is no browser: follow docs/HPRC_PIPELINE.md step 3.")
    flags = ["--drive-shared-with-me"] if d["shared"] else []
    return flags, f"{d['remote']}:{d['folder']}"


def local_clips(dest: Path) -> set[str]:
    return {p.name.lower() for p in dest.rglob("*") if p.suffix.lower() in camlib.VIDEO_EXTS} if dest.exists() else set()


def needed(who: str, have: set[str], lead_min: float = 5.0) -> list[str]:
    import run_share as rs
    plan_p = SHARES / "plan.json"
    if not plan_p.exists():
        raise SystemExit("no outputs/shares/plan.json yet: get it from the team (made once on HPRC), or make one with\n"
                         "  python fetch_clips.py --list   then   python run_share.py --plan-only --videos outputs/shares/all_clips.txt")
    plan = json.loads(plan_p.read_text(encoding="utf-8"))
    if who not in plan["shares"]:
        raise SystemExit(f"{who} is not in plan.json (shares: {', '.join(plan['shares'])})")
    sh = plan["shares"][who]
    start = rs.lead_in(sh["start"], plan["span"][0], lead_min)     # the clip running into the first chunk's lead-in too
    return rs.missing_for_window(plan, have, start, sh["end"])


def copy(rc: str, flags: list[str], src: str, dest: Path, include_lines: list[str]) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    inc = SHARES / "_fetch_include.txt"
    inc.parent.mkdir(parents=True, exist_ok=True)
    inc.write_text("".join(f"{l}\n" for l in include_lines), encoding="utf-8")
    cmd = [rc, "copy", *flags, src, str(dest), "--include-from", str(inc), "--transfers", "8", "--checkers", "16",
           "--drive-chunk-size", "64M", "--retries", "5", "--low-level-retries", "20", "-P"]
    print("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    return subprocess.run(cmd).returncode


def fetch_for(who: str, dest: Path, rclone: str | None = None) -> bool:
    """Used by run_share.py when clips are missing. True when nothing is missing afterwards."""
    rc = rclone_bin(rclone)
    flags, src = base_args(rc, drive_cfg())
    miss = needed(who, local_clips(dest))
    if not miss:
        print(f"all clips for {who} are already in {dest}")
        return True
    print(f"downloading {len(miss)} clip(s) for {who} from {src} into {dest} (about {len(miss) * 35 / 1024:.1f} GB)")
    rcode = copy(rc, flags, src, dest, [f"/**/{n}" for n in miss])
    left = needed(who, local_clips(dest))
    if left:
        print(f"still missing {len(left)} clip(s), e.g. {left[:3]} (rclone exit {rcode}): they may not be on the Drive; "
              f"rerun, or check the Drive folder")
        return False
    print(f"done: every clip for {who} is in {dest}")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--who", help="share name from plan.json (naufal_training, afeef_training, nooh_training, hprc_day2)")
    ap.add_argument("--dest", help="your video folder (clips land in <dest>/Day N/Cam 0N/)")
    ap.add_argument("--check", action="store_true", help="only test that the Drive folder is reachable")
    ap.add_argument("--list", action="store_true", help="write every clip path to outputs/shares/all_clips.txt")
    ap.add_argument("--all", action="store_true", help="download every clip")
    ap.add_argument("--rclone", help="path to the rclone binary (default: PATH, then tools/rclone)")
    a = ap.parse_args(argv)
    os.chdir(REPO)
    rc = rclone_bin(a.rclone)
    flags, src = base_args(rc, drive_cfg())
    if a.check or a.list:
        p = subprocess.run([rc, "lsf", "-R", "--files-only", "--include", "Event*", *flags, src], capture_output=True, text=True)
        if p.returncode:
            print(p.stderr[-600:])
            print(f"FAIL: cannot list {src}. Is the folder shared with the Google account rclone uses? "
                  f"Does the path in configs/pipeline.yaml (drive.folder) match what you see under 'Shared with me'?")
            return 1
        names = [l for l in p.stdout.splitlines() if l.strip()]
        days = sorted({n.split("/")[0] for n in names if "/" in n})
        print(f"OK: {len(names)} clips on the Drive under {src} ({', '.join(days)})")
        if a.list:
            SHARES.mkdir(parents=True, exist_ok=True)
            (SHARES / "all_clips.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
            print("written to outputs/shares/all_clips.txt; make the plan with:\n"
                  "  python run_share.py --plan-only --replan --videos outputs/shares/all_clips.txt")
        return 0
    if not a.dest:
        raise SystemExit("give --dest <your video folder>")
    dest = Path(a.dest).expanduser().resolve()
    if a.all:
        return copy(rc, flags, src, dest, ["/**/Event*"])
    if not a.who:
        raise SystemExit("give --who <share> (or --all)")
    return 0 if fetch_for(a.who, dest, a.rclone) else 1


if __name__ == "__main__":
    sys.exit(main())
