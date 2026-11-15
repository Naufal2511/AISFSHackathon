"""Sort a folder of Event clips into cameras, and lay them out the way stage 1 reads them (one folder per camera).

Stage 1 (main.py) takes one source per camera and plays each camera's clips as one continuous stream. The footage
may arrive flat (all clips in one folder) or in the DVR layout ("Day 1/Cam 01/..."). Rules, in order:
  1. a folder on the clip's path matches cameras.folder_pattern ("Cam 01", "cam2", "Camera 3");
  2. the 3 digits after the 14-digit timestamp map through cameras.suffix (006 -> cam1, 017 -> cam2, 003 -> cam3).
A clip no rule places, or two rules that disagree, stops the run: a wrong camera silently corrupts every position.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

EVENT_RE = re.compile(r"Event(\d{14})(\d{3})")
VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".mov"}
CAMERAS = ("cam1", "cam2", "cam3")


class CameraError(RuntimeError):
    pass


@dataclass
class ClipInfo:
    path: str          # absolute source path
    name: str          # file name (never changed: both stages key on it)
    camera: str        # cam1 | cam2 | cam3
    rule: str          # how the camera was decided: folder | suffix
    start: str         # wall-clock start from the name, YYYY-mm-ddTHH:MM:SS


def _folder_camera(path: Path, root: Path, pattern: re.Pattern) -> Optional[str]:
    try:
        parts = path.parent.relative_to(root).parts
    except ValueError:
        parts = path.parent.parts
    hits = {f"cam{m.group(1)}" for p in parts if (m := pattern.match(p.strip()))}
    if len(hits) > 1:
        raise CameraError(f"{path}: folders name more than one camera {sorted(hits)}")
    return hits.pop() if hits else None


def assign_cameras(root: str | Path, cfg: dict) -> list[ClipInfo]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise CameraError(f"--videos {root} is not a folder")
    pattern = re.compile(cfg.get("folder_pattern", r"^cam(?:era)?[\s_-]*0?([1-3])$"), re.I)
    suffix = {str(k): v for k, v in (cfg.get("suffix") or {}).items()}
    clips, seen, unknown, conflicts = [], {}, [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS or p.name.startswith("."):
            continue
        m = EVENT_RE.search(p.name)
        if not m:
            continue
        if p.name.lower() in seen:            # same clip twice (e.g. a copy in another folder): both stages key on the name
            continue
        seen[p.name.lower()] = p
        by_folder = _folder_camera(p, root, pattern)
        by_suffix = suffix.get(m.group(2))
        if by_folder and by_suffix and by_folder != by_suffix:
            conflicts.append(f"{p.relative_to(root)}: folder says {by_folder}, suffix {m.group(2)} says {by_suffix}")
            continue
        cam, rule = (by_folder, "folder") if by_folder else (by_suffix, "suffix")
        if cam not in CAMERAS:
            unknown.append(str(p.relative_to(root)))
            continue
        s = m.group(1)
        clips.append(ClipInfo(str(p), p.name, cam, rule, f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:{s[10:12]}:{s[12:14]}"))
    problems = []
    if conflicts:
        problems.append("camera rules disagree:\n  " + "\n  ".join(conflicts[:10]))
    if unknown:
        problems.append(f"{len(unknown)} clip(s) with no camera (no 'Cam 0N' folder, unknown suffix):\n  "
                        + "\n  ".join(unknown[:10]))
    if problems:
        raise CameraError("\n".join(problems) + "\nFix: put the clips in 'Cam 01' / 'Cam 02' / 'Cam 03' folders, "
                          "or extend cameras.suffix in configs/pipeline.yaml.")
    if not clips:
        raise CameraError(f"no Event*.Avi clips under {root}")
    return clips


def ffprobe_resolution(path: str) -> Optional[tuple[int, int]]:
    if shutil.which("ffprobe") is None:
        return None
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "json", path], capture_output=True, timeout=60)
    try:
        st = json.loads(r.stdout or b"{}")["streams"][0]
        return int(st["width"]), int(st["height"])
    except (KeyError, IndexError, ValueError):
        return None


def check_resolutions(clips: list[ClipInfo], expected: dict,
                      probe: Callable[[str], Optional[tuple[int, int]]] = ffprobe_resolution) -> dict:
    """First clip of each camera must have that camera's resolution. Returns {camera: [w, h]}."""
    seen = {}
    for cam in CAMERAS:
        first = next((c for c in sorted(clips, key=lambda c: c.start) if c.camera == cam), None)
        if first is None or cam not in expected:
            continue
        wh = probe(first.path)
        if wh is None:
            raise CameraError(f"could not read the resolution of {first.path} (is ffprobe on PATH?)")
        if list(wh) != list(expected[cam]):
            raise CameraError(f"{first.name} was placed in {cam} ({first.rule} rule) but is {wh[0]}x{wh[1]}, "
                              f"{cam} footage is {expected[cam][0]}x{expected[cam][1]}. The camera mapping is wrong "
                              f"for this footage: use Cam 0N folders or fix cameras.suffix (or --no-camera-check).")
        seen[cam] = list(wh)
    return seen


def link_inputs(clips: list[ClipInfo], dest: str | Path) -> dict:
    """dest/<cam>/<original name>: hard links (no extra disk, same volume), else a copy. Existing correct links are kept;
    files from an earlier assignment that no longer applies are removed, so every camera folder holds exactly its clips."""
    dest = Path(dest)
    stats = {"linked": 0, "copied": 0, "kept": 0, "removed": 0}
    want = {cam: {c.name for c in clips if c.camera == cam} for cam in CAMERAS}
    for cam in CAMERAS:
        d = dest / cam
        if d.exists():
            for f in d.iterdir():
                if f.name not in want[cam]:
                    f.unlink()
                    stats["removed"] += 1
    for c in clips:
        d = dest / c.camera
        d.mkdir(parents=True, exist_ok=True)
        out = d / c.name
        if out.exists():
            if os.path.samefile(out, c.path) or out.stat().st_size == os.path.getsize(c.path):
                stats["kept"] += 1
                continue
            out.unlink()
        try:
            os.link(c.path, out)
            stats["linked"] += 1
        except OSError:
            shutil.copy2(c.path, out)
            stats["copied"] += 1
    return stats


def summary(clips: list[ClipInfo]) -> dict:
    out = {}
    for cam in CAMERAS:
        cs = sorted((c for c in clips if c.camera == cam), key=lambda c: c.start)
        if cs:
            out[cam] = {"clips": len(cs), "first": cs[0].start, "last": cs[-1].start,
                        "rules": sorted({c.rule for c in cs})}
    return out


def as_dicts(clips: list[ClipInfo]) -> list[dict]:
    return [asdict(c) for c in clips]
