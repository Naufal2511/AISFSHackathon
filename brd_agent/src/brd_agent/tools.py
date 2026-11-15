"""Tool registry. Each tool is bound to ONE cow and ONE minute (the agent cannot wander to other cows).
The agent chooses only camera / region / time offset / window. Args are validated and clamped here,
never trusted from the VLM. Tools return OBSERVED facts + images; they never return a behavior label."""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image, ImageOps, ImageStat

from .backend import DataBackend
from . import night
from .config import AgentConfig
from .schemas import CAMERAS, Brief, ToolCall, ToolResult


class ToolError(Exception):
    pass


@dataclass
class ToolContext:
    backend: DataBackend
    brief: Brief
    cfg: AgentConfig
    workdir: Path
    history: Optional[Any] = None          # HistoryStore (file store) or None
    _vis: Optional[dict] = field(default=None, repr=False)

    def visibility(self) -> dict[str, float]:
        if self._vis is None:
            self._vis = self.backend.visibility(self.brief.cow_id, self.brief.day, self.brief.minute)
        return self._vis

    def best_camera(self) -> str:
        vis = self.visibility()
        return max(vis, key=vis.get) if vis else "top"

    def resolve_camera(self, cam: str) -> str:
        return self.best_camera() if cam == "auto" else cam


@dataclass
class Tool:
    name: str
    description: str
    args: dict[str, dict]
    fn: Callable[..., dict]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description,
                "args": {k: {kk: vv for kk, vv in v.items() if kk != "default"} | ({"default": v["default"]} if "default" in v else {})
                         for k, v in self.args.items()}}


TOOLS: dict[str, Tool] = {}
CAM = {"type": "enum", "values": [*CAMERAS, "auto"], "default": "auto"}
OFFSET = {"type": "float", "min": -30.0, "max": 30.0, "default": 0.0}


def register(name: str, description: str, args: Optional[dict] = None):
    def deco(fn):
        TOOLS[name] = Tool(name, description, args or {}, fn)
        return fn
    return deco


def validate_args(spec: dict[str, dict], args: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, s in spec.items():
        v = args.get(k, s.get("default"))
        if v is None:
            continue
        t = s["type"]
        if t == "enum":
            if v not in s["values"]:
                raise ToolError(f"arg '{k}'={v!r} not in {s['values']}")
        elif t in ("int", "float"):
            try:
                v = int(v) if t == "int" else float(v)
            except (TypeError, ValueError):
                raise ToolError(f"arg '{k}'={v!r} is not a number")
            v = max(s["min"], min(s["max"], v))
        out[k] = v
    return out


def tool_specs() -> list[dict]:
    return [t.spec() for t in TOOLS.values()]


def execute(ctx: ToolContext, call: ToolCall) -> ToolResult:
    tool = TOOLS.get(call.tool)
    if tool is None:
        return ToolResult(tool=call.tool, args=call.args, ok=False, error=f"unknown tool '{call.tool}'", reason=call.reason, source=call.source)
    try:
        args = validate_args(tool.args, call.args)
        out = tool.fn(ctx, **args)
        return ToolResult(tool=call.tool, args=args, ok=True, facts=out.get("facts", {}),
                          images=out.get("images", []), captions=out.get("captions", []),
                          reason=call.reason, source=call.source)
    except ToolError as e:
        return ToolResult(tool=call.tool, args=call.args, ok=False, error=str(e), reason=call.reason, source=call.source)
    except Exception as e:  # backend failure must never crash the agent loop
        return ToolResult(tool=call.tool, args=call.args, ok=False, error=f"{type(e).__name__}: {e}",
                          reason=call.reason, source=call.source)


def _b(ctx):  # shorthand
    return ctx.backend, ctx.brief.cow_id, ctx.brief.day, ctx.brief.minute


# ----------------------------------------------------------------------------- tools
@register("best_camera", "Report per-camera visible fraction of this cow and which camera sees her best. Cheap; use first when views are uneven.")
def best_camera(ctx: ToolContext):
    vis = ctx.visibility()
    return {"facts": {"visible_frac": vis, "best_camera": ctx.best_camera()}}


@register("get_frames", "Cow-centred crop (whole body) from one camera at a time offset within the minute.",
          {"camera": CAM, "t_offset_s": OFFSET})
def get_frames(ctx, camera="auto", t_offset_s=0.0):
    cam = ctx.resolve_camera(camera)
    be, c, d, m = _b(ctx)
    p = be.crop(c, d, m, cam, t_offset_s, "body", 1.0)
    return {"facts": {"camera": cam, "t_offset_s": t_offset_s, "available": p is not None},
            "images": [p] if p else [], "captions": [f"{cam} body t={t_offset_s:+.0f}s"] if p else []}


@register("zoom", "High-resolution crop of a body region (head | body | legs). Use head for muzzle/jaw/ear cues, legs for posture.",
          {"camera": CAM, "region": {"type": "enum", "values": ["head", "body", "legs"], "default": "head"},
           "t_offset_s": OFFSET, "scale": {"type": "float", "min": 1.0, "max": 4.0, "default": 2.0}})
def zoom(ctx, camera="auto", region="head", t_offset_s=0.0, scale=2.0):
    cam = ctx.resolve_camera(camera)
    be, c, d, m = _b(ctx)
    p = be.crop(c, d, m, cam, t_offset_s, region, scale)
    return {"facts": {"camera": cam, "region": region, "available": p is not None},
            "images": [p] if p else [], "captions": [f"{cam} {region} zoom x{scale:g} t={t_offset_s:+.0f}s"] if p else []}


@register("temporal_window", "Filmstrip of the cow over time (before/after the minute centre) to judge motion, transitions and duration.",
          {"camera": CAM, "before_s": {"type": "float", "min": 0, "max": 30, "default": 10},
           "after_s": {"type": "float", "min": 0, "max": 30, "default": 10},
           "n_frames": {"type": "int", "min": 2, "max": 6, "default": 4}})
def temporal_window(ctx, camera="auto", before_s=10.0, after_s=10.0, n_frames=4):
    cam = ctx.resolve_camera(camera)
    be, c, d, m = _b(ctx)
    ps = be.filmstrip(c, d, m, cam, before_s, after_s, n_frames)
    return {"facts": {"camera": cam, "n_frames": len(ps), "span_s": [-before_s, after_s]},
            "images": ps, "captions": [f"{cam} filmstrip frame {i + 1}/{len(ps)}" for i in range(len(ps))]}


@register("get_burst", "Short high-frame-rate burst motion summary of the head region (jaw/head/body motion scores).",
          {"camera": CAM, "seconds": {"type": "float", "min": 4, "max": 10, "default": 8}})
def get_burst(ctx, camera="auto", seconds=8.0):
    cam = ctx.resolve_camera(camera)
    be, c, d, m = _b(ctx)
    r = be.burst_motion(c, d, m, cam, seconds)
    return {"facts": {"camera": cam, "seconds": seconds, **r}}


@register("zone_query", "Where the cow is relative to the feed bunk / waterer / resting area (zone fractions, head-in-zone fractions, distances).")
def zone_query(ctx):
    be, c, d, m = _b(ctx)
    return {"facts": be.zone_info(c, d, m)}


@register("track_history", "Motion and posture summary over a window around the minute: mean/max speed, displacement, posture, plus previous final labels.",
          {"window_s": {"type": "float", "min": 30, "max": 300, "default": 90}})
def track_history(ctx, window_s=90.0):
    be, c, d, m = _b(ctx)
    return {"facts": {**be.track_history(c, d, m, window_s),
                      "prev_label": ctx.brief.prev_label, "next_label_tier1": ctx.brief.next_label_tier1}}


@register("neighbors", "Other cows near this one: nearest-neighbour distance, count within radius, contact seconds.")
def neighbors(ctx):
    be, c, d, m = _b(ctx)
    return {"facts": be.neighbors(c, d, m)}


@register("rumination_probe", "Quantitative test for cud chewing: jaw-motion score and periodicity from a high-fps burst. Stills cannot show rumination; use this.",
          {"camera": CAM, "seconds": {"type": "float", "min": 4, "max": 10, "default": 8}})
def rumination_probe(ctx, camera="auto", seconds=8.0):
    cam = ctx.resolve_camera(camera)
    be, c, d, m = _b(ctx)
    r = be.burst_motion(c, d, m, cam, seconds)
    lo, hi = ctx.cfg.verifier.rumination_hz
    jaw, hz = float(r.get("jaw_motion_score", 0.0)), float(r.get("periodicity_hz", 0.0))
    likely = jaw >= ctx.cfg.verifier.rumination_min_jaw and lo <= hz <= hi
    return {"facts": {"camera": cam, "jaw_motion_score": jaw, "periodicity_hz": hz, "rumination_likely": likely,
                      "note": "heuristic; thresholds are assumptions"}}


@register("enhance_night", "Contrast-enhanced (CLAHE + gamma) scene crop for dark or infrared frames. Reports brightness before/after and whether the frame looks infrared.",
          {"camera": CAM, "t_offset_s": OFFSET})
def enhance_night(ctx, camera="auto", t_offset_s=0.0):
    cam = ctx.resolve_camera(camera)
    be, c, d, m = _b(ctx)
    p = be.crop(c, d, m, cam, t_offset_s, "scene", 1.0)
    if p is None:
        return {"facts": {"camera": cam, "available": False}}
    im = Image.open(p).convert("RGB")
    st = night.frame_stats(im, ctx.cfg.night)
    out = night.enhance(im, ctx.cfg.night)
    after = ImageStat.Stat(out.convert("L")).mean[0]
    outp = ctx.workdir / f"enh_{cam}_{zlib.crc32(f'{c}{d}{m}{t_offset_s}'.encode()):08x}.png"
    out.save(outp)
    return {"facts": {"camera": cam, "brightness_before": st["brightness"], "brightness_after": round(after, 1),
                      "ir_like": st["ir_like"], "looks_like_night": st["is_night"]},
            "images": [str(outp)], "captions": [f"{cam} enhanced scene t={t_offset_s:+.0f}s"]}


@register("compare_to_exemplar", "Identity check: similarity between this cow's current crop and her gallery exemplars.",
          {"camera": CAM})
def compare_to_exemplar(ctx, camera="auto"):
    cam = ctx.resolve_camera(camera)
    be, c, d, m = _b(ctx)
    p = be.crop(c, d, m, cam, 0.0, "body", 1.0)
    if p is None:
        return {"facts": {"camera": cam, "similarity": None, "match": None}}
    raw = be.exemplar_similarity(c, p)
    if raw is None:                       # no gallery: say so instead of pretending the identity was checked
        return {"facts": {"camera": cam, "available": False, "note": "no exemplar gallery for identity check"}}
    sim = float(raw)
    facts = {"camera": cam, "similarity": round(sim, 3), "match": sim >= ctx.cfg.verifier.min_exemplar_sim}
    scores = be.exemplar_scores(p)
    if scores:
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        facts["ranking"] = [(k, round(v, 3)) for k, v in ranked[:3]]
        others = [(k, v) for k, v in ranked if k != c]
        if others and others[0][1] >= sim + 0.10:          # margin: ASSUMPTION
            facts["better_match"] = others[0][0]
    return {"facts": facts}


@register("cow_history", "Context from THIS cow's own recent labelled minutes in the database: run-length history, current bout length, minutes since feeding/drinking/lying/walking/rumination, and (offline only) next-minute tier-1 guesses. Context only, never proof: the current minute's evidence decides.",
          {"lookback_min": {"type": "int", "min": 5, "max": 120, "default": 30}})
def cow_history(ctx, lookback_min=30):
    if ctx.history is None:
        return {"facts": {"available": False}}
    b = ctx.brief
    return {"facts": ctx.history.summary(b.cow_id, b.day, b.minute, lookback_min)}
