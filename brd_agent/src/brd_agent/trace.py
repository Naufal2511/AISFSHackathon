"""Reasoning traces: one JSON + one readable Markdown per decision. Pick the three submission traces
(success / failure / night) from these via scripts/ or `select_traces`."""
from __future__ import annotations

import json
import shutil
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .schemas import Brief, Decision


@dataclass
class Trace:
    trace_id: str
    brief: dict
    events: list[dict] = field(default_factory=list)
    t0: float = field(default_factory=time.time)
    decision: Optional[dict] = None

    def add(self, type_: str, **payload: Any):
        self.events.append({"t": round(time.time() - self.t0, 3), "type": type_, **payload})


class TraceLogger:
    def __init__(self, out_dir: Optional[str | Path], copy_images: bool = True):
        self.dir = Path(out_dir) if out_dir else None
        self.copy_images = copy_images
        if self.dir:
            (self.dir / "img").mkdir(parents=True, exist_ok=True)

    def start(self, b: Brief) -> Trace:
        return Trace(trace_id=f"{b.cow_id}_d{b.day}_m{b.minute:04d}", brief=b.model_dump(mode="json"))

    def _stash(self, path: str) -> str:
        if not (self.dir and self.copy_images):
            return path
        dst = self.dir / "img" / f"{zlib.crc32(path.encode()):08x}_{Path(path).name}"
        if not dst.exists():
            shutil.copy(path, dst)
        return f"img/{dst.name}"

    def finish(self, tr: Trace, d: Decision):
        tr.decision = d.model_dump(mode="json")
        if not self.dir:
            return
        for e in tr.events:
            if "images" in e:
                e["images"] = [self._stash(p) for p in e["images"]]
        (self.dir / f"{tr.trace_id}.json").write_text(json.dumps(
            {"trace_id": tr.trace_id, "brief": tr.brief, "events": tr.events, "decision": tr.decision}, indent=1, default=str), encoding="utf-8")
        (self.dir / f"{tr.trace_id}.md").write_text(render_md(tr), encoding="utf-8")


def render_md(tr: Trace) -> str:
    b, d = tr.brief, tr.decision or {}
    L = [f"# Trace {tr.trace_id}",
         f"**Cow** {b['cow_id']} · **day** {b['day']} · **minute** {b['minute']} · night={b['is_night']}",
         f"**Tier-1:** {b['tier1_top'][:3]}  · id_conf={b['id_conf']} · visibility={b['visible_frac']}", ""]
    for i, e in enumerate(tr.events, 1):
        t = e["type"]
        if t == "gate":
            L.append(f"{i}. **Gate** → {e['action']} ({', '.join(e['reasons']) or 'no reasons'})")
        elif t == "plan":
            L.append(f"{i}. **Plan** ({e.get('source')}): {e.get('rationale', '')}  steps={[s['tool'] for s in e['steps']]}")
        elif t == "tool":
            L.append(f"{i}. **Tool** `{e['tool']}` {e['args']} _(why: {e.get('reason') or 'n/a'}; by {e.get('source') or 'planner'})_ → ok={e['ok']} facts={json.dumps(e['facts'], default=str)[:300]}"
                     + (f" error={e['error']}" if e.get("error") else ""))
            for p, c in zip(e.get("images", []), e.get("captions", [])):
                L.append(f"   ![{c}]({p})")
        elif t == "hypothesis":
            L.append(f"{i}. **Hypothesis** ({e['stage']}): `{e['label']}` conf={e['confidence']} cues={e['cues']} alts={e['alternatives']}")
        elif t == "verifier":
            bad = [f"{c['name']}[{c['severity']}]: {c['detail']}" for c in e['checks'] if not c['passed']]
            L.append(f"{i}. **Verifier** ({e['stage']}) → hard={e['n_hard']} soft={e['n_soft']}" + (f" · failures: {bad}" if bad else " · all passed"))
        elif t == "critique":
            L.append(f"{i}. **Critique** → {e['verdict']} (alt={e.get('best_alternative')}) {e.get('reason', '')}")
        elif t == "vlm":
            L.append(f"{i}. _vlm call_ `{e['task']}` tokens={e['prompt_tokens']}+{e['completion_tokens']} {e['latency_s']}s cached={e['cached']} repaired={e['repaired']}")
        else:
            L.append(f"{i}. **{t}** {json.dumps({k: v for k, v in e.items() if k not in ('t', 'type')}, default=str)[:300]}")
    if d.get("actions"):
        L += ["", "## Every action and the reason for it", "", "| # | actor | action | reason | outcome |", "|---|---|---|---|---|"]
        for a in d["actions"]:
            L.append(f"| {a['seq']} | {a['actor']} | {a['action']} | {a['reason']} | {str(a['outcome'])[:160]} |")
    L += ["", f"## Decision: `{d.get('behavior')}` conf={d.get('confidence')} · reason={d.get('reason')} · source={d.get('source')} · flags={d.get('flags')}", f"**Correction:** {d.get('correction')} · **Why:** {d.get('rationale')}"]
    return "\n".join(L)
