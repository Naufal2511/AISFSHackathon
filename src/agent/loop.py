"""The agent loop: for every flag  plan -> call tools -> decide -> VERIFY -> act -> trace.

Self-check (deterministic, independent of the model):
  * the decision is well formed (known cow IDs, a valid behavior class, times in range);
  * a correction is backed by >= 2 different kinds of evidence the agent actually gathered
    (visual, appearance_score, trajectory, classifier, chew_motion);
  * relabel: the appearance ranking does not contradict it (new ID in the top 2 when it was
    scored) and it does not create a duplicate ID in the same camera at the same time;
  * set_behavior: rumination needs measured chewing; feeding / drinking need a visual check.
If the check fails, the reasons go back to the model for one revision; if it still fails the
decision is downgraded to "uncertain" (never a silent guess).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from ..behavior.taxonomy import BEHAVIORS
from .audit import audit_text, build_sheet
from .memory import Memory
from .tools import Toolbox
from .triggers import Flag, audit_schedule, find_flags

CORRECTING = ("relabel", "set_behavior")
DECISIONS = ("confirm", "relabel", "uncertain", "set_behavior", "note", "escalate")


def playbook(f: Flag, hms) -> str:
    """Suggested plan per flag type, with the arguments filled in. The model may deviate,
    but small local VLMs follow a concrete plan far more reliably than an open toolbox."""
    cid = (f.cow_ids or ["COW_01"])[0]
    t0, t1, mid = hms(f.t0), hms(f.t1), hms((f.t0 + f.t1) / 2)
    cam, tid = f.camera, f.track_id
    if f.kind in ("id_jump", "weak_id", "new_cow_id") and tid is not None:
        return (f"1) all_views {{\"time\": \"{t1}\", \"cow_id\": \"{cid}\"}}  2) compare_to_cards "
                f"{{\"camera\": \"{cam}\", \"track_id\": {tid}, \"time\": \"{t1}\"}}  3) trajectory "
                f"{{\"cow_id\": \"{cid}\", \"t0\": \"{hms(f.t0 - 60)}\", \"t1\": \"{hms(f.t1 + 10)}\"}}. "
                f"Then: confirm if track {tid} really is {cid}; relabel (camera {cam}, track_id {tid}) only if the "
                "appearance score AND the views/trajectory point to the same other cow; otherwise uncertain.")
    if f.kind == "duplicate_id":
        return (f"1) scene {{\"camera\": \"{cam}\", \"time\": \"{t0}\"}}  2) compare_to_cards for each track listed "
                "in the flag. Then relabel the wrong track or mark it uncertain.")
    if f.kind in ("extra_cow", "cow_missing"):
        return (f"1) all_views {{\"time\": \"{t1}\"}}  2) who_is_near for the camera that matters  3) trajectory "
                f"{{\"cow_id\": \"{cid}\", \"t0\": \"{hms(f.t0 - 120)}\", \"t1\": \"{t1}\"}}. Then note / escalate / confirm.")
    if f.kind == "behavior_check":
        beh = (f.data or {}).get("behavior", "")
        chew = (f"  3) chew_motion {{\"cow_id\": \"{cid}\", \"time\": \"{mid}\"}} (chewing at the feed bunk is "
                "EATING, not rumination)") if "rumin" in beh else ""
        return (f"1) behavior {{\"cow_id\": \"{cid}\", \"t0\": \"{t0}\", \"t1\": \"{t1}\"}}  2) cow_strip "
                f"{{\"cow_id\": \"{cid}\", \"time\": \"{mid}\", \"before\": 6, \"after\": 6}}{chew}. "
                f"Then confirm, or set_behavior (cow_id {cid}, t0 {t0}, t1 {t1}) if the evidence shows another class.")
    if f.kind == "id_audit":
        return ("Look at the attached audit sheet: rows whose labels are red or whose sim-to-own is low are suspects. "
                "Use compare_to_cards on at most 2 suspects, then give a final confirm with a 'mismatches' list "
                "(empty if all rows look right).")
    return "Gather two kinds of evidence, then decide."


class Agent:
    def __init__(self, toolbox: Toolbox, memory: Memory, backend, out_dir: str = "outputs/agent",
                 max_steps: int = 6, cfg: Optional[Dict] = None):
        self.tb, self.memory, self.backend = toolbox, memory, backend
        self.out_dir = out_dir
        self.max_steps = max_steps
        self.cfg = cfg or {}
        self.corrections: List[Dict] = []
        self.traces: List[Dict] = []
        os.makedirs(os.path.join(out_dir, "traces"), exist_ok=True)

    # ------------------------------------------------------------------ one flag
    def investigate(self, flag: Flag, intro_images: Optional[List[np.ndarray]] = None, intro_text: str = "") -> Dict:
        hms = self.tb.hms
        fd = flag.to_dict()
        intro = (f"FLAG {flag.kind} (severity {flag.severity}) {hms(flag.t0)}-{hms(flag.t1)}"
                 f"{' camera ' + flag.camera if flag.camera else ''}"
                 f"{' track ' + str(flag.track_id) if flag.track_id is not None else ''}: {flag.detail}\n"
                 f"Memory cards of the cows involved:\n"
                 + "\n".join(self.memory.describe(c) for c in flag.cow_ids[:10])
                 + (f"\n{intro_text}" if intro_text else "")
                 + f"\nKnown cows: {', '.join(self.memory.known_ids())}\n"
                 f"Tools (max {self.max_steps} calls):\n{self.tb.describe()}\n"
                 f"SUGGESTED PLAN: {playbook(flag, hms)}\n"
                 "Decisions are NOT tools: to decide, reply with a \"final\" object.")
        state = {"flag": fd, "steps": [], "hms": hms,
                 "turns": [{"role": "user", "text": intro, "images": intro_images or []}]}
        trace = {"flag": fd, "backend": getattr(self.backend, "label", self.backend.name), "steps": [],
                 "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        t_start = time.time()
        final, revisions = None, 0
        for step in range(self.max_steps + 2):
            try:
                act = self.backend.next_action(state)
            except Exception as e:
                final = {"decision": "escalate", "reason": f"backend error: {e}", "confidence": 0.0,
                         "evidence": [], "summary": "backend unavailable"}
                break
            thought = act.get("thought", "")
            state["turns"].append({"role": "assistant", "text": json.dumps({k: v for k, v in act.items() if k != "_raw"})})
            if "final" in act or len(state["steps"]) >= self.max_steps:
                final = act.get("final") or {"decision": "uncertain", "summary": "tool budget used up",
                                             "evidence": [], "confidence": 0.3}
                ok, reasons = self.verify(final, state)
                trace["steps"].append({"thought": thought, "final": final, "verified": ok, "check": reasons})
                if ok or revisions >= 1:
                    if not ok:
                        final = self._downgrade(final, flag, reasons)
                    break
                revisions += 1
                state["turns"].append({"role": "user", "text": "SELF-CHECK FAILED: " + "; ".join(reasons)
                                       + ". Gather the missing evidence with a tool, or change the decision."})
                continue
            name, args = act.get("tool"), act.get("args") or {}
            if name == "final" and isinstance(args, dict) and args.get("decision") in DECISIONS:
                name, args = args["decision"], {k: v for k, v in args.items() if k != "decision"}
            if name in DECISIONS:  # small models sometimes "call" the decision
                act["final"] = {"decision": name, **args, "evidence": args.get("evidence", []),
                                "summary": args.get("summary", thought[:200])}
                state["turns"].pop()
                state["turns"].append({"role": "assistant", "text": json.dumps({"thought": thought, "final": act["final"]})})
                final = act["final"]
                ok, reasons = self.verify(final, state)
                trace["steps"].append({"thought": thought, "final": final, "verified": ok, "check": reasons})
                if ok or revisions >= 1:
                    if not ok:
                        final = self._downgrade(final, flag, reasons)
                    break
                revisions += 1
                state["turns"].append({"role": "user", "text": "SELF-CHECK FAILED: " + "; ".join(reasons)
                                       + ". Gather the missing evidence with a tool, or change the decision."})
                continue
            if any(s["tool"] == name and s["args"] == args for s in state["steps"]):
                state["turns"].append({"role": "user", "text": f"You already called {name} with these arguments "
                                       "(result above). Call a DIFFERENT tool or give your final decision."})
                state["steps"].append({"tool": name, "args": args, "kind": "none", "text": "repeat", "data": {}})
                continue
            res = self.tb.call(name, args)
            kind = self.tb.tools[name].evidence if name in self.tb.tools else "none"
            state["steps"].append({"tool": name, "args": args, "kind": kind, "text": res.text, "data": res.data})
            img_paths = self._save_images(res.images, flag, len(trace["steps"]))
            trace["steps"].append({"thought": thought, "tool": name, "args": args, "evidence_kind": kind,
                                   "result": res.text, "images": img_paths})
            left = self.max_steps - len(state["steps"])
            nudge = ("\nYou have used all tool calls: reply now with your FINAL decision." if left <= 0 else
                     f"\n({left} tool call{'s' if left > 1 else ''} left.)")
            state["turns"].append({"role": "user", "text": f"Result of {name}:\n{res.text}{nudge}", "images": res.images})
        if final is None:
            final = self._downgrade({"decision": "none", "evidence": []}, flag, ["no decision within the step budget"])
        trace["final"] = final
        trace["seconds"] = round(time.time() - t_start, 1)
        self.apply(final, flag)
        self._write_trace(trace)
        return trace

    # ------------------------------------------------------------------ self-check
    def verify(self, final: Dict, state: Dict):
        reasons = []
        d = final.get("decision")
        kinds = {s["kind"] for s in state["steps"] if s["kind"] not in ("none", "memory")}
        known = set(self.memory.known_ids())
        if d not in DECISIONS:
            return False, [f"decision must be one of {DECISIONS}"]
        if d == "confirm" and not state["steps"]:
            reasons.append("confirm without looking at any evidence")
        if d in CORRECTING and len(kinds) < 2:
            reasons.append(f"a correction needs 2 kinds of evidence, you used {sorted(kinds) or 'none'}")
        if d == "relabel":
            for k in ("camera", "track_id", "new_cow_id", "t0", "t1"):
                if final.get(k) in (None, ""):
                    reasons.append(f"relabel needs '{k}'")
            new = final.get("new_cow_id")
            if new and known and new not in known:
                reasons.append(f"{new} is not a known cow ({', '.join(sorted(known))})")
            for s in state["steps"]:
                if s["tool"] == "compare_to_cards" and s["data"].get("ranking") and \
                        str(s["args"].get("track_id")) == str(final.get("track_id")):
                    top2 = [c for c, _ in s["data"]["ranking"][:2]]
                    if new not in top2:
                        reasons.append(f"appearance ranking puts {new} outside the top 2 ({top2})")
            if not reasons:
                try:
                    a, b = self.tb.t(final["t0"]), self.tb.t(final["t1"])
                    r = self.tb.rec
                    clash = r[(r.camera_id == final["camera"]) & (r.global_cow_id == new) & r.t.between(a, b)
                              & (r.local_track_id != int(final["track_id"]))]
                    if len(clash):
                        reasons.append(f"{new} is already on track {int(clash.local_track_id.iloc[0])} in "
                                       f"{final['camera']} at the same time: would create a duplicate ID")
                except Exception as e:
                    reasons.append(f"bad times: {e}")
        if d == "set_behavior":
            beh = final.get("behavior")
            if beh not in BEHAVIORS:
                reasons.append(f"behavior must be one of {BEHAVIORS}")
            if beh and str(beh).startswith("ruminating"):
                chew = [s for s in state["steps"] if s["tool"] == "chew_motion"]
                if not chew or chew[-1]["data"].get("peak_ratio", 0) < 3.0:
                    reasons.append("rumination needs measured chewing (chew_motion peak_ratio >= 3)")
            if beh in ("feeding", "drinking") and "visual" not in kinds:
                reasons.append(f"{beh} needs a visual check of the head at the bunk / waterer")
        return (not reasons), reasons

    def _downgrade(self, final: Dict, flag: Flag, reasons: List[str]) -> Dict:
        return {"decision": "uncertain", "cow_id": (flag.cow_ids or [None])[0], "camera": flag.camera,
                "track_id": flag.track_id, "t0": self.tb.hms(flag.t0), "t1": self.tb.hms(flag.t1),
                "confidence": 0.3, "evidence": final.get("evidence", []),
                "summary": f"proposed {final.get('decision')} failed self-check ({'; '.join(reasons)}); marked uncertain",
                "proposed": final}

    # ------------------------------------------------------------------ actions
    def apply(self, final: Dict, flag: Flag):
        d = final.get("decision")
        when = self.tb.hms(flag.t0)
        row = {"flag": flag.kind, "flag_time": str(pd.Timestamp(flag.t0, unit="s")), "decision": d,
               "confidence": final.get("confidence"), "summary": final.get("summary", "")}
        for k in ("camera", "track_id", "new_cow_id", "cow_id", "behavior", "t0", "t1", "text", "reason"):
            if final.get(k) is not None:
                row[k] = final[k]
        for k in ("t0", "t1"):
            if k in row:
                try:
                    row[k + "_epoch"] = self.tb.t(row[k])
                except Exception:
                    pass
        if d in ("relabel", "uncertain", "set_behavior", "escalate"):
            self.corrections.append(row)
        if d == "note" and final.get("cow_id"):
            self.memory.add_note(final["cow_id"], when, final.get("text", ""))
        if d in ("relabel", "set_behavior", "uncertain", "escalate"):
            for cid in {final.get("cow_id"), final.get("new_cow_id")} - {None}:
                if cid in self.memory.cards:
                    self.memory.add_note(cid, when, f"[{d}] {final.get('summary', '')}")
        for m in final.get("mismatches", []) or []:
            self.memory.add_note(m, when, "30-min audit: identity looked inconsistent")

    # ------------------------------------------------------------------ traces
    def _save_images(self, images, flag: Flag, step: int) -> List[str]:
        out = []
        d = os.path.join(self.out_dir, "traces", "img")
        os.makedirs(d, exist_ok=True)
        for i, im in enumerate(images):
            p = os.path.join(d, f"{flag.kind}_{int(flag.t0)}_{flag.camera}_{flag.track_id}_s{step}_{i}.jpg")
            s = 1280 / max(im.shape[:2])
            cv2.imwrite(p, cv2.resize(im, None, fx=s, fy=s) if s < 1 else im)
            out.append(p)
        return out

    def _write_trace(self, trace: Dict):
        self.traces.append(trace)
        f = trace["flag"]
        name = f"{pd.Timestamp(f['t0'], unit='s'):%Y%m%d_%H%M%S}_{f['kind']}_{'_'.join(f['cow_ids'][:2])}"
        base = os.path.join(self.out_dir, "traces", name)
        with open(base + ".json", "w") as fh:
            json.dump(trace, fh, indent=1, default=str)
        with open(base + ".md", "w") as fh:
            fh.write(render_trace(trace, self.tb.hms))

    # ------------------------------------------------------------------ whole run
    def run(self, window_sec: int = 300, audit_every_min: int = 30, trigger_cfg: Optional[Dict] = None,
            embedder=None, max_flags: Optional[int] = None, log=print) -> List[Dict]:
        rec, tl = self.tb.rec, self.tb.timeline
        t_lo, t_hi = float(rec.t.min()), float(rec.t.max()) + 1
        audits = audit_schedule(t_lo, t_hi, audit_every_min)
        if not audits:  # short runs still get one ID check (mid-span)
            audits = [t_lo + (t_hi - t_lo) / 2]
        last_seen: Dict[str, float] = {}
        handled, n = set(), 0
        for w0 in np.arange(t_lo, t_hi, window_sec):
            w1 = min(w0 + window_sec, t_hi)
            flags = find_flags(rec, w0, w1, self.memory.known_ids(), trigger_cfg, tl, last_seen, audits)
            flags = [f for f in flags if f.key not in handled]
            log(f"[AGENT] window {self.tb.hms(w0)}-{self.tb.hms(w1)}: {len(flags)} flags "
                + ", ".join(f"{f.kind}({','.join(f.cow_ids[:2])})" for f in flags))
            for f in flags:
                if max_flags is not None and n >= max_flags:
                    break
                handled.add(f.key)
                imgs, txt = [], ""
                if f.kind == "id_audit":
                    path, rows, sheet = build_sheet(f.t0, rec, self.tb.frames, self.memory, embedder,
                                                    os.path.join(self.out_dir, "id_checks"))
                    imgs, txt = [sheet], ("Audit sheet: one row per cow = card reference | cam1 | cam2 | cam3 "
                                          "box carrying that ID now (red label = closer to another card).\n"
                                          + audit_text(rows))
                    log(f"[AGENT]   ID-check sheet -> {path}")
                tr = self.investigate(f, imgs, txt)
                n += 1
                fin = tr["final"] or {}
                log(f"[AGENT]   {f.kind}: {fin.get('decision')} ({tr['seconds']} s, {len(tr['steps'])} steps) "
                    f"- {fin.get('summary', '')}")
            self.memory.update_last_seen(rec[(rec.t >= w0) & (rec.t < w1)])
        self.memory.save()
        return self.traces


def render_trace(tr: Dict, hms) -> str:
    f = tr["flag"]
    lines = [f"# {f['kind']} - {', '.join(f['cow_ids'][:4])} at {hms(f['t0'])}",
             f"*Backend:* {tr['backend']}  |  *time:* {tr.get('seconds')} s  |  *trigger:* {f['detail']}", ""]
    for i, s in enumerate(tr["steps"], 1):
        if "tool" in s:
            lines += [f"**{i}. Thought:** {s['thought']}",
                      f"**Tool:** `{s['tool']}` {json.dumps(s['args'])} *(evidence: {s['evidence_kind']})*",
                      "```", s["result"][:1200], "```"]
            lines += [f"![]({os.path.relpath(p, os.path.dirname(os.path.dirname(p)))})" for p in s.get("images", [])]
        else:
            lines += [f"**{i}. Decision:** {s['thought']}", "```json", json.dumps(s["final"], indent=1), "```",
                      f"**Self-check:** {'passed' if s['verified'] else 'FAILED: ' + '; '.join(s['check'])}"]
    fin = tr.get("final") or {}
    lines += ["", f"**Outcome:** {fin.get('decision')} - {fin.get('summary', '')}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- corrections
def apply_id_corrections(rec: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
    """relabel -> new ID on that track/time; uncertain -> ID removed (counted as not visible
    rather than credited to a possibly wrong cow)."""
    out = rec.copy()
    out["agent_action"] = ""
    if corr is None or corr.empty:
        return out
    for c in corr.itertuples():
        t0 = getattr(c, "t0_epoch", np.nan)
        t1 = getattr(c, "t1_epoch", np.nan)
        if pd.isna(t0) or pd.isna(t1):
            continue
        if c.decision == "relabel" and pd.notna(getattr(c, "track_id", np.nan)):
            m = (out.camera_id == c.camera) & (out.local_track_id == int(c.track_id)) & out.t.between(t0, t1)
            out.loc[m, "global_cow_id"] = c.new_cow_id
            out.loc[m, "agent_action"] = "relabel"
        elif c.decision == "uncertain":
            if pd.notna(getattr(c, "track_id", np.nan)) and isinstance(getattr(c, "camera", None), str):
                m = (out.camera_id == c.camera) & (out.local_track_id == int(c.track_id)) & out.t.between(t0, t1)
            elif isinstance(getattr(c, "cow_id", None), str):
                m = (out.global_cow_id == c.cow_id) & out.t.between(t0, t1)
            else:
                continue
            out.loc[m, "global_cow_id"] = np.nan
            out.loc[m, "agent_action"] = "uncertain"
    return out


def apply_behavior_overrides(tl: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
    """tl: classifier timeline with global_cow_id and sec (epoch seconds)."""
    out = tl.copy()
    if corr is None or corr.empty:
        return out
    for c in corr[corr.decision == "set_behavior"].itertuples():
        m = (out.global_cow_id == c.cow_id) & out.sec.between(c.t0_epoch, c.t1_epoch)
        out.loc[m, "behavior"] = c.behavior
        out.loc[m, "evidence"] = "agent_verified"
        out.loc[m, "confidence"] = c.confidence if pd.notna(c.confidence) else 0.7
    return out
