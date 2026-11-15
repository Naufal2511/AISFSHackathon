"""BRD risk agent: for each cow, a heavy reasoning model (TAMU, Claude Opus 4.8 by default) audits the
behavior indicators and looks for visible clinical signs with inspection tools, then reports
structured findings that cite evidence ids. Code (not the model) turns findings into the score,
after a deterministic self-check that downgrades any claim the evidence cannot support."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..agent.evidence import to_jpeg_b64
from ..agent.vlm import OpenAICompatVLM, load_dotenv, parse_json
from .data import hms
from .score import fmt
from .store import EvidenceStore
from .tools import RiskTools

SIGNS = ["head_low", "ears_droop", "nasal_discharge", "ocular_discharge", "labored_breathing", "cough", "gaunt"]
DISCHARGE = {"nasal_discharge", "ocular_discharge"}

SYSTEM = """You are a feedlot veterinarian and cattle behaviourist auditing a camera-based early-warning
system for bovine respiratory disease (BRD) in one pen of up to 10 Angus cattle (Texas A&M Pen 6).
Cameras: cam1 front over the feed bunk looking down the pen, cam2 close over the bunk, cam3 back
corner. Some, all, or none of the animals may have BRD. Most pens have few or no sick animals.

You review ONE cow. Your job:
1. AUDIT the behavior indicators. Are the cow's deviations real, or artifacts? Artifacts include:
   little time in view (low hours), a mis-identified cow, or a behavior label the images contradict
   (e.g. labelled feeding but standing away from the bunk). Check with images before you call one.
2. LOOK for visible clinical signs (DART pen-riding signs):
   head_low (head carried below the topline while standing and not eating), ears_droop,
   nasal_discharge, ocular_discharge, labored_breathing (open mouth, extended neck, strong flank
   effort), cough (coughing posture), gaunt (hollow left flank, empty look).
3. REPORT structured findings. You do NOT set the score. Code combines your findings with the
   behavior evidence, using weights from the literature.

Evidence rules (an automatic self-check enforces them):
- Cite fact ids for everything: F-<cow>-<indicator> (behavior numbers), I#### (images a tool showed
  you), V#### (observations you saved with record_fact).
- A sign can be "present" or "absent" only if you saw THIS cow in an image whose box is >= {min_px} px.
  Discharge needs >= {dis_px} px in daylight colour. Otherwise use "not_assessable". IR night video is
  grey and soft: no discharge, no ear detail.
- Before calling posture abnormal, compare with herdmates at the same moment (compare_herdmates).
  Low heads while eating, drinking or lying are normal. Relaxed ears while ruminating are normal.
- Short coverage is ALREADY discounted by the score (shrinkage by hours observed). Do not call an
  indicator an artifact because coverage is short. An artifact needs image evidence (I#### / V####)
  that the identity, visibility or behavior label behind it is wrong.
- Absence of evidence is not evidence of absence. "not_assessable" is a good, honest answer.
Suggested order (long-term first, images second):
  a. trend(cow, indicator) for the deviations that matter: is it sustained, and when did it start
     (CUSUM onset)? daily_budget for the day/night pattern; feed_response (appetite after feed delivery);
     bouts (fewer, longer lying bouts; shorter meals); social (drifting from usual neighbours).
  b. breathing (breaths/min from still close windows; resting adult cattle ~26-50/min, and compare with
     the herd) and jerks (cough / head-shake candidates: confirm with get_sequence around the time).
  c. gallery = the closest daylight crops saved over 48 h (best for ears, eyes, nose, discharge);
     otherwise find_moments(kind="close_view") + get_crop(region="head", enhance=true).
  d. compare_herdmates before calling posture abnormal.

Tools:
{tools}

Reply with ONE JSON object per turn, nothing else:
{{"thought": "short reasoning", "tool": "<name>", "args": {{...}}}}
or, when done:
{{"thought": "...", "final": {{
  "signs": {{"head_low": {{"status": "present|absent|not_assessable", "evidence": ["I0003"], "note": "..."}}, ... all 7 signs ...}},
  "artifacts": [{{"indicator": "<indicator name>", "evidence": ["I0005"], "why": "..."}}],
  "behavior_assessment": "which behavior deviations you believe and why",
  "summary": "2-3 plain sentences a producer can act on",
  "key_evidence": ["F-COW_03-feeding_share", "V0002", "I0004"],
  "confidence": 0.0-1.0 }}}}
You have {budget} tool calls."""


class Backends:
    """Ordered TAMU models; a call falls through on errors, empty replies or invalid JSON."""

    def __init__(self, agent_cfg: Dict, names: List[str]):
        load_dotenv()
        self.models, self.calls, self.seconds, self.used = [], 0, 0.0, []
        for n in names:
            p = agent_cfg.get(n, {})
            key = os.environ.get(p.get("api_key_env", ""), None)
            if not p or not key:
                continue
            model = (os.environ.get(p["model_env"]) if p.get("model_env") else None) or p["model"]
            self.models.append((n, OpenAICompatVLM(p["base_url"], model, key,
                                                   temperature=None if "temperature" in p and p["temperature"] is None else p.get("temperature", 0.1),
                                                   max_tokens=p.get("max_tokens", 2000), timeout=p.get("timeout", 180),
                                                   image_max_side=p.get("image_max_side", 1024),
                                                   token_key=p.get("token_key", "max_tokens"))))
        if not self.models:
            raise RuntimeError("no reasoning backend available (TAMU_API_KEY in .env?)")
        self.image_max_side = self.models[0][1].image_max_side

    def complete(self, messages) -> str:
        """Raw text from the first model that returns parseable JSON (used by batch labelers)."""
        act, text, model = self.ask(messages)
        return text

    def ask(self, messages) -> (Dict, str, str):
        err = None
        for n, m in self.models:
            t = time.time()
            try:
                text = m.complete(messages)
            except Exception as e:
                err = e
                continue
            act = parse_json(text)
            self.calls += 1
            self.seconds += time.time() - t
            if act is not None:
                self.used.append(m.model)
                return act, text, m.model
            err = f"{m.model}: not JSON"
        raise RuntimeError(f"all reasoning backends failed: {err}")


class RiskAgent:
    def __init__(self, tools: RiskTools, store: EvidenceStore, backends: Backends, cfg: Dict, out_dir: str):
        self.tb, self.store, self.b, self.cfg = tools, store, backends, cfg
        self.out = out_dir
        os.makedirs(os.path.join(out_dir, "traces"), exist_ok=True)

    def _system(self, budget: int) -> str:
        return SYSTEM.format(min_px=self.cfg["visual_min_box_px"], dis_px=self.cfg["discharge_min_box_px"],
                             tools=self.tb.describe(), budget=budget)

    def _messages(self, system: str, turns: List[Dict], keep_images: int = 3):
        msgs = [{"role": "system", "content": system}]
        n_img = sum(1 for t in turns if t.get("images"))
        seen = 0
        for t in turns:
            if t["role"] == "assistant":
                msgs.append({"role": "assistant", "content": t["text"]})
                continue
            content = [{"type": "text", "text": t["text"]}]
            if t.get("images"):
                seen += 1
                if n_img - seen < keep_images:
                    content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                                                                    to_jpeg_b64(im, self.b.image_max_side)}} for im in t["images"]]
                else:
                    content[0]["text"] += "\n[older image omitted; its fact id is still valid]"
            msgs.append({"role": "user", "content": content})
        return msgs

    def brief(self, cow: str, herd_rank: pd.DataFrame, notes: str) -> str:
        prof = self.tb.t_cow_profile(cow).text
        ranks = ", ".join(f"{c} {r.behavior_score:.2f}" for c, r in herd_rank.iterrows())
        return (f"Cow under review: {cow}\n\nBehavior evidence (herd-relative robust z; positive = sick direction):\n{prof}\n\n"
                f"Herd behavior scores: {ranks}\n\nData notes: {notes}\n\nStart by deciding which deviations matter, "
                f"then look at the cow. Cite fact ids.")

    def investigate(self, cow: str, herd_rank: pd.DataFrame, notes: str, budget: int) -> Dict:
        system = self._system(budget)
        turns = [{"role": "user", "text": self.brief(cow, herd_rank, notes)}]
        steps, t0, final = [], time.time(), None
        calls = 0
        while final is None:
            try:
                act, raw, model = self.b.ask(self._messages(system, turns))
            except RuntimeError as e:
                steps.append({"error": str(e)})
                break
            turns.append({"role": "assistant", "text": raw[:4000]})
            if "final" in act and isinstance(act["final"], dict):
                final = act["final"]
                steps.append({"thought": act.get("thought"), "final": final, "model": model})
                break
            name, args = act.get("tool"), act.get("args") or {}
            if calls >= budget or not name:
                turns.append({"role": "user", "text": "Tool budget used. Reply with your final JSON now."})
                calls += 1
                if calls > budget + 2:
                    break
                continue
            res = self.tb.call(name, args)
            calls += 1
            steps.append({"thought": act.get("thought"), "tool": name, "args": args, "result": res.text[:1500],
                          "facts": [res.data.get("fact")] if res.data.get("fact") else [], "model": model})
            turns.append({"role": "user", "text": f"Result of {name}:\n{res.text}\n({budget - calls} tool calls left)",
                          "images": res.images})
        final = final or {}
        errors = self.verify(cow, final)
        if errors and final:  # one chance to fix, like a reviewer sending it back
            turns.append({"role": "user", "text": "Self-check failed:\n- " + "\n- ".join(errors) +
                                                  "\nFix these and resend the final JSON (no more tools)."})
            try:
                act, raw, model = self.b.ask(self._messages(system, turns))
                if isinstance(act.get("final"), dict):
                    steps.append({"thought": act.get("thought"), "final": act["final"], "model": model, "revision": True})
                    final = act["final"]
            except RuntimeError as e:
                steps.append({"error": str(e)})
        clean, remaining = self.apply_rules(cow, final)
        trace = {"cow_id": cow, "seconds": round(time.time() - t0, 1), "tool_calls": calls, "steps": steps,
                 "self_check_first": errors, "self_check_after_revision": remaining, "final": clean,
                 "models": sorted(set(s.get("model") for s in steps if s.get("model")))}
        json.dump(trace, open(os.path.join(self.out, "traces", f"{cow}.json"), "w"), indent=1, default=str)
        return trace

    # ------------------------------------------------------------------ self-check
    def _image_support(self, cow: str, ids: List[str]) -> List[Dict]:
        """Image facts of this cow behind the cited ids (directly, or through V facts)."""
        out = []
        for i in ids or []:
            f = self.store.get(str(i))
            if not f:
                continue
            if f["kind"] == "visual":
                out += self._image_support(cow, f["data"].get("based_on", []))
            elif f["kind"] == "image" and f["cow_id"] == cow:
                out.append(f)
        return out

    def verify(self, cow: str, final: Dict) -> List[str]:
        errs = []
        if not final:
            return ["no final answer"]
        signs = final.get("signs") or {}
        for s in SIGNS:
            v = signs.get(s) or {}
            st = v.get("status", "not_assessable")
            if st not in ("present", "absent", "not_assessable"):
                errs.append(f"{s}: status must be present|absent|not_assessable")
                continue
            if st == "not_assessable":
                continue
            sup = self._image_support(cow, v.get("evidence", []))
            if not sup:
                errs.append(f"{s}={st} cites no image of {cow} (cite the I#### you looked at)")
                continue
            need = self.cfg["discharge_min_box_px"] if s in DISCHARGE else self.cfg["visual_min_box_px"]
            ok = [f for f in sup if f["data"].get("box_px", 0) >= need and
                  (s not in DISCHARGE or not self.cfg["discharge_needs_daylight"] or f["data"].get("daylight"))]
            if not ok:
                errs.append(f"{s}={st}: cited images are too small or not daylight (need box >= {need} px"
                            f"{', daylight colour' if s in DISCHARGE else ''}); use not_assessable or a closer view")
        known = set(self.cfg["indicators"])
        for a in final.get("artifacts") or []:
            if a.get("indicator") not in known:
                errs.append(f"artifact indicator '{a.get('indicator')}' unknown")
            elif not self._image_support(cow, a.get("evidence", [])) and not [
                    i for i in a.get("evidence", []) if (self.store.get(str(i)) or {}).get("kind") == "image"]:
                errs.append(f"artifact {a.get('indicator')} needs image evidence (I#### / V####) that the data is wrong; "
                            f"short coverage is already discounted, so drop it if that is the only reason")
        for i in final.get("key_evidence") or []:
            if not (self.store.get(str(i)) or (str(i).startswith(f"F-{cow}-"))):
                errs.append(f"key_evidence {i} does not exist")
        return errs

    def apply_rules(self, cow: str, final: Dict):
        """Downgrade whatever still fails the self-check; return the clean findings + remaining errors."""
        final = json.loads(json.dumps(final or {}, default=str))
        signs = final.get("signs") or {}
        remaining = self.verify(cow, final)
        for s in SIGNS:
            v = signs.setdefault(s, {"status": "not_assessable", "evidence": []})
            if any(e.startswith(f"{s}=") or e.startswith(f"{s}:") for e in remaining):
                v["note"] = (v.get("note", "") + " [downgraded by self-check]").strip()
                v["status"] = "not_assessable"
        final["signs"] = signs
        final["artifacts"] = [a for a in final.get("artifacts") or [] if a.get("indicator") in self.cfg["indicators"]
                              and not any(e.startswith(f"artifact {a.get('indicator')}") for e in remaining)]
        final["key_evidence"] = [i for i in final.get("key_evidence") or [] if self.store.get(str(i)) or str(i).startswith(f"F-{cow}-")]
        return final, remaining


def visual_logit(final: Dict, cfg: Dict) -> (float, List[str]):
    total, why = 0.0, []
    for s, spec in cfg["visual_signs"].items():
        st = ((final.get("signs") or {}).get(s) or {}).get("status")
        if st in ("present", "absent"):
            total += spec[st]
            if st == "present":
                why.append(spec["label"])
    return max(total, cfg["visual_max_down"]), why
