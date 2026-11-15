"""Reasoning backends for the agent.

`OpenAICompatVLM` speaks the OpenAI chat-completions API with images, which covers:
  * Ollama on this Mac (Qwen2.5-VL):   base_url http://localhost:11434/v1, model qwen2.5vl:7b
  * TAMU Chat / TAMU AI API:           base_url + key from the TAMU portal
  * vLLM on HPRC (Qwen2.5-VL / Qwen3-VL): base_url http://<node>:8000/v1
`RulePolicy` is a deterministic offline fallback (no model): it gathers the same evidence and
decides from the numeric tools only. Its traces are labelled backend=rules so they are never
mistaken for VLM reasoning.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from .evidence import to_jpeg_b64

SYSTEM_PROMPT = """You are the supervisor agent of a cattle-monitoring pipeline for Pen 6:
10 Black Angus steers in a 12 m x 30 m pen. Feed bunk along the front fence (Y = 0 to 1.5 m),
waterer near X 8.5-11.5 m, Y 14-18 m. Cameras: cam1 = front fisheye, cam2 = overhead over the
feed bunk, cam3 = back of the pen (its metre coordinates are NOT reliable).
The perception models run on every frame: a YOLO cow detector, ByteTrack (box tracks per
camera), an appearance identity model (DINOv2 + coat colour) assigning COW_01..COW_10, and a
behavior classifier with 10 classes: lying, standing, walking, feeding, drinking,
ruminating (lying), ruminating (standing), grooming, social interaction, other / not visible.
You are only called when something looks wrong or uncertain. Find out what really happened
with the tools, then decide.

Rules:
- Black Angus look nearly identical. Never decide identity from one still image. Combine the
  appearance score, trajectory continuity (where the cow was just before), memory-card traits
  (ear tags, white marks, size) and the frames themselves.
- Chewing (rumination) cannot be seen in still frames: use chew_motion for rumination.
- Feeding means head in the feed bunk; drinking means head at the waterer.
- A correction (relabel / set_behavior) needs at least TWO different kinds of evidence that
  agree. If evidence conflicts or is weak, answer "uncertain" instead of guessing.
- A relabel must NEVER assign an ID that is already active on another track in the same camera
  at the same time (check who_is_near / cow_state).
- Be brief. One tool call per reply.

Reply with exactly ONE JSON object and nothing else, either
{"thought": "<what you know and what you check next>", "tool": "<tool name>", "args": {...}}
or, when done,
{"thought": "<reasoning>", "final": {"decision": "<confirm|relabel|uncertain|set_behavior|note|escalate>",
  ... decision fields ..., "evidence": ["<fact 1>", "<fact 2>"], "confidence": <0-1>,
  "summary": "<one sentence for the log>"}}
Decision fields: relabel -> camera, track_id, new_cow_id, t0, t1 ; uncertain -> cow_id or
(camera, track_id), t0, t1 ; set_behavior -> cow_id, behavior, t0, t1 ; note -> cow_id, text ;
escalate -> reason (a human must look); confirm -> nothing extra (the pipeline was right).
For a scheduled ID audit you may add "mismatches": ["COW_xx", ...] to the final object.
Times are HH:MM:SS.
Example of a final reply:
{"thought": "Appearance ranks COW_07 first by 0.08 and the trajectory shows COW_04 was 9 m away 5 s earlier",
 "final": {"decision": "relabel", "camera": "cam1", "track_id": 16, "new_cow_id": "COW_07", "t0": "16:02:10",
 "t1": "16:05:30", "evidence": ["appearance best COW_07 (+0.08)", "COW_04 elsewhere at 16:02:05"],
 "confidence": 0.7, "summary": "track 16 is COW_07, not COW_04"}}"""


def parse_json(text: str) -> Optional[Dict]:
    """First balanced {...} in the text (models sometimes wrap JSON in prose or ``` fences)."""
    text = re.sub(r"```(json)?", "", text or "")
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


class OpenAICompatVLM:
    name = "vlm"

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None, temperature: float = 0.1,
                 max_tokens: int = 600, timeout: float = 180, image_max_side: int = 1024,
                 keep_images: int = 2, extra_body: Optional[Dict] = None, token_key: str = "max_tokens"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self.image_max_side, self.keep_images = image_max_side, keep_images
        self.extra_body = extra_body or {}
        self.token_key = token_key
        self.calls, self.seconds = 0, 0.0

    @property
    def label(self):
        return f"{self.model} @ {self.base_url}"

    def _messages(self, state) -> List[Dict]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        n_img_turns = sum(1 for s in state["turns"] if s.get("images"))
        seen_img = 0
        for turn in state["turns"]:
            if turn["role"] == "assistant":
                msgs.append({"role": "assistant", "content": turn["text"]})
                continue
            content = [{"type": "text", "text": turn["text"]}]
            if turn.get("images"):
                seen_img += 1
                if n_img_turns - seen_img < self.keep_images:  # only the latest images are re-sent
                    for im in turn["images"]:
                        content.append({"type": "image_url", "image_url": {
                            "url": "data:image/jpeg;base64," + to_jpeg_b64(im, self.image_max_side)}})
                else:
                    content[0]["text"] += "\n[image from this step omitted to save time; re-run the tool if needed]"
            msgs.append({"role": "user", "content": content})
        return msgs

    def complete(self, messages: List[Dict]) -> str:
        body = {"model": self.model, "messages": messages, self.token_key: self.max_tokens,
                "stream": False, **self.extra_body}  # some gateways stream by default
        if self.temperature is not None:  # GPT-5 / o-series accept only their default temperature
            body["temperature"] = self.temperature
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "Mozilla/5.0 (Macintosh) pen6-agent/1.0",  # some gateways 403 urllib's default
                                              **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        last = None
        for attempt in range(3):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                self.calls += 1
                self.seconds += time.time() - t0
                return out["choices"][0]["message"]["content"] or ""
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"VLM request failed: {last}")

    def next_action(self, state) -> Dict:
        text = self.complete(self._messages(state))
        act = parse_json(text)
        if act is None:  # one repair attempt
            state["turns"].append({"role": "assistant", "text": text[:2000]})
            state["turns"].append({"role": "user", "text": "Reply with ONE valid JSON object only."})
            text = self.complete(self._messages(state))
            state["turns"].pop(); state["turns"].pop()
            act = parse_json(text) or {"thought": text[:500], "final": {"decision": "escalate",
                                                                         "reason": "agent reply was not valid JSON: "
                                                                                   + (text or "")[:300]}}
        act["_raw"] = text
        return act


class RulePolicy:
    """Offline fallback with the same tools and verifier but no model (scripted plan)."""

    name = "rules"
    label = "rule policy (no VLM)"
    calls, seconds = 0, 0.0

    def next_action(self, state) -> Dict:
        f, done = state["flag"], {s["tool"]: s for s in state["steps"]}
        cid = (f.get("cow_ids") or [None])[0]
        hms = state["hms"]
        if f["kind"] in ("weak_id", "id_jump", "new_cow_id") and f.get("track_id") is not None:
            if "compare_to_cards" not in done:
                return {"thought": "score the box against all memory cards", "tool": "compare_to_cards",
                        "args": {"camera": f["camera"], "track_id": f["track_id"], "time": hms(f["t1"])}}
            if "trajectory" not in done and cid:
                return {"thought": "check continuity of the assigned ID", "tool": "trajectory",
                        "args": {"cow_id": cid, "t0": hms(f["t0"] - 60), "t1": hms(f["t1"] + 10)}}
            rank = done["compare_to_cards"]["data"].get("ranking") or []
            if len(rank) >= 2 and rank[0][0] != cid and rank[0][1] - rank[1][1] > 0.05:
                return {"thought": "appearance clearly prefers another cow", "final": {
                    "decision": "relabel", "camera": f["camera"], "track_id": f["track_id"], "new_cow_id": rank[0][0],
                    "t0": hms(f["t0"]), "t1": hms(f["t1"] + 300), "confidence": 0.6,
                    "evidence": [f"appearance best {rank[0][0]} by {rank[0][1] - rank[1][1]:.3f}", "trajectory checked"],
                    "summary": f"track {f['track_id']} looks like {rank[0][0]}, not {cid}"}}
            if rank and rank[0][0] == cid:
                return {"thought": "appearance agrees with the assigned ID", "final": {
                    "decision": "confirm", "confidence": 0.6, "evidence": [f"appearance best = {cid}"],
                    "summary": f"{cid} assignment consistent with its card"}}
            return {"thought": "scores do not separate the candidates", "final": {
                "decision": "uncertain", "camera": f["camera"], "track_id": f["track_id"],
                "t0": hms(f["t0"]), "t1": hms(f["t1"]), "confidence": 0.5,
                "evidence": ["appearance margin too small"], "summary": "identity ambiguous; marked uncertain"}}
        if f["kind"] == "behavior_check" and cid:
            beh = (f.get("data") or {}).get("behavior", "")
            if "behavior" not in done:
                return {"thought": "look at classifier evidence", "tool": "behavior",
                        "args": {"cow_id": cid, "t0": hms(f["t0"]), "t1": hms(f["t1"])}}
            if beh.startswith("ruminating") and "chew_motion" not in done:
                return {"thought": "measure chewing", "tool": "chew_motion",
                        "args": {"cow_id": cid, "time": hms((f["t0"] + f["t1"]) / 2)}}
            if beh.startswith("ruminating"):
                chew = done["chew_motion"]["data"].get("peak_ratio", 0)
                if chew < 3:
                    base = "lying" if "lying" in beh else "standing"
                    return {"thought": "no chewing rhythm", "final": {
                        "decision": "set_behavior", "cow_id": cid, "behavior": base, "t0": hms(f["t0"]),
                        "t1": hms(f["t1"]), "confidence": 0.6,
                        "evidence": [f"chew peak_ratio {chew:.2f} < 3", "classifier posture"],
                        "summary": f"{cid}: no measurable chewing, {beh} -> {base}"}}
            return {"thought": "nothing contradicts the classifier", "final": {
                "decision": "confirm", "confidence": 0.5, "evidence": ["classifier"], "summary": f"{cid} {beh} kept"}}
        if f["kind"] == "id_audit":
            return {"thought": "no VLM: audit sheet saved for review", "final": {
                "decision": "note", "cow_id": cid or "PEN", "text": "audit sheet saved (no VLM review)",
                "confidence": 0.3, "evidence": ["audit sheet"], "summary": "audit sheet saved"}}
        return {"thought": "needs a person", "final": {"decision": "escalate", "reason": f["detail"], "confidence": 0.3,
                                                       "evidence": [], "summary": f["detail"]}}


class FallbackVLM:
    """Primary reasoning model with automatic fallbacks (e.g. TAMU GPT-5.6 Luna -> local Qwen).
    A backend is skipped for that call when it errors (network, quota, 4xx/5xx) or returns an
    empty / non-JSON reply; the trace records which model answered."""

    name = "fallback"

    def __init__(self, backends):
        self.backends = backends
        self.fallbacks = 0
        self.last_used = None

    @property
    def label(self):
        return " -> ".join(getattr(b, "label", b.name) for b in self.backends)

    @property
    def calls(self):
        return sum(getattr(b, "calls", 0) for b in self.backends)

    @property
    def seconds(self):
        return sum(getattr(b, "seconds", 0.0) for b in self.backends)

    @staticmethod
    def _bad(act):
        fin = act.get("final") or {}
        return fin.get("decision") == "escalate" and str(fin.get("reason", "")).startswith("agent reply was not valid JSON")

    def next_action(self, state):
        err = None
        for i, b in enumerate(self.backends):
            try:
                act = b.next_action(state)
            except Exception as e:  # network, quota, gateway errors
                err = e
                self.fallbacks += i < len(self.backends) - 1
                continue
            if self._bad(act) and i < len(self.backends) - 1:
                self.fallbacks += 1
                continue
            act["_model"] = getattr(b, "label", b.name)
            self.last_used = act["_model"]
            return act
        raise RuntimeError(f"all reasoning backends failed: {err}")

    def complete(self, messages):
        err = None
        for b in self.backends:
            if not hasattr(b, "complete"):
                continue
            try:
                out = b.complete(messages)
                if out and out.strip():
                    return out
            except Exception as e:
                err = e
        raise RuntimeError(f"all reasoning backends failed: {err}")


class TieredDecisionBackend:
    """Decision model that routes routine/quick checks to a fast tier and critical or
    ambiguous cases to a major reasoning model.

    * Quick Tier: handles lower-severity flags (severity < escalate_severity, e.g. weak_id, behavior_check)
      with a fast model (e.g. RulePolicy, Qwen fast, or Luna fast) with a short step budget.
    * Major Decision Tier: handles critical flags (severity >= escalate_severity, e.g. duplicate_id,
      id_jump, extra_cow, id_audit) OR takes over when the quick tier is uncertain, low-confidence,
      or fails verification.
    The major model inherits all tool outputs and conversation history from the quick tier,
    so no tool calls or evidence are wasted.
    """

    name = "decision"

    def __init__(self, fast_backend, major_backend, escalate_severity: int = 3,
                 min_confidence: float = 0.70, max_quick_steps: int = 2,
                 escalate_kinds: Optional[List[str]] = None):
        self.fast = fast_backend
        self.major = major_backend
        self.escalate_severity = escalate_severity
        self.min_confidence = min_confidence
        self.max_quick_steps = max_quick_steps
        self.escalate_kinds = set(escalate_kinds or ["duplicate_id", "extra_cow", "new_cow_id", "id_audit"])
        self.escalations = 0

    @property
    def label(self):
        fast_lbl = getattr(self.fast, "label", getattr(self.fast, "name", "fast"))
        major_lbl = getattr(self.major, "label", getattr(self.major, "name", "major"))
        return f"tiered[quick: {fast_lbl} | major: {major_lbl}]"

    @property
    def calls(self):
        return getattr(self.fast, "calls", 0) + getattr(self.major, "calls", 0)

    @property
    def seconds(self):
        return getattr(self.fast, "seconds", 0.0) + getattr(self.major, "seconds", 0.0)

    def _should_escalate_initial(self, flag: Dict) -> bool:
        if flag.get("severity", 0) >= self.escalate_severity:
            return True
        if flag.get("kind") in self.escalate_kinds:
            return True
        return False

    def next_action(self, state: Dict) -> Dict:
        flag = state.get("flag", {})
        steps = state.get("steps", [])
        turns = state.get("turns", [])

        is_major = bool(state.get("_escalated_to_major"))
        if not is_major and self._should_escalate_initial(flag):
            is_major = True
            state["_escalated_to_major"] = True

        if not is_major and turns:
            last_text = turns[-1].get("text", "")
            if "SELF-CHECK FAILED" in last_text:
                is_major = True
                state["_escalated_to_major"] = True
                self.escalations += 1

        if not is_major and len(steps) >= self.max_quick_steps:
            is_major = True
            state["_escalated_to_major"] = True
            self.escalations += 1

        if is_major:
            act = self.major.next_action(state)
            act["_tier"] = "major"
            act["_model"] = getattr(self.major, "label", self.major.name)
            return act

        # Fast tier execution
        act = self.fast.next_action(state)
        act["_tier"] = "quick"
        act["_model"] = getattr(self.fast, "label", self.fast.name)

        if "final" in act and isinstance(act["final"], dict):
            fin = act["final"]
            dec = fin.get("decision")
            conf = float(fin.get("confidence") or 0.0)
            if dec in ("uncertain", "escalate") or conf < self.min_confidence:
                state["_escalated_to_major"] = True
                self.escalations += 1
                act_major = self.major.next_action(state)
                act_major["_tier"] = "major"
                act_major["_model"] = getattr(self.major, "label", self.major.name)
                return act_major

        return act

    def complete(self, messages):
        if hasattr(self.major, "complete"):
            return self.major.complete(messages)
        if hasattr(self.fast, "complete"):
            return self.fast.complete(messages)
        raise RuntimeError("no underlying complete method in tiered decision backend")


def load_dotenv(path: str = ".env"):
    """KEY=value lines from a git-ignored .env into the environment (never printed)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_backend(cfg: Dict, name: Optional[str] = None, with_fallback: bool = True):
    load_dotenv()
    b = (name or os.environ.get("AGENT_BACKEND") or cfg.get("backend", "ollama")).lower()
    if b in ("decision", "tiered"):
        p = cfg.get(b, {})
        fast_name = p.get("fast_backend", "tamu_luna")
        major_name = p.get("major_backend", "tamu_luna")
        fast_b = make_backend(cfg, fast_name, with_fallback=True)
        major_b = make_backend(cfg, major_name, with_fallback=True)
        return TieredDecisionBackend(
            fast_backend=fast_b,
            major_backend=major_b,
            escalate_severity=p.get("escalate_severity", 3),
            min_confidence=p.get("min_confidence", 0.70),
            max_quick_steps=p.get("max_quick_steps", 2),
            escalate_kinds=p.get("escalate_kinds"),
        )
    chain = [x for x in cfg.get("fallback", []) if x != b] if with_fallback and not name else []
    if chain:
        backends = [make_backend(cfg, b, False)]
        for fb in chain:
            try:
                backends.append(make_backend(cfg, fb, False))
            except Exception:
                pass
        return FallbackVLM(backends) if len(backends) > 1 else backends[0]
    if b == "rules":
        return RulePolicy()
    p = cfg.get(b, {})
    base = os.environ.get("AGENT_VLM_BASE_URL") or p.get("base_url")
    model = os.environ.get("AGENT_VLM_MODEL") or p.get("model")
    key = os.environ.get("AGENT_VLM_API_KEY") or (os.environ.get(p["api_key_env"]) if p.get("api_key_env") else None)
    if not base or not model:
        raise ValueError(f"backend '{b}' needs base_url and model (configs/agent.yaml or AGENT_VLM_* env vars)")
    return OpenAICompatVLM(base, model, key, temperature=p.get("temperature", 0.1) if "temperature" not in p or p["temperature"] is not None else None,
                           token_key=p.get("token_key", "max_tokens"),
                           max_tokens=p.get("max_tokens", 600), timeout=p.get("timeout", 180),
                           image_max_side=p.get("image_max_side", 1024), keep_images=p.get("keep_images", 2),
                           extra_body=p.get("extra_body"))
