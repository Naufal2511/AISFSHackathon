"""VLM access. `OpenAICompatVLM` talks to any OpenAI-compatible server (e.g. vLLM on HPRC).
Features: on-disk cache keyed by (model, prompt_version, system, user, image bytes), JSON extraction,
schema validation, one automatic repair attempt. `MockVLM` lets you run/test the whole loop offline."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, Type

import requests
from PIL import Image
from pydantic import BaseModel, ValidationError

from .config import AgentConfig, VLMCfg
from .env import ConfigError, Settings


class VLMParseError(Exception):
    pass


def extract_json(text: str) -> dict:
    """Pull the first balanced JSON object out of a model reply (handles ``` fences and chatter)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    start = t.find("{")
    if start < 0:
        raise VLMParseError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(t[start:i + 1])
    raise VLMParseError("unbalanced JSON")


class VLMHTTPError(Exception):
    pass


class VLMClient(ABC):
    provider: str = "unknown"
    model: str = "unknown"

    @abstractmethod
    def ask(self, task: str, system: str, user: str, images: list[str], schema: Type[BaseModel],
            meta: Optional[dict] = None) -> tuple[BaseModel, dict]:
        """Return (validated object, info). info: {raw, prompt_tokens, completion_tokens, latency_s, cached, repaired}.
        `meta` is ignored by real clients (MockVLM uses it)."""


@dataclass
class ProviderConn:
    """Everything provider-specific. The rest of the code never branches on the provider."""
    provider: str                       # 'openai' | 'local'
    url: str                            # full .../chat/completions URL
    model: str
    api_key: Optional[str] = None
    image_detail: Optional[str] = None  # OpenAI only
    temperature: Optional[float] = None # None -> omitted from the request
    max_tokens: int = 800
    max_tokens_param: str = "max_tokens"          # OpenAI uses max_completion_tokens
    json_mode: bool = False                       # OpenAI response_format=json_object
    guided_json: bool = False                     # local servers that support it


def _chat_url(base: str) -> str:
    b = base.rstrip("/")
    return b if b.endswith("/chat/completions") else b + "/chat/completions"


def make_vlm(settings: Settings, cfg: AgentConfig) -> VLMClient:
    """USE_LOCAL_VLM=false (default) -> OpenAI; true -> local OpenAI-compatible server (e.g. vLLM)."""
    if settings.use_local_vlm:
        if not settings.local_model:
            raise ConfigError("USE_LOCAL_VLM=true but LOCAL_VLM_MODEL is empty (set it in .env)")
        conn = ProviderConn("local", _chat_url(settings.local_base_url), settings.local_model,
                            api_key=settings.local_api_key or None, temperature=cfg.vlm.temperature,
                            max_tokens=cfg.vlm.max_tokens, max_tokens_param="max_tokens",
                            guided_json=settings.local_guided_json)
    else:
        if not settings.openai_api_key:
            raise ConfigError("USE_LOCAL_VLM=false (OpenAI) but OPENAI_API_KEY is empty (set it in .env)")
        if not settings.openai_model:
            raise ConfigError("OPENAI_MODEL is empty: set the OpenAI vision model you want in .env")
        conn = ProviderConn("openai", _chat_url(settings.openai_base_url), settings.openai_model,
                            api_key=settings.openai_api_key, image_detail=settings.openai_image_detail or None,
                            temperature=settings.openai_temperature, max_tokens=settings.openai_max_tokens,
                            max_tokens_param="max_completion_tokens", json_mode=True)
    return OpenAICompatVLM(conn, cfg.vlm, cfg.prompt_version)


class OpenAICompatVLM(VLMClient):
    RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, conn: ProviderConn, cfg: VLMCfg, prompt_version: str,
                 max_retries: int = 3, backoff_s: float = 2.0):
        self.conn, self.cfg, self.pv = conn, cfg, prompt_version
        self.provider, self.model = conn.provider, conn.model
        self.max_retries, self.backoff_s = max_retries, backoff_s
        self.cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _encode(self, path: str) -> str:
        im = Image.open(path).convert("RGB")
        s = self.cfg.max_image_side / max(im.size)
        if s < 1:
            im = im.resize((int(im.width * s), int(im.height * s)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    def _key(self, system, user, b64s, schema) -> str:
        c = self.conn
        h = hashlib.sha256()
        for part in (c.provider, c.model, c.url, str(c.image_detail), str(c.temperature), str(c.json_mode),
                     self.pv, schema.__name__, system, user, *b64s):
            h.update(str(part).encode())
        return h.hexdigest()

    def _post(self, messages, schema) -> dict:
        c = self.conn
        payload: dict[str, Any] = {"model": c.model, "messages": messages, c.max_tokens_param: c.max_tokens}
        if c.temperature is not None:
            payload["temperature"] = c.temperature
        if c.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if c.guided_json:
            payload["guided_json"] = schema.model_json_schema()   # server-specific; verify for your server version
        headers = {"Content-Type": "application/json"}
        if c.api_key:
            headers["Authorization"] = f"Bearer {c.api_key}"
        last: Optional[str] = None
        for i in range(self.max_retries + 1):
            try:
                r = requests.post(c.url, json=payload, headers=headers, timeout=self.cfg.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if r.status_code < 400:
                    return r.json()
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in self.RETRY_STATUS:
                    raise VLMHTTPError(last)
                ra = r.headers.get("Retry-After")
                if ra and ra.replace(".", "", 1).isdigit() and i < self.max_retries:
                    time.sleep(min(float(ra), 30.0))
                    continue
            if i < self.max_retries:
                time.sleep(self.backoff_s * (2 ** i))
        raise VLMHTTPError(f"{c.provider} request failed after {self.max_retries + 1} attempts: {last}")

    def ask(self, task, system, user, images, schema, meta=None):
        b64s = [self._encode(p) for p in images]
        key = self._key(system, user, b64s, schema)
        cpath = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cpath and cpath.exists():
            c = json.loads(cpath.read_text(encoding="utf-8"))
            return schema.model_validate(c["parsed"]), {**c["info"], "cached": True}

        def img_part(b):
            d = {"url": f"data:image/jpeg;base64,{b}"}
            if self.conn.image_detail:
                d["detail"] = self.conn.image_detail
            return {"type": "image_url", "image_url": d}

        content = [{"type": "text", "text": user}] + [img_part(b) for b in b64s]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        t0, repaired, ptok, ctok = time.time(), False, 0, 0
        raw, err = "", None
        for attempt in range(2):                       # 1 try + 1 repair
            resp = self._post(messages, schema)
            raw = resp["choices"][0]["message"]["content"] or ""
            u = resp.get("usage", {})
            ptok += u.get("prompt_tokens", 0); ctok += u.get("completion_tokens", 0)
            try:
                obj = schema.model_validate(extract_json(raw))
                info = {"raw": raw, "prompt_tokens": ptok, "completion_tokens": ctok,
                        "latency_s": round(time.time() - t0, 2), "cached": False, "repaired": repaired}
                if cpath:
                    cpath.write_text(json.dumps({"parsed": obj.model_dump(mode="json"), "info": info}), encoding="utf-8")
                return obj, info
            except (VLMParseError, ValidationError, json.JSONDecodeError, ValueError) as e:
                err = str(e)[:400]
                repaired = True
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Your reply was invalid ({err}). Reply again with ONLY the JSON object in the required shape, using only allowed labels."}]
        raise VLMParseError(f"{task}: could not parse after repair: {err}")


class MockVLM(VLMClient):
    """Deterministic stand-in. `policy(task, user, images, meta) -> dict | str` returns the reply
    (a dict is dumped to JSON; a str is returned raw so you can test the repair path)."""

    provider = "mock"
    model = "mock"

    def __init__(self, policy: Optional[Callable] = None):
        self.policy = policy or default_mock_policy
        self.calls: list[tuple[str, int]] = []

    def ask(self, task, system, user, images, schema, meta=None):
        self.calls.append((task, len(images)))
        reply = self.policy(task, user, images, meta or {})
        raw = reply if isinstance(reply, str) else json.dumps(reply)
        repaired = False
        try:
            obj = schema.model_validate(extract_json(raw))
        except Exception:
            repaired = True
            obj = schema.model_validate(extract_json(json.dumps(self.policy(task + ":repair", user, images, meta or {}))))
        return obj, {"raw": raw, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0,
                     "cached": False, "repaired": repaired}


def default_mock_policy(task: str, user: str, images: list[str], meta: dict):
    facts = meta.get("facts", {})
    brief = meta.get("brief")
    task = task.split(":")[0]
    if task == "plan":
        return {"rationale": "mock plan", "steps": [
            {"tool": "zone_query", "args": {}, "reason": "check whether the head is at the bunk or waterer"},
            {"tool": "track_history", "args": {}, "reason": "measure speed and posture"},
            {"tool": "zoom", "args": {"camera": "auto", "region": "head"}, "reason": "look closely at the head"}]}
    zq, th, rp = facts.get("zone_query", {}), facts.get("track_history", {}), facts.get("rumination_probe", {})
    posture = th.get("posture_mode") or (brief.posture if brief else None) or "unknown"
    speed = th.get("speed_mean", brief.speed_px_s if brief else 0.0)
    if task == "critique":
        alt = meta.get("hint")
        return {"verdict": "revise" if meta.get("hard_failed") else "accept", "best_alternative": alt,
                "reason": "mock critique", "requested_camera": None}
    # hypothesis
    if rp.get("rumination_likely"):
        lab = "ruminating_lying" if posture == "lying" else "ruminating_standing"
        cues = ["rhythmic jaw motion measured", f"periodicity {rp.get('periodicity_hz')} Hz"]
    elif zq.get("head_in_bunk_frac", 0) >= 0.4:
        lab, cues = "feeding", ["head inside feed bunk zone"]
    elif zq.get("head_in_waterer_frac", 0) >= 0.1:
        lab, cues = "drinking", ["head at waterer"]
    elif posture == "lying":
        lab, cues = "lying", ["recumbent posture"]
    elif speed >= 20:
        lab, cues = "walking", [f"speed {speed} px/s"]
    else:
        lab, cues = "standing", ["upright, little displacement"]
    return {"label": lab, "confidence": 0.8, "cues": cues, "alternatives": [["standing", 0.1]], "needs": None,
            "rationale": f"{lab}: " + "; ".join(cues)}
