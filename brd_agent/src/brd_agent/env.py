"""Environment/.env handling. One boolean (USE_LOCAL_VLM) selects the provider; the rest are connection details.
Real environment variables override values in the .env file. Secrets are never logged (see Settings.describe)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

_TRUE = {"1", "true", "yes", "on", "y", "t"}


class ConfigError(Exception):
    pass


def parse_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in _TRUE


def parse_dotenv(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        k, sep, v = line.partition("=")
        if not sep:
            continue
        v = v.strip()
        if v[:1] in ("'", '"') and v[-1:] == v[:1] and len(v) >= 2:
            v = v[1:-1]
        elif v.startswith("#"):
            v = ""
        elif " #" in v:
            v = v.split(" #")[0].strip()
        out[k.strip()] = v
    return out


def _opt_float(v: Optional[str]) -> Optional[float]:
    if v is None or v.strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        raise ConfigError(f"expected a number, got {v!r}")


@dataclass(frozen=True)
class Settings:
    use_local_vlm: bool
    openai_api_key: str
    openai_model: str
    openai_base_url: str
    openai_image_detail: str
    openai_temperature: Optional[float]
    openai_max_tokens: int
    local_base_url: str
    local_model: str
    local_api_key: str
    local_guided_json: bool
    store_dir: str

    @property
    def provider(self) -> str:
        return "local" if self.use_local_vlm else "openai"

    def describe(self) -> str:
        """Safe one-liner for logs: never contains the key."""
        if self.use_local_vlm:
            return f"provider=local model={self.local_model or '(unset)'} url={self.local_base_url}"
        k = self.openai_api_key
        masked = f"...{k[-4:]}" if len(k) >= 8 else ("(set)" if k else "(missing)")
        return f"provider=openai model={self.openai_model or '(unset)'} key={masked} url={self.openai_base_url}"


def load_settings(env_path: str | Path = ".env", environ: Optional[Mapping[str, str]] = None) -> Settings:
    env = dict(os.environ if environ is None else environ)
    m = {**parse_dotenv(env_path), **env}          # real env wins over the file
    g = lambda k, d="": (m.get(k) if m.get(k) is not None else d).strip()
    return Settings(
        use_local_vlm=parse_bool(m.get("USE_LOCAL_VLM"), False),      # default: OpenAI
        openai_api_key=g("OPENAI_API_KEY"),
        openai_model=g("OPENAI_MODEL"),
        openai_base_url=g("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_image_detail=g("OPENAI_IMAGE_DETAIL", "low"),
        openai_temperature=_opt_float(m.get("OPENAI_TEMPERATURE")),
        openai_max_tokens=int(g("OPENAI_MAX_COMPLETION_TOKENS", "800") or 800),
        local_base_url=g("LOCAL_VLM_BASE_URL", "http://localhost:8000/v1"),
        local_model=g("LOCAL_VLM_MODEL"),
        local_api_key=g("LOCAL_VLM_API_KEY"),
        local_guided_json=parse_bool(m.get("LOCAL_VLM_GUIDED_JSON"), False),
        store_dir=g("BRD_STORE_DIR", "data/store"),
    )
