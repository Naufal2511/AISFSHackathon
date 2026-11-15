"""Prompt loading/rendering. Prompts are versioned files (prompts/<version>/<name>.md) with {{var}} placeholders.
The version is part of the VLM cache key and of every trace; bump it whenever a prompt or behaviors.yaml changes."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from .schemas import TAGS

PROMPT_DIR = Path(__file__).parent / "prompts"


@lru_cache(maxsize=None)
def _load(version: str, name: str) -> str:
    return (PROMPT_DIR / version / f"{name}.md").read_text(encoding="utf-8")


def render(version: str, name: str, **vars) -> str:
    text = _load(version, name)
    for k, v in vars.items():
        text = text.replace("{{" + k + "}}", str(v))
    left = re.findall(r"\{\{(\w+)\}\}", text)
    if left:
        raise KeyError(f"unfilled placeholders in prompt '{name}': {left}")
    return text


def behavior_texts(behaviors_path: str | Path) -> tuple[str, str]:
    d = yaml.safe_load(Path(behaviors_path).read_text(encoding="utf-8"))
    defs = "\n".join(f"- {k}: {v['definition']}" for k, v in d["classes"].items())
    rules = "\n".join(f"- {r}" for r in d.get("tie_break_rules", []))
    return defs, rules


def system_prompt(version: str, behaviors_path: str | Path) -> str:
    defs, rules = behavior_texts(behaviors_path)
    tags = "\n".join(f"- {k}: {v['desc']}" + (" [clinical screening cue]" if v["clinical"] else "") for k, v in TAGS.items())
    return render(version, "system", behavior_definitions=defs, tie_break_rules=rules, tag_vocabulary=tags)
