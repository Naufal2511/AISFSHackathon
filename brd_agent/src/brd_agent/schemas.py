"""Typed contracts. Every input/output of the agent is one of these models (validated, serialisable)."""
from __future__ import annotations

import math
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

BEHAVIORS = [
    "lying", "standing", "walking", "feeding", "drinking",
    "ruminating_lying", "ruminating_standing", "grooming",
    "social_interaction", "other_not_visible",
]
CAMERAS = ("back", "front", "top")
LYING_LABELS = {"lying", "ruminating_lying"}
UPRIGHT_LABELS = {"standing", "walking", "feeding", "drinking", "ruminating_standing"}
Source = Literal["observed", "model_estimate", "assumed", "not_visible"]

# Optional annotation vocabulary. Tags NEVER replace the 10 required labels; they sit beside them
# (activity.csv is unchanged). `clinical` tags are unvalidated screening cues for Level 3, not diagnoses.
TAGS: dict[str, dict] = {
    "self_licking":        {"applies_to": ["grooming"], "clinical": False, "desc": "licking own flank/legs/body"},
    "rubbing_on_fixture":  {"applies_to": ["grooming"], "clinical": False, "desc": "rubbing/scratching against a pen fixture"},
    "scratching_hind_leg": {"applies_to": ["grooming", "standing"], "clinical": False, "desc": "scratching head/neck with a hind leg"},
    "licking_other":       {"applies_to": ["social_interaction"], "clinical": False, "desc": "licking another cow"},
    "head_butting":        {"applies_to": ["social_interaction"], "clinical": False, "desc": "head-to-head or head-to-body pushing"},
    "mounting":            {"applies_to": ["social_interaction"], "clinical": False, "desc": "mounting or being mounted"},
    "displaced_from_bunk": {"applies_to": ["social_interaction", "standing", "walking"], "clinical": False, "desc": "pushed away from the feed bunk by another cow"},
    "pacing":              {"applies_to": ["walking"], "clinical": False, "desc": "repetitive back-and-forth walking"},
    "walking_to_bunk":     {"applies_to": ["walking"], "clinical": False, "desc": "travelling toward the feed bunk"},
    "walking_to_water":    {"applies_to": ["walking"], "clinical": False, "desc": "travelling toward the waterer"},
    "at_bunk_not_eating":  {"applies_to": ["standing", "other_not_visible"], "clinical": False, "desc": "at the bunk, head up or sniffing, no intake seen"},
    "tongue_rolling":      {"applies_to": ["standing", "feeding", "ruminating_standing"], "clinical": False, "desc": "tongue rolling / oral stereotypy"},
    "lying_sternal":       {"applies_to": ["lying", "ruminating_lying"], "clinical": False, "desc": "upright chest-down lying"},
    "lateral_recumbency":  {"applies_to": ["lying"], "clinical": True, "desc": "lying flat on one side (notable if sustained)"},
    "head_tucked":         {"applies_to": ["lying", "ruminating_lying"], "clinical": False, "desc": "head turned back onto flank"},
    "stretching":          {"applies_to": ["standing", "walking", "lying"], "clinical": False, "desc": "stretching after rising"},
    "elimination":         {"applies_to": ["standing", "walking"], "clinical": False, "desc": "urinating or defecating"},
    "head_low":            {"applies_to": ["standing", "walking", "ruminating_standing"], "clinical": True, "desc": "head carried low, not at feed/water"},
    "ear_droop":           {"applies_to": [], "clinical": True, "desc": "ears drooping/asymmetric"},
    "nasal_discharge":     {"applies_to": [], "clinical": True, "desc": "visible nasal discharge"},
    "coughing":            {"applies_to": [], "clinical": True, "desc": "cough (needs a burst; stills cannot show it)"},
    "labored_breathing":   {"applies_to": [], "clinical": True, "desc": "visibly increased respiratory effort"},
    "drooling":            {"applies_to": [], "clinical": True, "desc": "visible salivation/drool"},
}


def norm_label(v: str) -> str:
    """'Ruminating (lying)' -> 'ruminating_lying'; 'other / not visible' -> 'other_not_visible'."""
    return re.sub(r"[^a-z]+", "_", str(v).lower()).strip("_")


def _clip01(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


def _norm_pairs(v: Any) -> list:
    out = []
    for item in v or []:
        try:
            if isinstance(item, dict):
                lab = item.get("label") or item.get("behavior")
                p = item.get("confidence", item.get("p", item.get("probability", 0.0)))
            else:
                lab, p = item[0], item[1]
            lab = norm_label(lab)
            if lab in BEHAVIORS:
                out.append((lab, _clip01(p)))
        except Exception:
            continue
    return out


# ----------------------------------------------------------------------------- INPUT
class Brief(BaseModel):
    """One cow-minute escalated to the agent. Produced by the tier-1 pipeline (tracker + fast classifier)."""
    cow_id: str
    day: int
    minute: int = Field(ge=0, le=1439)
    tier1_top: list[tuple[str, float]] = Field(description="Calibrated top-k [(label, prob)], descending")
    tier1_entropy: float = 0.0
    zone_fractions: dict[str, float] = Field(default_factory=dict)   # feed_bunk/waterer/resting/open over the minute
    speed_px_s: float = 0.0
    posture: Optional[Literal["lying", "standing", "unknown"]] = None
    visible_frac: dict[str, float] = Field(default_factory=dict)      # per camera 0..1
    id_conf: float = 1.0
    is_night: bool = False
    prev_label: Optional[str] = None            # final label of the previous minute
    next_label_tier1: Optional[str] = None      # tier-1 guess for the next minute
    camera_disagreement: bool = False
    task: Literal["verify", "find_missing"] = "verify"   # find_missing: YOLO produced no record for this cow-minute
    src_file: Optional[str] = None          # video file the representative YOLO record came from
    src_camera: Optional[str] = None
    src_offset_s: Optional[float] = None    # seconds from the start of src_file
    abs_time: Optional[str] = None          # ISO wall-clock time of that record
    yolo_flags: list[str] = Field(default_factory=list)   # e.g. duplicate_id, low_roster_count

    @field_validator("posture", mode="before")
    @classmethod
    def _posture(cls, v):
        return v if v in ("lying", "standing", "unknown") else None

    @field_validator("tier1_top", mode="before")
    @classmethod
    def _t1(cls, v):
        return _norm_pairs(v)

    @property
    def top_label(self) -> Optional[str]:
        return self.tier1_top[0][0] if self.tier1_top else None

    @property
    def top_prob(self) -> float:
        return self.tier1_top[0][1] if self.tier1_top else 0.0

    @property
    def max_visible(self) -> float:
        return max(self.visible_frac.values()) if self.visible_frac else 1.0


# ----------------------------------------------------------------------------- TOOLS / PLAN
class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""                  # WHY this action is taken: required from the planner, filled for guardrail actions
    source: str = "planner"           # planner | guardrail | second_look | needs | critic_request

    @field_validator("reason", "source", mode="before")
    @classmethod
    def _none_to_default(cls, v):
        return "" if v is None else str(v)


class Plan(BaseModel):
    rationale: str = ""
    steps: list[ToolCall] = Field(default_factory=list)


class ToolResult(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    reason: str = ""
    source: str = ""
    facts: dict[str, Any] = Field(default_factory=dict)     # JSON-serialisable measurements (OBSERVED)
    images: list[str] = Field(default_factory=list)         # file paths
    captions: list[str] = Field(default_factory=list)       # parallel to images
    error: Optional[str] = None


# ----------------------------------------------------------------------------- VLM OUTPUTS
class Hypothesis(BaseModel):
    label: str
    confidence: float = 0.0
    cues: list[str] = Field(default_factory=list)
    alternatives: list[tuple[str, float]] = Field(default_factory=list)
    needs: Optional[str] = None     # what extra evidence would resolve doubt (drives an extra evidence round)
    rationale: str = ""             # why this label: ties the cues to the evidence
    tags: list[str] = Field(default_factory=list)   # optional annotations from TAGS (never change the label)

    @field_validator("tags", mode="before")
    @classmethod
    def _tags(cls, v):
        out: list[str] = []
        for t in v or []:
            t = norm_label(t)
            if t in TAGS and t not in out:
                out.append(t)
        return out[:3]

    @field_validator("label", mode="before")
    @classmethod
    def _label(cls, v):
        v = norm_label(v)
        if v not in BEHAVIORS:
            raise ValueError(f"label '{v}' not in allowed behaviors")
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v):
        return _clip01(v)

    @field_validator("alternatives", mode="before")
    @classmethod
    def _alts(cls, v):
        return _norm_pairs(v)


class Critique(BaseModel):
    verdict: Literal["accept", "revise"]
    best_alternative: Optional[str] = None
    reason: str = ""
    requested_camera: Optional[str] = None

    @field_validator("best_alternative", mode="before")
    @classmethod
    def _alt(cls, v):
        if v in (None, "", "none", "null"):
            return None
        v = norm_label(v)
        return v if v in BEHAVIORS else None

    @field_validator("verdict", mode="before")
    @classmethod
    def _verdict(cls, v):
        return str(v).strip().lower()


# ----------------------------------------------------------------------------- VERIFIER / OUTPUT
class CheckResult(BaseModel):
    name: str
    passed: bool
    severity: Literal["hard", "soft"] = "soft"
    detail: str = ""


class VerifierReport(BaseModel):
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def hard_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "hard"]

    @property
    def soft_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "soft"]


class Decision(BaseModel):
    """Final per-cow-minute output; maps 1:1 onto the `minute_labels` table."""
    cow_id: str
    day: int
    minute: int
    behavior: str
    confidence: float
    source: Source = "model_estimate"
    tier: int = 2
    # agent_accept | agent_revised | unresolved_fallback_tier1 | budget_fallback | auto_abstain | keep_tier1 | error_fallback
    reason: str = "agent_accept"
    cameras_used: list[str] = Field(default_factory=list)
    tier1_label: Optional[str] = None
    n_tool_calls: int = 0
    n_vlm_calls: int = 0
    trace_id: Optional[str] = None
    flags: list[str] = Field(default_factory=list)   # e.g. id_unverified, soft:temporal_blip
    behavior_tags: list[str] = Field(default_factory=list)
    used_history: bool = False       # True when the cow_history tool contributed evidence
    tokens_in: int = 0
    tokens_out: int = 0
    provider: Optional[str] = None   # openai | local | mock (None for tier-1 passthrough)
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    rationale: str = ""              # human-readable WHY for the final label (and for any correction of YOLO)
    correction: Optional[str] = None # confirmed | behavior_corrected | abstained | recovered_miss | miss_not_found | unresolved
    is_night: bool = False
    id_verified: Optional[bool] = None
    id_suggested: Optional[str] = None   # another cow the crop matches better (flag only; never auto-applied)
    actions: list[dict] = Field(default_factory=list)   # every action taken, each with its reason (-> agent_actions table)
