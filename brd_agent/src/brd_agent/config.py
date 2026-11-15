from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class BudgetCfg(BaseModel):
    max_tool_calls: int = 8
    max_vlm_calls: int = 5          # plan + hypothesis + needs-round hypothesis + critique + revised hypothesis


class GateCfg(BaseModel):
    conf_threshold: float = 0.60
    entropy_threshold: float = 1.20
    pair_gap: float = 0.25
    rare_conf: float = 0.75
    min_id_conf: float = 0.60
    night_vis: float = 0.40
    night_conf: float = 0.75          # stricter confidence needed to keep a tier-1 label at night
    abstain_vis: float = 0.08


class VerifierCfg(BaseModel):
    min_bunk_frac_feeding: float = 0.30
    min_waterer_frac_drinking: float = 0.10
    walk_speed_px_s: float = 20.0
    still_speed_px_s: float = 6.0
    # tracking input measures speed in m/s (world coordinates); run_db swaps these in when the store says speed_unit=m_s
    walk_speed_m_s: float = 0.4         # ASSUMPTION from the data: fused walking median ~0.7 m/s, standing/feeding/lying ~0.03-0.12 m/s
    still_speed_m_s: float = 0.2
    min_visible: float = 0.15
    premature_abstain_vis: float = 0.60
    rumination_min_jaw: float = 0.35
    rumination_hz: tuple[float, float] = (0.6, 1.4)
    min_exemplar_sim: float = 0.50
    min_id_conf: float = 0.60
    soft_penalty: float = 0.75
    bout_break_min: int = 8      # ASSUMPTION: a bout at least this long is "established"; leaving it is flagged softly


class HistoryCfg(BaseModel):
    enabled: bool = True
    lookback_min: int = 30          # run-length history shown to the agent
    bout_lookback_min: int = 180    # how far back to measure the current bout
    since_lookback_min: int = 240   # "minutes since last feeding/drinking/..." horizon
    lookahead_min: int = 2          # next-minute TIER-1 guesses (offline only). Set 0 for live/streaming operation.


class NightCfg(BaseModel):
    brightness_thresh: float = 70.0     # ASSUMPTION: mean luminance below this => night (tune on your frames)
    ir_saturation_thresh: float = 0.04  # ASSUMPTION: near-grayscale frames are infrared-like
    max_confidence: float = 0.85        # cap on final confidence for night minutes (less visual evidence)
    min_visible: float = 0.30           # stricter visibility floor at night
    clahe_clip: float = 3.0
    gamma: float = 0.6
    needs_measurement: list[str] = ["grooming", "drinking", "social_interaction"]   # fine labels that need a measured cue at night


class VerifyCfg(BaseModel):
    audit_rate: float = 0.05            # fraction of otherwise-confident minutes sent to the VLM agent as a random audit
    audit_seed: int = 13
    night_audit_rate: float = 0.25      # night minutes are less reliable, so audit more of the confident ones
    verify_all: bool = False            # True = every minute goes through the agent (cost: ~10x more VLM calls)
    find_missing: bool = True           # try to recover cow-minutes with no YOLO record
    rumination_screen: bool = True      # measure jaw motion for confident standing/lying minutes (the commonest hidden YOLO miss)
    screen_seconds: float = 20.0
    expected_cows: int = 10


class CalmCfg(BaseModel):
    """Skip the VLM (and the video decode) for cows that are not doing anything: stable label, high confidence, fully observed,
    stationary, cameras agree. One minute per bout is verified (the anchor); the minutes after it inherit that verified result
    until `recheck_every_min`. Applies only to tracking input, which carries the stability metrics."""
    enabled: bool = True
    min_stability: float = 0.98         # share of the minute's seconds that carry the dominant label
    min_conf: float = 0.90              # lowest fused per-second confidence in the minute
    min_obs_frac: float = 0.95          # share of the minute's seconds actually observed (not gap-filled / not visible)
    max_speed_p90: float = 0.25         # m/s, 90th percentile of the fused per-second speed
    min_id_conf: float = 0.80
    recheck_every_min: int = 10         # a calm bout is re-verified at least this often
    labels: list[str] = ["lying", "standing", "feeding"]
    blocking_flags: list[str] = ["camera_conflict", "label_unstable", "gap_fill", "not_visible", "duplicate_id", "id_unresolved"]


class VideoCfg(BaseModel):
    root: Optional[str] = None          # folder with the .avi files
    pen_map: Optional[str] = None       # optional YAML with zone polygons per camera
    camera_alias: dict[str, str] = {"cam1": "front", "cam2": "top", "cam3": "back"}   # tracking camera id -> internal slot (cam2 = Front Camera 2 / bunk uses the 'top' slot)
    camera_map: Optional[dict[str, str]] = None   # folder name -> back|front|top (default: sorted folders -> back, front, top)
    record_hz: Optional[float] = None   # None = infer from the YOLO records (median spacing); set to override
    recovered_visibility: float = 0.5   # ASSUMPTION: visibility credited to a cow found again by the detector
    day_origin: str = "midnight"        # midnight | first_file_start  (how day/minute indices are derived from timestamps)
    crop_pad: float = 0.15
    head_frac: float = 0.35             # ASSUMPTION: head region = top 35% of the cow box
    frame_max_side: int = 1280


class VLMCfg(BaseModel):
    """Non-secret, provider-independent settings. Connection details (key/model/URL) live in .env."""
    temperature: float = 0.0        # local provider only; OpenAI temperature comes from OPENAI_TEMPERATURE
    max_tokens: int = 600           # local provider only; OpenAI uses OPENAI_MAX_COMPLETION_TOKENS
    timeout_s: int = 120
    max_image_side: int = 768
    cache_dir: Optional[str] = "cache/vlm"


class AgentConfig(BaseModel):
    prompt_version: str = "v3"
    planner: str = "llm"
    blind_first_look: bool = True
    critique_below: float = 0.70
    max_images: int = 6
    behaviors_path: str = "configs/behaviors.yaml"
    budget: BudgetCfg = Field(default_factory=BudgetCfg)
    gate: GateCfg = Field(default_factory=GateCfg)
    verifier: VerifierCfg = Field(default_factory=VerifierCfg)
    vlm: VLMCfg = Field(default_factory=VLMCfg)
    history: HistoryCfg = Field(default_factory=HistoryCfg)
    night: NightCfg = Field(default_factory=NightCfg)
    verify: VerifyCfg = Field(default_factory=VerifyCfg)
    video: VideoCfg = Field(default_factory=VideoCfg)
    calm: CalmCfg = Field(default_factory=CalmCfg)


def load_config(path: Optional[str | Path] = None) -> AgentConfig:
    if path is None:
        return AgentConfig()
    with open(path, encoding="utf-8") as f:
        return AgentConfig(**(yaml.safe_load(f) or {}))
