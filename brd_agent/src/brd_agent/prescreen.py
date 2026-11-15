"""Rule pre-verification of EVERY YOLO result (no VLM): the agent does not get to skip a minute just because YOLO was confident.
Cheap measurements (track speed, zone fractions) are checked against the claimed label; contradictions, YOLO flags and a
random audit sample escalate to the full agent. Night minutes are audited at a higher rate because night detections are less reliable."""
from __future__ import annotations

import zlib

from .config import AgentConfig
from .schemas import Brief, Hypothesis
from .verifier import Verifier

UPRIGHT_STILL = {"standing", "feeding", "drinking", "grooming", "ruminating_standing", "social_interaction"}
NOT_ESCALATING = {"zone_unmeasured", "id_unverified", "night_unmeasured"}      # "could not measure" is not "contradicted"


def _frac(cow: str, day: int, minute: int, seed: int) -> float:
    return (zlib.crc32(f"{seed}:{cow}:{day}:{minute}".encode()) % 100000) / 100000.0


def prescreen(b: Brief, backend, cfg: AgentConfig, verifier: Verifier) -> tuple[list[str], list[dict]]:
    """Returns (escalation reasons, action records). Empty reasons = YOLO's label survived every cheap check."""
    reasons: list[str] = []
    acts: list[dict] = []
    label = b.top_label
    F: dict[str, dict] = {}
    try:
        F["track_history"] = backend.track_history(b.cow_id, b.day, b.minute, 90.0)
        if label in ("feeding", "drinking"):
            F["zone_query"] = backend.zone_info(b.cow_id, b.day, b.minute)
    except Exception as e:                                    # a measurement failure must not hide a minute from review
        reasons.append("prescreen_measurement_error")
        acts.append({"actor": "verifier", "action": "rule_verify", "args": {}, "reason": "measure speed/zone to test YOLO's label",
                     "outcome": f"measurement failed: {type(e).__name__}", "source": ""})
        return reasons, acts
    h = Hypothesis(label=label or "other_not_visible", confidence=b.top_prob, cues=[])
    rep = verifier.run(h, b, F)
    bad = [c for c in rep.checks if not c.passed and c.name not in NOT_ESCALATING]
    for c in bad:
        reasons.append(f"rule_{'contradiction' if c.severity == 'hard' else 'flag'}:{c.name}")
    th = F.get("track_history", {})
    if label in UPRIGHT_STILL and th.get("speed_mean", 0.0) >= 2 * cfg.verifier.walk_speed_px_s:
        reasons.append("rule_contradiction:moving_but_labelled_stationary")
        bad.append(None)
    if cfg.verify.rumination_screen and label in ("standing", "lying"):
        cam = b.src_camera or (max(b.visible_frac, key=b.visible_frac.get) if b.visible_frac else None)
        if cam:
            try:
                bm = backend.burst_motion(b.cow_id, b.day, b.minute, cam, cfg.verify.screen_seconds)
                jaw = float(bm.get("jaw_motion_score", 0.0))
                acts.append({"actor": "verifier", "action": "rumination_screen", "args": {"camera": cam, "seconds": cfg.verify.screen_seconds},
                             "reason": f"YOLO said '{label}'; resting cows are often ruminating, which a label of '{label}' hides, so measure jaw motion",
                             "outcome": f"jaw_motion_score={jaw:.2f}, periodicity={bm.get('periodicity_hz')}Hz", "source": ""})
                if bm.get("available", True) and jaw >= cfg.verifier.rumination_min_jaw:
                    reasons.append("rule_contradiction:jaw_motion_in_" + label)
            except Exception as e:
                acts.append({"actor": "verifier", "action": "rumination_screen", "args": {}, "reason": "measure jaw motion for a resting label",
                             "outcome": f"measurement failed: {type(e).__name__}", "source": ""})
    for f in b.yolo_flags:
        if f in ("duplicate_id", "camera_conflict", "label_unstable", "id_unresolved"):
            reasons.append(f"yolo_flag:{f}")
    outcome = "no contradiction found" if not reasons else "; ".join(reasons)
    acts.append({"actor": "verifier", "action": "rule_verify", "args": {"label": label, "measured": list(F)},
                 "reason": "test YOLO's label against measured speed/zone/flags before trusting it, even though YOLO was confident",
                 "outcome": outcome, "source": ""})
    v = cfg.verify
    if v.verify_all:
        reasons.append("verify_all")
    else:
        rate = v.night_audit_rate if b.is_night else v.audit_rate
        if rate > 0 and _frac(b.cow_id, b.day, b.minute, v.audit_seed) < rate:
            reasons.append("night_audit_sample" if b.is_night else "audit_sample")
            acts.append({"actor": "verifier", "action": "audit_select",
                         "args": {"rate": rate}, "reason": "a random share of confident minutes is re-checked by the full agent, "
                         "so silent YOLO errors are measured rather than assumed absent", "outcome": "selected", "source": ""})
    return reasons, acts
