"""Layer-1 verification: deterministic checks of a hypothesis against measurements (not against the VLM).
hard failure  -> the label is contradicted by evidence: trigger critique / second look / fallback.
soft failure  -> suspicious or unverified: multiply confidence by soft_penalty and add a flag."""
from __future__ import annotations

import re

from .config import NightCfg, VerifierCfg
from .schemas import LYING_LABELS, UPRIGHT_LABELS, Brief, CheckResult, Hypothesis, VerifierReport

CHEW_WORDS = re.compile(r"\b(chew|chewing|jaw|cud|rumin\w*)\b", re.I)
WALK_WORDS = re.compile(r"\b(walk\w*|moving|striding|travel\w*)\b", re.I)
# "no jaw motion", "not chewing", "without visible chewing", "cannot see ... jaw": a NEGATED cue is not a claim of that cue
NEGATION = re.compile(r"\b(no|not|non|without|absent|absence of|never|cannot|can't|isn't|aren't|nor|lack of|lacking|unable to see)\b"
                      r"(?:\W+\w+){0,4}?\W+(?:(?:chew\w*|jaw\w*|cud|rumin\w*|walk\w*|moving|striding|travel\w*)\b\W*)+", re.I)


def _affirmed(text: str) -> str:
    """Drop the part of each cue that is negated, so only cues the model actually CLAIMS are checked against measurements."""
    return NEGATION.sub(" ", text)
IMPLAUSIBLE = {("lying", "drinking"), ("lying", "feeding")}
SAME_POSTURE_FAMILY = [{"lying", "ruminating_lying"}, {"standing", "ruminating_standing"}]   # refinement, not a bout break   # must pass through standing/walking first


class Verifier:
    NIGHT_MEASURE = {"grooming": "get_burst", "social_interaction": "neighbors", "drinking": "zone_query"}

    def __init__(self, cfg: VerifierCfg, night_cfg: NightCfg | None = None):
        self.c = cfg
        self.n = night_cfg or NightCfg()

    def run(self, h: Hypothesis, b: Brief, F: dict[str, dict]) -> VerifierReport:
        checks = [self.visibility(h, b), self.zone(h, b, F), self.posture(h, b, F),
                  self.rumination(h, F), self.cue_grounding(h, b, F), self.temporal(h, b), self.history(h, F), self.identity(b, F), self.night(h, b, F)]
        return VerifierReport(checks=[c for c in checks if c is not None])

    # --- individual checks ------------------------------------------------
    def visibility(self, h, b):
        vmax = b.max_visible
        if h.label == "other_not_visible":
            if vmax >= self.c.premature_abstain_vis:
                return CheckResult(name="premature_abstention", passed=False, severity="soft",
                                   detail=f"abstained but a camera sees her at {vmax:.2f}")
            return CheckResult(name="visibility", passed=True, severity="hard")
        floor = max(self.c.min_visible, self.n.min_visible) if b.is_night else self.c.min_visible
        ok = vmax >= floor
        return CheckResult(name="visibility", passed=ok, severity="hard",
                           detail=f"max visible fraction {vmax:.2f} (min {floor}{', night' if b.is_night else ''})")

    def zone(self, h, b, F):
        zq = F.get("zone_query", {})
        if h.label in ("feeding", "drinking") and zq.get("available") is False and not b.zone_fractions:
            return CheckResult(name="zone_unmeasured", passed=False, severity="soft",
                               detail=f"no zone data (pen map missing): '{h.label}' cannot be confirmed by position, only by images")
        zf = zq.get("zone_fractions") or b.zone_fractions
        if h.label == "feeding":
            v = zq.get("head_in_bunk_frac", zf.get("feed_bunk", 0.0))
            return CheckResult(name="zone_feeding", passed=v >= self.c.min_bunk_frac_feeding, severity="hard",
                               detail=f"head/cow in bunk fraction {v:.2f} (min {self.c.min_bunk_frac_feeding})")
        if h.label == "drinking":
            v = zq.get("head_in_waterer_frac", zf.get("waterer", 0.0))
            return CheckResult(name="zone_drinking", passed=v >= self.c.min_waterer_frac_drinking, severity="hard",
                               detail=f"head at waterer fraction {v:.2f} (min {self.c.min_waterer_frac_drinking})")
        return None

    def posture(self, h, b, F):
        th = F.get("track_history", {})
        posture = th.get("posture_mode") or b.posture or "unknown"
        speed = th.get("speed_mean", b.speed_px_s)
        d = f"posture={posture}, speed={speed:.1f}px/s"
        if h.label in LYING_LABELS:
            bad = posture == "standing" or speed >= self.c.walk_speed_px_s
            return CheckResult(name="posture_lying", passed=not bad, severity="hard", detail=d)
        if h.label == "walking":
            bad = posture == "lying" or speed < self.c.still_speed_px_s
            return CheckResult(name="posture_walking", passed=not bad, severity="hard", detail=d)
        if h.label in UPRIGHT_LABELS:
            return CheckResult(name="posture_upright", passed=posture != "lying", severity="hard", detail=d)
        return None

    def rumination(self, h, F):
        if not h.label.startswith("ruminating"):
            return None
        rp = F.get("rumination_probe")
        if not rp:
            return CheckResult(name="rumination_unverified", passed=False, severity="soft",
                               detail="no rumination_probe run; stills cannot show cud chewing")
        return CheckResult(name="rumination_support", passed=bool(rp.get("rumination_likely")), severity="hard",
                           detail=f"jaw={rp.get('jaw_motion_score')}, hz={rp.get('periodicity_hz')}")

    def cue_grounding(self, h, b, F):
        text = _affirmed(" ".join(h.cues))
        rp = F.get("rumination_probe") or F.get("get_burst")
        if CHEW_WORDS.search(text) and h.label != "other_not_visible":
            if rp is None:
                return CheckResult(name="cue_chewing_ungrounded", passed=False, severity="soft",
                                   detail="chewing cue claimed without any burst measurement")
            if float(rp.get("jaw_motion_score", 0.0)) < self.c.rumination_min_jaw:
                return CheckResult(name="cue_chewing_hallucinated", passed=False, severity="hard",
                                   detail=f"claims chewing but jaw_motion_score={rp.get('jaw_motion_score')}")
        th = F.get("track_history", {})
        if WALK_WORDS.search(text) and th and th.get("speed_mean", 99) < self.c.still_speed_px_s:
            return CheckResult(name="cue_walking_hallucinated", passed=False, severity="soft",
                               detail=f"claims movement but speed_mean={th.get('speed_mean')}")
        return None

    def temporal(self, h, b):
        if b.prev_label and b.next_label_tier1 and b.prev_label == b.next_label_tier1 != h.label:
            return CheckResult(name="temporal_blip", passed=False, severity="soft",
                               detail=f"isolated between {b.prev_label} and {b.next_label_tier1}")
        if b.prev_label and (b.prev_label, h.label) in IMPLAUSIBLE:
            return CheckResult(name="temporal_transition", passed=False, severity="soft",
                               detail=f"{b.prev_label} -> {h.label} without intermediate upright state")
        return None

    def history(self, h, F):
        """Soft only: leaving a long established bout is allowed (cows do get up) but is flagged unless the next
        minute's tier-1 guess confirms the change. Uses same-cow recent history only."""
        hs = F.get("cow_history") or {}
        cb = hs.get("current_bout")
        if not cb or cb.get("ends_minutes_ago", 99) > 1 or h.label == cb["label"]:
            return None
        if any(h.label in g and cb["label"] in g for g in SAME_POSTURE_FAMILY):
            return None
        if cb["minutes"] >= self.c.bout_break_min:
            nxt = hs.get("next_tier1") or []
            if nxt and nxt[0] == h.label:
                return None
            return CheckResult(name="breaks_long_bout", passed=False, severity="soft",
                               detail=f"{cb['minutes']} min of {cb['label']} just before; proposed {h.label} is not confirmed by the next minute")
        return None

    def night(self, h, b, F):
        """Soft: a fine-grained label at night must be backed by a measurement, not by appearance in an IR frame."""
        if not b.is_night or h.label not in self.n.needs_measurement:
            return None
        tool = self.NIGHT_MEASURE.get(h.label)
        got = F.get(tool) if tool else None
        if got and "_error" not in got:
            return None
        return CheckResult(name="night_unmeasured", passed=False, severity="soft",
                           detail=f"'{h.label}' proposed at night without a {tool} measurement")

    def identity(self, b, F):
        ce = F.get("compare_to_exemplar", {})
        if ce.get("available") is False:
            return CheckResult(name="id_unverified", passed=False, severity="soft", detail="no gallery: identity could not be checked")
        if ce.get("better_match"):
            return CheckResult(name="id_swap_suspected", passed=False, severity="soft",
                               detail=f"crop matches {ce['better_match']} better than {b.cow_id} (ranking {ce.get('ranking')})")
        sim = ce.get("similarity")
        if b.id_conf >= self.c.min_id_conf and (sim is None or sim >= self.c.min_exemplar_sim):
            return None
        verified = sim is not None and sim >= self.c.min_exemplar_sim
        return CheckResult(name="id_unverified", passed=verified, severity="soft",
                           detail=f"id_conf={b.id_conf:.2f}, exemplar_sim={sim}")
