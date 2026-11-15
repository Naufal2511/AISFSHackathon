"""Do not spend the VLM (or a 4K video decode) on cows that are not doing anything.

A cow-minute is CALM when the tracking metrics say nothing happened and nothing is in doubt:
  * one label all minute (stability), every second confidently labelled (min_conf),
  * every second actually observed (not gap-filled, not "not visible"),
  * the cameras agree, the identity is solid,
  * stationary (90th-percentile speed below a threshold) and a resting/feeding label.
One minute per calm bout is verified normally (the ANCHOR: rule checks, jaw-motion screen, audits, full agent if flagged). The calm
minutes that follow inherit the anchor's verified result until the bout ends, anything changes, or `recheck_every_min` passes.
Every inherited minute records why it was not re-verified. Trade-off: a calm minute between two anchors is not looked at on its own;
`--verify-all` turns this off.
"""
from __future__ import annotations

from typing import Optional

from .config import AgentConfig, CalmCfg
from .schemas import Brief, Decision

USABLE_REASONS = {"keep_tier1", "agent_accept", "agent_revised"}


def calm_metrics_present(row: dict) -> bool:
    return row.get("stability") is not None and row.get("speed_p90") is not None


def is_calm(row: dict, brief: Brief, c: CalmCfg) -> tuple[bool, str]:
    """(calm?, why/why not)."""
    if not calm_metrics_present(row):
        return False, "no stability metrics (not tracking input)"
    label = brief.top_label
    if label not in c.labels:
        return False, f"label '{label}' is not a resting/feeding label"
    flags = set(brief.yolo_flags)
    hit = sorted(flags & set(c.blocking_flags))
    if hit:
        return False, "flags: " + ",".join(hit)
    if row["stability"] < c.min_stability:
        return False, f"label switched within the minute (stability {row['stability']:.2f})"
    if (row.get("min_conf") or 0.0) < c.min_conf:
        return False, f"confidence dipped to {row.get('min_conf')}"
    if (row.get("obs_frac") if row.get("obs_frac") is not None else 1.0) < c.min_obs_frac:
        return False, f"only {row.get('obs_frac'):.2f} of the seconds were observed"
    if row["speed_p90"] > c.max_speed_p90:
        return False, f"moving (p90 speed {row['speed_p90']:.2f})"
    if brief.id_conf < c.min_id_conf:
        return False, f"identity confidence {brief.id_conf:.2f}"
    if brief.is_night:
        return False, "night"
    return True, "stable, stationary, fully observed, cameras agree"


class CalmChain:
    """Per-cow state while walking its minutes in time order."""

    def __init__(self):
        self.anchor: Optional[dict] = None

    def try_carry(self, brief: Brief, row: dict, cfg: AgentConfig) -> Optional[Decision]:
        c = cfg.calm
        a = self.anchor
        if not c.enabled or cfg.verify.verify_all or a is None or brief.task != "verify":
            return None
        ok, why = is_calm(row, brief, c)
        t = row["t_abs"]
        if not ok or t != a["last_t"] + 1 or brief.top_label != a["label"] or a["carried"] >= c.recheck_every_min - 1:
            return None
        ad: Decision = a["d"]
        a["last_t"], a["carried"] = t, a["carried"] + 1
        changed = ad.correction == "behavior_corrected"
        why_txt = (f"calm bout: {brief.top_label} all minute (stability {row['stability']:.2f}, min confidence {row.get('min_conf')}, "
                   f"speed p90 {row['speed_p90']:.2f}, {row.get('obs_frac', 1.0):.2f} observed, cameras agree). Not re-verified: it inherits the "
                   f"verified result of minute {a['minute']} ({ad.correction}: {a['label']} -> {ad.behavior}); re-checked every {c.recheck_every_min} min.")
        return Decision(
            cow_id=brief.cow_id, day=brief.day, minute=brief.minute, behavior=ad.behavior,
            confidence=round(min(ad.confidence, brief.top_prob) * 0.98, 4), source=ad.source, tier=1, reason="stable_carry",
            tier1_label=brief.top_label, flags=["stable_carry"], is_night=False, correction="behavior_corrected" if changed else "confirmed",
            rationale=why_txt,
            actions=[{"seq": 1, "actor": "gate", "action": "stable_carry",
                      "args": {"anchor_minute": a["minute"], "carried_so_far": a["carried"], "recheck_every_min": c.recheck_every_min},
                      "reason": "the cow is not doing anything (see metrics): spending video decoding or a VLM call here would only repeat the "
                                "anchor minute's verified answer",
                      "outcome": f"inherited '{ad.behavior}' from minute {a['minute']}", "source": ""}])

    def observe(self, brief: Optional[Brief], row: dict, d: Decision, cfg: AgentConfig) -> None:
        """Called after a minute was processed the normal way: decide whether it can anchor the following calm minutes."""
        if d.reason == "stable_carry":
            return
        c = cfg.calm
        if not c.enabled or brief is None or brief.task != "verify":
            self.anchor = None
            return
        ok, _ = is_calm(row, brief, c)
        usable = (d.reason in USABLE_REASONS and d.correction in ("confirmed", "behavior_corrected") and d.behavior != "other_not_visible"
                  and not d.is_night and "over_budget_unverified" not in d.flags)
        self.anchor = ({"label": brief.top_label, "last_t": row["t_abs"], "minute": row["minute"], "d": d, "carried": 0}
                       if ok and usable else None)
