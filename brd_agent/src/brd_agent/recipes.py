"""Behavior evidence recipes: for each behavior, which tools (and whether DB history) are needed to confirm it.
Used by (a) the planner guardrail (a candidate label's required tools can never be skipped), (b) the second look,
(c) plan-prompt hints. Recipes encode *what evidence is needed*, not biological thresholds: numbers stay in agent.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import Brief, ToolCall


@dataclass(frozen=True)
class Recipe:
    behavior: str
    must_run: tuple[tuple[str, dict], ...]      # tools that must run when this label is a top-2 candidate / is proposed
    uses_history: bool                          # also run cow_history (the label has duration structure)
    note: str                                   # one-line hint shown to the planner


RECIPES: dict[str, Recipe] = {r.behavior: r for r in [
    Recipe("ruminating_lying", (("rumination_probe", {}), ("track_history", {})), True,
           "Rumination comes in bouts lasting many minutes. Stills cannot show chewing: run rumination_probe; cow_history shows whether she is already in a lying/rumination bout."),
    Recipe("ruminating_standing", (("rumination_probe", {}), ("track_history", {})), True,
           "Same as rumination while lying but upright; confirm posture with track_history and jaw cycling with rumination_probe."),
    Recipe("feeding", (("zone_query", {}),), True,
           "Needs her head at the bunk (zone_query head_in_bunk_frac). cow_history shows whether a feeding bout is ongoing or just ended."),
    Recipe("drinking", (("zone_query", {}),), True,
           "Short events: need her head at the waterer (zone_query) and a head zoom. cow_history helps tell a drink from standing near the water."),
    Recipe("walking", (("track_history", {}),), True,
           "Needs real displacement (track_history), not one blurry frame. cow_history shows whether she just stood up or is travelling between zones."),
    Recipe("lying", (("track_history", {}),), True,
           "Confirm recumbent posture with track_history; cow_history shows how long she has been down."),
    Recipe("standing", (("track_history", {}), ("zone_query", {})), False,
           "Default for upright and still; make sure she is not at the bunk/waterer or chewing."),
    Recipe("grooming", (("get_burst", {}),), False,
           "Needs motion: head turned to flank/legs or rubbing a fixture; zoom the head and read the burst."),
    Recipe("social_interaction", (("neighbors", {}),), False,
           "Needs contact with another cow (neighbors contact seconds) and a filmstrip to see licking, pushing or mounting."),
    Recipe("other_not_visible", (("best_camera", {}),), False,
           "Only after checking that no camera sees her well enough."),
]}


def required_calls(label: str, include_history: bool = True) -> list[ToolCall]:
    r = RECIPES.get(label)
    if not r:
        return []
    calls = [ToolCall(tool=t, args=dict(a)) for t, a in r.must_run]
    if include_history and r.uses_history:
        calls.append(ToolCall(tool="cow_history"))
    return calls


def recipe_hints(b: Brief) -> str:
    seen, out = set(), []
    for lab, _ in b.tier1_top[:2]:
        r = RECIPES.get(lab)
        if r and lab not in seen:
            seen.add(lab)
            out.append(f"- {lab}: {r.note}")
    return "\n".join(out) or "- (none)"
