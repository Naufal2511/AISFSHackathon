# Saved agent reasoning traces

Picked automatically from this run's BRD validation-agent traces (pipeline/deliverables.py `pick_traces`).

## success: COW_09 at 2026-11-08 14:22

Why this one: the agent changed or confirmed the stage 1 label from what it saw in the frames.

Result: **walking** (verified, confidence 0.96; agent_accept).

Trace: [success_COW_02_d1_m0862.md](success_COW_02_d1_m0862.md)

## failure: COW_03 at 2026-11-07 13:33

Why this one: the agent could not settle the minute and kept the stage 1 label, marked unverified.

Result: **standing** (unverified, confidence 0.23; unresolved_fallback_tier1).

Trace: [failure_COW_01_d1_m0813.md](failure_COW_01_d1_m0813.md)

## night: COW_10 at 2026-11-08 18:05

Why this one: an agent check on infrared night footage.

Result: **other / not visible** (not_visible, confidence 0.85; agent_accept).

Trace: [night_COW_05_d1_m1085.md](night_COW_05_d1_m1085.md)
