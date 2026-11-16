# Submission results: 48 h, 10 cows

The final run covers **11-07 00:04 to 11-09 00:01** (48.0 h, 3 cameras). It joins 33 chunks: 8 laptop chunks for Day 1 and 25 one-hour HPRC chunks for Day 2. They were stitched by `stitch_shares.py`, then scored by `scripts/run_level2.py`.

These files contain event data: the evidence crops, clips and trace images are cut from the event video. **Keep this repo private. Share these files only within the team, unless the organizers say otherwise.**

## Open first

| File | What it is |
|---|---|
| `level2/index.html` | BRD risk ranking with reasons, confidence and a card per flagged cow |
| `level2/brd_risk.csv` | The same ranking as a table |
| `stitched/viewer.html` | 48 h ethogram per cow, the camera joins, time budgets and the id map (self-contained) |

## BRD risk (Level 2)

| Rank | Cow | risk_score | p_top3 | Confidence |
|---|---|---|---|---|
| 1 | COW_02 | 0.372 | 0.92 | high |
| 2 | COW_01 | 0.325 | 0.67 | medium |
| 3 | COW_10 | 0.259 | 0.88 | high |
| 4-10 | the rest | 0.10 to 0.19 | | |

- No cow showed a visible BRD sign in the agent's review. The top 3 are flagged on behavior only: slow to reach the bunk after feed calls, short bunk visits, and less feeding than the herd at the same hours.
- **risk_score is an index for ranking, not a probability.** Each cow gets its own score: a herd-typical cow is about 0.10, and the scores do not add up to 1.
- **p_top3** is the share of time-block resamples in which the cow ranks in the top 3. Across cows it adds up to 3.
- Method: `docs/LEVEL2_METHOD.md`.

| File | What it is |
|---|---|
| `level2/cards/COW_*.html` | Per-cow card: indicator chart, two 12 s clips, and the evidence the agent cited |
| `level2/indicators.csv` | Each indicator, per cow |
| `level2/contributions.csv` | How each indicator moved the score: value, herd reference, z, weight, contribution |
| `level2/features_hourly.csv`, `features_summary.csv` | Per-hour and summary features |
| `level2/agent/traces/COW_*.json` | The risk agent's tool calls, findings and self-check for every cow |
| `level2/evidence/` | `facts.jsonl` (each cited fact) and `img/` (the frame crops the agent looked at) |
| `level2/risk_meta.json` | Time span, runs used, model, and where each clip came from |

## Stitched behavior (Level 1)

| File | What it is |
|---|---|
| `stitched/final/activity.csv` | Behavior bouts per cow (start, end, behavior) |
| `stitched/final/activity_minute.csv` | One row per cow per minute: behavior, confidence, cameras used, and the agent's rationale |
| `stitched/final/time_budgets.csv` | Minutes and share of time per behavior, per cow per day |
| `stitched/final/actions.csv` | Per-detection actions (55 MB) |
| `stitched/final/traces/` | Example reasoning traces from the identity and behavior agents: one success, one failure, one at night |
| `stitched/id_map.csv` | How each chunk's local track ids were mapped to COW_01..COW_10, and the evidence for each join (box overlap, position or guess) |
| `stitched/stitch_manifest.json` | Chunk order, time spans and the source of each chunk |

The id map flags joins that are uncertain: `overlap_split`, `position` and `guess`. The viewer shows them in the join panel.
