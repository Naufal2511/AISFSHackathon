# Submission results: 48 h, 10 cows

The final run covers **11-07 00:04 to 11-09 00:01** (48.0 h, 3 cameras). It joins 33 chunks: 8 laptop chunks for Day 1 and 25 one-hour HPRC chunks for Day 2. They were stitched by `stitch_shares.py`, then scored by `scripts/run_level2.py`. Level 3 (`live_watch.py`) turns that risk into alerts and keeps it current from new video.

These files contain event data: the evidence crops, clips and trace images are cut from the event video. **Keep this repo private. Share these files only within the team, unless the organizers say otherwise.**

## Open first

| File | What it is |
|---|---|
| [`SHOWCASE.md`](SHOWCASE.md) | **Everything on one page, with pictures** (renders on GitHub) |
| `index.html` | The same showcase with buttons to every full page (open in a browser after download) |
| `level2/index.html` | BRD risk ranking with reasons, confidence and a card per flagged cow |
| `level2/brd_risk.csv` | The same ranking as a table |
| `stitched/viewer.html` | 48 h ethogram per cow, the camera joins, time budgets and the id map (self-contained) |
| `level3/clip_1600/report.html` | Live early warning on one 5-min clip: every decision with its evidence, alert tiers, clips and the simulated email outbox |

## BRD risk (Level 2)

| Rank | Cow | risk_score | p_top3 | Confidence | Relevant Decision Clips |
|---|---|---|---|---|---|
| 1 | COW_02 | 0.372 | 0.92 | high | [`COW_02_1.mp4`](level2/cards/clips/COW_02_1.mp4) (feed latency) · [`COW_02_2.mp4`](level2/cards/clips/COW_02_2.mp4) (closest view) |
| 2 | COW_01 | 0.325 | 0.67 | medium | [`COW_01_1.mp4`](level2/cards/clips/COW_01_1.mp4) (bunk feeding) · [`COW_01_2.mp4`](level2/cards/clips/COW_01_2.mp4) (closest view) |
| 3 | COW_10 | 0.259 | 0.88 | high | [`COW_10_1.mp4`](level2/cards/clips/COW_10_1.mp4) (bunk visit) · [`COW_10_2.mp4`](level2/cards/clips/COW_10_2.mp4) (closest view) |
| 4-10 | the rest | 0.10 to 0.19 | | | *(Not flagged)* |

- No cow showed a visible BRD sign in the agent's review. The top 3 are flagged on behavior only: slow to reach the bunk after feed calls, short bunk visits, and less feeding than the herd at the same hours.
- **risk_score is an index for ranking, not a probability.** Each cow gets its own score: a herd-typical cow is about 0.10, and the scores do not add up to 1.
- **p_top3** is the share of time-block resamples in which the cow ranks in the top 3. Across cows it adds up to 3.
- Method: `docs/LEVEL2_METHOD.md`.

| File | What it is |
|---|---|
| [`level2/README.md`](level2/README.md) | Dedicated Level 2 guide with clip breakdowns and direct video links |
| [`level2/cards/clips/`](level2/cards/clips/) | The 6 decision evidence clips for COW_02, COW_01, COW_10 |
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

## Live early warning (Level 3)

`live_watch.py` watches new video, updates every cow's BRD risk and sends an alert with video evidence. Method: `docs/LEVEL3_LIVE.md`.

**Alert tiers from the 48 h risk above** (issued on start; each live window re-evaluates them):

| Tier | Rule | Cows |
|---|---|---|
| Pull and examine | a confirmed visible sign | none |
| Check on the next pen ride | top 3, in the top 3 in ≥ 60 % of resamples, medium/high confidence | COW_02, COW_01, COW_10 |
| Watch list | rank ≤ 5 or top 3 in ≥ 20 % of resamples | COW_09, COW_03, COW_04 |
| Normal | everyone else | the other 4 |

Each live window maps the cows to the 48 h ids and compares them with the herd in the same window. The VLM checks close crops for drooping ears, head carried low and nasal discharge, and every flag is checked again: rejected by tracker speed or the standing rule, or confirmed by a second look next to a herdmate. The window then updates the risk and sends alerts with a 12 s clip. Every alert and the daily BRD report are also written as emails (simulated, never sent).

| File | What it is |
|---|---|
| `level3/clip_1600/report.html` | **One 5-min clip end to end** (11-07 16:00, 3 cameras), the way the judges' clip runs. 7 cows matched to the 48 h ids. The VLM checked 6 crops: 3 "head low" flags were rejected by measurement (cows walking at 0.36–0.87 m/s) and 1 "nasal discharge" by the second look. Fresh 12 s clips of the check-list cows COW_02 and COW_01. |
| `level3/day2_replay/report.html` | **Three Day 2 hours fed in as if live** (11-08 12:28–15:23). COW_10 got a behavior alert: lying 76 % vs 22 % for the herd, alone 80 % vs 36 %. Flags rejected by measurement, and COW_03's low ear was not flagged because she was lying (the ear carries an ear tag). |
| `level3/*/alerts/*.html` | The alert cards a pen rider gets: what to do, the clip, the frame, the crops, why, risk |
| `level3/*/outbox/*.eml` | Simulated emails: one per alert plus the daily BRD report, evidence attached (they open in any mail client) |
| `level3/*/index.html` | Live dashboard: tier and risk per cow vs the 48 h baseline, trend, alert feed |
| `level3/README.md` | What each folder shows and how it was produced |
