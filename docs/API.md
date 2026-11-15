# Pen 6 API: for the behavior model and the validation agent

Everything the tracking side produces is served per processed time slot (a **run**), and the
downstream steps send their results back through the same API. Columns are defined in
[DATA_CONTRACT.md](DATA_CONTRACT.md).

```
tracking + identity + world state ──► observations, cow_state, frames, crops, agent tools
                                            │
         behavior model ◄───────────────────┘   POST /behavior  (per-box p_* )  ──► fused per cow per second
                                                                                     ──► behavior_model/activity.csv
         validation agent ◄── cow_state, tools, frames          POST /verdicts  ──► final/activity.csv, time_budgets.csv
```

## Start the server

```bash
python -m src.api.server --port 8800                          # this machine only
python -m src.api.server --host 0.0.0.0 --port 8800 --token SECRET   # team network (token required)
```

Runs are folders under `outputs/` that contain `contract/observations.csv`. Create or refresh one with
`python scripts/run_level1.py --records <tracking_records.csv> --out outputs/level1_<slot> --skip-agent`.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{"ok": true}` |
| GET | `/runs` | available runs |
| GET | `/runs/{run}/manifest` | span, cows, which outputs exist |
| GET | `/runs/{run}/cows` | `COW_01` … |
| GET | `/runs/{run}/observations?t0&t1&cow_id&camera&limit&offset` | per box per second per camera |
| GET | `/runs/{run}/cow_state?t0&t1&cow_id&limit&offset` | per cow per second (fused position, visibility, zone, nearest cow, anomalies, baseline behavior) |
| GET | `/runs/{run}/frame.jpg?camera&t&boxes=1&width=1280` | full frame (optionally with IDs drawn) |
| GET | `/runs/{run}/crop.jpg?camera&t&cow_id=…` (or `track_id`) `&pad&size` | crop of one box |
| GET | `/runs/{run}/tools` | agent tools and arguments |
| GET | `/runs/{run}/tools/{name}?arg=value…` | `{text, images:[base64 jpeg], data}` |
| POST | `/runs/{run}/behavior?model=NAME` | upload per-box predictions (CSV or JSON) → fused outputs |
| POST | `/runs/{run}/verdicts` | upload validation verdicts (CSV or JSON) → final outputs |
| GET | `/runs/{run}/activity?stage=best\|final\|model\|baseline` | per cow per minute |
| GET | `/runs/{run}/time_budgets?stage=…` | per cow per day |

Paged tables return `{"total", "offset", "count", "rows": [...]}` (`limit` ≤ 5000 per page; the
Python client pages automatically). Times are ISO wall clock (`2026-11-07T16:01:00`) or epoch
seconds. With a token: header `Authorization: Bearer <token>`.

### Agent tools (same ones the supervisor agent uses)
`memory, scene, cow_strip, track_strip, compare_to_cards, all_views, references, trajectory,
who_is_near, behavior, cow_state, visibility, best_view, trajectory_map, anomalies, chew_motion`.
List arguments are passed as JSON, e.g. `cow_ids=["COW_02","COW_03"]`.

## Behavior model → `POST /runs/{run}/behavior`

Send one row per box you classified. Required: `wall_time, camera, track_id` (+ `track_piece`
if present in observations) and all 10 probabilities:
`p_lying, p_standing, p_walking, p_feeding, p_drinking, p_rum_lying, p_rum_standing, p_grooming, p_social, p_other`.
Easiest: take `observations`, fill those columns, send it back. Rows are matched to boxes,
probabilities are renormalised, and fused per cow per second (weighted by detection confidence
and visibility). Seconds where a known cow has no box become `other / not visible`.
Output: `outputs/<run>/behavior_model/{activity.csv, time_budgets.csv, behavior_timeline_1s.csv, behavior_bouts.csv}`.

## Validation agent → `POST /runs/{run}/verdicts`

`cow_id, t0, t1, behavior_in, verdict (confirm|override|uncertain), behavior_out, confidence, evidence, trace_file`.
Overrides replace the behavior for those seconds; confirms / uncertain are recorded in `evidence`.
Applied on top of the behavior model's output (or the baseline classifier if no model output yet).
Output: `outputs/<run>/final/{activity.csv, time_budgets.csv, ...}` = the Level 1 deliverables.

## Python

```python
from src.api.client import Pen6Client          # over HTTP
c = Pen6Client("http://127.0.0.1:8800")
obs = c.observations("level1_d1_1600_h", t0="2026-11-07T16:01:00", t1="2026-11-07T16:02:00", camera="cam1")
img = c.crop("level1_d1_1600_h", "cam1", "2026-11-07T16:01:10", cow_id="COW_03")      # JPEG bytes
view = c.tool("level1_d1_1600_h", "best_view", cow_id="COW_03", time="16:01:10", purpose="head")
c.submit_behavior("level1_d1_1600_h", predictions_df, model="beh-yolo-v1")
c.submit_verdicts("level1_d1_1600_h", verdicts_df)
final = c.activity("level1_d1_1600_h", stage="final")

from src.api.flow import Pen6Run                # in-process, same methods (no server)
run = Pen6Run("outputs/level1_d1_1600_h")
```

## curl

```bash
curl -s localhost:8800/runs
curl -s "localhost:8800/runs/level1_d1_1600_h/cow_state?cow_id=COW_02&t0=2026-11-07T16:01:00&t1=2026-11-07T16:01:05"
curl -s -o crop.jpg "localhost:8800/runs/level1_d1_1600_h/crop.jpg?camera=cam1&t=2026-11-07T16:01:10&cow_id=COW_03"
curl -s -X POST -H "Content-Type: text/csv" --data-binary @predictions.csv "localhost:8800/runs/level1_d1_1600_h/behavior?model=beh-yolo-v1"
```

Event data rule: the server binds to this machine by default; sharing it on a network requires a
token, and the video / outputs never go to GitHub.
