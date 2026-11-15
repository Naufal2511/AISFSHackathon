# brd_agent: agentic behavior labeling (tier 2) with a file-based store (CSV + JSONL) and a switchable VLM

Selective, evidence-gathering second opinion on cow-minutes the cheap tier-1 pipeline is unsure about. Every finished
minute is **written back to its row in a CSV file in the store folder**, and the agent can read the same cow's recent history from it.

```
Student 1 tier-1 output --import--> store/minutes/<cow>_d<day>.csv (UPSTREAM cols)
                                          |
   per cow, in time order:  row -> Brief -> GATE -> keep_tier1 | auto_abstain | AGENT
   AGENT: PLAN -> ACT (tools incl. cow_history) -> HYPOTHESIZE (blind) -> VERIFY -> [SECOND LOOK -> CRITIQUE -> REVISE] -> RESOLVE
                                          |
                      write_decision -> same minutes CSV (AGENT cols) + runs/ + actions/ (JSONL audit)  ->  bouts -> activity.csv
```

## 0. Level 1 flow with the standalone YOLO process (read this first)

YOLO gives you, per record: **video file name + timestamp + labelled cow + behaviour (+ box)**, e.g. `Event20261107033930006.Avi`
(= 2026-11-07 03:39:30.006). The agent treats every YOLO result as a *claim to verify*:

```
YOLO CSV --import-yolo--> store/yolo/*.jsonl --aggregate--> store/minutes/*.csv (+ find_missing rows for cows YOLO skipped)
   for EVERY cow-minute:  night check (from the frame) -> gate -> RULE PRE-VERIFICATION (no VLM) -> full agent only if flagged/audited
   full agent: plan -> tools on the real video -> blind hypothesis -> rule verifier -> needs-driven round -> second look -> critique -> resolve
   result per minute: behavior, confidence, correction (confirmed | behavior_corrected | abstained | recovered_miss | miss_not_found | unresolved),
                      rationale, and EVERY action with its reason (table agent_actions, trace .md)
```

| Requirement | Where it is done |
|---|---|
| Verify YOLO genuinely | `prescreen.py` checks the label of *every* minute against measured speed/zone, a jaw-motion screen for confident standing/lying, YOLO flags (duplicate IDs, camera disagreement), plus a random audit (`verify.audit_rate`) and `--verify-all` |
| Rectify mistakes | the full agent re-measures on video (`VideoBackend`), a verifier can contradict the VLM, final `correction` says what changed |
| Rectify misses | cow-minutes YOLO skipped become `find_missing`: `recover_box` runs a detector (optional `--detector-weights`) and continues her last known position; else honestly `miss_not_found` |
| Cow identification | low ID confidence or a duplicate-ID flag forces `compare_to_exemplar`; a crop that matches another cow better sets `id_suggested` + flag `id_swap_suspected` (never auto-reassigned). Needs a gallery: `--exemplars <cow>/*.jpg` |
| Night | night is **measured from the frame** (brightness, saturation), re-gates the minute, forces `enhance_night` (CLAHE+gamma), stricter visibility, confidence cap, soft check that fine labels at night have a measurement, night audit rate, accuracy reported by light (`eval` -> `by_light`) |
| A reason for every action | every plan step carries a `reason`; guardrail/needs/second-look/critic actions state theirs; every row in `agent_actions` has actor, action, reason, outcome (also for passed-through minutes) |

Try it with no real data: `python scripts/make_demo_video.py demo/` then
`python -m brd_agent.dbcli --store demo/store import-yolo --yolo demo/yolo_output.csv --video-root demo/videos`, `aggregate --video-root demo/videos --cows C01,C02`,
`run --backend video --video-root demo/videos --vlm mock`, `export --out demo/out`. (The demo YOLO says C02 is "standing"; the clip shows 1 Hz head motion at night, and the agent corrects it.)

**ASSUMPTIONS to confirm against the real files** (nothing here was tested on real data): camera is not in the file name, so it comes from the parent folder
(unmapped folders -> back, front, top in sorted order; override with `video.camera_map: {folder_name: back}` in `configs/agent.yaml`); `timestamp` is seconds from file start, `HH:MM:SS(.fff)` or an absolute datetime;
day 1 starts at midnight of the first date seen (`video.day_origin`); box pixels refer to the video frame; posture is *not* inferred from box shape; zone checks need a pen map
(without one they are reported unavailable, not zero); night/jaw thresholds in `configs/agent.yaml` are placeholders to calibrate on labelled frames.
YOLO column names are matched through aliases in `yolo.py` (`ALIASES`).


## 0b. Tracking input (observations.csv + cow_state.csv) and skipping idle cows

`python -m brd_agent.dbcli --store S import-tracking --observations observations.csv --cow-state cow_state.csv` loads the tracking
pipeline's two files (re-importing is safe). `wall_time` (= file start from the file name + `offset_s`, checked on the sample) is the common
clock, so camera files with different names and start times line up; `file -> camera` is stored and used by the video index.
Per cow-minute it builds the tier-1 row from the fused state (top-k from mean per-second probabilities, speed in m/s, zone fractions,
stability, lowest confidence, observed fraction, flags from `anomaly`/`beh_evidence`) and the source pointer (best camera, file, offset)
from the boxes. Cows with no state in a minute of the window become `find_missing` rows.

**Not doing anything -> not verified** (`calm:` in `configs/agent.yaml`, `src/brd_agent/calm.py`). A minute is calm when one label holds all
minute, every second is confident and observed, the cameras agree, the identity is solid and the cow is stationary
(p90 speed < 0.25 m/s) with a lying/standing/feeding label. One minute per calm bout (the anchor) is verified normally; the following
calm minutes inherit its verified result (`reason=stable_carry`, `status=carried`, no video decode, no VLM) and re-check every
`recheck_every_min` (10). On the sample, 44% of cow-minutes are carried. `--verify-all` or `calm.enabled: false` turns it off.
Trade-off: a calm minute between two anchors is not looked at on its own (a cow that starts ruminating mid-bout is caught at the next anchor).

## 1. Choose the VLM provider (.env)
```bash
cp .env.example .env        # .env is git-ignored; real environment variables override it
```
| Variable | Meaning |
|---|---|
| `USE_LOCAL_VLM` | **`false` (default) = OpenAI**, `true` = local OpenAI-compatible server (e.g. vLLM). The only switch. |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | Required when OpenAI is selected. **`OPENAI_MODEL` has no default**: set the vision model you want. |
| `OPENAI_BASE_URL`, `OPENAI_IMAGE_DETAIL`, `OPENAI_TEMPERATURE`, `OPENAI_MAX_COMPLETION_TOKENS` | Optional tuning. Empty temperature = omitted from the request (some models reject it). `low` image detail is cheapest. |
| `LOCAL_VLM_BASE_URL`, `LOCAL_VLM_MODEL`, `LOCAL_VLM_API_KEY`, `LOCAL_VLM_GUIDED_JSON` | Local server details. Pin the exact model/version in your repo README. |
| `BRD_STORE_DIR` | Folder holding the CSV/JSONL store (default `data/store`). |

```bash
python -m brd_agent.dbcli provider        # prints the provider/model in use; the key is masked
```
Both providers share one client (`vlm_client.OpenAICompatVLM`); only the request shape differs (OpenAI: `max_completion_tokens`,
`response_format=json_object`, image `detail`; local: `max_tokens`, optional `guided_json`). Retries on 429/5xx, one JSON-repair
retry, and an on-disk cache keyed by provider+model+prompt+images. Every decision records `provider`, `model`, `prompt_version`, tokens.

> **Data policy:** the brief says event data may not be shared without the organizers' permission. Sending event frames to
> the OpenAI API is arguably sharing. Confirm with aisfstamu@gmail.com first, or test OpenAI mode on non-event images. The local mode is the safe path.

## 2. Workflow
```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m brd_agent.dbcli init --cows C01,C02,C03,C04,C05,C06,C07,C08,C09,C10 --days 1,2   # 28,800 pre-created rows
python -m brd_agent.dbcli import-minutes --csv upstream_minutes.csv                         # Student 1's output
python -m brd_agent.dbcli import-human --csv human_labels.csv                               # optional, for evaluation
python -m brd_agent.dbcli run --workers 10 --backend yourpkg.adapter:RealBackend            # provider from .env
python -m brd_agent.dbcli run --vlm mock                                                    # plumbing test, no VLM
python -m brd_agent.dbcli status | export --out outputs/level1 | eval
```
Useful flags for `run`: `--max-agent-runs N` (cap VLM-escalated minutes, highest priority kept), `--live` (no look-ahead),
`--rerun` (clear agent columns and reprocess; `agent_runs` history is kept), `--cows`, `--days`, `--limit-per-cow`.
The run is **resumable**: it only processes `pending` and `stale` rows. Re-importing upstream data that materially changes an
already-processed row marks it `stale`, so it is reprocessed next run.

## 3. File store (`store.py`): CSV + JSON/JSONL, no database
`--store <folder>` (env `BRD_STORE_DIR`). Plain files you can open, diff, zip and hand to the organizers:

| File | Format | Purpose |
|---|---|---|
| `minutes/<cow>_d<day>.csv` | CSV | **One row per (cow, day, minute), 1440 per cow-day.** UPSTREAM columns (tier-1 top-k, zone fractions, speed, posture, per-camera visibility, id_conf, is_night) plus the source pointer (`src_file`, `src_camera`, `src_offset_s`) and the AGENT columns (`agent_status, behavior, confidence, source, tier, reason, correction, rationale, cameras_used, flags, ...`). Opens in Excel. |
| `yolo/<cow>_d<day>.jsonl` | JSONL | Raw records of the standalone YOLO process (file, camera, time, cow, behaviour, box). Append-only, read lazily and cached. |
| `actions/<cow>_d<day>.jsonl` | JSONL | Every action the agent took, with its reason and outcome. |
| `runs/<cow>_d<day>.jsonl` | JSONL | Append-only audit of every decision (compare models/prompts/reruns). |
| `meta.json`, `cows.json`, `imports.jsonl` | JSON | Clock origin, inferred record rate, cow list, import log. |
| `human_labels.csv`, `track_obs.csv` | CSV | Optional hand labels (`dbcli eval` reports final vs tier-1-only accuracy per class and per day/night) and tracker output. |

Why these formats: the per-minute table is small (28,800 rows for 10 cows x 2 days) and should be human-readable, so CSV; raw YOLO
records are numerous and append-only, so JSONL; logs and metadata are nested, so JSON/JSONL. Each worker owns one cow's files and writes
them atomically (temp file + rename), so a crash loses at most the last ~150 unflushed minutes, which the resumable run redoes.
Re-importing changed upstream data marks finished rows `stale` and only those are re-processed.

## What Student 1 must deliver: `upstream_minutes.csv`
Required: `cow_id, day, minute` and `tier1_top` (`standing:0.41;feeding:0.33` or JSON `[["standing",0.41],...]`, calibrated, labels from the 10 classes;
or `tier1_label` + `tier1_conf`). Optional but valuable: `tier1_entropy, is_night, zone_feed_bunk, zone_waterer, zone_resting, zone_open, speed_px_s, posture (lying|standing|unknown), vis_back, vis_front, vis_top, id_conf, camera_disagreement`.
See `examples/upstream_minutes_example.csv`. Minutes with no tier-1 data stay in the table and are labelled `other_not_visible` / `no_upstream_data` (never guessed).
Per-second data for the real backend goes through `import-obs` (columns in `db.import_track_obs_csv`).

## 4. History (the `cow_history` tool)
SQL-backed, **same cow, short window only**: recent run-length labels (`feeding x12 > standing x3`), current bout length, minutes since
feeding/drinking/lying/walking/rumination, counts over the last hour, and (offline only) the next minutes' *tier-1* guesses. Minutes not yet
processed appear as provisional tier-1 labels marked `?`. Policy choices (state them in the brief):
- **No Day-1 habit priors and no herd priors.** They would smooth away the very behavioral change Level 2 looks for.
- History is **context, not proof** (prompt rule 6); this minute's images/measurements decide. Decisions that used it have `used_history=1` so you can ablate it.
- Verifier soft check `breaks_long_bout`: leaving a bout of >= `bout_break_min` minutes is flagged (confidence reduced) unless the next minute's tier-1 confirms it.
  `lying <-> ruminating_lying` and `standing <-> ruminating_standing` are refinements, not breaks.
- Per-cow time order is what makes "previous final label" available; parallelism is across cows. `--live` removes look-ahead.

## 5. Behavior recipes and tags
`recipes.py` states, per behavior, the tools (and whether history) needed to confirm it. A top-2 candidate's recipe tools are **forced into the plan**
(a planner cannot skip `rumination_probe` for a rumination candidate, `zone_query` for feeding/drinking, `neighbors` for social, ...), reused for the second look,
and summarised as hints in the plan prompt. Recipes list *what evidence is needed*; numeric thresholds stay in `agent.yaml`.

The 10 required labels remain the output. **Tags** are optional annotations beside the label (`schemas.TAGS`, filtered to the label they apply to, max 3):
grooming (`self_licking`, `rubbing_on_fixture`, `scratching_hind_leg`), social (`licking_other`, `head_butting`, `mounting`, `displaced_from_bunk`),
walking (`pacing`, `walking_to_bunk`, `walking_to_water`), others (`at_bunk_not_eating`, `tongue_rolling`, `lying_sternal`, `head_tucked`, `stretching`, `elimination`)
and **clinical screening cues** (`lateral_recumbency`, `head_low`, `ear_droop`, `nasal_discharge`, `coughing`, `labored_breathing`, `drooling`), which add a
`clinical_candidate:<tag>` flag for Level 3. Clinical tags are **unvalidated**: you have no labeled examples to measure them against.

## 6. Output contract: `Decision` (-> the minutes CSV agent columns)
`reason`: `agent_accept | agent_revised | unresolved_fallback_tier1 | budget_fallback | error_fallback | auto_abstain | keep_tier1 | no_upstream_data`.
`flags`: `soft:<check>` (confidence reduced), `hard:<check>` (evidence contradicted the label, fell back), `disagrees_with_tier1`, `gate:<reason>`, `clinical_candidate:<tag>`.
Confidence = VLM confidence x `soft_penalty`^(soft failures). **Not a calibrated probability until checked on held-out labels.**

## 7. Code map
`env.py` (.env, provider switch) | `vlm_client.py` (client, factory, mock) | `store.py` (CSV/JSONL store, importers, write-back, history, export, eval) |
`pipeline.py` + `dbcli.py` (DB workflow) | `cli.py` (JSONL workflow) | `gate.py` | `recipes.py` | `tools.py` (12 tools incl. `cow_history`) |
`verifier.py` | `orchestrator.py` (the loop) | `trace.py` | `prompts/v3/*.md` (v1, v2 kept for reproducibility) | `schemas.py` | `backend.py` (interface for Student 1's real data).

## 8. Tests and what they do / do not show
```bash
PYTHONPATH=src python -m pytest tests -q      # 55 tests, no GPU, no API key
```
They cover the env/provider switch, request shape for both providers against a local fake server, caching, retries, DB write-back, history facts,
staleness, export and the CLI. A full-scale mock run (28,800 cow-minutes, 10 workers) writes every row in about 30 s.
**They do not show label accuracy**: the mock backend/VLM return canned answers, and no prompt has been run against a real model. The OpenAI request shape is checked
against a fake server, not the live API, so confirm your first real call. Expect to iterate on prompts (add `v3`) once you see real traces.

## 9. Known limits and assumptions
- Thresholds in `agent.yaml` marked ASSUMPTION (walk speed, jaw score, rumination Hz band, `bout_break_min`) are uncalibrated defaults.
- Rumination from the jaw-motion probe is a heuristic estimate, weakest at night.
- Gate thresholds assume calibrated tier-1 probabilities (`calibration.fit_temperature`).
- With `--max-agent-runs`, escalation priority is computed in a pre-pass using tier-1 neighbours (finals do not exist yet), so it is approximate.
- Look-ahead (`next_tier1`) is offline-only; a live system cannot use it, so report offline and `--live` results separately.
