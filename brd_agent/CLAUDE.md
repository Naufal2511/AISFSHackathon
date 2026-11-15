# CLAUDE.md: handoff context for this project

Read this first. It is the full context of the project, what exists, how to run it, and what is left.

## 1. The project
**Merck-TAMU Animal Science Hackathon 2026, "Agentic Vision for Angus Behavior and BRD Prediction".** Team of 3 students. This repo is the
**agentic VLM component** (owned by the user). Teammate "Student 1" owns detection, tracking and cow identification (YOLO, tracker, fusion).
The BRD PDF (`Hackathon_Challenge_BRD_Agentic_Vision_2026.pdf`) is in the claude.ai Project "AIFS Hackathon"; read it for the exact rules.
* Deadline: **submission Sun Nov 15, 2026, 11 PM CST**; finalists present Nov 18. Contact: aisfstamu@gmail.com.
* Data per the BRD: ~48 h from each of 3 cameras of 10 Angus cows in one pen, day and night footage.
* **Level 1 judging:** behavior accuracy, correct cow identification, a genuine agent loop that uses tools and checks itself, complete and clean output.
* **Level 1 outputs:** `activity.csv` (one row per behavior bout, every cow, every minute of both days); ID-check frames showing cow IDs every
  30 min on each camera; per-cow time budgets; three saved agent traces (one success, one failure, one at night).
* Levels 2 (BRD risk) and 3 are not built.

## 2. What the agent must do (the user's requirement)
Upstream (tracking/YOLO) supplies: video file name + timestamp + labelled cow + behavior. The agent must **genuinely verify** that, correct
mistakes and recover misses, cover **night** footage, and give a **reason for every action** it takes. It should **not** spend the VLM on cows that
are not doing anything.

## 3. Architecture (src/brd_agent)
Per cow-minute: night check (from the frame) -> gate -> **rule pre-verification (no VLM)** -> full agent only if flagged/audited.
Full agent loop: plan -> tools on real video -> blind hypothesis -> rule verifier -> needs-driven round -> second look -> critique -> resolve.

| Module | Role |
|---|---|
| `store.py` | **File store, no database** (replaced SQLite): `minutes/<cow>_d<day>.csv` (1440 rows/cow-day, the working table), `yolo/*.jsonl` (raw records), `actions/`, `runs/` (JSONL logs), `meta.json`, `cows.json`, `imports.jsonl`. Atomic writes, per-cow-day caches, `HistoryStore`, `row_to_brief`, `export_activity`, `evaluate_vs_human`. |
| `ingest.py` | `import_tracking`: **observations.csv + cow_state.csv** -> records + minute rows (see section 5). |
| `yolo.py` | older path: standalone YOLO CSV -> `import_yolo` / `aggregate_minutes`. |
| `calm.py` | skip the VLM for calm cows: anchor-and-inherit (section 6). |
| `prescreen.py` | cheap rule checks on EVERY minute: label vs measured speed/zone, jaw-motion screen for confident standing/lying, YOLO flags, random audit (`audit_rate` 0.05, night 0.25). |
| `orchestrator.py` | `BehaviorAgent` loop, guardrails (`FLAG_TOOLS`), critique, `tier1_decision`. |
| `verifier.py`, `gate.py` | rule checks; escalation gate. Negated cues ("no chewing") are not claims (fixed). |
| `tools.py`, `backend.py` | the agent's tools; `MockBackend` for tests. |
| `video.py`, `video_backend.py` | `Event<YYYYMMDDHHMMSSmmm>.Avi` parsing, `VideoIndex`, `Clock`, frame/burst reading (ffmpeg fallback for HEVC "GHEC"), `VideoBackend` (crops, zoom, jaw motion, zones, track history, miss recovery, exemplar ID check). |
| `night.py` | frame-based night detection + CLAHE/gamma enhancement. |
| `pipeline.py` | `run_db`: per-cow workers, prepass gate, find_missing, passthrough, calm chain, budget. |
| `dbcli.py` | the CLI (below). `env.py`/`.env` select the VLM provider. `vlm_client.py`: OpenAI or local OpenAI-compatible server, disk cache, `MockVLM`. |
| `prompts/v3/*.md`, `configs/*.yaml` | prompts, behavior definitions, all thresholds. |

Outcome per minute (`correction`): confirmed | behavior_corrected | abstained | recovered_miss | miss_not_found | unresolved, plus `rationale` and an
`actions` log (actor, action, args, reason, outcome). Final reasons include `keep_tier1`, `agent_accept`, `agent_revised`, `auto_abstain`,
`no_upstream_data`, `miss_not_recovered`, `error_fallback`, `stable_carry`.

## 4. How to run (Windows PowerShell, which the user uses)
```powershell
conda activate langchain                      # optional
$env:PATH = "<ffmpeg full_build bin>;" + $env:PATH   # a STANDALONE ffmpeg+ffprobe; the conda ffmpeg on this machine is broken (exit -1073741511)
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"                       # needed in every new terminal
python -m pytest -q                           # 64 tests, no network, no videos needed
python -m brd_agent.dbcli --store tracking_store import-tracking --observations <observations.csv> --cow-state <cow_state.csv>
python -m brd_agent.dbcli --store tracking_store run --backend video --video-root videos --vlm auto --days 1 --trace-dir out/traces
python -m brd_agent.dbcli --store tracking_store status
python -m brd_agent.dbcli --store tracking_store export --out out/level1
```
`--vlm mock` = no API calls (plumbing only). Useful `run` flags: `--cows`, `--verify-all` (every minute through the agent, disables calm),
`--rerun`, `--audit-rate`, `--max-agent-runs`, `--workers`, `--pen-map`, `--detector-weights`, `--exemplars`, `--live`.
Config: copy `.env.example` to `.env` (`USE_LOCAL_VLM`, `OPENAI_API_KEY`, `OPENAI_MODEL` (a vision model), local server settings). The final
solution is meant to use a **local VLM** (`USE_LOCAL_VLM=true`, any OpenAI-compatible server such as vLLM).
Windows gotchas already handled: all file IO is UTF-8 (cp1252 crashed on "→"); OpenCV cannot open the HEVC clips, so **ffmpeg and ffprobe must be on PATH**.
Outputs: `out/level1/activity.csv` (bouts), `activity_minute.csv` (per cow-minute), `actions.csv` (every action + reason), `out/traces/*.md|json`.

## 5. Input data (the real files, from the previous agent / tracking pipeline)
* `observations.csv`: per camera, per second, per box. Columns incl. `wall_time, file, offset_s, camera (cam1..3), camera_name, track_id, cow_id, id_conf, id_evidence,
  x1..y2, frame_w/h, world_x/world_y (m), visibility, occluded_by`. Behavior columns are EMPTY here. ~1,086 boxes have no cow_id (counted, unused).
* `cow_state.csv`: per cow, per second, fused over cameras: `speed_mps, zone (bunk|rest|other), n_cams, behavior_fused, beh_conf_fused, beh_evidence
  (observed|estimated_gap_fill|estimated_heuristic|not_visible), p_lying ... p_other, anomaly (camera_offset|camera_conflict)`.
* Sample window: 2026-11-07 16:00:30-16:09:59, 8 cows `COW_01..08`, cams: cam1 Front Camera 1, cam2 Front Camera 2 (bunk), cam3 Back Camera.
  Each camera has its own file names and start times (e.g. `...160019006`, `...160029017`, `...160008003`), 5-minute files.
  **Verified: `wall_time` = file start (from the name) + `offset_s`**, so wall_time is the common clock. The fused state has **no rumination** probability at all.
* A 3-minute trimmed copy is in `examples/tracking_sample/` (used by `tests/test_tracking.py`).
* Videos: put all `Event*.Avi` in one folder (`videos/`), **never rename**; the camera comes from the `file -> camera` map stored at import (not from folders).
* Day/minute indexing: day 1 starts at midnight of the first date; `minute` = minute of day (16:00 = 960). `t_abs=(day-1)*1440+minute`.
* Camera slots are `back/front/top`; `video.camera_alias` maps cam1->front, cam2->top, cam3->back (cam2 uses the "top" slot as a naming choice).
* Speeds are metric (m/s) for tracking input (`speed_unit=m_s` in meta; verifier swaps in `walk_speed_m_s`/`still_speed_m_s`).

## 6. Skipping the VLM for idle cows (`calm.py`, `calm:` in `configs/agent.yaml`)
A minute is calm when one label holds all minute (stability >= 0.98), min confidence >= 0.9, observed fraction >= 0.95, no flags
(camera_conflict, label_unstable, gap_fill, not_visible, duplicate_id), id_conf >= 0.8, p90 speed < 0.25 m/s, label in lying/standing/feeding, not night.
One minute per calm bout (the **anchor**) is verified normally; later calm minutes inherit its verified result (`stable_carry`, status `carried`, no video decode,
no VLM, reason logged) until the label changes, anything stops being calm, or `recheck_every_min` (10). On the sample, 32 of 72 cow-minutes were carried.
Trade-off: a cow that starts ruminating mid-bout is caught at the next anchor. Thresholds were calibrated on only 10 minutes of data.

## 7. Findings from the real test runs (important)
* The real VLM (OpenAI) ran end to end on the fake 7-cow clip. Result: plumbing fine, but (a) a verifier bug rejected correct VLM answers because the
  cue text contained the word "chewing" ("no chewing"); **fixed** (negation handling + test). (b) Several minutes ended `error_fallback`: the exception text is now
  written into `rationale`; **the cause is still unknown**, check it first. (c) Many `abstained: not visible enough`: the fake YOLO boxes were hand-placed, so judge again
  with the real tracking boxes. (d) The cheap rules cannot detect lying-vs-standing (posture is not inferred from box shape); only the VLM can, so the audit/anchor sampling matters.
* The first tracking-input run on the real videos has NOT happened yet (the sandbox had no 16:00 videos). Run it and read the traces and `status` output.
* `find_missing` without `--detector-weights` reports `miss_not_found` (no VLM spent); cows entering/leaving the pen show up this way (COW_06/07/08 start late in the sample).

## 8. Known gaps / TODO (rough priority)
1. Run the tracking flow on the real videos; inspect traces; check the `error_fallback` cause; check calm skipping did not hide real changes.
2. **Speed of 4K HEVC decoding** (~1.8 s per frame, ~6.5 s per 10 s burst): build a 1280-wide H.264 proxy transcode + share bursts per minute/camera before any 48 h run.
3. **ID-check frames every 30 min per camera** (BRD deliverable, not built): draw boxes + cow IDs from observations onto frames.
4. Trace selection script (one success, one failure, one night) and per-cow time budgets (from `activity.csv`).
5. Few-shot examples in prompts; model bake-off script (compare models on a hand-labelled subset via `dbcli eval`: `import-human --csv labels.csv`, then `eval`).
6. Level 2 BRD risk code and Level 3 (not started). Optional: `planner: rules` to save one VLM call per agent run.
7. Cow re-identification when cows leave and re-enter the pen: deferred (the tracking input already carries identity + evidence; add appearance-based re-ID only if needed).
8. Email organizers (aisfstamu@gmail.com) as required by the BRD; check file naming rules ("named by day, camera and start time" in the BRD vs the real `Event...Avi` names).
Assumptions to confirm with the real data: day origin = midnight, night thresholds (luminance < 70), calm thresholds, zone mapping (bunk->feed_bunk, rest->resting, other->open; no waterer zone), camera alias.
