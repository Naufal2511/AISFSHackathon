# End-to-end pipeline: videos in, agent CSVs out

`run_pipeline.py` runs the whole Level 1 chain from one input (a folder of `Event*.Avi` clips). Both stages run
through their existing entry points, unchanged, as subprocesses, so `main.py`, `scripts/run_level1.py` and
`brd_agent`'s `dbcli` keep working on their own exactly as before.

```
videos ─► prepare ─► STAGE 1 (this repo)                          ─► STAGE 2 (brd_agent/)          ─► final/*.csv ─► LEVEL 2
          cameras    track   main.py (YOLO + ByteTrack + DINOv2 IDs     import  dbcli import-tracking       activity*.csv    run_level2.py
          + links            + risk assets: daylight, motion, crops)   agent   dbcli run --backend video   actions.csv      brd_risk.csv
                     level1  identity → world state → anchors →         export  dbcli export                                 index.html
                             IDENTITY AGENT (ID flags only) →                                                                cards/
                             behavior (DINOv2 classifier + jaw probe) → contract
                     assets  build_risk_assets.py (breathing, jerks, feed events, clips)
```

Each check is done once:

| what | done by | reused by |
|---|---|---|
| identity (duplicate_id, id_jump, extra_cow, cow_missing, 30-min ID audit + ID-check sheets) | stage 1 identity agent (`run_agent.py` without `--behavior`) | everything after it runs on the corrected IDs |
| behavior per second, rumination | stage 1 classifier + `--probe-rumination` | stage 2 skips its 20 s jaw-motion burst when `cow_state` has rumination |
| day / night | stage 1 asset recorder (`assets_raw/frames.csv`, every tracked frame) | stage 2 reads it at import instead of decoding a frame per minute |
| behavior validation, recovering missed minutes, reasons | stage 2 (`brd_agent`) | Level 2 reads its verified minutes (`stage1/level1/final/activity.csv`, with wall time) |
| BRD risk | Level 2 (`run_level2.py`), once over all footage | |

All VLM calls go to TAMU with the repo `.env`'s `TAMU_API_KEY` (identity agent: `stage1.id_agent_backend`; stage 2:
`stage2.tamu_profile`, used when `brd_agent/.env` does not exist).

## Run it

```powershell
conda activate langchain                       # one env for both stages (torch, ultralytics, pydantic, ...)
python run_pipeline.py --videos "C:\data\Day 1" --name d1 --ffmpeg-dir "C:\path\to\ffmpeg\bin"
```

| Option | Meaning |
|---|---|
| `--videos` | folder of clips: flat, or `Cam 01/Cam 02/Cam 03` subfolders (searched recursively) |
| `--name` | run folder `outputs/pipeline/<name>/`; same name = resume |
| `--vlm auto\|mock` | `auto` = provider from `brd_agent/.env`; `mock` = no API calls (plumbing check) |
| `--start/--end` | wall-clock window for stage 1, e.g. `"2026-11-07 16:00"` |
| `--from STEP` | redo from `prepare/track/level1/assets/import/agent/export/level2` (and everything after) |
| `--until STEP` | stop after a step (e.g. `--until level1` to get stage 1's contract only; chunks use `--until export`) |
| `--no-id-agent` / `--id-backend` | skip the identity agent (memory cards are still built) / pick its configs/agent.yaml profile |
| `--level2-skip-agent` | Level 2 behavior score only |
| `--detector` | default `models/angus_det_hprc_m1280.pt` (Git LFS: `git lfs pull`) |
| `--agent-args ...` | passed to `dbcli run`, e.g. `--agent-args --max-agent-runs 500 --audit-rate 0.1` (keep it last) |
| `--id-gallery` | share one appearance gallery across runs so `COW_xx` stays the same animal across days |
| `--dry-run` | show the camera assignment and the plan, run nothing |

Defaults live in `configs/pipeline.yaml`.

## Output

```
outputs/pipeline/<name>/
  final/activity.csv            one row per behavior bout (every cow, every minute)   <- the deliverable
  final/activity_minute.csv     one row per cow-minute: behavior, confidence, correction, rationale
  final/actions.csv             every agent action with its reason
  stage2/traces/                agent reasoning traces (.md + .json)
  level2/                       brd_risk.csv, index.html, cards/ (BRD risk; per-share chunks stop before this)
  stage1/level1/contract/       observations.csv + cow_state.csv (the hand-off between the stages)
  stage1/level1/agent/          identity agent: traces, id_checks/ (30-min ID-check sheets), memory/ (cards, prototypes)
  stage1/level1/assets/         risk assets (gallery, motion, breathing, jerks, feed events, clips)
  stage1/level1/final/activity.csv   stage 2's verified minutes with wall time (Level 2 / stitcher input)
  stage1/level1/identity|state|behavior/   stage 1 intermediates (identity sheet, time budgets, ...)
  input/cam1..cam3/             hard links to your clips (no copies)
  logs/<n>_<step>.log           full output of each step
  pipeline_state.json           step status (resume); pipeline_manifest.json: settings, timings, git commit
```

## Cameras

Camera is not part of the useful file name, so each clip is placed by (1) a `Cam 0N` / `camN` folder on its path, else
(2) the 3 digits after the timestamp: `006` = cam1, `017` = cam2, `003` = cam3 (true for every real clip so far). The
first clip of each camera is then checked against that camera's resolution (cam2 is 2592×1944, cam1/cam3 4000×3000).
Unknown suffixes, disagreeing rules or a wrong resolution stop the run before any compute is spent.

## Speed

* One lossless HEVC→MP4 remux per clip (~2 s), shared by tracking, identity crops, rumination probes and the agent
  (`PEN6_REMUX_DIR`), deleted at the end unless `--keep-cache`. The agent reads single frames from it with OpenCV,
  2.5× faster than an ffmpeg process per frame, pixel-identical; bursts keep the exact ffmpeg path.
* Stage 1 samples 1 fps (`stage1.fps`) and skips the dashboard video (`--render-video` to get it).
* Every step is resumable, so a failure in the agent never reruns tracking.
* The risk-asset post-pass runs before stage 2, while the remuxed clips are still cached.
* Stage 2 decodes no frame for day/night and no jaw-motion burst when stage 1 supplied them (see the table above),
  and verifies `stage2.workers` cows in parallel (threads waiting on TAMU and ffmpeg).
* Stage 1 on CPU is the slow part (detector + DINOv2 on 3 × 4K streams); use a CUDA machine for the 48 h set.

## 48 h on three laptops

```bash
python run_share.py            # pick naufal_training / afeef_training / nooh_training, the video folder, chunks in parallel
```

* `outputs/shares/plan.json` splits the recording into 3 shares with equal footage (camera-hours, whole minutes). It
  depends only on the clip names, so laptops with the full folder agree; a laptop with only its own clips needs the
  plan.json from one that has them all (`python run_share.py --plan-only --videos <all clips>`).
* Each share is cut into ~4 h chunks (`--chunk-hours`), `--parallel` at a time (suggested from cores, RAM and
  GPU/MPS), CPU threads divided between them, `--until export` (Level 2 waits for the stitch). Only the clips a chunk
  touches are linked into it. Rerun the same command to resume.
* `python run_share.py --who <you> --pack` → `outputs/shares/<who>.zip` (results only, no clips or remux cache).

On one machine, unzip the three zips at the repo root, then:

```bash
python stitch_shares.py --videos "<all clips>"
```

Each chunk numbers its cows itself. Every chunk also tracks the 2 minutes before the window it owns
(`--overlap-min`), so the stitcher joins ids where both chunks saw the same boxes in the same frames (IoU >= 0.5,
Hungarian on the votes). Cows not seen in that overlap fall back to memory-card appearance prototypes; black Angus
look alike to DINOv2 (different cows score 0.75-0.94), so those matches are flagged. Weak or new matches are listed in `outputs/stitched/id_map.csv` and can be fixed
with `--id-map overrides.yaml` (`{chunk: {local_id: global_id}}`). It writes `outputs/stitched/final/`
(`activity.csv` bouts joined across chunk edges, `activity_minute.csv`, `actions.csv`, `time_budgets.csv`), traces
and ID-check sheets per chunk (sheet images keep the chunk's own ids; `legend.csv` maps them), and runs Level 2 over
all chunks (`outputs/stitched/level2/`). Without `--videos`, Level 2 runs behavior-only.
