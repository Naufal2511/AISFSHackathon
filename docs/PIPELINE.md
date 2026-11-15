# End-to-end pipeline: videos in, agent CSVs out

`run_pipeline.py` runs the whole Level 1 chain from one input (a folder of `Event*.Avi` clips). Both stages run
through their existing entry points, unchanged, as subprocesses, so `main.py`, `scripts/run_level1.py` and
`brd_agent`'s `dbcli` keep working on their own exactly as before.

```
videos ─► prepare ─► STAGE 1 (this repo)                       ─► STAGE 2 (brd_agent/)                     ─► final/*.csv
          cameras    track   main.py (YOLO + ByteTrack + identity)  import  dbcli import-tracking
          + links    level1  run_level1.py --skip-agent             agent   dbcli run --backend video
                             identity → world state → anchors →     export  dbcli export
                             behavior → contract export
                             = observations.csv + cow_state.csv  ───┘
```

Stage 1's own supervisor agent (`run_agent.py`) is skipped: stage 2 is the validation agent.

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
| `--from STEP` | redo from `prepare/track/level1/import/agent/export` (and everything after) |
| `--until STEP` | stop after a step (e.g. `--until level1` to get stage 1's contract only) |
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
  stage1/level1/contract/       observations.csv + cow_state.csv (the hand-off between the stages)
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
* Stage 1 on CPU is the slow part (detector + DINOv2 on 3 × 4K streams); use a CUDA machine for the 48 h set.
