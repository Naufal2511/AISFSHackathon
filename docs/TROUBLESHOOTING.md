# When a chunk fails: finding out why, and fixing it

## 1. Ask the script first

```bash
python why_failed.py --who <you>                  # e.g. --who afeef_training; no --who = every share here
python why_failed.py --run outputs/pipeline/<name>  # a single run_pipeline.py run
```

It only reads files, so it is safe while other chunks are running. For each chunk it prints every step
(ok / FAILED / INTERRUPTED / running), the error and traceback of the step that stopped (from the **latest**
attempt only), the seconds tracking skipped, whether a tracking checkpoint exists, then:

```
LIKELY CAUSE: ...
FIX:          ...
THEN:         rerun the same command
```

Most of the time that is all you need. The rest of this page is for when it says "not a known one", or you want
to look yourself.

## 2. After the fix: rerun the same command

`python run_share.py` (same answers) or the same `run_pipeline.py` line. Nothing done is redone:

| What | Kept on rerun |
|---|---|
| finished chunks | skipped |
| finished steps inside a chunk (`pipeline_state.json`) | skipped |
| tracking (the long step) | continues from `tracking_checkpoint.pkl`, saved every 15 min of footage: at most ~15 min lost |
| converted video (`<chunk>/cache/remux/`) | reused |

A chunk that crashed on code older than commit `1aa69e4` has no checkpoint: that one rerun of tracking starts from
the beginning. `git pull` first so it cannot happen again.

## 3. Where everything is

For a laptop share, `<chunk>` = `outputs/shares/<you>/<you>_cNN/`:

| File | What it tells you |
|---|---|
| `outputs/shares/<you>/<you>_cNN.log` | the chunk as a whole: start, the step that failed, Ctrl+C / kill |
| `<chunk>/pipeline_state.json` | each step's status, time taken and error message |
| `<chunk>/logs/1_track.log` | tracking (YOLO + ByteTrack + re-ID); progress line every 300 steps |
| `<chunk>/logs/2_level1.log` | identities, identity agent (TAMU), behavior, contract |
| `<chunk>/logs/2b_assets.log` | Level 2 risk assets |
| `<chunk>/logs/3_import.log`, `4_agent.log`, `5_export.log` | BRD agent (TAMU) and `activity.csv` export |
| `<chunk>/stage1/tracking/step_errors.log` | full traceback of every second tracking skipped |
| `<chunk>/stage1/tracking/tracking_checkpoint.pkl` | the resume point (exists only while tracking is unfinished) |
| `outputs/shares/<you>/setup_check.json` | the last `check_setup.py` result |

Logs are appended to on every attempt; each attempt starts with a `===== <step> <time>` line. Read from the last
one. Watch a running step live with `tail -f <chunk>/logs/1_track.log`.

Reading a traceback: the **last line** is the error (`ValueError: ...`); the `File "..."` line just above it is
where it happened. Lines under `.venv/` or `site-packages/` are libraries; the first line in our code
(`src/`, `scripts/`, `brd_agent/`) is usually the place to look.

## 4. Known causes

| You see (in the log / traceback) | Cause | Fix |
|---|---|---|
| `operands could not be broadcast together with shapes (384,) (420,)` | tracking bug in code before `4124414` | `git pull`, rerun |
| `KeyboardInterrupt`, `Killed: 9`, step INTERRUPTED | stopped: Ctrl+C, terminal closed, laptop slept, or out of memory | rerun; `caffeinate -dims python run_share.py` (Mac), Sleep: Never (Windows); if `Killed` repeats: `--parallel 1` |
| `out of memory`, `MPS backend out of memory`, `MemoryError` | RAM / GPU memory | fewer chunks at once (`--parallel 1`), close other apps |
| `No space left on device` | disk full | delete `cache/remux/` of finished chunks and old `outputs/pipeline/` runs; need ≥ 30 GB free |
| `401`, `403`, `error code: 1010`, `Forbidden` from TAMU | key wrong or expired | fix `TAMU_API_KEY` in `.env` (no quotes or spaces), then `python check_setup.py --who <you> --videos <clips>` |
| `429`, `Too Many Requests` | TAMU rate limit for this key | rerun; if repeated, `--stage2-workers 2` and `PEN6_ID_PARALLEL=2` |
| `ConnectionError`, `Max retries exceeded`, `SSLError`, name resolution | no network to TAMU | Wi-Fi / VPN; on HPRC compute nodes the WebProxy module (`scripts/hprc/env.sh`) |
| `OPENAI_API_KEY`, `model_not_found` | TAMU failed 3 times **and** the OpenAI fallback failed | wait and rerun (TAMU outage), or set `OPENAI_API_KEY` / `OPENAI_MODEL` in `.env` |
| `ffmpeg: command not found` | ffmpeg missing | install (LAPTOP_SETUP.md §3) or `--ffmpeg-dir <folder>` |
| `No clips of ... fall in`, `missing_clips.txt` written | footage for the window is not in your folder | `python fetch_clips.py --who <you> --dest <clips>`, or fix `--videos` |
| `invalid load key`, `PytorchStreamReader failed` | model file is a Git LFS pointer | `git lfs install && git lfs pull` |
| `ModuleNotFoundError` | package missing / wrong venv | activate the venv; `pip install -r requirements.txt -r brd_agent/requirements.txt` |
| `plan.json` / fingerprint mismatch | your plan differs from the team's | copy the team's `outputs/shares/plan.json`; check_setup prints the fingerprint |
| `N steps in a row failed` (tracking) | something is really broken, not one bad frame | read `stage1/tracking/step_errors.log`, fix that, rerun (resumes from the checkpoint) |

### Errors that do not stop the run

* `step_errors.log` with a handful of entries: tracking skipped those seconds (one bad frame, a broken clip) and
  carried on. A few in a 4 h chunk are harmless; if there are hundreds, look at what they have in common.
* `read error ... skipping 60 s of this camera`: a clip could not be decoded; that camera resumes a minute later.
* TAMU retries in the agent logs: each call is tried 3 times (2 s, 4 s), then OpenAI; only a step that ends with an
  error matters.

## 5. HPRC

Each array task writes `/scratch/user/$USER/pen6_logs/pen6_pipe_<jobid>_<task>.out` (task 0 = smoke test). The
chunk's own logs are the same as above under `/scratch/user/$USER/pen6/outputs/shares/hprc_day2/`, and
`venv/bin/python why_failed.py --who hprc_day2` works there too.

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS     # why SLURM ended it
```

| `State` | Meaning | Fix |
|---|---|---|
| `TIMEOUT` | hit `--time` (8 h) | resubmit (tracking resumes from its checkpoint), or `PEN6_CHUNK_H=2` |
| `OUT_OF_MEMORY` | hit `--mem` | raise `#SBATCH --mem` in `scripts/hprc/pipeline.slurm` |
| `FAILED` | the pipeline exited with an error | `why_failed.py` as above |
| `CANCELLED by ...` / `NODE_FAIL` / `PREEMPTED` | not our fault | resubmit: `bash scripts/hprc/submit_pipeline.sh` (finished chunks are skipped) |

## 6. Still stuck

Send the team the output of `python why_failed.py --who <you>` and the step log it points at. **Never send `.env`**
(it holds your TAMU key). Logs contain clip names and times only, no keys and no video.
