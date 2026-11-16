# Agentic Vision for Angus Behavior and BRD Risk
## Team 404 - Naufal, Afeef, Nooh

**Merck–TAMU AISFS Hackathon 2026 · Pen 6 · 10 Angus cattle · 3 cameras · 48 h**

We watch 10 Angus cattle in Pen 6 through three cameras for 48 hours and turn the raw video into three things:
1. **What every cow did, every minute**, with one ID per cow across cameras and days (Level 1).
2. **A BRD risk ranking** a veterinarian can check, with reasons and video evidence (Level 2).
3. **A live early-warning loop** that alerts a pen rider with a clip (Level 3).

Deterministic computer vision measures; tool-using agents check that record; code checks the agents. Every number says whether it was observed, verified, estimated or not visible.

**Our finding:** COW_02, COW_01 and COW_10 are **early suspects, not sick**. They are slower to the bunk after feeding and eat less than the herd at the same hours, but no visible clinical sign was confirmed. Next step: take their temperatures.

**Our results are already in [`submission/`](submission/README.md). Open these first:**
- [`submission/SHOWCASE.md`](submission/SHOWCASE.md): **every result on one page, with pictures, readable right here on GitHub** (`submission/index.html` is the same page with buttons to every full page, after download)
- `submission/level2/index.html`: the ranking
- `submission/stitched/viewer.html`: the 48 h ethogram
- `submission/level3/clip_1600/report.html`: one live window, end to end

## 🎬 Submission video (5 min 39 s)

[![Team 404 submission video: click to watch on YouTube](docs/media/team404_submission_video_poster.jpg)](https://youtu.be/a2jZN3BnDJE)

**[▶ Watch on YouTube](https://youtu.be/a2jZN3BnDJE)** · a copy is also in this repository: [`docs/media/team404_submission_video.mp4`](docs/media/team404_submission_video.mp4) (54 MB; *View raw* / *Download* to play).

| | |
|---|---|
| 🎬 **Submission video** | [YouTube](https://youtu.be/a2jZN3BnDJE) · [`docs/media/team404_submission_video.mp4`](docs/media/team404_submission_video.mp4) |
| 📄 **Technical brief** (2 pages) | [`docs/TECHNICAL_BRIEF.pdf`](docs/TECHNICAL_BRIEF.pdf) |
| 📊 **All results** | [`submission/`](submission/README.md) |
| ▶️ **Run it** | [section 1](#1-quick-start) (Docker or Python), then [Level 3 on any clip](#3-level-3-on-a-clip-you-choose-the-judges-clip) |

## Submission at a glance

Every deliverable the brief asks for, and where it is:

| Required | File in this repository |
|---|---|
| **Level 1:** `activity.csv` (one row per behavior bout, every cow, every minute of both days) | [`submission/stitched/final/activity.csv`](submission/stitched/final/activity.csv) |
| ↳ the same per minute, with confidence, basis and the agent's rationale | [`submission/stitched/final/activity_minute.csv`](submission/stitched/final/activity_minute.csv) |
| **Level 1:** ID-check frames every 30 min on each camera (97 sheets, 48 h) | [`submission/stitched/id_checks/`](submission/stitched/id_checks/) |
| ↳ how to read them: each sheet is a JPG + CSV per 30 min; `legend.csv` maps the chunk's IDs to the final `COW_01..10` | e.g. [`hprc_day2_c18/id_check_20261108_163000.jpg`](submission/stitched/id_checks/hprc_day2_c18/id_check_20261108_163000.jpg) |
| **Level 1:** per-cow time budgets | [`submission/stitched/final/time_budgets.csv`](submission/stitched/final/time_budgets.csv) |
| **Level 1:** agent reasoning trace, **success** | [`success_COW_02_d1_m0862.md`](submission/stitched/final/traces/success_COW_02_d1_m0862.md): stage 1 said drinking (0.69); the agent verified walking (0.96) |
| **Level 1:** agent reasoning trace, **failure** | [`failure_COW_01_d1_m0813.md`](submission/stitched/final/traces/failure_COW_01_d1_m0813.md): a weak "standing" the agent could not settle, kept as unverified |
| **Level 1:** agent reasoning trace, **night** | [`night_COW_05_d1_m1085.md`](submission/stitched/final/traces/night_COW_05_d1_m1085.md): infrared footage; the agent rejected "ruminating" and set not visible |
| ↳ which cow each trace is after stitching | [`traces/README.md`](submission/stitched/final/traces/README.md) |
| **Level 2:** `brd_risk.csv` (cow ID, risk 0–1, rank, top reasons) | [`submission/level2/brd_risk.csv`](submission/level2/brd_risk.csv) |
| **Level 2:** visualization tool | [`submission/level2/index.html`](submission/level2/index.html) (ranking, stability, cards) and [`submission/stitched/viewer.html`](submission/stitched/viewer.html) (48 h ethogram, camera joins, time budgets) |
| **Level 2:** one-page risk card per top-3 cow, with 2–3 clips | [`COW_02.html`](submission/level2/cards/COW_02.html) · [`COW_01.html`](submission/level2/cards/COW_01.html) · [`COW_10.html`](submission/level2/cards/COW_10.html) |
| ↳ the risk agent's reasoning and every piece of evidence it cited | [`submission/level2/agent/traces/`](submission/level2/agent/traces/) · [`evidence/`](submission/level2/evidence/) |
| **Level 3:** live alert on one 5-min clip, end to end | [`submission/level3/clip_1600/report.html`](submission/level3/clip_1600/report.html) |
| ↳ three Day 2 hours replayed as live | [`submission/level3/day2_replay/report.html`](submission/level3/day2_replay/report.html) |
| **Code** that reproduces everything from raw video | sections [1–5](#1-quick-start) below |
| **Models, versions, outside data and models** | [section 6](#6-models-versions-and-external-resources) |
| **AI tools used** | [section 7](#7-ai-tools-and-how-we-used-claude) |
| **Submission video** | [YouTube](https://youtu.be/a2jZN3BnDJE) · [`docs/media/team404_submission_video.mp4`](docs/media/team404_submission_video.mp4) |

**Viewing the HTML pages:** GitHub shows HTML files as source code. Clone or download the repository and open them in a browser; they are self-contained apart from the video clips.

> **Event data:** the footage, frames and results stay within the event. Keep this repository private. Apart from the submission video, video clips are not in the repository; the commands below regenerate them.

---

## 1. Quick start

You need the footage the way the organizers shared it (`Day 1/Cam 01/Event*.Avi`, …) and, for the agents, a TAMU AI Chat API key.

### Option A: Docker (recommended)

```bash
git clone <this repository> pen6 && cd pen6
git lfs install && git lfs pull                 # model weights (models/*.pt, ~170 MB)
cp .env.example .env                            # then put your key in it: TAMU_API_KEY=...
docker build -t pen6 .                          # CPU image, ~4 GB
```

NVIDIA GPU (much faster tracking), on Linux with the NVIDIA Container Toolkit:

```bash
docker build -t pen6 --build-arg BASE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime .
# then add --gpus all to every docker run below
```

Every command in sections 3–5 runs inside the container. Mount the footage read-only at `/videos` and keep the results on your machine:

```bash
docker run --rm -it --env-file .env \
  -v "/path/to/footage:/videos:ro" -v "$PWD/outputs:/app/outputs" \
  pen6 <command>
```

### Option B: Python on your machine

Requirements: Python 3.10–3.12, `ffmpeg` with HEVC decoding (macOS `brew install ffmpeg`; Ubuntu `apt install ffmpeg`), and Git LFS.

```bash
git clone <this repository> pen6 && cd pen6
git lfs install && git lfs pull
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r brd_agent/requirements.txt      # exact tested versions: requirements-lock.txt
cp .env.example .env                                                 # TAMU_API_KEY=...
```

The models run on an NVIDIA GPU (CUDA) if there is one, else the Apple GPU (MPS), else the CPU. Set `PEN6_DEVICE=cpu` to force the CPU.

### See every result in your browser

The image's default command serves the results showcase on this computer only:

```bash
docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/outputs:/app/outputs" pen6
```

Without Docker (needs only Python, no packages): `python serve_showcase.py`. Then open:

- **http://localhost:8000/**: the showcase. It covers Levels 1, 2 and 3 with links to every page: the 48 h viewer, the BRD ranking and risk cards, the Level 3 evidence reports, alert cards and emails, and the technical brief.
- **http://localhost:8000/live/**: your own Level 3 run, after `live_watch.py --clip …` (section 3).

It serves only `submission/`, `docs/`, this README and `outputs/live/`. It never serves the code folders or `.env`.

---

## 2. Check before you run (1 minute, nothing is processed)

```bash
python run_pipeline.py --videos "/videos/Day 2" --name check --dry-run
```

This prints which camera each clip was assigned to and the plan. Cameras are recognised from the `Cam 0N` folder, or from the 3 digits after the timestamp in the file name (`006` = cam1 front, `017` = cam2 bunk/top, `003` = cam3 back). The first clip's resolution is then checked. A clip that cannot be placed stops the run; it is never guessed.

---

## 3. Level 3 on a clip you choose (the judges' clip)

Pass **one camera's file** of a 5-minute window. The same window from the other two cameras is found next to it (`Cam 01/02/03`).

```bash
python live_watch.py --clip "/videos/Day 2/Cam 01/Event20261108142130006.Avi"
```

For that window it:
1. tracks the cows and maps them to the 48 h IDs;
2. compares each cow with the herd in the same window;
3. checks close crops for drooping ears, a low head and nasal discharge (VLM). Every flag gets a second look and a measurement check;
4. updates each cow's risk, starting from the 48 h Level 2 result, and sends alerts with a 12 s clip.

| Output (in `outputs/live/`) | What it is |
|---|---|
| `index.html` | Live dashboard: tier and risk per cow, trend, alert feed |
| `alerts/<cow>_<time>.html` + clip, frame, crops | The alert a pen rider gets: what to do, the evidence, why |
| `outbox/*.eml` | Simulated alert emails and the daily BRD report (never sent) |
| `report.html`, `risk_history.csv`, `state.json`, `windows.jsonl`, `live.log` | Every decision with its evidence |

To view them in your browser, start the showcase (`python serve_showcase.py`, or the image's default command above) and open **http://localhost:8000/live/**.

Ranch mode watches the folder the cameras write to and processes each new window as it completes:

```bash
python live_watch.py --watch /videos --poll 30
```

Useful flags:
- `--verify`: also runs the stage 2 behavior agent on the window.
- `--no-signs`: skips the VLM sign check.
- `--reset`: starts again from the 48 h baseline.

Settings: `configs/live.yaml`. Method: [`docs/LEVEL3_LIVE.md`](docs/LEVEL3_LIVE.md).

---

## 4. Levels 1 and 2 on any footage

One command takes a folder of clips (any length) to the Level 1 and Level 2 deliverables:

```bash
python run_pipeline.py --videos "/videos/Day 2" --name d2_afternoon \
    --start "2026-11-08 14:00" --end "2026-11-08 16:00"     # optional time window
```

| Step | What happens | Output (in `outputs/pipeline/<name>/`) |
|---|---|---|
| `prepare` | sort clips into cameras (links, no copies) | `input/cam1..cam3/` |
| `track` | fine-tuned YOLO26 detector + ByteTrack per camera, 1 frame/s | `stage1/tracking/` |
| `level1` | identity by cross-camera agreement, world state, identity agent, behavior model | `stage1/level1/` (+ `agent/id_checks/`, the 30-min ID sheets) |
| `assets` | breathing, sudden movements, feed events, evidence clips | `stage1/level1/assets/` |
| `import` / `agent` / `export` | stage 2 behavior agent re-checks flagged minutes on the video | `final/activity.csv`, `activity_minute.csv`, `time_budgets.csv`, `actions.csv`, `traces/` |
| `level2` | 13 BRD indicators vs the herd, risk agent, risk cards | `level2/brd_risk.csv`, `index.html`, `cards/` |

- **Resuming:** runs resume after an interruption. The same `--name` skips finished steps, and `--from STEP` redoes a step and everything after it.
- **Logs:** each step's log is in `logs/`. `python why_failed.py` says which step stopped and why ([`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)).
- **Viewer:** `python scripts/run_viewer.py --run outputs/pipeline/<name>` makes an interactive `viewer.html`.
- **`basis` column:** every row of `activity.csv` says how it is known: `verified` (an agent looked at the frames), `observed`, `unverified`, `estimated` or `not_visible`.

**Running time:** 5 minutes of 3-camera footage took about 12 min on a MacBook (Apple GPU):

| Step | Time |
|---|---|
| tracking | 4 min |
| identity + behavior | 5 min |
| assets | 1 min |
| behavior agent | 2 min |

An NVIDIA GPU is several times faster. For the full 48 h, see section 5.

### Without an API key

Everything still runs on rules and the local models; only the agents are switched off:

```bash
python run_pipeline.py --videos /videos/... --name test --vlm mock --no-id-agent --level2-skip-agent
python live_watch.py --clip /videos/.../Event....Avi --no-signs     # live_watch drops the identity agent itself when no key is set
```

---

## 5. Reproducing the 48 h results in `submission/`

48 h × 3 cameras of 4K footage is split into chunks, run in parallel, and stitched.

```bash
# 1. plan: split the footage into shares (all on one machine here)
python run_share.py --plan-only --layout all-laptops --videos /videos
# 2. run each share's chunks (resumable; repeat for each --who)
python run_share.py --who naufal_training --videos /videos --yes
python run_share.py --who afeef_training  --videos /videos --yes
python run_share.py --who nooh_training   --videos /videos --yes
# 3. join the chunks into one 48 h result and run Level 2 on it
python stitch_shares.py --videos /videos            # -> outputs/stitched/ (final/, level2/, id_map.csv)
python scripts/stitched_viewer.py                   # -> outputs/stitched/viewer.html
```

Each chunk also tracks the 2 minutes before its window. The stitcher joins cow IDs where neighbouring chunks saw the same boxes in the same frames. Uncertain joins are listed in `id_map.csv` and can be overridden with `--id-map overrides.yaml`.

- **On a cluster:** we ran Day 2 as a Slurm GPU array job on TAMU HPRC: [`docs/HPRC_PIPELINE.md`](docs/HPRC_PIPELINE.md).
- **On laptops:** [`docs/LAPTOP_SETUP.md`](docs/LAPTOP_SETUP.md).
- **Level 3 baseline:** Level 3 starts from `submission/level2` (`configs/live.yaml: baseline`).

---

## 6. Models, versions and external resources

| What | Model / version | Where |
|---|---|---|
| Cow detector | YOLO26-m (Ultralytics 8.4.212), fine-tuned at 1280 px on 216 hand-labelled frames from this pen | `models/angus_det_hprc_m1280.pt` (Git LFS) |
| Tracker | ByteTrack (Ultralytics), per camera | |
| Cow appearance (identity tie-break, memory cards) | Meta DINOv2 ViT-S/14 via `torch.hub` (`facebookresearch/dinov2`) | downloaded on first use; baked into the Docker image |
| Behavior classifier | DINOv2-S (`timm` `vit_small_patch14_dinov2.lvd142m`), last 4 blocks fine-tuned on our ~330 labelled clips + 9,269 CBVD-5 crops | `models/behavior/beh_finetuned.pt` (Git LFS), report in `models/behavior/ft_report.md` |
| Identity agent, stage 2 behavior agent | GPT-5.6 Luna (`protected.gpt-5.6-luna`) via the TAMU AI Chat API | `configs/agent.yaml`, `configs/pipeline.yaml` |
| Level 2 risk agent | Claude Opus 4.8 (`protected.Claude Opus 4.8`) via TAMU | `configs/risk.yaml` |
| Level 3 sign check | Gemini 3.5 Flash and Claude Opus 4.8 via TAMU | `configs/live.yaml` |
| Fallbacks | the same model on the OpenAI API (`OPENAI_API_KEY`), then local Qwen2.5-VL 7B through Ollama | `configs/agent.yaml` |

**External data and code:**
- CBVD-5 cattle behavior dataset (Kaggle, CC0)
- COCO-pretrained YOLO26 weights (Ultralytics)
- DINOv2 (Meta)
- OpenCV, ffmpeg, Label Studio (local, for our labelling)
- BRD literature for the indicator weights (sources in `configs/risk.yaml` and on every risk card)

**Tested environment:** Python 3.12.15 with ffmpeg 9.0 on macOS (Apple GPU), and Linux on TAMU HPRC (NVIDIA GPUs). The Docker image uses Debian's ffmpeg. Exact package versions: `requirements-lock.txt`.


---

## 7. AI tools and how we used Claude

**Inside the system** (every call is logged in the traces):

| Where | Model | What it does | What keeps it honest |
|---|---|---|---|
| Identity agent | GPT-5.6 Luna (TAMU) | Investigates duplicate IDs, ID jumps and camera conflicts; audits every 30 min | Code accepts a relabel only with ≥ 2 kinds of evidence, a top-2 appearance match and no duplicate ID |
| Stage 2 behavior agent | GPT-5.6 Luna (TAMU) | Re-checks flagged cow-minutes on the video with tools | Rule verifier; rumination needs measured chewing; it abstains rather than guess |
| Level 2 risk agent | **Claude Opus 4.8** (TAMU) | Audits each cow's indicators and looks for DART signs on 4K crops | Code, not the model, sets the score; a sign needs a cited image of that cow at a minimum size |
| Level 3 sign check | Gemini 3.5 Flash and **Claude Opus 4.8** (TAMU) | Ears, head carriage and nasal discharge on close crops | Each flag gets a second look and a measurement check (tracker speed, posture) |
| Benchmarked, not used | Claude Sonnet 5, local Qwen2.5-VL 7B | Compared on 8 identical identity cases (speed, first-try self-check pass) | |

**Building it:** we wrote the code with **Claude Code** (Anthropic's coding agent, Claude Opus models) as a pair programmer. 54 of our 58 commits are co-authored by Claude.

It helped with:
- the pipeline, agents, stitching and viewers;
- debugging the 48 h runs;
- the technical brief's layout and wording;
- this README and the Dockerfile.

The team chose the methods, labelled the training data, ran the footage and checked every number in the brief and in `submission/` against the outputs. We remain responsible for every claim.

## 8. Repository map

```
run_pipeline.py        videos -> Level 1 + Level 2 (one command, resumable)
live_watch.py          Level 3 live early warning
run_share.py           48 h in chunks;  stitch_shares.py  joins them
main.py                stage 1 tracking (detector + ByteTrack)
scripts/               run_level1.py, run_level2.py, viewers, calibration, anchors, identity tools
src/                   detection, tracking, identity, world state, behavior, agents, risk, live
brd_agent/             stage 2 behavior agent (verifies flagged cow-minutes on the video)
pipeline/              camera sorting, step orchestration, deliverables
configs/               cameras, pen layout, models, agents, risk weights and sources, live alert rules
models/                fine-tuned detector and behavior model (Git LFS)
submission/            our 48 h results (Levels 1–3), traces, ID-check sheets, evidence images
docs/                  technical brief, method notes per level, setup and troubleshooting
tests/                 pytest -q
```

Method notes:
- pipeline: [`docs/PIPELINE.md`](docs/PIPELINE.md)
- Level 2: [`docs/LEVEL2_METHOD.md`](docs/LEVEL2_METHOD.md)
- Level 3: [`docs/LEVEL3_LIVE.md`](docs/LEVEL3_LIVE.md)
- column definitions: [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md)

---

## 9. What this can and cannot support

- **Observed vs estimated:** every minute in `activity.csv` carries its basis. About 30 % of cow-minutes are not visible to any camera, and those are reported as such, never guessed.
- **Risk score:** the BRD risk is a ranked suspicion index against the herd, not a diagnosis or a calibrated probability. 10 cows over 48 h, with no clinical outcomes, cannot validate a BRD model.
- **Our 48 h call:** COW_02, COW_01 and COW_10 are early suspects on behavior (slower to the bunk, eat less), with no visible clinical sign confirmed. Confirm with temperature and examination.
