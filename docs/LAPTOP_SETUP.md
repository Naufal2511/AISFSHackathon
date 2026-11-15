# Running your share of the 48 h on your laptop

Three laptops, three shares: `naufal_training`, `afeef_training`, `nooh_training`. Each laptop runs its share in
chunks (2 h or 4 h of footage each), then sends one zip to whoever stitches. Follow the steps in order. Do not skip
step 5: it catches setup mistakes in 15 minutes instead of 6 hours.

## 1. Get the code (everyone the same commit)

```bash
git clone https://github.com/Naufal2511/AISFSHackathon.git
cd AISFSHackathon
git lfs install
git lfs pull
```

Already cloned? `git checkout main && git pull && git lfs pull`.

## 2. Python environment (Python 3.10–3.12)

Mac / Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r brd_agent/requirements.txt
```

Windows (PowerShell):
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r brd_agent/requirements.txt
```

NVIDIA laptop: install the CUDA build of torch first (pytorch.org, "Get started"), then the line above.

## 3. ffmpeg

* Mac: `brew install ffmpeg`
* Windows: download `ffmpeg-release-full.7z` from gyan.dev, unzip it (for example to `C:\ffmpeg`). You will pass
  `--ffmpeg-dir "C:\ffmpeg\bin"` to every command below. The conda ffmpeg is broken on Windows; do not use it.

## 4. The `.env` file (TAMU key)

Create a file named `.env` in the repo folder (next to `README.md`) with one line:

```
TAMU_API_KEY=<your own TAMU AI Chat key>
OPENAI_API_KEY=<your own OpenAI key>
```

No quotes, no spaces. Each person uses their own keys. Every agent tries TAMU 3 times (waiting 2 s, then 4 s) and
then switches to the same model on OpenAI; the check below sends one tiny request to each to prove they work. There must be **no** `brd_agent/.env` (it would switch stage 2 away from TAMU).

## 5. The plan and the footage

* All laptops use the **same** `outputs/shares/plan.json`: Day 1 split over the three laptops, Day 2 on HPRC. It is
  made once on HPRC from the full Google Drive listing (`docs/HPRC_PIPELINE.md` step 3) and sent to everyone; put
  it in `outputs/shares/` (create the folder). (Without HPRC: `python run_share.py --plan-only --layout all-laptops
  --videos "<folder with ALL clips>"`.)
* Your video folder needs every clip of your share. Missing ones come straight from the organizers' Drive (Shared
  with me > AISFS HACKATHON > Problem 2 > Agentic Computer Vision Hackathon Dataset), only the ones you need:

  ```bash
  brew install rclone                                   # Windows: winget install Rclone.Rclone (then a new terminal)
  rclone config create gdrive drive scope=drive.readonly   # a browser opens: sign in with the account it's shared with
  python fetch_clips.py --who <you> --dest "<your video folder>"
  ```

  It keeps the Drive layout (`Day 1/Cam 01/...`), downloads only what is missing, and can be rerun. `run_share.py`
  also offers to do this itself when a chunk is missing clips.

## 5b. GPU

Nothing to set: every model uses the NVIDIA GPU (CUDA) if there is one, else the Apple GPU (MPS) on a Mac, else
the CPU. The check below runs a computation on it. If the Mac GPU misbehaves, force the CPU for that laptop:
`export PEN6_DEVICE=cpu` (Windows PowerShell: `$env:PEN6_DEVICE="cpu"`) before running anything.

## 6. Check the laptop

```bash
python check_setup.py --who <you> --videos "<your video folder>"          # Windows: add --ffmpeg-dir "C:\ffmpeg\bin"
```

It ends with **READY** or **NOT READY**. Every problem has a `FIX:` line; fix them and run it again until READY.
Read the warnings too. Post the line `plan.json fingerprint ...` in the team chat: all three must match, and the
`code is up to date` line must show the same commit on all three.

## 7. Run your share

```bash
python run_share.py                         # Windows: python run_share.py --ffmpeg-dir "C:\ffmpeg\bin"
```

It asks who you are and where the clips are, then:

1. runs the setup checks again (stops on any FAIL), including that **every clip your share needs is in your video
   folder**: if not, it stops, says which chunks are affected and writes the list to
   `outputs/shares/<you>/missing_clips.txt`;
2. runs a **5-minute smoke test** through the whole pipeline with real TAMU (10–20 min) and prints ✔ / ✖ for each
   check plus an estimate for your whole share. It stops if anything fails;
3. asks the chunk length (**2** or **4** hours) and how many chunks at once (it suggests a number for your
   machine), then runs them. Each chunk prints when it starts and ends; logs are in `outputs/shares/<you>/`.

If anything stops (crash, sleep, lost Wi-Fi): run the same command again. Finished steps and chunks are skipped.
Keep the laptop plugged in and stop it from sleeping (Mac: `caffeinate -dims python run_share.py`; Windows: Power
settings → Sleep: Never).

Look at any finished chunk: `python scripts/run_viewer.py --run outputs/shares/<you>/<you>_c01`, then open the
`viewer.html` it writes.

## 8. Send the results

```bash
python run_share.py --who <you> --pack
```

Send `outputs/shares/<you>.zip` to the stitcher (it holds results only, no video). The stitcher unzips all three at
the repo root and runs `python stitch_shares.py --videos "<all clips>"`.

## Settings everyone shares (configs/pipeline.yaml)

* Frame rate `stage1.fps` (1 fps unless the team agrees otherwise; 1 frame / 5 s was tested and breaks cow IDs),
  detector `models/angus_det_hprc_m1280.pt` at 1280 px.
* Identity agent and BRD agent on TAMU `protected.gpt-5.6-luna`; at most 120 full BRD-agent checks per hour of
  footage, so TAMU load is predictable.
* Each chunk also tracks the 2 minutes before its window; that overlap is how the stitcher keeps cow IDs the same
  across chunks and laptops.

Do not change these on one laptop only: the three results must be comparable.
