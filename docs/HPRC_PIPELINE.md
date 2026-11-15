# Day 2 on TAMU HPRC (Google Drive → /scratch → GPU array job)

Day 1 runs on the three laptops (`docs/LAPTOP_SETUP.md`); Day 2 runs here as share `hprc_day2`, one 2-hour chunk per
GPU task, several at once. The footage comes straight from Google Drive to `/scratch` with rclone (campus bandwidth,
no laptop upload). Everything stays in your own `/scratch/user/<NetID>` (folders are `chmod 700`): the event data
must stay within the event.

## 1. Code + weights (from a laptop that has run `git lfs pull`)

```bash
bash scripts/hprc/push_code.sh <NetID>               # rsync of the git-tracked files to /scratch/user/<NetID>/pen6
```

The repo is private, so this avoids setting up a GitHub key and git-lfs on the cluster. Rerun after `git pull`.

## 2. One-time setup on a login node

```bash
ssh <NetID>@grace.hprc.tamu.edu
cd /scratch/user/$USER/pen6
bash scripts/hprc/pipeline_setup.sh                  # venv + CUDA torch, static ffmpeg, rclone, DINOv2 cache
printf 'TAMU_API_KEY=%s\n' '<key>' > .env && chmod 600 .env
```

## 3. Footage from Google Drive

Connect rclone to Drive once (HPRC has no browser, so one step happens on your laptop):

```bash
tools/rclone config
#  n  -> name: gdrive -> storage: drive -> client_id / secret: Enter -> scope: 2 (read-only) -> service account: Enter
#  advanced: n -> "Use web browser to authenticate?": n
#  it prints:  rclone authorize "drive" "eyJ..."
#  run THAT line on your laptop (Mac: brew install rclone), sign in with the account the hackathon folder is shared
#  with, paste the result back -> shared drive: n -> y -> q
```

The organizers' folder is already in `configs/pipeline.yaml` (`drive:`: Shared with me > AISFS HACKATHON >
Problem 2: BRD Agentic Vision > Agentic Computer Vision Hackathon Dataset, with `Day 1/`, `Day 2/` and `Cam 01..03`).

```bash
bash scripts/hprc/fetch_footage.sh check    # lists the Drive folder: should say "Day 1, Day 2" and the clip count
# put the team's plan.json (made on a laptop with all of Day 1) in outputs/shares/ first, then:
bash scripts/hprc/fetch_footage.sh share    # Drive listing + download ALL of Day 2 (+ the Day 1 clips around midnight)
```

Use the **same `plan.json` as the laptops** (do not run `fetch_footage.sh plan` once they have started): the HPRC
share always runs to the end of Day 2, and the clips for it come from the Drive listing.

* `downloadQuotaExceeded`: Google throttles shared files many people download. Wait and rerun (it resumes), or copy
  the folder into your own Drive server-side and point `drive.folder` / `shared_with_me` at the copy.
* Want to stitch and run Level 2 here too? `fetch_footage.sh all` downloads everything.
* `.env` on HPRC: `TAMU_API_KEY=` and `OPENAI_API_KEY=` (the fallback), `chmod 600 .env`.

## 4. Run

```bash
bash scripts/hprc/submit_pipeline.sh                 # checks, then the 5-min smoke test, then every 2 h chunk
squeue -u $USER                                      # watch
tail -f /scratch/user/$USER/pen6_logs/pen6_pipe_<jobid>_0.out    # the smoke test's results
```

* The chunk tasks start only if the smoke test passes (`afterok`). At most 8 run at once (`MAX_AT_ONCE`), each on
  one T4 (`GPU=a100` for another type), with 3 BRD-agent and 3 identity-agent threads each so TAMU is not flooded.
* A chunk that fails or times out: submit again; finished steps are skipped.
* `PEN6_CHUNK_H=4` for 4-hour chunks (fewer, longer tasks).

## 5. Results

```bash
venv/bin/python run_share.py --who hprc_day2 --pack     # outputs/shares/hprc_day2.zip (results, no video)
```

Stitch where the footage is: here (after `fetch_footage.sh all`, with the laptops' zips copied up and unzipped in
this folder) or on a laptop (copy `hprc_day2.zip` down with `scp`). Then `python stitch_shares.py --videos <clips>`.
