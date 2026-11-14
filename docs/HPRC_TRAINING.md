# Training the cow detector on TAMU HPRC (Tesla T4)

One Slurm array job fine-tunes three COCO-pretrained YOLO26 sizes (s, m, l) on the
hand-labelled frames, one T4 each, in parallel. On the Mac (16 GB unified memory) yolo26m at
1280 needs batch 2 and ~2 min/epoch. A T4 runs it at batch 8 with fp16.

> Data rule: event data must stay within the event. Keep it in your own `/scratch/user/<NetID>`
> (setup makes the folder `chmod 700`), do not put it in shared/project space, and check with the
> organizers if you're unsure whether TAMU HPRC counts as "within the event".

## 1. On the Mac: bundle (≈ 575 MB)

```bash
./scripts/hprc/pack.sh
```

Contents: the 216 reviewed frames + labels, train/val/test lists, COCO weights
(`yolo26n/s/m/l/x.pt`), and `scripts/hprc/`.

Split (by clip, so no near-duplicate frames leak across):

| split | frames | boxes | used for |
|---|---|---|---|
| train | 150 | 1,594 | training |
| val | 43 | 354 | early stopping, picking best.pt, choosing s/m/l |
| test | 23 | 212 | final, untouched score |

## 2. Copy it to the cluster and set up once (on a login node)

```bash
ssh <NetID>@grace.hprc.tamu.edu          # FASTER: faster.hprc.tamu.edu
mkdir -p /scratch/user/$USER/angus
```

From the Mac:

```bash
scp outputs/hprc_bundle.tar.gz <NetID>@grace.hprc.tamu.edu:/scratch/user/<NetID>/angus/
```

On the cluster:

```bash
cd /scratch/user/$USER/angus && tar -xzf hprc_bundle.tar.gz && bash scripts/hprc/setup.sh
```

`setup.sh` loads Python via `scripts/hprc/env.sh`, creates `venv/` with `ultralytics==8.4.212`,
writes `dataset_hprc.yaml` with the cluster path, and creates `logs/ runs/ test_preds/`.
If the module line fails, run `module spider Python` and edit `env.sh`.

## 3. Train

```bash
sbatch scripts/hprc/train.slurm            # tasks 0,1,2 = yolo26s, yolo26m, yolo26l
squeue -u $USER
tail -f logs/yolo26m_1280_hand_train_b8.log
```

- **GPU**: `--partition=gpu --gres=gpu:t4:1`. Check the names with `sinfo -o "%P %G"` if the job
  pends with an invalid gres.
- **Logs**: `/scratch/user/<NetID>/angus/logs/`, holding the Slurm output plus one log per training attempt.
- **Checkpoints**: `runs/<model>_1280_hand/weights/last.pt` (every epoch), `epochN.pt` (every 10), `best.pt`.
- **Resume**: 5 min before the time limit the job saves, requeues itself and continues from
  `last.pt`. Re-running `sbatch` also resumes. Delete `runs/<name>` to start over.
- **Out of memory**: the batch is halved automatically (s 16 → 8, m 8 → 4, l 4 → 2).
- **Test images**: after training, each task scores `best.pt` on the held-out test split
  (`runs/<name>_test/`) and saves the test images with predicted boxes (`test_preds/<name>/`).
- Only one model: `sbatch --array=1 scripts/hprc/train.slurm` (m). yolo26x: `--array=3 --time=08:00:00`.
- Quick try: `EPOCHS=30 sbatch ...`.

## 4. Back on the Mac: fetch and pick the model

```bash
./scripts/hprc/fetch.sh <NetID>
```

Then the per-camera, day/night recall on the TEST split against COCO-x:

```bash
.venv/bin/python scripts/compare_detectors.py --hprc-split test \
  --model s=outputs/hprc/runs/yolo26s_1280_hand/weights/best.pt@1280 \
  --model m=outputs/hprc/runs/yolo26m_1280_hand/weights/best.pt@1280 \
  --model l=outputs/hprc/runs/yolo26l_1280_hand/weights/best.pt@1280 \
  --model coco_x=yolo26x.pt@1280
```

Pick the smallest model within ~1–2 recall points of the best, since speed matters for processing
whole days of 3-camera video. Then run the pipeline with it:

```bash
python main.py --front "input/Day 1/Cam 01" --top "input/Day 1/Cam 02" --back "input/Day 1/Cam 03" \
  --start "2026-11-07 07:00" --end "2026-11-07 09:00" --fps 1 \
  --model outputs/hprc/runs/yolo26m_1280_hand/weights/best.pt --imgsz 1280 --no-video
```

Camera folders are read as one continuous stream per camera, so the tracker doesn't restart at
the 5-minute file boundaries.
