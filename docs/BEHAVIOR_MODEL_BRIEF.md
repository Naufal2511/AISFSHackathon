# Behavior model brief: what exists, what we need, where it plugs in

This is for whoever trains the **per-box behavior classifier**. It explains the pipeline around the
model, the data you can train on, what the model must take in and give back, and how its output is
used afterwards. The exact column definitions are in [DATA_CONTRACT.md](DATA_CONTRACT.md).

> **Event data rule:** videos, frames, crops and labels stay on our own machines (or your private
> HPRC scratch). No cloud labelling, no Drive/Colab uploads, no public model hubs with event data.

---

## 1. The task in one paragraph

Pen 6 has about 10 Angus steers filmed by 3 fixed cameras for 2 days, day and infrared night. The
Level 1 deliverable is **"what is each cow doing, every minute"** over the 10 official behavior
classes. Detection, tracking and cow identity are already done. Your model looks at **one cow's box
(one camera, one moment)** and says what that cow is doing, as **10 probabilities**. Everything
after that (combining cameras, smoothing over time, minute labels, checking by the agent) is already
built and only needs your probabilities.

---

## 2. The pipeline and where your model sits

```
 raw video (3 cameras, 5-min .Avi files)
   │
   ├─ 1. detection        fine-tuned YOLO26 cow detector (models/angus_det_hprc_m1280.pt)
   ├─ 2. tracking         ByteTrack per camera, tracks split at ID switches
   ├─ 3. identity         COW_01..COW_N by cross-camera position agreement + anchors
   ├─ 4. world state      pen position (metres), speed, zone, visibility/occlusion, facing, anomalies
   │
   ├─ 5. observations.csv one row per box per second per camera  ◄── YOUR MODEL fills p_* here
   │
   ├─ 6. fusion           all cameras of one cow in one second → one 10-class vector
   │                      (weighted by camera expertise + detection quality, product of experts
   │                       with the geometric cues)
   ├─ 7. HMM smoothing    per cow, 1 s steps, bout-length priors, forbids lying→walking jumps
   ├─ 8. validation agent re-checks low-confidence / hard bouts with frames + VLM,
   │                      writes behavior_verdicts.csv (confirm / override / uncertain)
   └─ 9. deliverables     activity.csv (per cow per minute), time_budgets.csv
```

Code map:

| Step | Code |
|---|---|
| 1–3 | `main.py`, `src/detection/`, `src/tracking/`, `scripts/resolve_identities.py` |
| 4 | `scripts/build_world_state.py`, `src/state/world_state.py` |
| 5 | `scripts/export_contract.py` → `<run>/contract/observations.csv` |
| 6–7 | `src/behavior/fusion.py` (`fuse_cameras`, `behavior_probabilities`, `smooth_cow`) |
| 6 (learned slot) | `src/behavior/appearance.py` (`predict_rows` → `app::<class>` columns) |
| 8 | `run_agent.py`, `src/agent/` (`triggers.py`, `loop.py`, `tools.py`) |
| 9 | `src/behavior/export.py` |
| end-to-end | `scripts/run_level1.py` |

---

## 3. What the system does today without you (the baseline you need to beat)

`classify_behavior.py` is a **geometry-only** classifier (`model = heuristic-fusion-v1`). It is
decent at what geometry can see and blind to the rest:

| Class | How it is decided today | Quality |
|---|---|---|
| lying / standing | box height vs. the expected standing height at that image row (ratio < 0.66 = lying) | good on cam1/cam3, useless on cam2 (overhead) |
| walking | ground-point speed > 0.3 m/s | good |
| feeding | box overlaps the bunk band + dwell ≥ 20 s | ok, but confuses queueing with eating |
| drinking | near the waterer + dwell ≥ 5 s | ok |
| ruminating (lying/standing) | only if a jaw-motion spectrum probe (`src/behavior/rumination.py`) measures 45–95 chews/min on long still bouts | sparse, slow |
| grooming | box width oscillating while the cow stays still | weak (capped at p = 0.55) |
| social interaction | another cow within 1.2 m and moving | weak (capped at p = 0.7) |
| other / not visible | poor detection quality or long unseen gaps | ok |

**So the value of your model is mostly in: rumination, grooming, social interaction, feeding vs.
standing-in-queue, and posture on cam2.** Matching geometry on lying/standing/walking is fine;
beating it on the hard classes is the point.

---

## 4. The 10 classes (exact strings, exact order)

Defined in `src/behavior/taxonomy.py` (`BEHAVIORS`). Use this order for the output vector.

| # | Class (exact string) | Short column | Folder name (`to-cls`) | Labelling rule |
|---|---|---|---|---|
| 0 | `lying` | `p_lying` | `lying` | lying, not chewing |
| 1 | `standing` | `p_standing` | `standing` | standing still, incl. waiting behind the bunk |
| 2 | `walking` | `p_walking` | `walking` | moving |
| 3 | `feeding` | `p_feeding` | `feeding` | head in or over the bunk |
| 4 | `drinking` | `p_drinking` | `drinking` | at the waterer, head down |
| 5 | `ruminating (lying)` | `p_rum_lying` | `ruminating_lying` | lying **and visibly rhythmic chewing** |
| 6 | `ruminating (standing)` | `p_rum_standing` | `ruminating_standing` | standing **and visibly rhythmic chewing** |
| 7 | `grooming` | `p_grooming` | `grooming` | licking / scratching itself |
| 8 | `social interaction` | `p_social` | `social_interaction` | head butting, allogrooming, mounting, pushing |
| 9 | `other / not visible` | `p_other` | `other_not_visible` | can't tell / mostly hidden / anything else |

Rumination is never guessed: it needs chewing to be seen. If the model can't see the jaw, it should
put the mass on `lying` / `standing`, not on rumination.

---

## 5. Training data we have

### 5a. Our own labelled clips (the domain that matters)

Built by `scripts/build_behavior_tasks.py build` into `data/behavior_annotation/`, labelled in a
local Label Studio (see [HANDOFF_TEAMMATE.md](HANDOFF_TEAMMATE.md), [HANDOFF_TOPUP.md](HANDOFF_TOPUP.md)).

One **task = one cow, one camera, one 5-second window**:

| File | What it is |
|---|---|
| `clips/<task>.mp4` | zoomed, smoothed 5 s crop that follows the cow, 10 fps, target outlined (for the human) |
| `context/<task>.jpg` | full frame, target cow in a thick box (for the human) |
| `crops/<task>/fNN.jpg` | **5 clean crops (no drawing), 512×512, spread over the 5 s → training images** |
| `crops/<task>/boxes.json` | source clip path, `t_start`, fps, frame size, crop side, IR flag, and the **per-frame box of the cow** (all 50 frames). Use this to re-decode the raw video at any resolution/length you want (e.g. for a temporal model). |
| `manifest.csv` | task_id, split, camera, clip, clip_start, t_start, night/ir, n_seen, guess, … |

Status: about **330 tasks built** across cam1/cam2/cam3, day and night, both days; a high-quality batch
of 75 (25 per camera, contrast-enhanced at night) is labelled first. Labels are being done now; the
export comes back as a Label Studio JSON.

Turn an export into a training set:

```bash
.venv/bin/python scripts/build_behavior_tasks.py to-cls --export <behavior-export>.json
```

This writes

- `data/behavior_annotation/cls/{train,val}/<class_folder>/<task>_fNN.jpg` (YOLO-cls / ImageFolder layout, every class folder exists in both splits),
- `data/behavior_annotation/behavior_labels.csv` (one row per task: label, split, camera, flags).

Rules baked in:
- **Split is by source clip** (md5 hash, 25% val), so the 5 crops of a task, and nearby windows of the
  same clip, never land on both sides. Keep it that way. Splitting by image will give fake-high scores.
- `bad track` and `unsure` tasks are dropped; `mostly occluded` is kept (use it as a hard case or as `other`).
- Pre-selected guesses (feeding / drinking / walking) were checked by the labeller, not trusted.

Expect class imbalance: lots of lying/standing/feeding, few grooming/social/drinking. Plan for
class weights or resampling, and report per-class numbers.

### 5b. Public datasets (optional, to cover rare classes)

Already wired in `src/behavior/appearance.py` with label maps in `taxonomy.py`:

| Dataset | Licence | Classes mapped |
|---|---|---|
| CVB (CSIRO, 8 Angus on pasture, AVA format) | CC BY-NC-SA 4.0 | all 10 incl. rumination lying/standing, grooming |
| CBVD-5 (Kaggle) | CC0 | standing, lying, feeding, drinking, rumination (split by box aspect) |

Different domain (pasture, daylight). Cap them per class (the probe script caps at 400) so they can't
swamp our data, and always **evaluate on our event val split only**.

### 5c. More data if you need it

Ask and we'll build more windows (new `--seed`, `--only-guess feeding`, `--per-camera`, etc.), or
pull crops from any tracked run: every box in `observations.csv` has `file`, `offset_s` and
`x1..y2`, so any second of any cow is one seek away.

---

## 6. What the model must do (the interface)

### Input

For one detected cow box: the raw frame(s) from `file` at `offset_s` and the box `x1,y1,x2,y2`
in native pixels.

| Camera | Resolution | View | Notes |
|---|---|---|---|
| `cam1` Front Camera 1 | 4000×3000 | front overview | good posture, sees bunk and waterer |
| `cam2` Front Camera 2 | 2592×1944 | above the feed bunk, near-overhead | best for feeding; posture from box height is useless here |
| `cam3` Back Camera | 4000×3000 | back-left corner | far cows are small |

- Night is **infrared greyscale**; it has to work on both. Day/night is in `manifest.csv` (`night`, `ir`).
- Boxes can be `partial` (cut by another cow or the frame edge) — see `visibility` / `occluded_by`.
- `box_role = part` rows are a head/limb box of a cow that already has a body box; you can skip them
  or use them (a head box is useful for chewing).
- Pad the crop (we use ~10–60% context) so the bunk/waterer/neighbour stays in view.

**Single frame vs. short clip:** a single crop is enough for lying / standing / feeding / drinking.
Rumination, grooming, social interaction and even walking are **motion** behaviors, so a temporal
input (e.g. 8–16 frames over 2–5 s around the second, using the per-frame boxes) will help a lot.
Either is fine; a single-frame model first is a perfectly good v1.

### Output

Per box: **10 probabilities in the order of §4, summing to 1**, plus top class and its probability.

- **Calibration matters more than top-1.** Fusion multiplies your vector with the geometric one and
  the HMM integrates it over time. An over-confident wrong 0.99 is much worse than an honest 0.6.
  Please check calibration on val (reliability plot / ECE) and apply temperature scaling if needed.
- If the crop is unusable (blurred, mostly hidden, not a cow), put the mass on `other / not visible`
  rather than guessing.
- Give the model a version string (e.g. `beh-yolo-v1`); it goes in the `model` column.

### Throughput

The pipeline runs at **1 box per cow per camera per second**. A 10-minute run is ~11k box rows; a
full day of 3 cameras is on the order of 1–2 million. A per-crop model at ~224–384 px is fine; we can
also score every 2nd–10th second and let the HMM fill in (`--appearance-every-sec`).

---

## 7. How to hand the model over (pick one)

**Option A — fill the contract (preferred, cleanest).** A script that reads
`<run>/contract/observations.csv`, decodes each `file` at `offset_s`, crops `x1..y2`, and writes back
the same file with `behavior`, `beh_conf`, `p_lying … p_other`, `model` filled. Nothing else changes.
Example row with the behavior part filled:

```
2026-11-07T16:00:30,Event20261107160019006.Avi,11.0,cam1,Front Camera 1,2,0,COW_02,0.95,agreement,body,645,947,1436,1632,4000,3000,0.74,4.59,4.06,visible,,unknown,lying,0.97,0.97,0.02,0,0,0,0.01,0,0,0,0,beh-yolo-v1
```

A real run to test on already exists: `outputs/level1_d1_1600/contract/observations.csv`
(Day 1, 16:00–16:10, ~11k rows, ~90% of boxes have a cow ID).

**Option B — a Python predictor.** A function we can call from `src/behavior/appearance.py`:

```python
def predict(crops: list[np.ndarray]  # BGR crops (or list of frame stacks for a temporal model)
            ) -> np.ndarray:          # (N, 10) probabilities, order = taxonomy.BEHAVIORS
```

plus the weights file. We wrap it so it produces the `app::<class>` columns that fusion already
consumes (product of experts with geometry, `fusion.behavior_probabilities`).

Either way the model's output is also exposed to the validation agent as `behavior_2`: wherever your
model and the geometric classifier disagree for a while, the agent gets a `behavior_check` flag and
goes to look (`src/agent/triggers.py`). So disagreement is useful, not a failure.

What happens downstream (so you know what not to build):
- combining the 3 cameras — done (camera weights in `configs/behavior.yaml`, e.g. cam2 trusted for feeding, not posture),
- smoothing over time / bout lengths / impossible transitions — done (HMM),
- per-minute labels and time budgets — done,
- observed vs. estimated tagging — done (`evidence_label`).

Your model only has to be good **per box, per moment**.

---

## 8. How to evaluate (what to send back with the model)

On the **event val split** (by clip, from `to-cls`), never on images seen in training:

1. Per-class precision / recall / F1 and **macro-F1** (the headline number; accuracy is dominated by lying/standing).
2. Confusion matrix (we care most about feeding↔standing, lying↔ruminating (lying), standing↔ruminating (standing), grooming/social↔anything).
3. Broken down **per camera** and **day vs. night (IR)**.
4. Calibration (reliability diagram or ECE) before/after temperature scaling.
5. The same numbers for the current geometric baseline on the same tasks, if you can (we can compute it for you from the task windows).

Plus: weights file, the exact class order, input size / preprocessing (padding, normalisation,
grey-to-3-channel at night), and the version string.

---

## 9. Suggested starting points (not requirements)

- **Fastest baseline:** YOLO-cls on the `to-cls` folders —
  `yolo classify train model=yolo11s-cls.pt data=data/behavior_annotation/cls imgsz=224 epochs=60 batch=64`.
- **Already in repo:** frozen DINOv2 ViT-S/14 embedding + logistic regression (`train_behavior_probe.py`),
  trains in seconds on CPU, can mix in CVB/CBVD-5. Good sanity check / fallback.
- **For the motion classes:** a small video model (e.g. X3D/SlowFast/VideoMAE-small, or a frame
  encoder + temporal pooling) on 2–5 s cow-following clips re-decoded from `boxes.json`.
- GPU: TAMU HPRC T4s are set up for the detector ([HPRC_TRAINING.md](HPRC_TRAINING.md)); the same
  pack/sbatch pattern works for this model. Keep data in your private `/scratch/user/<NetID>`.

---

## 10. Quick checklist

- [ ] Train on `data/behavior_annotation/cls` (by-clip split), optionally + capped CVB/CBVD-5
- [ ] Output = 10 calibrated probabilities in `taxonomy.BEHAVIORS` order
- [ ] Works on cam1/cam2/cam3, day and IR night, partial boxes
- [ ] Unusable crops → `other / not visible`; no rumination without visible chewing
- [ ] Deliver as Option A (fill `observations.csv`) or Option B (`predict()` + weights)
- [ ] Send back: weights, version string, preprocessing, per-class/per-camera/day-night metrics, confusion matrix, calibration
- [ ] No event data leaves our machines / private scratch
