# Data contract: tracking → behavior model → validation agent

All tables are CSV, one header row, UTF-8. They join on **`wall_time` + `cow_id`** (and `camera`
for per-box rows). Produce them with:

```bash
python scripts/export_contract.py --run outputs/level1_<slot>
```

which writes `<run>/contract/observations.csv`, `<run>/contract/cow_state.csv` and `manifest.json`.

## Conventions

| Thing | Convention |
|---|---|
| Time | `wall_time` = ISO local wall clock, 1 s resolution (`2026-11-07T16:00:30`). `offset_s` = seconds into `file` (for seeking the video). |
| Cameras | `cam1` = Front Camera 1, `cam2` = Front Camera 2 (bunk), `cam3` = Back Camera (back-left corner). |
| Cows | `COW_01` … `COW_N`; **empty = no identity** (never guessed). N comes from the anchor herd-size estimate. |
| Boxes | pixels at native resolution; `frame_w`, `frame_h` given (cam1/cam3 4000×3000, cam2 2592×1944). |
| Pen metres | X 0–12 (0 = chute side), Y 0–30 (0 = feed bunk / front). |
| Behaviors | exactly: lying, standing, walking, feeding, drinking, ruminating (lying), ruminating (standing), grooming, social interaction, other / not visible. Probability columns use the short names below, always all 10, summing to 1. |

## 1. `observations.csv`: one row per box per second per camera

Filled by the tracking pipeline; **the per-box behavior model fills only the behavior columns**.

| Column | Meaning |
|---|---|
| wall_time, file, offset_s | when / which clip / where in it |
| camera, camera_name | see conventions |
| track_id, track_piece | tracker track and piece (a track is split where it switched cows) |
| cow_id | identity (empty = unknown) |
| id_conf, id_evidence | 0.95 `agreement` (2+ cameras agree on position), 0.85 `continuity`, 0.8 `seed`, 0.55 `attached`, 0.5 `part` |
| box_role | `body` or `part` (head / limb box of a cow that already has a body box) |
| x1, y1, x2, y2, frame_w, frame_h | box |
| det_conf | detector confidence |
| world_x, world_y | this camera's ground position estimate (empty for parts) |
| visibility, occluded_by, facing | `visible` / `partial` (+ `occluded_by` = cow or `frame edge`) ; facing `toward` / `away` / `side-on` / `unknown` relative to this camera |
| behavior, beh_conf | **behavior model**: top class and its probability |
| p_lying, p_standing, p_walking, p_feeding, p_drinking, p_rum_lying, p_rum_standing, p_grooming, p_social, p_other | **behavior model**: all 10 probabilities |
| model | behavior model name / version |

Example:
```
2026-11-07T16:00:30,Event20261107160019006.Avi,11.0,cam1,Front Camera 1,2,0,COW_02,0.95,agreement,body,645,947,1436,1632,4000,3000,0.74,4.59,4.06,visible,,unknown,lying,0.97,0.97,0.02,0,0,0,0.01,0,0,0,0,beh-yolo-v1
```

## 2. `cow_state.csv`: one row per cow per second (fused over cameras)

| Column | Meaning |
|---|---|
| wall_time, cow_id | key |
| x, y, pos_sd_m | fused pen position (Kalman over all cameras) and its uncertainty |
| speed_mps, heading_deg | motion (heading empty when standing still) |
| predicted | true = no camera saw the cow this second (position predicted) |
| n_cams, cams_visible | cameras that observed it / that see it at least partly |
| best_cam_identity, best_cam_head | camera to look at for identity (clear, large) and for the head (facing it) |
| zone | `bunk` / `waterer` / `rest` / `other` |
| nearest_cow, nearest_m | closest other cow and distance |
| behavior_fused, beh_conf_fused, beh_evidence | fused per-cow behavior (`observed`, `observed_chewing`, `estimated_*`, `not_visible`) |
| p_lying … p_other | fused probabilities |
| model | fusion / classifier version |
| anomaly | `jump`, `camera_conflict`, `camera_offset`, `speed`, `outside_pen` when present |

## 3. `behavior_verdicts.csv`: written by the validation agent

| Column | Meaning |
|---|---|
| cow_id, t0, t1 | interval checked (wall_time) |
| behavior_in | what the classifier said |
| verdict | `confirm` / `override` / `uncertain` |
| behavior_out | final behavior (= behavior_in when confirmed) |
| confidence | agent confidence 0–1 |
| evidence | short text: the facts used (camera, frames, chew rate, zone…) |
| trace_file | path of the reasoning trace (markdown + images) |

Verdicts are applied to the per-second fused behavior to produce the final `activity.csv`
(per cow per minute) and `time_budgets.csv`.

## Flow

```
tracking (YOLO + ByteTrack, 3 cameras)
  -> identity by agreement (+ anchors)          identity/tracking_records_resolved.csv
  -> world state (positions, visibility, anomalies)   state/
  -> observations.csv + cow_state.csv            contract/      <- behavior model fills p_* per box
  -> fusion per cow per second                   cow_state.csv behavior_fused
  -> validation agent                            behavior_verdicts.csv + traces
  -> activity.csv, time_budgets.csv, ID-check frames (Level 1 deliverables)
```
