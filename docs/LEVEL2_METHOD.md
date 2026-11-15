# Level 2: BRD risk from behavior, checked on video

```
activity (per cow per minute)  ─┐
cow_state (per cow per second) ─┼─► 13 indicators ─► herd-relative z ─► weighted, coverage-shrunk ─► behavior score
                                │                    (+ day-to-day trend)       (literature weights)        │
frames (4000x3000 source)  ─────┴─► risk agent (TAMU Claude Opus 4.8) with inspection tools ──────────────►│
                                    audits indicators, looks for DART signs, cites evidence ids              ▼
                                    deterministic self-check ─► signs + artifacts ─► final risk = behavior logit + sign log-odds
                                                                                     ─► brd_risk.csv, risk cards + clips, index.html
```

Run: `python scripts/run_level2.py --runs <level1 run dirs...> --activity activity.json --out outputs/level2`
(`--skip-agent` for behavior only). Configuration, weights and sources: `configs/risk.yaml`.

## Inputs

* **activity**: one row per cow per minute, JSON (`[{...}]`, `{"rows": [...]}` or `{"COW_01": [...]}`)
  or CSV. Required: `cow_id`, a minute time (`minute_start` / `wall_time` / `timestamp`, ISO or epoch)
  and `behavior` (the 10 classes; common spellings such as `rum_lying` are mapped). Optional:
  `confidence`, `sec_<behavior>` seconds per class, `seconds_observed`.
* **cow_state** from each Level 1 run (`contract/cow_state.csv`): fused position, zone, nearest cow and
  distance, speed, and whether the position was observed or predicted.
* **observations** and source video, used by the tools for frames, crops and clips.

## Indicators (per cow, over all observed time and per day)

| indicator | sick direction | weight | why (sources in `configs/risk.yaml`) |
|---|---|---|---|
| time feeding | lower | 1.0 | Bunk time and head-down time fall days before BRD diagnosis [tamu_cusum, wolfger2015]. Feeding time falls by the diagnosis day [toaff2018]. |
| minutes to the bunk after a herd feed rush | higher | 0.8 | DART appetite sign: "slow to come to eat" [dart]. |
| metres walked per hour | lower | 0.7 | Activity drops ≥ 6 d before diagnosis in feedlot calves [feedlot_acc], and inactivity rises in calves [ear_acc2022]. |
| bunk visits per hour, minutes per visit | lower | 0.6 each | Bunk-visit frequency and duration are among the best early traits [tamu_cusum, wolfger2015]. |
| time lying | higher | 0.5 | Longer lying before diagnosis [ear_acc2022]. The direction differs between studies, hence the moderate weight. |
| metres to nearest herdmate; time alone (> 5 m) | higher | 0.5 each | Social withdrawal is part of sickness behavior [hart1988, scirep2022]. |
| time out of step with the herd | higher | 0.5 | For example, lying while the herd feeds: withdrawal and anorexia [hart1988, toaff2018]. |
| time ruminating | lower | 0.3 | Reduced rumination [marchesini2018]. Down-weighted because rumination from video is hard. |
| minutes per lying bout | higher | 0.3 | Lying-bout changes [feedlot_acc, ear_acc2022]. |
| waterer visits per hour | lower | 0.2 | Water intake falls but visit counts barely change [padua_water], so the weight is low. |
| time self-grooming | lower | 0.2 | Self-maintenance drops in sickness behavior [hart1988]. |

A **feed rush** is the herd moving to the bunk: occupancy of at least 50 % after at least 15 min at or below 20 %. It stands in for feed delivery, which the cameras cannot see.

## Score

1. **Herd-relative robust z.** For each indicator, `(value - herd median) / max(1.4826·MAD, floor)`. The herd shares feed, weather and handling, so comparing within the pen removes those effects.
2. **Sick direction only.** The z is signed so that positive means the sick direction, then clipped at ±3. Healthier-than-herd values offset at ¼ strength, because sickness signals are one-sided.
3. **Coverage shrinkage.** Each z is multiplied by `h / (h + 2)`, where h is hours observed. A cow seen for 20 minutes cannot top the ranking on noise.
4. **Self-relative trend** (with ≥ 2 days). The day-to-day change of each indicator is scored against the herd's own change, at half weight. A cow that declines while the herd does not stands out.
5. **Combination.** A weighted Stouffer sum `Σ w·z / √Σw²` gives the logit `-2.2 + 1.0·z`. A herd-typical cow scores about 0.10. This is a ranked suspicion index, not a calibrated probability: there are no labels to calibrate against.
6. **Visible signs** from the agent add fixed log-odds. Present: nasal discharge or laboured breathing +0.9, head low or ear droop +0.6, cough +0.7, ocular discharge or gaunt +0.4. Small negatives apply when a sign was checked and found absent, floored at -0.5. The model reports what it sees; it never sets the number.
7. **Rank stability.** Recompute after resampling 30-min blocks of time 200 times. `p_top3` is how often the cow stays in the top 3.

## The agent (TAMU, Claude Opus 4.8; GPT-5.6 Sol and then Luna as fallback)

For each cow, the agent receives the indicator profile with fact ids. The top N cows get up to 9 tool calls; the rest get 3. Its jobs:

* **Audit.** Decide whether a deviation is real or an artifact: a wrong ID, a cow out of view, or a behavior label the images contradict. Artifacts must cite image evidence; short coverage alone is not an artifact.
* **Look.** Check the DART signs on close views.
* **Report.** Return structured findings that cite evidence ids.

| tool | gives the model |
|---|---|
| `cow_profile`, `herd_compare` | indicator values vs herd, hours, contributions, fact ids |
| `timeline` | per-minute behavior of the cow vs the herd, and distance from the bunk (chart) |
| `find_moments` | times worth looking at: close_view (largest box, daylight first), feeding, lying, isolated, out_of_step, walking, at_bunk |
| `camera_info` | which cameras see the cow; box size in source pixels; distance; facing; occlusion; day or IR night |
| `get_boxes`, `get_frame` | all detections at a second; the whole frame with IDs |
| `get_crop` | crop from the 4000×3000 source: body, head (from facing), halves or ends. Enhanced with Lanczos upscale, denoise, CLAHE and unsharp masking. |
| `get_sequence` | 4–8 crops over a few seconds (head carriage, flank effort, cough posture) |
| `compare_herdmates` | the cow beside up to 3 pen-mates at the same moment (posture is relative) |
| `record_fact` | saves an observation that cites the images it is based on |

**Evidence store** (`<out>/evidence/facts.jsonl` + `img/`):
* every number (`F-<cow>-<indicator>`);
* every image the model was shown (`I####`, with camera, time, box px and daylight);
* every observation (`V####`).

The cards show exactly what the model saw.

**Self-check** (deterministic; one revision is allowed, then downgrades):
* A sign is present or absent only with a cited image of that cow, box ≥ 250 px.
* Discharge needs a box ≥ 450 px in daylight colour. IR night video cannot show it.
* Artifacts need image evidence.
* Cited ids must exist.

Anything else becomes `not_assessable`.

## Outputs

* `brd_risk.csv`: cow_id, risk_score (0–1), rank, top_reasons, behavior_score, visual_adjustment, p_top3, hours_observed, confidence, signs_present, artifacts.
* `cards/<COW>.html`, one page each for the top 3:
  * the action line;
  * reasons with numbers vs the herd;
  * an indicator chart;
  * 2–3 clips (strongest behavior deviation, the image behind a visible sign, the closest view);
  * a signs table with evidence thumbnails;
  * the agent's assessment;
  * sources.
* `index.html`: the ranking with stability, the visualization page. `indicators.csv`, `contributions.csv`: the full numbers.

## Honest limits

* This is screening from behavior, not diagnosis. Clinical-sign diagnosis of BRD is itself only about 62 % sensitive and 63 % specific [white2009]. Confirm with temperature and auscultation.
* Herd-relative scoring always ranks someone first. Absolute risk stays low when no cow deviates much, which is why the brief allows for no BRD.
* Behavior quality limits everything downstream. Feeding and lying come from the behavior model. Rumination from video is the weakest signal and is weighted accordingly.
* 48 h is short for self-relative baselines. The trend term is half weight and needs both days.

## Feature table (team list) and where each comes from

`features_hourly.csv` (per cow per hour) and `features_summary.csv` (all / day / night / day1 / day2 /
change), each feature with a herd z (vs the pen median the same hour) and a rank among the cows.

| # | feature | source | scored? |
|---|---|---|---|
| 1–3 | lying min, lying bouts, longest lying bout | behavior classifier | yes (lying share, bout length, longest bout) |
| 4 | standing min | classifier | table only |
| 5–8 | feeding min, meals, meal length, longest gap between meals | classifier (meals = feeding minutes merged across ≤ 5 min pauses) | yes |
| 9–10 | drinking min, drinking visits | classifier / waterer zone | visits (low weight) |
| 11 | walking min | classifier | table only (distance is scored) |
| 12 | ruminating min | classifier | yes (low weight) |
| 13 | social interaction min | classifier | table only (direction unclear) |
| 14–15 | distance, speed | fused tracking; speeds < 0.25 m/s are jitter | distance |
| 16–17 | bunk-zone and water-zone time | tracking + pen layout | table only (feeding is scored) |
| 18 | time to bunk after feed delivery | herd rush + trough-change events | yes |
| 19–21 | nearest-cow distance, cows within 3 m, time at edge or alone | tracking | yes |
| 22 | stand-up / lie-down duration | needs per-second posture | not computed |
| 23–26 | head down while standing and not eating, head height, ear droop, arched back | VLM labels on close daylight gallery crops (2 per cow per hour), with the bunk position overriding "not feeding" | yes, as sampled shares |
| 27–28 | breathing rate and effort | 30 s windows when a cow is still and close: FFT of flank intensity after a 4 s high-pass | yes; weight 0.5 / 0.3 until checked by eye |
| — | discharge | VLM "yes", or VLM "unclear" plus a streak in the muzzle box found by the image check (light / yellow-green elongated blobs below the nostrils); a VLM "no" is never overruled | yes, as a sampled share; often not assessable from overhead views |

## Order of work for the 48 h data

1. **Tracking pass**, in HPRC chunks: `main.py --fps 1`. Assets are on by default and written to `assets_raw/`.
2. **Level 1** on the merged records: identity, world state and behavior, giving `activity.json`.
3. **`scripts/build_risk_assets.py --raw <tracking run> --run <level1 run> --head-signs`**: assigns cow IDs; computes breathing, sudden movements, feed events and clips; labels head signs.
4. **`scripts/run_level2.py --runs <level1 run> --activity activity.json`**: features, score, agent, cards.
