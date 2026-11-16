# Level 3: live early warning, example runs

Saved outputs of `live_watch.py` (code: `live_watch.py`, `src/live/`; design, speed and limits: `docs/LEVEL3_LIVE.md`).
Everything opens in a browser from disk; nothing needs to run. The frames come from the
hackathon's event video.

**Start with `report.html` in each folder.** It shows every decision with its evidence. At the top: **the alerts from the 48 h BRD risk** (the morning list: each cow's tier, the rule that put her there, Level 2's reasons and clips). Then for each window:
* the alerts: the pen-rider action, the 12 s clip cropped to the cow, the full frame, the sign crops and the reason;
* **every cow**: a close photo, how her id was found, her numbers vs the herd in the same window (feeding, at the bunk,
  lying, alone, walked, with z-scores), her risk before → after, any visible signs, and the decision (alert or not)
  with its reason in words;
* **every close crop the vision model checked**: what the VLM saw (ears, head carriage, discharge), the tracker's
  speed at that second, the herdmate control crop, and the outcome (nothing flagged / flag rejected by measurement /
  second look did not confirm / confirmed).

| Folder | What it shows |
|---|---|
| `clip_1600/` | **One 5-min clip end to end**, the way the judges' clip runs. The three cameras' files for 11-07 16:00–16:05 (`videos/day_1/`) went through `run_pipeline.py` (tracking seeded from the last Day 2 chunk's appearance gallery, identity, behavior, close crops). `live_watch.py --run` then scored the window (`--clip <file>` runs both steps in one command). 7 cows were mapped to the 48 h ids; the VLM checked 6 close crops. Three "head low" flags were rejected by measurement (the cows were walking at 0.36–0.87 m/s) and a "nasal discharge" flag by the second look ("the muzzle is blurry"). No new alert was warranted, so the window shows **fresh 12 s clips and frames of the cows already on the check list** (COW_02 and COW_01), cut from the raw video. |
| `day2_replay/` | **Three consecutive hours of Day 2** (11-08 12:28–15:23), fed in time order as if arriving live, with the real VLM sign check (TAMU). It starts with the **morning list**: every cow's tier from the 48 h Level 2 risk, and push alerts with Level 2's clips for COW_02, COW_01 and COW_10. The agent checks itself: two "head low" flags were rejected by measurement (walking at 0.18 and 0.56 m/s), and COW_03's "drooping ears" was not flagged because she was lying and the low ear carries an ear tag (ear carriage is only judged on a standing cow). Behavior alert: COW_10 lay 76% of the hour vs 22% for the herd and was alone 80% vs 36%, so she got a "Watch closely" card. The 1 h chunks were scored after the fact, so their live cards have no clip (the raw Day 2 video is on HPRC); the morning-list cards carry Level 2's clips. |

**Email (simulated).** Every alert and the daily BRD report are also written as real email files (`outbox/*.eml`, standard format: they open in Outlook, Mail or Thunderbird) to placeholder ranch addresses, with the evidence attached. The report's "Email outbox" section shows each one as it would arrive. Nothing is sent: `delivery.email.mode: smtp` in `configs/live.yaml` plus an SMTP login in `.env` would send the same messages.

Also in each folder:
* `index.html`: the live dashboard (risk now vs the 48 h baseline per cow, trend, alert feed)
* `alerts/<cow>_<time>.html` + `.json`: the alert card a pen rider gets
* `live.log`: every decision as it happened; `risk_history.csv`: each cow's risk before and after each window
* `windows/<time>/`: `ids.csv`, `features.csv`, `decisions.csv`, `signs.csv` and the crops the VLM saw
