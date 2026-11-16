# Level 3: live early warning

`live_watch.py` turns the pipeline into a pen monitor: it takes new video, updates every cow's BRD risk, and sends
an alert with video evidence that a pen rider can act on. Settings: `configs/live.yaml`.

## Run it

```bash
# the judges' clip: any one camera's file of a 5-min window; the other cameras' files for the same 5 min are found next to it
python live_watch.py --clip "<videos>/Day 2/Cam 01/Event20261108142130006.Avi"

# ranch mode: watch the folder the DVR writes to; every completed 5-min window is processed as it lands
python live_watch.py --watch "<videos>"            # --poll 30 (s), --once for a single pass

# re-score windows that already went through run_pipeline.py (e.g. the 48 h chunks), in time order
python live_watch.py --run <run1> <run2> ... [--video-root <videos>]
```

Open `outputs/live/report.html` (every decision with its evidence) or `outputs/live/index.html` (dashboard, refreshes every 30 s). Each alert is `outputs/live/alerts/<cow>_<time>.html` with
its clip, frame and crops, plus a `.json` copy. Every alert and the daily BRD report are also written as real emails (`outbox/*.eml`, attachments included) to the addresses in `delivery.email` (placeholders by default). `mode: simulate` never sends; `mode: smtp` with `SMTP_HOST/PORT/USER/PASSWORD` in `.env` sends the same messages. Set `LIVE_WEBHOOK_URL` to also post alerts to Slack, Teams or ntfy. In every window, cows already on the check list get a fresh 12 s clip and frame even when nothing new happens.
Clinical signs need `TAMU_API_KEY` in `.env`. Without it, the alert says so and the signs are not checked; it never
reports a sign it did not look for. `--reset` starts again from the 48 h baseline.

## Alert tiers: from the 48 h BRD risk, then live

On a fresh start (or with `--baseline`, which can also run without video) every cow gets a tier from the Level 2
result (`brd_risk.csv`). Push tiers get an alert card with Level 2's top reasons and its 2 video clips:

| Tier | Rule | Push? | Pen-rider action |
|---|---|---|---|
| Pull and examine | a confirmed visible sign (Level 2 `signs_present`, or a live confirmed sign) | yes | pull, take a temperature, treat by protocol |
| Check on the next pen ride | rank ≤ 3, top 3 in ≥ 60% of time-block resamples, confidence medium/high (live: risk ≥ 0.25 and top 3) | yes | look at her at the bunk: appetite, dullness, nose, eyes, ears, breathing |
| Watch list | rank ≤ 5 or top 3 in ≥ 20% of resamples (live: rank ≤ 5 or risk ≥ 0.18) | dashboard only | — |
| Normal | everyone else | — | — |

On the final 48 h result this puts COW_02, COW_01 and COW_10 on "Check on the next pen ride". Each live window
re-evaluates every tier. Moving up into a push tier is an alert; moving down is logged with its reason.

## What happens to each 5-min window

```
3 camera clips ─► run_pipeline.py (stage 1: detect, track, identity, behavior, close crops)     [--verify: + stage 2 BRD agent]
              ─► ids: the 48 h COW_xx (tracking seeded from the last Day 2 chunk's appearance gallery → stitched id map)
              ─► behavior vs the herd IN THE SAME WINDOW: feeding, at the bunk, lying, alone (> 5 m), metres walked
              ─► visible clinical signs on close crops: VLM "pose sensor" (ears, head carriage, discharge)
                  + a pixel check for discharge in the muzzle box
              ─► agent self-check: every flagged crop gets a second, stricter look next to a herdmate crop from the
                  same minute; only confirmed signs count
              ─► risk update: the 48 h Level 2 score is the prior; window evidence moves it (EWMA);
                  confirmed signs add at once (discharge +0.25, ears +0.15, head low +0.10)
              ─► alert? new confirmed sign → "Check now"; risk ≥ 0.45 and rising, or 2 strong deviations
                  while in the top 3 → "Watch closely"; 2 h cooldown per cow unless a new sign appears
              ─► card: what to do, the 12 s clip cropped to the cow, where she is in the pen, the sign crops next to
                  the control crop, why (each indicator vs the herd), risk now / before / 48 h, identity route
```

Every step writes its reason: `live.log`, `windows.jsonl`, `risk_history.csv` (each cow's risk before and after each
window, with the evidence), `windows/<time>/{ids,features,signs}.csv`, and `state.json`.

## Why it is built this way

* **Time-matched herd reference.** A cow is compared with the other cows in the same 5 minutes, so feed delivery,
  weather and time of day cancel out. A short window says little alone, so the 48 h Level 2 score is the prior and
  each window only moves it (EWMA 0.35). One odd window cannot make an alert. A run of them can.
* **Signs are judged like a pen rider would.** `head_low` only counts for a cow the tracker measures as standing still, and ear carriage only for a standing cow (a lying, cud-chewing cow's relaxed ear, or an ear pulled down by its tag, is not droop). Every rejected flag is shown in the report with its reason.
* **Signs need two looks.** The screening VLM is tuned to flag. The confirming look must see the sign next to a
  herdmate under the same light, and it is told the usual false alarms: feed crumbs, glare, IR reflections, ears
  back while eating, head down while eating.
* **Identity.** Appearance alone cannot tell black Angus apart (on Day 2, prototype matching got 1 of 8 cows right).
  Live tracking therefore starts from the appearance gallery of the last 48 h chunk, so its ids carry over, and the
  stitched id map turns them into the 48 h COW_xx. The card states which route each id took. A cow that matches
  nothing gets a `LIVE_` id instead of a guessed one.
* **Producer-ready.** One command and one config. It runs on the clips the DVR already writes (no new hardware) and
  resumes from `state.json` after a restart. Outputs are plain files plus an optional webhook. The action line uses
  pen-rider language (DART signs, take a temperature, pull by your BRD protocol).

## Speed (honest numbers)

A window is 5 min × 3 cameras of 4K HEVC at 1 frame/s. On HPRC (16 CPU cores), stage 1 tracking runs at about real
time: 1 h of footage in 70 min, then 35–55 min for identity and behavior. A 5-min window is therefore about 10 min
end to end on CPU, and much less on a GPU. On a laptop CPU it is slower. `--fps 0.5` halves tracking time, at a
cost in id accuracy. For a ranch, one GPU box keeps up with 3 cameras in real time. For the judges' clip, expect
minutes, not seconds.

## What it can and cannot claim

* **Observed:** positions, zones, who is near whom (from the boxes); the clip and crops on the card.
* **Estimated:** behavior labels (video model), the risk index, the VLM's sign labels (screening, unvalidated: no
  labelled sick animals).
* **Not possible from video:** temperature, lung sounds, appetite in kg. The alert asks the pen rider to check these.
* Signs are checked on close daylight crops only. At night the alert relies on behavior and says so.
* Risk is a ranking index (0–1), not a probability of disease.
