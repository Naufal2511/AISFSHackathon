# Pen 6 media kit: clips, pictures, data and agent traces

Everything here comes from our real pipeline outputs (Merck–TAMU AISFS Hackathon 2026, Pen 6, 10 Angus cattle, 3 cameras, 7–8 Nov 2026, 48 h).

> **Event data rule:** this kit contains pen footage and frames. Keep it within the team and the event. Do not post it publicly.

## What's where

| Folder | Contents | Use it for |
|---|---|---|
| `clips/annotated/` | Real footage with our boxes, cow IDs (one colour per cow, the same in every camera), live behavior labels and the top-down world-state map, all synced | The "this is the system running" shots |
| `clips/annotated/feed_rush_panels/` | The same feed rush cut into single panels: front, bunk (top) and back camera, plus the map and per-cow table | Picture-in-picture, split screens |
| `clips/animations/` | 15 motion-graphics scenes from our draft cut (no voice), 1080p | Explainers: pipeline, world state, where to look, identity, memory cards, agent trace, results, stitching, risk ranking |
| `clips/raw_footage/` | 10 s clean clips, no overlays: day feed rush (8 Nov 16:28) and infrared night (8 Nov 02:01), each camera | Cold open, B-roll |
| `clips/risk_evidence/` | The risk agent's evidence clips for the top 3 cows (`COW_02_1.mp4` …) | Level 2 scene |
| `pictures/id_check_sheets/<chunk>/` | All 97 real 30-minute ID-check sheets, per chunk, with `legend.csv` (chunk ID → global ID) | Identity scene |
| `pictures/risk/` | Risk cards (PNG) for COW_02, COW_01, COW_10, their indicator charts, and the ranking page | Level 2 scene |
| `pictures/risk_agent_evidence_images/` | Every image the risk agent was shown (`I####`) | Close-ups |
| `pictures/memory_cards/` | Enrolment sheet, reference crops per cow, `cards.json` (ear tags, traits, facts with time and source) | Memory-card scene |
| `pictures/behavior_label_crops/`, `best_view_crops_8nov/` | Crops from our labelling set; one cow from all three cameras at the same second | Behavior / where-to-look |
| `pictures/animation_stills/` | One 1080p still per animation | Thumbnails, storyboard |
| `pictures/architecture_diagram.png` | The pipeline diagram from the technical brief | Pipeline scene |
| `traces/behavior_agent/` | The three required traces: **success** (stage 1 said drinking 0.69 → agent verified walking 0.96), **failure** (weak "standing" 0.23 the agent could not settle → kept as unverified), **night** (IR: "ruminating" → not visible). `README.md` maps file IDs to stitched IDs | Agent scene |
| `traces/identity_agent/` | Two **successful relabels** (camera conflict, ID jump), a **refusal** (a duplicate COW_05 box, similarity 0.204, the ID is not forced), a **night** confirm (01:44), and a **30-min audit**. Each has its images and a legend. | Identity / agent scenes |
| `traces/risk_agent/` | Per-cow risk-agent traces (Claude Opus 4.8) and `evidence_facts.jsonl` (every number, image and observation it cited) | Level 2 scene |
| `data/level1/` | `activity.csv` (4,858 bouts, every cow and minute, with a `basis` column), `activity_minute.csv`, `time_budgets.csv` | On-screen tables |
| `data/level2/` | `brd_risk.csv`, indicators, contributions, hourly features | Ranking charts |
| `data/stitching/` | `id_map.csv` (290 cross-chunk joins and how each was made), `stitch_manifest.json` (33 chunks) | Stitching scene |
| `data/animation_windows/` | Per-second cow state and boxes for the two animated windows (`win_feed` = 8 Nov 16:28–16:30; `win_world` = 7 Nov 15:58–16:02) | Re-animating |
| `docs/` | Technical brief (PDF) and our draft video script | Wording, numbers |

## Numbers you can put on screen (all from these files)

- **48 h, 3 cameras, 10 cows; 28,800 cow-minutes.** Basis: observed 16,963 (59 %) · agent-verified 509 · unverified 2,142 · estimated 441 · not visible 8,745 (30 %).
- **Behavior agent:** 1,815 minutes confirmed on video, 335 corrected, 5,003 abstained rather than guess.
- **Where to look:** 17,385 cow-minutes (60 %) passed cheap rule checks and needed no model call. The agent looked at 4,633 (16 %).
- **Detector** (YOLO26-m fine-tuned on 216 hand-labelled frames): precision / recall 0.84 / 0.88 by day and 0.93 / 0.88 at night. Off-the-shelf COCO recall was 0.71 / 0.73.
- **Behavior classifier** (DINOv2-S): ~330 hand-labelled 5 s clips plus 9,269 CBVD-5 crops; accuracy 0.69 on held-out clips; lying F1 0.95, feeding 1.00.
- **Stitching:** 33 chunks (Day 1 on three laptops, Day 2 on TAMU HPRC). 204 of 290 joins came from identical boxes in 2-min overlaps; 73 are weak.
- **BRD risk:** COW_02 0.37 (top 3 in 92 % of resamples) · COW_01 0.33 (67 %) · COW_10 0.26 (88 %). A herd-typical cow scores ≈ 0.10.
  - COW_02 takes 27 vs 19 min to reach the bunk after 14 feed rushes, walks less (88 vs 101 m/h) and feeds less (7 vs 10 %).
  - Conclusion: **early suspects, not sick.** No visible clinical sign was confirmed. Next step: take temperatures.

## Caveats to keep the video honest

- **Two ID numberings:** identity traces and ID-check sheets use each chunk's own IDs; the `legend.csv` files map them to the global IDs in `activity.csv`. The behavior traces' file names also use chunk IDs (see `traces/behavior_agent/README.md`).
- **Rumination is over-called:** our classifier labels much lying as "ruminating", and we say so in the brief. In voice-over, prefer "lying".
- **No global anchor in this data:** no second has all 10 cows ≥ 2.5 m apart. The identity animation shows cows locked one at a time when each stands clear.
- **Illustrative parts of the animations:** the consensus bars start from example beliefs, though the update rule is the real one, and the chewing wave is an illustration. Both are labelled on screen.

**Palette:** teal `#0f5a6b` (system), orange `#a85a24` (agents and the focus cow), off-white `#fbfcfd`. **Font:** Avenir Next.
